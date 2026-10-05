#!/usr/bin/env python3
"""Download (optionally), validate, de-duplicate and freeze Pet CSV manifests.

The command consumes the official ``annotations/trainval.txt`` and
``annotations/test.txt`` together; its 70/15/15 partition is therefore a
custom split, not the Oxford-IIIT Pet benchmark partition.  It never mutates
an existing manifest. Use ``--download-only`` to fetch the official archives
without reading or writing any manifests. Use a new ``--output-dir`` when
creating a new version.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "oxford-iiit-pet"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "splits"
MAPPING_SOURCE = PROJECT_ROOT / "splits" / "class_mapping.json"
REQUIRED_COLUMNS = ["image_id", "relative_path", "class_name", "label", "content_hash"]
ARCHIVES = {
    "images.tar.gz": "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
    "annotations.tar.gz": "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
}


def load_mapping(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, list):
        raise ValueError(f"{path} has no 'classes' list")
    labels = [entry.get("label") for entry in classes]
    if labels != list(range(10)):
        raise ValueError(f"{path} labels must be exactly the contiguous range 0..9")
    by_official = {entry["official_class_name"]: entry for entry in classes}
    if len(by_official) != 10:
        raise ValueError(f"{path} must contain 10 unique official class names")
    return by_official, {entry["class_name"]: entry for entry in classes}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"Unsafe path in archive {archive}: {member.name}")
        tar.extractall(destination)


def download_dataset(root: Path) -> None:
    """Fetch the two official archives only when --download was requested."""
    root.mkdir(parents=True, exist_ok=True)
    for filename, url in ARCHIVES.items():
        archive = root / filename
        valid_archive = False
        if archive.exists():
            try:
                # ``is_tarfile`` alone accepts a truncated gzip stream; walking
                # headers verifies the archive reaches its end-of-stream marker.
                with tarfile.open(archive, "r:gz") as tar:
                    tar.getmembers()
                valid_archive = True
            except (tarfile.TarError, EOFError, OSError):
                archive.unlink()
        if not valid_archive:
            print(f"Downloading {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, archive)
        _safe_extract(archive, root)


def read_official_ids(dataset_root: Path, expected_official: set[str]) -> list[tuple[str, str]]:
    """Return selected (image_id, official_class_name) from both official splits."""
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for filename in ("trainval.txt", "test.txt"):
        path = dataset_root / "annotations" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Official annotation file is missing: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id = line.split()[0]
            if "_" not in image_id:
                raise ValueError(f"Unexpected annotation ID in {path}:{line_number}: {image_id!r}")
            official_name, suffix = image_id.rsplit("_", 1)
            if not suffix.isdigit():
                raise ValueError(f"Unexpected annotation ID in {path}:{line_number}: {image_id!r}")
            if official_name not in expected_official:
                continue
            if image_id in seen:
                raise ValueError(f"Duplicate image_id across official annotations: {image_id}")
            seen.add(image_id)
            records.append((image_id, official_name))
    present = {name for _, name in records}
    missing = sorted(expected_official - present)
    if missing:
        raise ValueError(f"Selected official classes absent from annotations: {missing}")
    return sorted(records)


def decode_record(dataset_root: Path, image_id: str, mapping_entry: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    relative_path = Path("images") / f"{image_id}.jpg"
    image_path = dataset_root / relative_path
    base = {"image_id": image_id, "relative_path": relative_path.as_posix()}
    if not image_path.is_file():
        return None, {**base, "reason": "missing_file"}
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, {**base, "reason": "decode_error", "detail": str(exc)}
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(f"{width}x{height}\0".encode("ascii"))
    digest.update(image.tobytes())
    # A compact average-hash is intentionally only a candidate detector. It is
    # not evidence sufficient to remove a sample.
    thumb = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(thumb.getdata())
    mean = sum(pixels) / len(pixels)
    phash = 0
    for pixel in pixels:
        phash = (phash << 1) | int(pixel >= mean)
    return {
        **base,
        "class_name": mapping_entry["class_name"],
        "label": int(mapping_entry["label"]),
        "content_hash": digest.hexdigest(),
        "width": width,
        "height": height,
        "perceptual_hash": f"{phash:016x}",
    }, None


def candidate_near_duplicates(rows: list[dict[str, Any]], max_distance: int) -> list[dict[str, Any]]:
    """Report, but never remove, visually similar candidate pairs."""
    candidates: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: row["image_id"])
    values = [int(row["perceptual_hash"], 16) for row in ordered]
    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            distance = (values[left_index] ^ values[right_index]).bit_count()
            if distance <= max_distance:
                candidates.append({
                    "image_id_a": left["image_id"],
                    "image_id_b": ordered[right_index]["image_id"],
                    "hamming_distance": distance,
                    "same_class": left["label"] == ordered[right_index]["label"],
                })
    return candidates


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: str, second: str) -> None:
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def load_confirmed_groups(path: Path | None, canonical_by_id: dict[str, str], canonical_ids: set[str]) -> list[list[str]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("confirmed-groups JSON must be an object with a 'groups' list")
    applied: list[list[str]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, list) or len(group) < 2 or not all(isinstance(item, str) for item in group):
            raise ValueError(f"confirmed group {index} must list at least two image IDs")
        unknown = sorted(set(group) - set(canonical_by_id))
        if unknown:
            raise ValueError(f"confirmed group {index} has unknown image IDs: {unknown}")
        canonical = sorted({canonical_by_id[item] for item in group})
        if any(item not in canonical_ids for item in canonical):
            raise AssertionError("Internal canonicalization failure")
        if len(canonical) > 1:
            applied.append(canonical)
    return applied


def make_grouped_split(rows: list[dict[str, Any]], confirmed_groups: list[list[str]], seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Stratify by class while keeping explicitly confirmed groups intact."""
    ids = sorted(row["image_id"] for row in rows)
    by_id = {row["image_id"]: row for row in rows}
    union_find = UnionFind(ids)
    for group in confirmed_groups:
        for member in group[1:]:
            union_find.union(group[0], member)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_id in ids:
        groups[union_find.find(image_id)].append(by_id[image_id])
    for members in groups.values():
        labels = {member["label"] for member in members}
        if len(labels) != 1:
            raise ValueError("A confirmed duplicate group spans classes; resolve it before creating a stratified split.")

    grouped_by_label: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
    for members in groups.values():
        grouped_by_label[members[0]["label"]].append(sorted(members, key=lambda row: row["image_id"]))
    split_names = ("train", "val", "test")
    output = {name: [] for name in split_names}
    per_class: dict[str, Any] = {}
    rng = random.Random(seed)
    for label in range(10):
        class_groups = sorted(grouped_by_label[label], key=lambda group: group[0]["image_id"])
        rng.shuffle(class_groups)
        total = sum(len(group) for group in class_groups)
        targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
        counts = {name: 0 for name in split_names}
        for group in class_groups:
            group_size = len(group)
            # The largest remaining deficit minimises rounding error for normal
            # singleton groups while correctly preserving non-singleton groups.
            destination = max(split_names, key=lambda name: (targets[name] - counts[name], -split_names.index(name)))
            output[destination].extend(group)
            counts[destination] += group_size
        per_class[str(label)] = {
            "total": total,
            "targets": targets,
            "actual": counts,
            "deviation_from_target": {name: counts[name] - targets[name] for name in split_names},
            "group_count": len(class_groups),
        }
    for name in split_names:
        output[name].sort(key=lambda row: row["image_id"])
    return output, {"seed": seed, "custom_split": True, "ratios": {"train": 0.70, "val": 0.15, "test": 0.15}, "per_class": per_class}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in REQUIRED_COLUMNS} for row in rows)


def ensure_new_output(output_dir: Path) -> None:
    protected = [output_dir / filename for filename in (
        "train.csv", "val.csv", "test.csv", "integrity_report.json", "exact_duplicates.json",
        "near_duplicate_candidates.json", "split_report.json", "confirmed_groups_applied.json",
    )]
    exists = [str(path) for path in protected if path.exists()]
    if exists:
        raise FileExistsError(
            "Refusing to overwrite frozen manifests or reports. Choose a new --output-dir. Existing: " + ", ".join(exists)
        )


def prepare(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.download_only:
        download_dataset(dataset_root)
        print(json.dumps({"dataset_root": str(dataset_root), "download_only": True}, indent=2))
        return
    if args.download:
        download_dataset(dataset_root)
    if not MAPPING_SOURCE.is_file():
        raise FileNotFoundError(f"Repository mapping is missing: {MAPPING_SOURCE}")
    official_mapping, _ = load_mapping(MAPPING_SOURCE)
    annotation_records = read_official_ids(dataset_root, set(official_mapping))
    valid_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for image_id, official_name in annotation_records:
        row, error = decode_record(dataset_root, image_id, official_mapping[official_name])
        if row is not None:
            valid_rows.append(row)
        else:
            assert error is not None
            exclusions.append(error)

    if not valid_rows:
        raise RuntimeError("No selected image could be decoded. Download/extract the official images archive before preparation.")

    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        hash_groups[row["content_hash"]].append(row)
    canonical_by_id: dict[str, str] = {}
    canonical_rows: list[dict[str, Any]] = []
    exact_groups: list[dict[str, Any]] = []
    for content_hash, members in sorted(hash_groups.items()):
        members.sort(key=lambda row: row["image_id"])
        canonical = members[0]
        canonical_rows.append(canonical)
        for member in members:
            canonical_by_id[member["image_id"]] = canonical["image_id"]
        if len(members) > 1:
            exact_groups.append({"content_hash": content_hash, "kept_image_id": canonical["image_id"], "removed_image_ids": [row["image_id"] for row in members[1:]]})
    canonical_rows.sort(key=lambda row: row["image_id"])
    missing_labels = sorted(set(range(10)) - {row["label"] for row in canonical_rows})
    if missing_labels:
        raise RuntimeError(f"At least one selected class has no valid retained image after cleanup: labels {missing_labels}")
    for group in exact_groups:
        for image_id in group["removed_image_ids"]:
            removed = next(row for row in valid_rows if row["image_id"] == image_id)
            exclusions.append({"image_id": image_id, "relative_path": removed["relative_path"], "reason": "exact_duplicate", "kept_image_id": group["kept_image_id"]})
    confirmed_groups = load_confirmed_groups(args.confirmed_groups, canonical_by_id, {row["image_id"] for row in canonical_rows})
    splits, split_report = make_grouped_split(canonical_rows, confirmed_groups, args.seed)

    ensure_new_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination_mapping = output_dir / "class_mapping.json"
    if destination_mapping.resolve() != MAPPING_SOURCE.resolve():
        if destination_mapping.exists():
            if destination_mapping.read_bytes() != MAPPING_SOURCE.read_bytes():
                raise FileExistsError(f"Refusing to replace a different existing mapping: {destination_mapping}")
        else:
            shutil.copyfile(MAPPING_SOURCE, destination_mapping)
    for name, rows in splits.items():
        write_csv(output_dir / f"{name}.csv", rows)
    before_counts = Counter(row["class_name"] for row in valid_rows)
    after_counts = Counter(row["class_name"] for row in canonical_rows)
    write_json(output_dir / "integrity_report.json", {
        "annotation_candidates": len(annotation_records), "valid_decoded": len(valid_rows), "retained_after_exact_dedup": len(canonical_rows),
        "counts_before_exact_dedup": dict(sorted(before_counts.items())), "counts_after_exact_dedup": dict(sorted(after_counts.items())),
        "excluded": sorted(exclusions, key=lambda row: row["image_id"]),
    })
    write_json(output_dir / "exact_duplicates.json", {"groups": exact_groups})
    write_json(output_dir / "near_duplicate_candidates.json", {
        "method": "8x8 grayscale average hash", "max_hamming_distance": args.near_duplicate_distance,
        "automatic_removal": False, "candidates": candidate_near_duplicates(canonical_rows, args.near_duplicate_distance),
    })
    split_report["confirmed_groups_applied"] = confirmed_groups
    split_report["split_sizes"] = {name: len(rows) for name, rows in splits.items()}
    write_json(output_dir / "split_report.json", split_report)
    write_json(output_dir / "confirmed_groups_applied.json", {"groups": confirmed_groups})
    print(json.dumps({"output_dir": str(output_dir), "split_sizes": split_report["split_sizes"], "excluded": len(exclusions)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=5)
    parser.add_argument("--confirmed-groups", type=Path, help="JSON {'groups': [[image_id, ...], ...]} approved after manual review")
    parser.add_argument("--download", action="store_true", help="Download official images and annotations before preparation")
    parser.add_argument("--download-only", action="store_true", help="Download official images and annotations without generating manifests")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        prepare(parse_args())
    except Exception as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
