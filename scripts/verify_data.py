#!/usr/bin/env python3
"""Validate frozen Pet manifests and (optionally) the DataLoader contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
REQUIRED_COLUMNS = {"image_id", "relative_path", "class_name", "label", "content_hash"}
EXPECTED_CLASSES = [
    "British Shorthair", "Russian Blue", "Siamese", "Birman", "American Bulldog",
    "Pug", "Sphynx", "Maine Coon", "Beagle", "Samoyed",
]


def content_hash(path: Path) -> str:
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError(f"cannot decode RGB image {path}: {exc}") from exc
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}\0".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def read_mapping(split_dir: Path) -> dict[int, str]:
    path = split_dir / "class_mapping.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes", [])
    labels = [item.get("label") for item in classes]
    names = [item.get("class_name") for item in classes]
    if labels != list(range(10)) or names != EXPECTED_CLASSES:
        raise AssertionError("class_mapping.json is not the fixed contiguous 0..9 project mapping")
    return dict(zip(labels, names))


def read_manifest(path: Path, dataset_root: Path, mapping: dict[int, str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise AssertionError(f"{path} misses required columns {sorted(REQUIRED_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise AssertionError(f"{path} is empty")
    for line, row in enumerate(rows, start=2):
        label = int(row["label"])
        if label not in mapping or row["class_name"] != mapping[label]:
            raise AssertionError(f"Mapping mismatch at {path}:{line}")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError(f"Path is not a safe dataset-relative path at {path}:{line}: {relative}")
        actual = content_hash(dataset_root / relative)
        if actual != row["content_hash"]:
            raise AssertionError(f"Content hash mismatch at {path}:{line}: {relative}")
    return rows


def assert_disjoint(manifests: dict[str, list[dict[str, str]]]) -> None:
    for field in ("image_id", "relative_path", "content_hash"):
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = {row[field] for row in manifests[first]} & {row[field] for row in manifests[second]}
            if overlap:
                example = sorted(overlap)[0]
                raise AssertionError(f"{field} overlap between {first}/{second}: {example}")


def assert_all_classes_present(manifests: dict[str, list[dict[str, str]]]) -> None:
    expected = set(range(10))
    for split, rows in manifests.items():
        present = {int(row["label"]) for row in rows}
        if present != expected:
            raise AssertionError(f"{split} does not contain every class 0..9; missing={sorted(expected - present)}")


def assert_confirmed_groups_do_not_cross(manifests: dict[str, list[dict[str, str]]], groups_path: Path) -> None:
    """Ensure manually confirmed duplicate/near-duplicate groups are split-safe."""
    if not groups_path.is_file():
        raise FileNotFoundError(f"Confirmed-groups file is missing: {groups_path}")
    payload = json.loads(groups_path.read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise AssertionError(f"{groups_path} must contain a 'groups' list")
    locations = {row["image_id"]: split for split, rows in manifests.items() for row in rows}
    for index, group in enumerate(groups):
        if not isinstance(group, list) or len(group) < 2:
            raise AssertionError(f"Invalid confirmed group {index} in {groups_path}")
        unknown = sorted(set(group) - set(locations))
        if unknown:
            raise AssertionError(f"Confirmed group {index} references unknown image IDs: {unknown}")
        locations_in_group = {locations[image_id] for image_id in group}
        if len(locations_in_group) != 1:
            raise AssertionError(f"Confirmed group {index} crosses splits: {group}")


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is needed to check loaders; install requirements.txt") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_loaders(config_path: Path, manifests: dict[str, list[dict[str, str]]], split_dir: Path, dataset_root: Path) -> None:
    import torch
    from src.dataset import get_loaders

    config = load_yaml_config(config_path)
    config["split_dir"] = str(split_dir)
    config["dataset_root"] = str(dataset_root)
    config["num_workers"] = 0  # deterministic, in-process contract check
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in split_dir.glob("*.csv")}
    loaders = get_loaders(config)
    if set(loaders) != {"train", "val", "test"}:
        raise AssertionError("get_loaders must return exactly train, val, test")
    for name, loader in loaders.items():
        if len(loader.dataset) != len(manifests[name]):
            raise AssertionError(f"{name} loader length does not match CSV")
        images, labels = next(iter(loader))
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise AssertionError(f"{name} batch shape is {tuple(images.shape)}, expected [B, 3, 224, 224]")
        if labels.dtype != torch.int64 or labels.ndim != 1 or not set(labels.tolist()).issubset(set(range(10))):
            raise AssertionError(f"{name} labels are not integer labels 0..9")
    deterministic = dict(config)
    deterministic["augment"] = False
    deterministic_loaders = get_loaders(deterministic)
    for name in ("val", "test"):
        first, _ = deterministic_loaders[name].dataset[0]
        second, _ = deterministic_loaders[name].dataset[0]
        if not torch.equal(first, second):
            raise AssertionError(f"{name} preprocessing is not deterministic")
    first, _ = deterministic_loaders["train"].dataset[0]
    second, _ = deterministic_loaders["train"].dataset[0]
    if not torch.equal(first, second):
        raise AssertionError("train preprocessing with augment=false is not deterministic")
    augmented = dict(config)
    augmented["augment"] = True
    augmented_loaders = get_loaders(augmented)
    for name in ("val", "test"):
        left, _ = deterministic_loaders[name].dataset[0]
        right, _ = augmented_loaders[name].dataset[0]
        if not torch.equal(left, right):
            raise AssertionError(f"augment flag changed {name} transforms")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in split_dir.glob("*.csv")}
    if before != after:
        raise AssertionError("get_loaders modified one or more manifests")


def check_reproducibility(dataset_root: Path, split_dir: Path) -> None:
    """Regenerate in a temporary directory and compare the frozen manifests bytewise."""
    with tempfile.TemporaryDirectory(prefix="pet-split-repro-") as temporary:
        generated = Path(temporary) / "generated"
        report = json.loads((split_dir / "split_report.json").read_text(encoding="utf-8"))
        near_report = json.loads((split_dir / "near_duplicate_candidates.json").read_text(encoding="utf-8"))
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_oxford_pets.py"),
            "--dataset-root", str(dataset_root), "--output-dir", str(generated),
            "--seed", str(report["seed"]),
            "--near-duplicate-distance", str(near_report["max_hamming_distance"]),
        ]
        confirmed = split_dir / "confirmed_groups_applied.json"
        if confirmed.is_file() and json.loads(confirmed.read_text(encoding="utf-8")).get("groups"):
            command.extend(["--confirmed-groups", str(confirmed)])
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            raise AssertionError(f"Temporary reproducibility generation failed:\n{result.stderr}")
        for filename in ("train.csv", "val.csv", "test.csv", "class_mapping.json"):
            if (generated / filename).read_bytes() != (split_dir / filename).read_bytes():
                raise AssertionError(f"Reproducibility mismatch in {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "oxford-iiit-pet")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--confirmed-groups", type=Path, help="Confirmed-groups JSON; defaults to <split-dir>/confirmed_groups_applied.json")
    parser.add_argument("--check-loaders", action="store_true")
    parser.add_argument("--check-reproducibility", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = read_mapping(args.split_dir)
    manifests = {name: read_manifest(args.split_dir / f"{name}.csv", args.dataset_root, mapping) for name in ("train", "val", "test")}
    assert_disjoint(manifests)
    assert_all_classes_present(manifests)
    assert_confirmed_groups_do_not_cross(manifests, args.confirmed_groups or args.split_dir / "confirmed_groups_applied.json")
    if args.check_loaders:
        check_loaders(args.config, manifests, args.split_dir, args.dataset_root)
    if args.check_reproducibility:
        check_reproducibility(args.dataset_root, args.split_dir)
    print(json.dumps({"status": "ok", "split_sizes": {name: len(rows) for name, rows in manifests.items()}, "loader_checks": args.check_loaders, "reproducibility_check": args.check_reproducibility}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
