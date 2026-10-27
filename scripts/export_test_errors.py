#!/usr/bin/env python3
"""Export every saved EVAL-1 test error, grouped by condition and breed pair.

Leakage note: these are read-only, post-evaluation diagnostic artifacts.
Never use them to select models, checkpoints, preprocessing, or parameters.
This script reads predictions and frozen manifests; it never runs a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TABLE_NAME = re.compile(r"(C[0-3])_augment_(on|off)_test_predictions\.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def display_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_errors(tables: list[Path], dataset_root: Path, split_dir: Path, output: Path) -> None:
    mapping = json.loads((split_dir / "class_mapping.json").read_text(encoding="utf-8"))
    names = {str(item["label"]): item["class_name"] for item in mapping["classes"]}
    manifest = {row["image_id"]: row for row in read_csv(split_dir / "test.csv")}
    dataset_root, output = dataset_root.resolve(), output.resolve()
    records, copies, seen_conditions = [], {}, set()
    groups = defaultdict(list)
    # Validate every table and source before copying anything. Fail on incomplete
    # evaluations rather than silently producing an incomplete "all errors" export.
    for table in sorted(tables):
        match = TABLE_NAME.fullmatch(table.name)
        if match is None:
            raise ValueError(f"Expected a condition-level test prediction table: {table}")
        condition, augment = match.groups()
        group = f"{condition}_augment_{augment}"
        if group in seen_conditions:
            raise ValueError(f"Duplicate condition/augmentation table: {group}")
        seen_conditions.add(group)
        rows, seen = read_csv(table), set()
        for row in rows:
            image_id, seed = row["image_id"], row["training_seed"]
            key = seed, image_id
            if key in seen or seed not in {"42", "43", "44"} or image_id not in manifest:
                raise ValueError(f"Invalid or duplicate seed/test image: {table}: {key}")
            seen.add(key)
            original = manifest[image_id]
            if (row["relative_path"], row["true_label"], row["true_class"]) != (
                original["relative_path"], original["label"], original["class_name"]
            ):
                raise ValueError(f"Prediction disagrees with frozen test manifest: {image_id}")
            if names.get(row["predicted_label"]) != row["predicted_class"]:
                raise ValueError(f"Predicted label/name mismatch: {image_id}")
            correct = row["true_label"] == row["predicted_label"]
            confidence = float(row["predicted_class_confidence"])
            if row["correct"] != str(int(correct)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"Invalid correctness/confidence: {image_id}")
            if correct:
                continue
            relative = Path(row["relative_path"])
            source = (dataset_root / relative).resolve()
            if relative.is_absolute() or not source.is_relative_to(dataset_root):
                raise ValueError(f"Unsafe source path: {relative}")
            pair = f"{slug(row['true_class'])}__as__{slug(row['predicted_class'])}"
            destination = output / group / pair / source.name
            if destination not in copies:
                with Image.open(source) as image:
                    rgb = image.convert("RGB")
                    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}\0".encode("ascii") + rgb.tobytes()).hexdigest()
                if digest != original["content_hash"]:
                    raise ValueError(f"Source image differs from frozen manifest: {source}")
                file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != file_hash:
                    raise ValueError(f"Refusing to overwrite a different existing image: {destination}")
                copies[destination] = source, file_hash
            source_info = copies[destination]
            if source_info[0] != source:
                raise ValueError(f"Export filename collision: {destination}")
            record = {
                "condition": condition, "augmentation": augment,
                **{key: row[key] for key in (
                    "training_seed", "checkpoint", "image_id", "relative_path", "true_label", "true_class",
                    "predicted_label", "predicted_class", "predicted_class_confidence", "correct",
                )},
                "prediction_table": display_path(table),
                "exported_path": display_path(destination), "source_image_sha256": source_info[1],
            }
            records.append(record)
            groups[group, pair].append(record)
        expected = {(seed, image_id) for seed in ("42", "43", "44") for image_id in manifest}
        if seen != expected:
            raise ValueError(f"Incomplete test predictions for seeds 42/43/44: {table}")

    output.mkdir(parents=True, exist_ok=True)
    for destination, (source, _) in copies.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
    fields = ["condition", "augmentation", "training_seed", "checkpoint", "image_id", "relative_path",
              "true_label", "true_class", "predicted_label", "predicted_class", "predicted_class_confidence",
              "correct", "prediction_table", "exported_path", "source_image_sha256"]
    records.sort(key=lambda row: (row["condition"], row["augmentation"], row["true_class"],
                                  row["predicted_class"], row["image_id"], row["training_seed"]))
    write_csv(output / "index.csv", fields, records)
    summaries = []
    for (group, pair), errors in sorted(groups.items()):
        summaries.append({"condition": errors[0]["condition"], "augmentation": errors[0]["augmentation"],
                          "pair": pair, "incorrect_predictions": len(errors),
                          "unique_images": len({r["image_id"] for r in errors}),
                          **{f"seed_{seed}_errors": sum(r["training_seed"] == seed for r in errors)
                             for seed in ("42", "43", "44")}})
    write_csv(output / "summary.csv", ["condition", "augmentation", "pair", "incorrect_predictions",
              "unique_images", "seed_42_errors", "seed_43_errors", "seed_44_errors"], summaries)
    for group in sorted(seen_conditions):
        errors = [r for r in records if f"{r['condition']}_augment_{r['augmentation']}" == group]
        print(f"{group}: {len(errors)} incorrect predictions; "
              f"{len({r['image_id'] for r in errors})} distinct images; "
              f"{len({r['exported_path'] for r in errors})} exported files")
    print(f"Total: {len(records)} incorrect predictions, {len(copies)} image copies, {len(groups)} pair folders")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", help="Default: all EVAL-1 condition-level test tables")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/oxford-iiit-pet")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits/v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/errors")
    args = parser.parse_args()
    tables = args.predictions if args.predictions else sorted((ROOT / "reports/tables").glob("C*_augment_*_test_predictions.csv"))
    if not tables:
        parser.error("No saved EVAL-1 test prediction tables found")
    export_errors(tables, args.dataset_root, args.split_dir, args.output_dir)


if __name__ == "__main__":
    main()
