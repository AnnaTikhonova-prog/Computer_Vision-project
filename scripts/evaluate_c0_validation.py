#!/usr/bin/env python3
"""Re-evaluate the nine authoritative C0 checkpoints on validation only."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_c0_grid import (  # noqa: E402
    LRS, SEEDS, expected_config, file_sha256, validate_completed_run,
)
from src.dataset import PetManifestDataset, _eval_transform  # noqa: E402


LR_NAMES = {1e-3: "1e-3", 1e-4: "1e-4", 1e-5: "1e-5"}
SCORE_TOLERANCE = 1e-12
PREDICTION_FIELDS = (
    "image_id", "relative_path", "true_label", "true_class",
    "predicted_label", "predicted_class",
)
METRIC_FIELDS = (
    "lr", "training_seed", "selected_epoch", "saved_val_macro_f1",
    "evaluated_val_macro_f1", "absolute_difference", "within_tolerance",
    "val_accuracy", "val_loss", "validation_examples", "checkpoint_sha256",
    "checkpoint_path", "predictions_path",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_inside(path: Path, parent: Path) -> Path:
    resolved, root = path.resolve(), parent.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes expected directory {root}: {resolved}")
    return resolved


def authoritative_runs(batch_dir: Path, config_path: Path) -> list[dict[str, Any]]:
    """Resolve grid cells from completed orchestrator records, never directory age."""
    manifest_path = batch_dir / "batch_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing orchestrator manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("Batch manifest is not completed")
    provenance = manifest.get("provenance", {})
    expected_grid = {
        "condition": "C0", "learning_rates": list(LRS), "training_seeds": list(SEEDS),
    }
    if provenance.get("grid") != expected_grid:
        raise ValueError("Batch manifest grid provenance is not the required C0 grid")
    for recorded_name, recorded_hash in provenance.get("files_sha256", {}).items():
        local_path = PROJECT_ROOT / Path(recorded_name.replace("\\", "/"))
        if not local_path.is_file() or file_sha256(local_path) != recorded_hash:
            raise ValueError(f"Current provenance input differs from the batch: {recorded_name}")

    found: dict[tuple[float, int], list[Path]] = {
        (lr, seed): [] for lr in LRS for seed in SEEDS
    }
    for attempt in manifest.get("attempts", []):
        if attempt.get("status") != "completed":
            continue
        key = (float(attempt.get("lr", math.nan)), attempt.get("training_seed"))
        if key not in found:
            raise ValueError(f"Unexpected completed attempt in manifest: {key}")
        run_dir = _resolved_inside(Path(attempt["output_dir"]), batch_dir / "runs")
        found[key].append(run_dir)
    missing = [key for key, paths in found.items() if not paths]
    duplicates = [key for key, paths in found.items() if len(paths) != 1]
    if missing or duplicates:
        raise ValueError(
            f"Orchestrator records are incomplete or ambiguous: missing={missing}, duplicates={duplicates}"
        )

    rows = []
    for lr in LRS:
        for seed in SEEDS:
            run_dir = found[(lr, seed)][0]
            row = validate_completed_run(run_dir, expected_config(config_path, lr, seed))
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def _csv_text(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_text_idempotent(path: Path, content: str) -> None:
    """Create a report, accept an identical report, reject a collision."""
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite conflicting evaluator output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def _load_model(checkpoint: dict[str, Any]) -> nn.Module:
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_run(
    row: dict[str, Any], dataset: PetManifestDataset, loader: DataLoader,
    class_mapping_payload: dict[str, Any], output_dir: Path,
) -> dict[str, Any]:
    checkpoint_path = Path(row["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_mapping") != class_mapping_payload:
        raise ValueError(f"Checkpoint class mapping differs: {checkpoint_path}")
    model = _load_model(checkpoint)
    predictions: list[int] = []
    targets: list[int] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            predictions.extend(logits.argmax(dim=1).tolist())
            targets.extend(labels.tolist())

    if len(targets) != len(dataset):
        raise RuntimeError("Validation loader did not return every manifest row exactly once")
    macro_f1 = float(f1_score(
        targets, predictions, labels=list(range(10)), average="macro", zero_division=0,
    ))
    accuracy = sum(left == right for left, right in zip(targets, predictions)) / len(targets)
    difference = abs(macro_f1 - row["selected_val_macro_f1"])
    seed = row["training_seed"]
    prediction_path = output_dir / f"seed_{seed}_predictions.csv"
    prediction_rows = []
    for manifest_row, target, prediction in zip(dataset.rows, targets, predictions):
        prediction_rows.append({
            "image_id": manifest_row["image_id"],
            "relative_path": manifest_row["relative_path"],
            "true_label": target,
            "true_class": dataset.mapping[target],
            "predicted_label": prediction,
            "predicted_class": dataset.mapping[prediction],
        })
    write_text_idempotent(prediction_path, _csv_text(prediction_rows, PREDICTION_FIELDS))
    return {
        "lr": row["lr"], "training_seed": seed, "selected_epoch": row["selected_epoch"],
        "saved_val_macro_f1": format(row["selected_val_macro_f1"], ".17g"),
        "evaluated_val_macro_f1": format(macro_f1, ".17g"),
        "absolute_difference": format(difference, ".17g"),
        "within_tolerance": difference <= SCORE_TOLERANCE,
        "val_accuracy": format(accuracy, ".17g"),
        "val_loss": format(loss_sum / len(targets), ".17g"),
        "validation_examples": len(targets), "checkpoint_sha256": row["checkpoint_sha256"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "predictions_path": str(prediction_path.resolve()),
    }


def evaluate_grid(batch_dir: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    rows = authoritative_runs(batch_dir, config_path)
    base_config = expected_config(config_path, LRS[0], SEEDS[0])
    split_dir = PROJECT_ROOT / base_config["split_dir"]
    dataset = PetManifestDataset(
        PROJECT_ROOT / base_config["dataset_root"], split_dir / "val.csv",
        split_dir / "class_mapping.json", _eval_transform(),
    )
    loader = DataLoader(
        dataset, batch_size=base_config["batch_size"], shuffle=False, drop_last=False,
        num_workers=base_config["num_workers"],
    )
    mapping_payload = _read_json(split_dir / "class_mapping.json")
    metrics: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for lr in LRS:
        lr_dir = output_root / f"lr_{LR_NAMES[lr]}"
        lr_metrics = [
            evaluate_run(row, dataset, loader, mapping_payload, lr_dir)
            for row in rows if row["lr"] == lr
        ]
        metrics.extend(lr_metrics)
        write_text_idempotent(lr_dir / "metrics.csv", _csv_text(lr_metrics, METRIC_FIELDS))
        scores = [float(item["evaluated_val_macro_f1"]) for item in lr_metrics]
        aggregate = {
            "lr": lr,
            **{f"seed_{seed}_val_macro_f1": format(score, ".17g")
               for seed, score in zip(SEEDS, scores)},
            "mean_val_macro_f1": format(statistics.fmean(scores), ".17g"),
            "sample_std_val_macro_f1": format(statistics.stdev(scores), ".17g"),
            "contributing_seeds": ",".join(map(str, SEEDS)),
            "metric_kind": "mean/std of seed-level macro-F1; not pooled or ensemble",
        }
        aggregates.append(aggregate)
        write_text_idempotent(
            lr_dir / "aggregate.csv", _csv_text([aggregate], tuple(aggregate))
        )

    winner = min(LRS, key=lambda lr: (-float(next(
        item["mean_val_macro_f1"] for item in aggregates if item["lr"] == lr
    )), lr))
    for item in aggregates:
        item["selected"] = item["lr"] == winner
    aggregate_fields = (
        "lr", *(f"seed_{seed}_val_macro_f1" for seed in SEEDS),
        "mean_val_macro_f1", "sample_std_val_macro_f1", "contributing_seeds",
        "metric_kind", "selected",
    )
    write_text_idempotent(
        output_root / "learning_rates.csv", _csv_text(aggregates, aggregate_fields)
    )
    mapping_rows = [{
        "lr": lr,
        "report_directory": str((output_root / f"lr_{LR_NAMES[lr]}").resolve()),
        "seeds": ",".join(map(str, SEEDS)),
        "checkpoint_sha256s": ",".join(
            row["checkpoint_sha256"] for row in rows if row["lr"] == lr
        ),
    } for lr in LRS]
    write_text_idempotent(
        output_root / "report_mapping.csv",
        _csv_text(mapping_rows, ("lr", "report_directory", "seeds", "checkpoint_sha256s")),
    )
    mismatches = [item for item in metrics if item["within_tolerance"] is False]
    return {"runs": len(metrics), "selected_lr": winner, "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "c0_cpu_grid")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "c0_cpu.yaml")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "c0_cpu_grid" / "validation_evaluation",
    )
    args = parser.parse_args()
    result = evaluate_grid(args.batch_dir.resolve(), args.config.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "status": "ok", "runs": result["runs"], "selected_lr": result["selected_lr"],
        "metric_tolerance": SCORE_TOLERANCE, "mismatch_count": len(result["mismatches"]),
        "output_dir": str(args.output_dir.resolve()), "test_split_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
