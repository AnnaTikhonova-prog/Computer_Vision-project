#!/usr/bin/env python3
"""Verify frozen result arithmetic and optionally repeat checkpoint inference.

This is read-only diagnostic work: no settings are changed based on test
results. Outputs go to a separate audit file; canonical metrics are not edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = (("C0", "on", "1e-3", "head"), ("C1", "on", "1e-4", "layer4"),
         ("C2", "on", "1e-5", "layer3_layer4"), ("C3", "on", "1e-5", "all"),
         ("C0", "off", "1e-3", "head"), ("C3", "off", "1e-5", "all"))
SEEDS = (42, 43, 44)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_equal(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{context}: {actual} != {expected}")


def summarize_predictions(rows: list[dict[str, str]], manifest: dict, mapping: dict) -> dict:
    """Recompute accuracy/F1 independently from the saved evaluator summaries."""
    seen, matrices = set(), {seed: [[0] * 10 for _ in range(10)] for seed in SEEDS}
    for row in rows:
        seed, image_id = int(row["training_seed"]), row["image_id"]
        key = seed, image_id
        if seed not in SEEDS or image_id not in manifest or key in seen:
            raise ValueError(f"Invalid/duplicate test prediction: {key}")
        seen.add(key)
        original = manifest[image_id]
        for field in ("relative_path", "class_name", "label"):
            prediction_field = {"class_name": "true_class", "label": "true_label"}.get(field, field)
            if row[prediction_field] != original[field]:
                raise ValueError(f"Test manifest mismatch: {key}: {field}")
        target, predicted = int(row["true_label"]), int(row["predicted_label"])
        if mapping.get(predicted) != row["predicted_class"] or mapping.get(target) != row["true_class"]:
            raise ValueError(f"Class mapping mismatch: {key}")
        confidence = float(row["predicted_class_confidence"])
        if row["correct"] != str(int(target == predicted)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid correctness/confidence: {key}")
        matrices[seed][target][predicted] += 1
    if seen != {(seed, image_id) for seed in SEEDS for image_id in manifest}:
        raise ValueError("Incomplete test prediction coverage for seeds 42/43/44")
    result = {}
    for seed, matrix in matrices.items():
        per_class = []
        for label in range(10):
            denominator = sum(matrix[label]) + sum(row[label] for row in matrix)
            per_class.append(2 * matrix[label][label] / denominator if denominator else 0.0)
        result[seed] = {"accuracy": sum(matrix[i][i] for i in range(10)) / len(manifest),
                        "macro_f1": statistics.mean(per_class), "per_class_f1": per_class}
    return result


def audit(inference: str, root: Path = ROOT) -> dict:
    manifest = {row["image_id"]: row for row in read_csv(root / "splits/v2/test.csv")}
    mapping = {item["label"]: item["class_name"] for item in
               json.loads((root / "splits/v2/class_mapping.json").read_text(encoding="utf-8"))["classes"]}
    report = (root / "reports/results.md").read_text(encoding="utf-8")
    recorded_runtime = json.loads((root / "reports/tables/latency_rerun_2026-11-02_runtime.json").read_text(encoding="utf-8"))
    recorded_hashes = {item["path"]: item["sha256"] for item in recorded_runtime["checkpoints"]}
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "inference_scope": inference,
              "policy": "No settings are changed based on test results; canonical artifacts are read-only.",
              "configurations": {}, "missing_checkpoints": [], "source_sha256": {},
              "environment": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for name in ("src/train.py", "src/dataset.py", "src/evaluate.py", "scripts/check_reproducibility.py", "reports/results.md"):
        result["source_sha256"][name] = sha256(root / name)
    for condition, augmentation, lr, unfreeze in SPECS:
        key = f"{condition}_augment_{augmentation}"
        table = root / f"reports/tables/{key}_test_predictions.csv"
        rows = read_csv(table)
        scores = summarize_predictions(rows, manifest, mapping)
        summary = {row["metric"]: row for row in read_csv(root / f"reports/tables/{key}_test_summary.csv")}
        metrics = {}
        for metric in ("accuracy", "macro_f1"):
            values = [scores[seed][metric] for seed in SEEDS]
            mean, std = statistics.mean(values), statistics.stdev(values)
            require_equal(float(summary[metric]["mean"]), mean, f"{key} {metric} mean")
            require_equal(float(summary[metric]["std"]), std, f"{key} {metric} sample SD")
            if int(summary[metric]["n_checkpoints"]) != 3:
                raise ValueError(f"{key}: summary must aggregate all three checkpoints")
            metrics[metric] = {"mean": mean, "std": std}
        class_rows = read_csv(root / f"reports/tables/{key}_test_per_class_f1.csv")
        if len(class_rows) != 10 or {int(item["label"]) for item in class_rows} != set(range(10)):
            raise ValueError(f"{key}: per-class table must contain every label exactly once")
        for item in class_rows:
            label = int(item["label"])
            support = sum(int(row["label"]) == label for row in manifest.values())
            require_equal(float(item["support_mean"]), support, f"{key} class {label} support")
            values = [scores[seed]["per_class_f1"][label] for seed in SEEDS]
            require_equal(float(item["f1_mean"]), statistics.mean(values), f"{key} class {label} mean")
            require_equal(float(item["f1_std"]), statistics.stdev(values), f"{key} class {label} SD")
        line = next(line for line in report.splitlines() if line.startswith(f"| {condition} · augmentation {augmentation} |"))
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        for metric, cell in zip(("accuracy", "macro_f1"), cells[-2:]):
            numbers = re.findall(r"\d+\.\d+", cell)
            expected = [f"{metrics[metric][name]:.4f}" for name in ("mean", "std")]
            if numbers != expected:
                raise ValueError(f"{key}: final report {metric} differs from prediction-derived values")
        entry = {"prediction_count": len(rows), "predictions_sha256": sha256(table),
                 "metrics": metrics, "saved_prediction_check": "passed", "inference": []}
        result["configurations"][key] = entry
        for seed in SEEDS:
            relative = (f"deliverables/{condition.lower()}/runs/lr_{lr}/seed_{seed}/best.pt" if augmentation == "on"
                        else f"artifacts/ablation_noaug/runs/{condition.lower()}_lr{lr}_seed{seed}/best.pt")
            checkpoint = root / relative
            available = checkpoint.is_file() and checkpoint.stat().st_size > 1024
            if not available:
                result["missing_checkpoints"].append(relative)
            requested = inference == "all" or (inference == "core" and augmentation == "on")
            if not requested:
                continue
            if not available:
                entry["inference"].append({"seed": seed, "status": "blocked", "checkpoint": relative})
                continue
            checkpoint_hash = sha256(checkpoint)
            if checkpoint_hash != recorded_hashes[relative]:
                raise ValueError(f"Checkpoint differs from the published artifact: {relative}")
            # Use the shared evaluator's model loading and forward pass, without
            # its report-writing or latency paths; frozen CSVs cannot be replaced.
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import torch
            from src.dataset import get_loaders
            from src.evaluate import _evaluate_checkpoint, _load_model, _resolve_config
            torch.set_num_threads(1)
            config = _resolve_config(root / f"configs/{condition.lower()}{'_noaug' if augmentation == 'off' else ''}.yaml", "cpu")
            loader = get_loaders(config)["test"]
            model, payload = _load_model(checkpoint, torch.device("cpu"))
            saved = payload["config"]
            saved_names = {item["label"]: item["class_name"] for item in payload["class_mapping"]["classes"]}
            if saved_names != mapping:
                raise ValueError(f"Checkpoint class mapping mismatch: {relative}")
            if (payload["training_seed"], saved["unfreeze"], saved["augment"], float(saved["lr"])) != (
                seed, unfreeze, augmentation == "on", float(lr)
            ):
                raise ValueError(f"Checkpoint config/seed mismatch: {relative}")
            measured = _evaluate_checkpoint(model, loader, torch.device("cpu"))
            by_id = {row["image_id"]: row for row in rows if int(row["training_seed"]) == seed}
            expected = [int(by_id[row["image_id"]]["predicted_label"]) for row in loader.dataset.rows]
            if len(measured["predictions"]) != len(expected):
                raise ValueError(f"Incomplete inference output: {relative}")
            differences = sum(a != b for a, b in zip(expected, measured["predictions"]))
            if differences:
                raise ValueError(f"{relative}: {differences} predictions differ; do not replace final metrics")
            for metric in ("accuracy", "macro_f1"):
                require_equal(measured[metric], scores[seed][metric], f"{relative} inference {metric}")
            entry["inference"].append({"seed": seed, "status": "passed", "checkpoint": relative,
                                        "checkpoint_sha256": checkpoint_hash, "prediction_differences": differences})
            del model
            print(f"PASS inference {key} seed={seed}: {len(expected)} predictions match", flush=True)
        print(f"PASS saved metrics and final summary: {key}", flush=True)
    # Verify the published timing arithmetic without claiming to reproduce
    # wall-clock latency on different hardware or overwriting the dated run.
    timings = read_csv(root / "reports/tables/latency_rerun_2026-11-02.csv")
    result["latency_aggregation"] = {}
    for condition, augmentation, _, _ in SPECS:
        key = f"{condition}_augment_{augmentation}"
        line = next(line for line in report.splitlines()
                    if line.startswith(f"| {condition} · augmentation {augmentation} |")
                    and len(line.split("|")) == 5)
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        aggregates = {}
        for batch_size in (1, 32):
            subset = [row for row in timings if row["configuration"] == key and int(row["batch_size"]) == batch_size]
            if len(subset) != 3 or {int(row["training_seed"]) for row in subset} != set(SEEDS):
                raise ValueError(f"{key}: incomplete latency checkpoint coverage")
            values = [float(row["mean_ms"]) for row in subset]
            aggregates[batch_size] = statistics.mean(values), statistics.stdev(values)
        expected = [f"{value:.2f}" for value in aggregates[1]]
        if re.findall(r"\d+\.\d+", cells[1]) != expected:
            raise ValueError(f"{key}: batch-1 timing aggregation differs from final report")
        expected = [f"{value:.2f}" for value in aggregates[32]] + [f"{value / 32:.2f}" for value in aggregates[32]]
        if re.findall(r"\d+\.\d+", cells[2]) != expected:
            raise ValueError(f"{key}: batch-32 timing aggregation differs from final report")
        result["latency_aggregation"][key] = "passed; recorded timings only, no new benchmark"
    blocked = any(item["status"] == "blocked" for entry in result["configurations"].values() for item in entry["inference"])
    result["status"] = "partial" if blocked else "passed"
    result["scope_note"] = "Saved-prediction arithmetic is checked for all six configurations. Inference covers only the explicitly requested scope. Training and hardware latency are not reproduced."
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", choices=("none", "core", "all"), default="none")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reproducibility/check.json")
    args = parser.parse_args()
    result = audit(args.inference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit {result['status']}: {args.output}")
    return 2 if result["status"] == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
