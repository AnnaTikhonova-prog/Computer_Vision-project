"""Evaluate one checkpoint or a retained three-checkpoint condition.

Examples::

    python -m src.evaluate --checkpoint artifacts/c2_seed43_lr1e-4/best.pt \
        --config configs/c2.yaml --split test
    python -m src.evaluate --condition C2 \
        --checkpoints artifacts/c2_seed42/best.pt artifacts/c2_seed43/best.pt artifacts/c2_seed44/best.pt \
        --config configs/c2.yaml --split test

All reports are written under reports/tables/ and reports/figures/.
"""

import argparse
import csv
import re
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torchvision.models import resnet50

from src.dataset import get_loaders
from src.train import CONDITIONS, PROJECT_ROOT, load_config

LABELS = list(range(10))
LATENCY_WARMUP = 3
LATENCY_REPEATS = 10


def _resolve_config(path: Path, device_override: str | None) -> dict[str, Any]:
    config = load_config(path)
    for key in ("dataset_root", "split_dir"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    if device_override is not None:
        config["device"] = device_override
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested by config but unavailable; pass --device cpu to evaluate on CPU.")
    # Windows CPU evaluation is more reliable without spawned DataLoader
    # processes; this only changes data transport, never sample order or metrics.
    if config["device"] == "cpu":
        config["num_workers"] = 0
    return config


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("architecture") != "resnet50":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    weight_name = payload.get("weights", "IMAGENET1K_V2")
    if weight_name != "IMAGENET1K_V2":
        raise ValueError(f"Unsupported ResNet50 pretraining weights: {weight_name}")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def _evaluate_checkpoint(model: nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    targets: list[int] = []
    predictions: list[int] = []
    confidence: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = logits.softmax(dim=1)
            batch_confidence, batch_predictions = probabilities.max(dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            confidence.extend(batch_confidence.cpu().tolist())
            targets.extend(labels.tolist())
    if not targets:
        raise ValueError("Selected evaluation split is empty")
    per_class = f1_score(targets, predictions, labels=LABELS, average=None, zero_division=0)
    return {
        "targets": targets,
        "predictions": predictions,
        "confidence": confidence,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=LABELS, average="macro", zero_division=0)),
        "per_class_f1": [float(value) for value in per_class],
        "confusion": confusion_matrix(targets, predictions, labels=LABELS),
    }


def _latency_ms(model: nn.Module, device: torch.device, batch_size: int) -> list[float]:
    sample = torch.zeros((batch_size, 3, 224, 224), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for _ in range(LATENCY_WARMUP):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(LATENCY_REPEATS):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "evaluation"


def evaluate(
    checkpoints: list[Path], config_path: Path, split: str, *,
    condition: str | None = None, device_override: str | None = None,
) -> dict[str, Path]:
    if len(checkpoints) not in (1, 3):
        raise ValueError("Supply exactly one checkpoint or exactly three retained checkpoints")
    if len({path.resolve() for path in checkpoints}) != len(checkpoints):
        raise ValueError("Checkpoint paths must be distinct")
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    if condition is not None and condition not in {"C0", "C1", "C2", "C3"}:
        raise ValueError("condition must be C0, C1, C2, or C3")

    config = _resolve_config(config_path, device_override)
    inferred_condition = CONDITIONS[config["unfreeze"]]
    if condition is not None and condition != inferred_condition:
        raise ValueError(f"--condition {condition} does not match config condition {inferred_condition}")
    condition = condition or inferred_condition
    device = torch.device(config["device"])
    loaders = get_loaders(config)
    loader = loaders[split]
    class_names = [loader.dataset.mapping[index] for index in LABELS]

    metric_results: list[dict[str, Any]] = []
    latency_results: list[dict[int, list[float]]] = []
    evaluated_seeds: list[int] = []
    for path in checkpoints:
        model, payload = _load_model(path, device)
        saved_config = payload.get("config")
        if not isinstance(saved_config, dict) or saved_config.get("unfreeze") not in CONDITIONS:
            raise ValueError(f"Checkpoint has no valid saved training config: {path}")
        if CONDITIONS[saved_config["unfreeze"]] != condition:
            raise ValueError(f"Checkpoint condition does not match {condition}: {path}")
        training_seed = payload.get("training_seed")
        if type(training_seed) is not int or training_seed not in (42, 43, 44):
            raise ValueError(f"Checkpoint has no valid training seed: {path}")
        if saved_config.get("training_seed") != training_seed:
            raise ValueError(f"Checkpoint seed does not match its saved config: {path}")
        evaluated_seeds.append(training_seed)
        checkpoint_mapping = payload.get("class_mapping")
        if not isinstance(checkpoint_mapping, dict) or not isinstance(checkpoint_mapping.get("classes"), list):
            raise ValueError(f"Checkpoint has no saved class mapping: {path}")
        try:
            saved_names = [item["class_name"] for item in checkpoint_mapping["classes"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Checkpoint has an invalid class mapping: {path}") from exc
        if saved_names != class_names:
            raise ValueError(f"Checkpoint class mapping does not match configured split: {path}")
        metrics = _evaluate_checkpoint(model, loader, device)
        metrics["training_seed"] = training_seed
        metrics["checkpoint"] = str(path)
        metric_results.append(metrics)
        latency_results.append({size: _latency_ms(model, device, size) for size in (1, 32)})
        del model
    if len(checkpoints) == 3 and set(evaluated_seeds) != {42, 43, 44}:
        raise ValueError("Condition aggregation requires one retained checkpoint for each training seed 42, 43, and 44")

    augmentation = "on" if config["augment"] else "off"
    report_name = f"{condition}_augment_{augmentation}"
    if len(checkpoints) == 1:
        report_name = f"{report_name}_{checkpoints[0].parent.name}"
    stem = f"{_safe_name(report_name)}_{split}"
    table_dir = PROJECT_ROOT / "reports" / "tables"
    figure_dir = PROJECT_ROOT / "reports" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for metric in ("macro_f1", "accuracy"):
        mean, std = _mean_std([result[metric] for result in metric_results])
        summary_rows.append({"metric": metric, "mean": mean, "std": std, "n_checkpoints": len(checkpoints)})
    for batch_size in (1, 32):
        checkpoint_means = [statistics.mean(result[batch_size]) for result in latency_results]
        mean_ms, std_ms = _mean_std(checkpoint_means)
        summary_rows.extend([
            {"metric": f"latency_batch_{batch_size}_ms", "mean": mean_ms, "std": std_ms, "n_checkpoints": len(checkpoints)},
            {"metric": f"latency_batch_{batch_size}_ms_per_image", "mean": mean_ms / batch_size,
             "std": std_ms / batch_size, "n_checkpoints": len(checkpoints)},
        ])
    summary_path = table_dir / f"{stem}_summary.csv"
    _write_csv(summary_path, ["metric", "mean", "std", "n_checkpoints"], summary_rows)

    class_rows = []
    for index, name in enumerate(class_names):
        scores = [result["per_class_f1"][index] for result in metric_results]
        supports = [sum(target == index for target in result["targets"]) for result in metric_results]
        mean, std = _mean_std(scores)
        class_rows.append({"label": index, "class_name": name, "support_mean": statistics.mean(supports),
                           "f1_mean": mean, "f1_std": std})
    per_class_path = table_dir / f"{stem}_per_class_f1.csv"
    _write_csv(per_class_path, ["label", "class_name", "support_mean", "f1_mean", "f1_std"], class_rows)

    # Keep sample-level test predictions so the final report can describe
    # representative errors without rerunning or changing any model settings.
    prediction_rows = []
    dataset_rows = getattr(loader.dataset, "rows", None)
    if dataset_rows is None or len(dataset_rows) != len(metric_results[0]["targets"]):
        raise ValueError("The evaluation dataset must expose one manifest row per prediction")
    for result in metric_results:
        for row_index, (target, prediction, confidence) in enumerate(zip(
            result["targets"], result["predictions"], result["confidence"]
        )):
            manifest_row = dataset_rows[row_index]
            prediction_rows.append({
                "training_seed": result["training_seed"],
                "checkpoint": result["checkpoint"],
                "image_id": manifest_row["image_id"],
                "relative_path": manifest_row["relative_path"],
                "true_label": target,
                "true_class": class_names[target],
                "predicted_label": prediction,
                "predicted_class": class_names[prediction],
                "predicted_class_confidence": confidence,
                "correct": int(target == prediction),
            })
    predictions_path = table_dir / f"{stem}_predictions.csv"
    _write_csv(predictions_path, [
        "training_seed", "checkpoint", "image_id", "relative_path", "true_label", "true_class",
        "predicted_label", "predicted_class", "predicted_class_confidence", "correct",
    ], prediction_rows)

    latency_rows = []
    for batch_size in (1, 32):
        for checkpoint_index, results in enumerate(latency_results):
            mean_ms, std_ms = _mean_std(results[batch_size])
            latency_rows.append({"checkpoint": str(checkpoints[checkpoint_index]), "batch_size": batch_size,
                                 "mean_ms": mean_ms, "std_ms": std_ms,
                                 "mean_ms_per_image": mean_ms / batch_size})
    latency_path = table_dir / f"{stem}_latency.csv"
    _write_csv(latency_path, ["checkpoint", "batch_size", "mean_ms", "std_ms", "mean_ms_per_image"], latency_rows)

    pooled_confusion = sum((result["confusion"] for result in metric_results), np.zeros((10, 10), dtype=int))
    figure_path = figure_dir / f"{stem}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(pooled_confusion, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Pooled prediction count")
    ax.set(xticks=LABELS, yticks=LABELS, xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted class", ylabel="True class",
           title=f"{condition} · {split} · pooled over {len(checkpoints)} checkpoint(s)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = pooled_confusion.max() / 2 if pooled_confusion.size else 0
    for row in range(10):
        for column in range(10):
            ax.text(column, row, str(pooled_confusion[row, column]), ha="center", va="center",
                    color="white" if pooled_confusion[row, column] > threshold else "black", fontsize=8)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return {
        "summary": summary_path,
        "per_class": per_class_path,
        "predictions": predictions_path,
        "latency": latency_path,
        "confusion_matrix": figure_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", type=Path, help="One checkpoint; report is per checkpoint")
    choice.add_argument("--condition", choices=("C0", "C1", "C2", "C3"), help="Aggregate its three retained checkpoints")
    parser.add_argument("--checkpoints", type=Path, nargs=3, help="Three checkpoints, required with --condition")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Override the device from YAML")
    args = parser.parse_args(argv)
    if args.condition and not args.checkpoints:
        parser.error("--condition requires exactly three paths in --checkpoints")
    if args.checkpoint and args.checkpoints:
        parser.error("Use either --checkpoint or --condition with --checkpoints")
    checkpoints = args.checkpoints if args.condition else [args.checkpoint]
    paths = evaluate(checkpoints, args.config, args.split, condition=args.condition, device_override=args.device)
    for name, path in paths.items():
        print(f"{name}: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
