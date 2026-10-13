#!/usr/bin/env python3
"""Validate and aggregate a complete nine-run C0 learning-rate grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import CONDITIONS, HISTORY_FIELDS, load_config  # noqa: E402


LRS = (1e-3, 1e-4, 1e-5)
SEEDS = (42, 43, 44)
RUN_FIELDS = (
    "condition", "lr", "training_seed", "split_seed", "selected_epoch",
    "selected_val_macro_f1", "checkpoint_path", "checkpoint_sha256", "history_path",
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_config(config_path: Path, lr: float, seed: int) -> dict[str, Any]:
    return load_config(config_path, training_seed=seed, lr=lr)


def validate_completed_run(
    run_dir: Path, expected: dict[str, Any], *, verify_checkpoint: bool = True
) -> dict[str, Any]:
    required = ("config.yaml", "metadata.json", "runtime.json", "history.csv", "best.pt")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"{run_dir}: missing artifacts {missing}")

    saved_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if saved_config != expected:
        raise ValueError(f"{run_dir}: effective config does not match the requested grid cell")
    if saved_config["unfreeze"] != "head" or CONDITIONS[saved_config["unfreeze"]] != "C0":
        raise ValueError(f"{run_dir}: run is not C0")

    metadata = _read_json(run_dir / "metadata.json")
    runtime = _read_json(run_dir / "runtime.json")
    if metadata.get("condition") != "C0":
        raise ValueError(f"{run_dir}: metadata condition is not C0")
    if metadata.get("training_seed") != expected["training_seed"] or metadata.get("split_seed") != 42:
        raise ValueError(f"{run_dir}: metadata seeds do not match")
    if runtime.get("status") != "completed":
        raise ValueError(f"{run_dir}: runtime status is not completed")

    with (run_dir / "history.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != HISTORY_FIELDS:
        raise ValueError(f"{run_dir}: invalid or empty history")
    parsed: dict[int, dict[str, float]] = {}
    for row in rows:
        values = {key: float(value) for key, value in row.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{run_dir}: non-finite history value")
        epoch = int(values["epoch"])
        if values["epoch"] != epoch or epoch in parsed:
            raise ValueError(f"{run_dir}: invalid or duplicate epoch")
        parsed[epoch] = values

    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected_epoch = checkpoint.get("epoch")
    selected_score = checkpoint.get("val_macro_f1")
    if type(selected_epoch) is not int or selected_epoch not in parsed:
        raise ValueError(f"{run_dir}: checkpoint epoch is absent from history")
    if not isinstance(selected_score, (int, float)) or not math.isfinite(float(selected_score)):
        raise ValueError(f"{run_dir}: checkpoint score is invalid")
    if parsed[selected_epoch]["val_macro_f1"] != float(selected_score):
        raise ValueError(f"{run_dir}: checkpoint score does not equal selected history row")
    history_best_epoch = max(parsed, key=lambda epoch: (parsed[epoch]["val_macro_f1"], -epoch))
    if selected_epoch != history_best_epoch:
        raise ValueError(f"{run_dir}: checkpoint is not the earliest best validation epoch")
    if checkpoint.get("config") != expected:
        raise ValueError(f"{run_dir}: checkpoint config does not match")
    if checkpoint.get("training_seed") != expected["training_seed"]:
        raise ValueError(f"{run_dir}: checkpoint seed does not match")
    if checkpoint.get("git") != metadata.get("git"):
        raise ValueError(f"{run_dir}: checkpoint and metadata Git provenance differ")
    if runtime.get("best_epoch") != selected_epoch or runtime.get("best_val_macro_f1") != selected_score:
        raise ValueError(f"{run_dir}: runtime best result disagrees with checkpoint")
    if runtime.get("bn_checks_passed") != runtime.get("epochs_completed"):
        raise ValueError(f"{run_dir}: BatchNorm invariance was not checked after every epoch")

    if verify_checkpoint:
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"{run_dir}: checkpoint state dict is missing")
        if tuple(state.get("fc.weight", torch.empty(0)).shape) != (10, 2048):
            raise ValueError(f"{run_dir}: checkpoint classifier weight shape is wrong")
        if tuple(state.get("fc.bias", torch.empty(0)).shape) != (10,):
            raise ValueError(f"{run_dir}: checkpoint classifier bias shape is wrong")
        if "conv1.weight" not in state or not any(name.startswith("layer4.") for name in state):
            raise ValueError(f"{run_dir}: checkpoint is not a complete ResNet50 state")

    return {
        "condition": "C0",
        "lr": float(expected["lr"]),
        "training_seed": expected["training_seed"],
        "split_seed": expected["split_seed"],
        "selected_epoch": selected_epoch,
        "selected_val_macro_f1": float(selected_score),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "history_path": str((run_dir / "history.csv").resolve()),
        "epochs_completed": runtime["epochs_completed"],
    }


def collect_grid(batch_dir: Path, config_path: Path) -> list[dict[str, Any]]:
    runs_root = batch_dir / "runs"
    found: dict[tuple[float, int], list[dict[str, Any]]] = {
        (lr, seed): [] for lr in LRS for seed in SEEDS
    }
    if runs_root.is_dir():
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            config_file = run_dir / "config.yaml"
            runtime_file = run_dir / "runtime.json"
            if not config_file.is_file() or not runtime_file.is_file():
                continue
            runtime = _read_json(runtime_file)
            if runtime.get("status") != "completed":
                continue
            saved = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            key = (float(saved.get("lr", math.nan)), saved.get("training_seed"))
            if key not in found:
                raise ValueError(f"Unexpected completed run in batch: {run_dir}")
            expected = expected_config(config_path, key[0], key[1])
            found[key].append(validate_completed_run(run_dir, expected))

    missing = [key for key, values in found.items() if not values]
    duplicates = [key for key, values in found.items() if len(values) > 1]
    if missing or duplicates:
        raise ValueError(f"Incomplete or ambiguous grid: missing={missing}, duplicates={duplicates}")
    return [found[(lr, seed)][0] for lr in LRS for seed in SEEDS]


def select_learning_rate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    summaries = []
    for lr in LRS:
        scores_by_seed = {
            row["training_seed"]: row["selected_val_macro_f1"]
            for row in rows if row["lr"] == lr
        }
        if set(scores_by_seed) != set(SEEDS):
            raise ValueError(f"LR {lr} does not have exactly seeds {SEEDS}")
        scores = [scores_by_seed[seed] for seed in SEEDS]
        summaries.append({
            "lr": lr,
            **{f"seed_{seed}_val_macro_f1": scores_by_seed[seed] for seed in SEEDS},
            "mean_val_macro_f1": statistics.fmean(scores),
            "sample_std_val_macro_f1": statistics.stdev(scores),
        })
    # Predeclared exact-mean tie rule: smaller learning rate wins.
    selected_lr = min(LRS, key=lambda lr: (-next(
        row["mean_val_macro_f1"] for row in summaries if row["lr"] == lr
    ), lr))
    for summary in summaries:
        summary["selected"] = summary["lr"] == selected_lr
    return summaries, selected_lr


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_batch(batch_dir: Path, config_path: Path) -> dict[str, Any]:
    rows = collect_grid(batch_dir, config_path)
    summaries, selected_lr = select_learning_rate(rows)
    summary_dir = batch_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "runs.csv", rows, RUN_FIELDS)
    lr_fields = (
        "lr", *(f"seed_{seed}_val_macro_f1" for seed in SEEDS),
        "mean_val_macro_f1", "sample_std_val_macro_f1", "selected",
    )
    write_csv(summary_dir / "learning_rates.csv", summaries, lr_fields)

    chosen = [row for row in rows if row["lr"] == selected_lr]
    delivery = {
        "condition": "C0", "selected_lr": selected_lr,
        "selection_metric": "mean selected-checkpoint validation macro-F1",
        "standard_deviation": "sample standard deviation (ddof=1)",
        "seed_selection": "none; all three seeds retained",
        "test_split_used": False,
        "checkpoints": [
            {key: row[key] for key in (
                "training_seed", "selected_epoch", "selected_val_macro_f1",
                "checkpoint_path", "checkpoint_sha256",
            )}
            for row in chosen
        ],
        "commit_status": "pending user action",
        "delivery_to_team_lead": "pending user action",
    }
    (summary_dir / "delivery_manifest.json").write_text(
        json.dumps(delivery, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected_summary = next(row for row in summaries if row["selected"])
    report_lines = [
        "# C0 production training report", "",
        "Nine validation-only runs: three learning rates × three training seeds.",
        "The test loader was not iterated and test metrics were not computed.", "",
        "## Per-run selected checkpoints", "",
        "| LR | Seed | Best epoch | Validation macro-F1 | Epochs run |", "|---:|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {row['lr']:.0e} | {row['training_seed']} | {row['selected_epoch']} | "
        f"{row['selected_val_macro_f1']:.17g} | {row['epochs_completed']} |"
        for row in rows
    )
    report_lines.extend(["", "## Learning-rate selection", "",
                         "| LR | Seed 42 | Seed 43 | Seed 44 | Mean | Sample std | Selected |",
                         "|---:|---:|---:|---:|---:|---:|:---:|"])
    report_lines.extend(
        f"| {row['lr']:.0e} | {row['seed_42_val_macro_f1']:.17g} | "
        f"{row['seed_43_val_macro_f1']:.17g} | {row['seed_44_val_macro_f1']:.17g} | "
        f"{row['mean_val_macro_f1']:.17g} | {row['sample_std_val_macro_f1']:.17g} | "
        f"{'yes' if row['selected'] else 'no'} |"
        for row in summaries
    )
    report_lines.extend([
        "", f"Selected LR: `{selected_lr:.0e}`.",
        f"Validation macro-F1: `{selected_summary['mean_val_macro_f1']:.17g} ± "
        f"{selected_summary['sample_std_val_macro_f1']:.17g}` (sample std, ddof=1).",
        "Exact mean ties select the smaller LR. No seed is ranked or selected.", "",
        "## Production protocol", "",
        "ImageNet V2 ResNet50; C0 head-only training; frozen BatchNorm statistics; "
        "AdamW; batch 32; microbatch 4; augmentation enabled; maximum 30 epochs; "
        "early stopping patience 5; earliest epoch retained on an exact score tie.", "",
        "## Limitations and delivery", "",
        "This is CPU training in the recorded local environment. It does not resolve the "
        "separate Task 1 GPU reproducibility blocker. Commit and delivery to the Team Lead "
        "remain pending user actions.", "",
    ])
    (summary_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8", newline="\n")
    return {"rows": rows, "learning_rates": summaries, "selected_lr": selected_lr, "delivery": delivery}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "c0_cpu_grid")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "c0_cpu.yaml")
    args = parser.parse_args()
    result = summarize_batch(args.batch_dir.resolve(), args.config.resolve())
    print(json.dumps({
        "status": "ok", "runs": len(result["rows"]), "selected_lr": result["selected_lr"],
        "summary_dir": str((args.batch_dir / "summary").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
