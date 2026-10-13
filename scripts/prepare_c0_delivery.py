#!/usr/bin/env python3
"""Build an idempotent, self-auditing local C0 delivery directory."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_c0_validation import (  # noqa: E402
    LR_NAMES, authoritative_runs, write_text_idempotent,
)
from scripts.summarize_c0_grid import LRS, SEEDS, file_sha256  # noqa: E402


def copy_idempotent(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or file_sha256(source) != file_sha256(destination):
            raise FileExistsError(f"Refusing to overwrite conflicting delivery file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_delivery(
    batch_dir: Path, config_path: Path, evaluation_dir: Path, delivery_dir: Path,
) -> dict[str, Any]:
    rows = authoritative_runs(batch_dir, config_path)
    if not (evaluation_dir / "learning_rates.csv").is_file():
        raise FileNotFoundError("Validation evaluation is missing; run evaluate_c0_validation.py first")
    manifest_files: list[dict[str, Any]] = []

    def add_copy(source: Path, relative: Path, category: str) -> Path:
        destination = delivery_dir / relative
        copy_idempotent(source, destination)
        manifest_files.append({
            "path": relative.as_posix(), "category": category,
            "source": str(source.resolve()), "size_bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
        })
        return destination

    delivered_by_cell: dict[tuple[float, int], Path] = {}
    for row in rows:
        lr, seed = row["lr"], row["training_seed"]
        relative_root = Path("runs") / f"lr_{LR_NAMES[lr]}" / f"seed_{seed}"
        source_root = Path(row["run_dir"])
        for name in ("best.pt", "config.yaml", "metadata.json", "history.csv"):
            delivered = add_copy(source_root / name, relative_root / name, "authoritative run artifact")
            if name == "best.pt":
                delivered_by_cell[(lr, seed)] = delivered
                if file_sha256(delivered) != row["checkpoint_sha256"]:
                    raise RuntimeError(f"Copied checkpoint hash mismatch: {delivered}")

    for lr in LRS:
        source_dir = evaluation_dir / f"lr_{LR_NAMES[lr]}"
        for source in sorted(source_dir.glob("*.csv")):
            add_copy(source, Path("evaluation") / f"lr_{LR_NAMES[lr]}" / source.name,
                     "validation evaluator report")
    for name in ("learning_rates.csv", "report_mapping.csv"):
        add_copy(evaluation_dir / name, Path("evaluation") / name, "validation evaluator report")
    for name in ("runs.csv", "learning_rates.csv", "report.md"):
        add_copy(batch_dir / "summary" / name, Path("training_summary") / name, "training summary")

    evaluated = read_csv(evaluation_dir / "learning_rates.csv")
    selected = next(item for item in evaluated if item["selected"] == "True")
    selected_lr = float(selected["lr"])
    chosen = [{
        "training_seed": seed,
        "checkpoint_path": delivered_by_cell[(selected_lr, seed)].relative_to(delivery_dir).as_posix(),
        "checkpoint_sha256": file_sha256(delivered_by_cell[(selected_lr, seed)]),
    } for seed in SEEDS]
    chosen_payload = {
        "condition": "C0", "selected_lr": selected_lr,
        "selection_metric": "mean validation macro-F1 across three selected checkpoints",
        "standard_deviation": "sample standard deviation (ddof=1)",
        "seed_selection": "none; all three seeds retained", "checkpoints": chosen,
    }
    chosen_path = delivery_dir / "chosen_lr_manifest.json"
    write_text_idempotent(chosen_path, json.dumps(chosen_payload, indent=2, sort_keys=True) + "\n")
    manifest_files.append({
        "path": "chosen_lr_manifest.json", "category": "chosen LR manifest", "source": None,
        "size_bytes": chosen_path.stat().st_size, "sha256": file_sha256(chosen_path),
    })

    readme = f"""# C0 local delivery

This directory contains the nine authoritative C0 validation-selected runs (three learning rates x seeds 42, 43, and 44), validation-only evaluator CSVs, and training summaries.

Selected LR: `{selected_lr:.0e}`. Re-evaluated validation macro-F1: `{selected['mean_val_macro_f1']} +/- {selected['sample_std_val_macro_f1']}` (mean and sample standard deviation across three seed-level scores, ddof=1; not pooled predictions and not an ensemble).

`runs/` preserves one `best.pt`, effective `config.yaml`, `metadata.json`, and `history.csv` for each authoritative completed run. The filenames are unchanged. Historical absolute paths inside copied metadata/config/history are preserved rather than rewritten. `evaluation/` keeps each LR separate and contains seed-specific predictions plus per-LR metrics and aggregates. `chosen_lr_manifest.json` references the same three selected-LR checkpoints already under `runs/`; no second checkpoint copies are made.

The interrupted `c0_lr1e-4_seed42` attempt, datasets, caches, logs, and environment files are excluded. Its source remains untouched under `artifacts/c0_cpu_grid`. No test-split evaluation was performed. CPU validation does not resolve the separate Task 1 GPU reproducibility check. Commit and delivery to the Team Lead remain pending user actions.
"""
    readme_path = delivery_dir / "README.md"
    write_text_idempotent(readme_path, readme)
    manifest_files.append({
        "path": "README.md", "category": "delivery documentation", "source": None,
        "size_bytes": readme_path.stat().st_size, "sha256": file_sha256(readme_path),
    })
    manifest_payload = {
        "condition": "C0", "authoritative_runs": 9, "selected_lr": selected_lr,
        "test_split_evaluated": False,
        "excluded_incomplete_attempt": str((batch_dir / "runs" / "c0_lr1e-4_seed42").resolve()),
        "files": sorted(manifest_files, key=lambda item: item["path"]),
    }
    write_text_idempotent(
        delivery_dir / "manifest.json", json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    )
    return {
        "files": len(manifest_files) + 1, "selected_lr": selected_lr,
        "total_bytes": sum(path.stat().st_size for path in delivery_dir.rglob("*") if path.is_file()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "c0_cpu_grid")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "c0_cpu.yaml")
    parser.add_argument(
        "--evaluation-dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "c0_cpu_grid" / "validation_evaluation",
    )
    parser.add_argument("--delivery-dir", type=Path, default=PROJECT_ROOT / "deliverables" / "c0")
    args = parser.parse_args()
    result = build_delivery(
        args.batch_dir.resolve(), args.config.resolve(),
        args.evaluation_dir.resolve(), args.delivery_dir.resolve(),
    )
    print(json.dumps({"status": "ok", **result, "delivery_dir": str(args.delivery_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
