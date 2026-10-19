"""Focused tests for C0 grid validation and aggregation."""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from scripts.summarize_c0_grid import (
    LRS, SEEDS, collect_grid, expected_config, select_learning_rate,
    summarize_batch, validate_completed_run,
)
from scripts.run_c0_grid import provenance
from src.train import CONDITIONS, HISTORY_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "c0_cpu.yaml"
C2_CONFIG = PROJECT_ROOT / "configs" / "c2_cpu.yaml"


def make_run(run_dir: Path, lr: float, seed: int, best_score: float, config_path: Path = CONFIG) -> None:
    run_dir.mkdir(parents=True)
    config = expected_config(config_path, lr, seed)
    condition = CONDITIONS[config["unfreeze"]]
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    git = {"commit": "test-commit", "dirty": True}
    (run_dir / "metadata.json").write_text(json.dumps({
        "condition": condition, "training_seed": seed, "split_seed": 42, "git": git,
    }), encoding="utf-8")
    (run_dir / "runtime.json").write_text(json.dumps({
        "status": "completed", "epochs_completed": 2, "bn_checks_passed": 2,
        "best_epoch": 1, "best_val_macro_f1": best_score,
    }), encoding="utf-8")
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for epoch, score in ((1, best_score), (2, best_score - 0.1)):
            writer.writerow({
                "epoch": epoch, "lr": lr, "train_loss": 1 / epoch,
                "train_accuracy": 0.5, "val_loss": 1.0,
                "val_accuracy": 0.5, "val_macro_f1": score,
            })
    torch.save({
        "epoch": 1, "val_macro_f1": best_score, "config": config,
        "training_seed": seed, "git": git,
        "model_state_dict": {
            "conv1.weight": torch.zeros(1), "layer4.0.conv1.weight": torch.zeros(1),
            "fc.weight": torch.zeros(10, 2048), "fc.bias": torch.zeros(10),
        },
    }, run_dir / "best.pt")


class C0AggregationTests(unittest.TestCase):
    def test_completed_c2_run_uses_shared_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run(run_dir, 1e-4, 42, 0.7, C2_CONFIG)
            row = validate_completed_run(run_dir, expected_config(C2_CONFIG, 1e-4, 42))
            self.assertEqual(row["condition"], "C2")

    def test_batch_provenance_survives_json_round_trip(self) -> None:
        current = provenance(CONFIG)
        self.assertEqual(json.loads(json.dumps(current)), current)

    def test_selected_checkpoint_can_differ_from_last_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run(run_dir, 1e-3, 42, 0.7)
            row = validate_completed_run(run_dir, expected_config(CONFIG, 1e-3, 42))
            self.assertEqual(row["selected_epoch"], 1)
            self.assertEqual(row["selected_val_macro_f1"], 0.7)
            self.assertEqual(row["epochs_completed"], 2)

    def test_sample_std_and_exact_mean_tie_choose_smaller_lr(self) -> None:
        rows = []
        scores = {
            1e-3: (0.8, 0.7, 0.6),
            1e-4: (0.71, 0.7, 0.69),
            1e-5: (0.8, 0.7, 0.6),
        }
        for lr, values in scores.items():
            rows.extend(
                {"lr": lr, "training_seed": seed, "selected_val_macro_f1": score}
                for seed, score in zip(SEEDS, values)
            )
        summaries, selected = select_learning_rate(rows)
        self.assertEqual(selected, 1e-5)
        first = next(row for row in summaries if row["lr"] == 1e-3)
        self.assertTrue(math.isclose(first["sample_std_val_macro_f1"], statistics.stdev(scores[1e-3])))

    def test_missing_and_duplicate_completed_cells_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            for lr in LRS:
                for seed in SEEDS:
                    make_run(batch / "runs" / f"lr{lr}_seed{seed}", lr, seed, 0.5)
            rows = collect_grid(batch, CONFIG)
            self.assertEqual(len(rows), 9)

            missing = batch / "runs" / f"lr{LRS[0]}_seed{SEEDS[0]}"
            shutil.rmtree(missing)
            with self.assertRaisesRegex(ValueError, "missing"):
                collect_grid(batch, CONFIG)

            make_run(missing, LRS[0], SEEDS[0], 0.5)
            duplicate = batch / "runs" / "duplicate"
            shutil.copytree(missing, duplicate)
            with self.assertRaisesRegex(ValueError, "duplicates"):
                collect_grid(batch, CONFIG)

    def test_complete_summary_retains_all_three_selected_lr_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            for lr in LRS:
                for seed in SEEDS:
                    score = {1e-3: 0.5, 1e-4: 0.8, 1e-5: 0.6}[lr] + (seed - 43) * 0.01
                    make_run(batch / "runs" / f"lr{lr}_seed{seed}", lr, seed, score)
            result = summarize_batch(batch, CONFIG)
            self.assertEqual(result["selected_lr"], 1e-4)
            self.assertEqual(
                [item["training_seed"] for item in result["delivery"]["checkpoints"]],
                list(SEEDS),
            )
            self.assertFalse(result["delivery"]["test_split_used"])
            self.assertTrue((batch / "summary" / "runs.csv").is_file())
            self.assertTrue((batch / "summary" / "learning_rates.csv").is_file())
            self.assertTrue((batch / "summary" / "report.md").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
