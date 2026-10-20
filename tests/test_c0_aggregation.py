"""Focused tests for C0 grid validation and aggregation."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from scripts.summarize_c0_grid import (
    LRS, SEEDS, collect_grid, expected_config, select_learning_rate,
    summarize_batch, validate_completed_run,
)
from scripts.evaluate_c0_validation import authoritative_runs
from scripts.run_c0_grid import (
    choose_output, execute_pipeline, pipeline_lock, provenance, set_authoritative,
)
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


def make_completed_grid(batch: Path) -> None:
    lr_names = {1e-3: "1e-3", 1e-4: "1e-4", 1e-5: "1e-5"}
    for lr in LRS:
        for seed in SEEDS:
            score = {1e-3: 0.5, 1e-4: 0.8, 1e-5: 0.6}[lr] + (seed - 43) * 0.01
            make_run(batch / "runs" / f"c0_lr{lr_names[lr]}_seed{seed}", lr, seed, score)


def pipeline_args(batch: Path) -> Namespace:
    return Namespace(batch_dir=batch, config=CONFIG, delivery_dir=batch / "delivery")


def require_completed_output(*args, **kwargs):
    output, reuse = choose_output(*args, **kwargs)
    if not reuse:
        raise AssertionError(f"Test attempted to start training for {output}")
    return output, reuse


class C0AggregationTests(unittest.TestCase):
    def test_completed_c2_run_uses_shared_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run(run_dir, 1e-4, 42, 0.7, C2_CONFIG)
            row = validate_completed_run(run_dir, expected_config(C2_CONFIG, 1e-4, 42))
            self.assertEqual(row["condition"], "C2")

    def test_authoritative_manifest_selects_one_attempt_per_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            manifest = {"authoritative_runs": []}
            for lr in LRS:
                for seed in SEEDS:
                    run_dir = batch / "runs" / f"c2_lr{lr}_seed{seed}"
                    make_run(run_dir, lr, seed, 0.5, C2_CONFIG)
                    set_authoritative(manifest, lr, seed, run_dir)
            duplicate = batch / "runs" / "c2_duplicate_completed"
            make_run(duplicate, LRS[0], SEEDS[0], 0.6, C2_CONFIG)
            (batch / "batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            rows = collect_grid(batch, C2_CONFIG)
            self.assertEqual(len(rows), 9)
            self.assertNotIn(str(duplicate), {row["checkpoint_path"] for row in rows})

    def test_reused_cell_replaces_authoritative_record(self) -> None:
        manifest = {"authoritative_runs": []}
        set_authoritative(manifest, 1e-3, 42, Path("first"))
        set_authoritative(manifest, 1e-3, 42, Path("replacement"))
        self.assertEqual(len(manifest["authoritative_runs"]), 1)
        self.assertTrue(manifest["authoritative_runs"][0]["output_dir"].endswith("replacement"))

    def test_restart_uses_completed_run_and_preserves_incomplete_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            expected = expected_config(C2_CONFIG, 1e-4, 42)
            incomplete = runs / "c2_lr1e-4_seed42"
            incomplete.mkdir()
            output, reuse = choose_output(runs, incomplete.name, expected)
            self.assertFalse(reuse)
            self.assertEqual(output.name, "c2_lr1e-4_seed42_attempt2")
            make_run(output, 1e-4, 42, 0.7, C2_CONFIG)
            selected, reuse = choose_output(runs, incomplete.name, expected)
            self.assertTrue(reuse)
            self.assertEqual(selected, output)
            self.assertTrue(incomplete.is_dir())

    def test_pipeline_lock_rejects_orchestrator_in_separate_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            code = (
                "import os,sys; from pathlib import Path; "
                "from scripts.run_c0_grid import pipeline_lock; "
                "lock=pipeline_lock(Path(os.environ['PIPELINE_TEST_BATCH'])); "
                "lock.__enter__(); print('LOCKED', flush=True); "
                "sys.stdin.readline(); lock.__exit__(None,None,None)"
            )
            environment = dict(os.environ, PIPELINE_TEST_BATCH=str(batch))
            child = subprocess.Popen(
                [sys.executable, "-c", code], cwd=PROJECT_ROOT, env=environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), "LOCKED")
                with self.assertRaisesRegex(RuntimeError, "Another pipeline"):
                    with pipeline_lock(batch):
                        self.fail("duplicate lock unexpectedly acquired")
            finally:
                assert child.stdin is not None
                child.stdin.write("release\n")
                child.stdin.flush()
                child.stdin.close()
                child.stdin = None
                _, stderr = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, stderr)

    def test_pipeline_lock_is_released_after_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with pipeline_lock(batch):
                    raise RuntimeError("fixture failure")
            with pipeline_lock(batch):
                self.assertTrue((batch / "pipeline.lock").is_file())

    def test_postprocessing_failure_is_failed_and_retry_reuses_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            make_completed_grid(batch)
            args = pipeline_args(batch)
            with (
                patch("scripts.run_c0_grid.prevent_system_sleep"),
                patch("scripts.run_c0_grid.choose_output", side_effect=require_completed_output),
                patch("scripts.evaluate_c0_validation.evaluate_grid", side_effect=RuntimeError("evaluation failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    execute_pipeline(args)

            failed = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["stage"], "validation_evaluation")
            self.assertTrue(failed["training_completed"])
            self.assertEqual(len(failed["authoritative_runs"]), 9)
            self.assertEqual(failed["attempts"], [])
            self.assertIn("evaluation failed", failed["error"])

            with (
                patch("scripts.run_c0_grid.prevent_system_sleep"),
                patch("scripts.run_c0_grid.choose_output", side_effect=require_completed_output),
                patch("scripts.evaluate_c0_validation.evaluate_grid", return_value={
                    "runs": 9, "selected_lr": 1e-4, "mismatches": [],
                }),
                patch("scripts.prepare_c0_delivery.build_delivery", return_value={
                    "files": 1, "selected_lr": 1e-4, "total_bytes": 1,
                }),
            ):
                execute_pipeline(args)

            completed = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["stage"], "completed")
            self.assertEqual(len(completed["authoritative_runs"]), 9)
            self.assertEqual(completed["attempts"], [])
            self.assertNotIn("error", completed)

    def test_completion_is_written_only_after_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            make_completed_grid(batch)
            stages = []

            def evaluate(*_args):
                manifest = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
                stages.append((manifest["status"], manifest["stage"]))
                self.assertEqual(len(authoritative_runs(batch, CONFIG)), 9)
                return {"runs": 9, "selected_lr": 1e-4, "mismatches": []}

            def deliver(*_args):
                manifest = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
                stages.append((manifest["status"], manifest["stage"]))
                return {"files": 1, "selected_lr": 1e-4, "total_bytes": 1}

            with (
                patch("scripts.run_c0_grid.prevent_system_sleep"),
                patch("scripts.run_c0_grid.choose_output", side_effect=require_completed_output),
                patch("scripts.evaluate_c0_validation.evaluate_grid", side_effect=evaluate),
                patch("scripts.prepare_c0_delivery.build_delivery", side_effect=deliver),
            ):
                execute_pipeline(pipeline_args(batch))

            final = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stages, [
                ("postprocessing", "validation_evaluation"),
                ("postprocessing", "delivery"),
            ])
            self.assertEqual((final["status"], final["stage"]), ("completed", "completed"))

    def test_delivery_failure_cannot_mark_pipeline_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            make_completed_grid(batch)
            with (
                patch("scripts.run_c0_grid.prevent_system_sleep"),
                patch("scripts.run_c0_grid.choose_output", side_effect=require_completed_output),
                patch("scripts.evaluate_c0_validation.evaluate_grid", return_value={
                    "runs": 9, "selected_lr": 1e-4, "mismatches": [],
                }),
                patch("scripts.prepare_c0_delivery.build_delivery", side_effect=RuntimeError("delivery failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "delivery failed"):
                    execute_pipeline(pipeline_args(batch))

            manifest = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
            status = json.loads((batch / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["stage"]), ("failed", "delivery"))
            self.assertTrue(manifest["training_completed"])
            self.assertIn("delivery failed", manifest["error"])
            self.assertEqual((status["status"], status["stage"]), ("FAILED", "delivery"))

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
