"""Focused tests for collision-safe C0 evaluator and delivery writes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_c0_validation import write_text_idempotent
from scripts.prepare_c0_delivery import copy_idempotent


class C0DeliveryTests(unittest.TestCase):
    def test_report_write_accepts_match_and_rejects_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "lr_1e-3" / "metrics.csv"
            write_text_idempotent(report, "score\n0.9\n")
            write_text_idempotent(report, "score\n0.9\n")
            with self.assertRaises(FileExistsError):
                write_text_idempotent(report, "score\n0.8\n")

    def test_copy_accepts_hash_match_and_rejects_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.pt", root / "delivery" / "best.pt"
            source.write_bytes(b"checkpoint-a")
            copy_idempotent(source, destination)
            copy_idempotent(source, destination)
            source.write_bytes(b"checkpoint-b")
            with self.assertRaises(FileExistsError):
                copy_idempotent(source, destination)


if __name__ == "__main__":
    unittest.main(verbosity=2)
