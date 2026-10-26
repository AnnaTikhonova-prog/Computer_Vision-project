import statistics
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from scripts.run_ablation import CELLS, SEEDS, aggregate_rows


def _baselines():
    return {
        condition: {"scores": {seed: 0.9 + seed / 10000 for seed in SEEDS}, "metrics": "baseline.csv"}
        for condition in ("C0", "C3")
    }


class AblationTests(unittest.TestCase):
    def test_ablation_cells_are_exactly_the_precommitted_six(self):
        self.assertEqual(CELLS, (
            ("C0", 1e-3, 42), ("C0", 1e-3, 43), ("C0", 1e-3, 44),
            ("C3", 1e-5, 42), ("C3", 1e-5, 43), ("C3", 1e-5, 44),
        ))

    def test_aggregation_refuses_an_incomplete_grid(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = {
                ("C0", seed): {"best_val_macro_f1": 0.8, "history": str(root / f"{seed}.csv")}
                for seed in SEEDS
            }
            with self.assertRaisesRegex(ValueError, "incomplete six-cell grid"):
                aggregate_rows(incomplete, _baselines())

    def test_aggregation_uses_all_seeds_and_sample_standard_deviation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scores = {
                ("C0", 42): 0.80, ("C0", 43): 0.82, ("C0", 44): 0.84,
                ("C3", 42): 0.90, ("C3", 43): 0.91, ("C3", 44): 0.95,
            }
            results = {
                key: {"best_val_macro_f1": value, "history": str(root / f"{key[0]}_{key[1]}.csv")}
                for key, value in scores.items()
            }
            rows = aggregate_rows(results, _baselines())
            c0_off = next(row for row in rows if row["condition"] == "C0" and row["augmentation"] == "off")
            self.assertAlmostEqual(c0_off["mean_val_macro_f1"], statistics.fmean((0.80, 0.82, 0.84)))
            self.assertAlmostEqual(c0_off["sample_std_val_macro_f1"], statistics.stdev((0.80, 0.82, 0.84)))


if __name__ == "__main__":
    unittest.main()

