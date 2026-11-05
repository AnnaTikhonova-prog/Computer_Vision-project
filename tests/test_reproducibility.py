"""Corruption detection for the read-only result audit."""
import copy
import unittest

from scripts.check_reproducibility import summarize_predictions


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {label: f"breed {label}" for label in range(10)}
        self.manifest = {f"pet_{i}": {"relative_path": f"images/pet_{i}.jpg",
                         "class_name": self.mapping[i], "label": str(i)} for i in range(10)}
        self.rows = [dict(training_seed=str(seed), image_id=image_id,
                     relative_path=item["relative_path"], true_label=item["label"],
                     true_class=item["class_name"], predicted_label=item["label"],
                     predicted_class=item["class_name"], correct="1", predicted_class_confidence="0.9")
                     for seed in (42, 43, 44) for image_id, item in self.manifest.items()]

    def summarize(self, rows):
        return summarize_predictions(rows, self.manifest, self.mapping)

    def test_known_confusion_has_expected_accuracy_and_macro_f1(self):
        self.rows[0].update(predicted_label="1", predicted_class="breed 1", correct="0")
        result = self.summarize(self.rows)
        self.assertEqual(result[42]["accuracy"], 0.9)
        self.assertAlmostEqual(result[42]["macro_f1"], (8 + 2 / 3) / 10)
        self.assertEqual(result[43]["accuracy"], 1)

    def test_duplicate_or_missing_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.summarize(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.summarize(self.rows[:-1])

    def test_invalid_confidence_correctness_or_label_is_rejected(self):
        for changes in ({"predicted_class_confidence": "nan"}, {"correct": "0"},
                        {"predicted_class": "wrong name"}, {"relative_path": "images/other.jpg"}):
            rows = copy.deepcopy(self.rows)
            rows[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.summarize(rows)
