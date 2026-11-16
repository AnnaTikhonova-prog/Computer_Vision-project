"""Real inference and UI checks for the separate demo environment."""

import csv
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

if importlib.util.find_spec("streamlit") is None:
    raise unittest.SkipTest("Install demo/requirements.txt to run the optional demo checks")

from streamlit.testing.v1 import AppTest

from demo.breeds import BREEDS
from demo.inference import ROOT, load_predictor, predict, read_image
from src.dataset import _eval_transform


class DemoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = load_predictor()

    def test_bundled_examples_and_frozen_preprocessing(self):
        self.assertEqual(len(BREEDS), 10)
        for filename, _ in BREEDS.values():
            with self.subTest(filename=filename):
                image = read_image((ROOT / "demo/examples" / filename).read_bytes())
                self.assertTrue(torch.equal(self.predictor[1](image), _eval_transform()(image)))
                predictions = predict(image, self.predictor)
                self.assertEqual(len(predictions), 3)
                self.assertEqual(len({name for name, _ in predictions}), 3)
                self.assertGreaterEqual(predictions[0][1], predictions[1][1])
                self.assertGreaterEqual(predictions[1][1], predictions[2][1])
                self.assertLessEqual(sum(value for _, value in predictions), 1.000001)

    def test_matches_frozen_evaluator_prediction(self):
        # Read-only regression against an existing diagnostic image and published
        # prediction, never used to choose model settings (test leakage policy).
        with (ROOT / "reports/tables/C1_augment_on_test_predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["training_seed"] != "42" or row["correct"] != "0":
                continue
            matches = list((ROOT / "reports/errors/C1_augment_on").glob(f"*/{row['image_id']}.jpg"))
            if matches:
                actual = predict(read_image(matches[0].read_bytes()), self.predictor)[0]
                self.assertEqual(actual[0], row["predicted_class"])
                self.assertAlmostEqual(actual[1], float(row["predicted_class_confidence"]), places=5)
                return
        self.fail("No exported frozen C1 seed-42 diagnostic image found")

    def test_rejects_invalid_or_oversized_images(self):
        for data in (b"", b"not an image", b"x" * (10 * 1024 * 1024 + 1)):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                read_image(data)
        image = Image.new("RGB", (10, 10))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        with patch("demo.inference.MAX_PIXELS", 50), self.assertRaisesRegex(ValueError, "megapixels"):
            read_image(buffer.getvalue())

    def test_phone_orientation_and_rgb_conversion(self):
        image = Image.new("RGB", (20, 10))
        exif = image.getexif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        self.assertEqual(read_image(buffer.getvalue()).size, (10, 20))
        buffer = io.BytesIO()
        Image.new("RGBA", (10, 10)).save(buffer, format="PNG")
        self.assertEqual(read_image(buffer.getvalue()).mode, "RGB")

    def test_rejects_lfs_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            pointer = Path(folder) / "best.pt"
            pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
            with patch("demo.inference.CHECKPOINT", pointer), self.assertRaisesRegex(ValueError, "Git LFS"):
                load_predictor()

    def test_live_app_examples(self):
        app = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        for example in BREEDS:
            with self.subTest(example=example):
                app.selectbox[0].select(example).run()
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertTrue(any("OUR BEST GUESS" in item.value for item in app.markdown))
                self.assertTrue(any("breed-description" in item.value for item in app.markdown))
        rendered = " ".join(item.value for item in app.markdown) + " ".join(item.value for item in app.caption)
        for removed in ("LIVE DEMO", "One pet. One photo.", "Built with ResNet50", "Examples: Oxford-IIIT"):
            self.assertNotIn(removed, rendered)


if __name__ == "__main__":
    unittest.main()
