"""Focused offline checks for the shared C0--C3 training contract.

These tests deliberately construct ResNet50 without pretrained weights so they
never download anything. Production ``build_model`` still requests ImageNet V2.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn
from torchvision.models import resnet50

import src.train as training


CONDITION_MODULES = {
    "head": {"fc"},
    "layer4": {"layer4", "fc"},
    "layer3_layer4": {"layer3", "layer4", "fc"},
    "all": {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"},
}


def offline_resnet50(*, weights: object) -> nn.Module:
    # Accept the production signature while avoiding network/cache dependence.
    del weights
    return resnet50(weights=None)


class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_all_configs_and_cli_overrides_are_typed_and_valid(self) -> None:
        expected = {"c0": "head", "c1": "layer4", "c2": "layer3_layer4", "c3": "all"}
        for name, unfreeze in expected.items():
            for seed in (42, 43, 44):
                for lr in (1e-3, 1e-4, 1e-5):
                    with self.subTest(config=name, training_seed=seed, lr=lr):
                        config = training.load_config(
                            training.PROJECT_ROOT / "configs" / f"{name}.yaml",
                            training_seed=seed,
                            lr=lr,
                        )
                        self.assertEqual(config["unfreeze"], unfreeze)
                        self.assertIs(type(config["training_seed"]), int)
                        self.assertIs(type(config["split_seed"]), int)
                        self.assertIs(type(config["lr"]), float)
                        self.assertEqual(config["training_seed"], seed)
                        self.assertEqual(config["split_seed"], 42)
                        self.assertEqual(config["lr"], lr)

        with patch.object(training, "train") as train_mock:
            training.main([
                "--config", str(training.PROJECT_ROOT / "configs" / "c2.yaml"),
                "--training-seed", "43", "--lr", "1e-3",
                "--output-dir", "unused-by-mock",
            ])
        cli_config = train_mock.call_args.args[0]
        self.assertIs(type(cli_config["training_seed"]), int)
        self.assertIs(type(cli_config["lr"]), float)
        self.assertEqual(cli_config["training_seed"], 43)
        self.assertEqual(cli_config["split_seed"], 42)
        self.assertEqual(cli_config["lr"], 1e-3)

        invalid = training.load_config(training.PROJECT_ROOT / "configs" / "c0.yaml")
        invalid["split_seed"] = 43
        with self.assertRaisesRegex(ValueError, "split_seed"):
            training.validate_config(invalid)

    def test_head_initialization_is_tensor_identical_for_every_seed_and_condition(self) -> None:
        with patch.object(training, "resnet50", side_effect=offline_resnet50):
            for seed in (42, 43, 44):
                expected = None
                for unfreeze in CONDITION_MODULES:
                    model = training.build_model(seed, unfreeze)
                    actual = {name: value.detach().clone() for name, value in model.fc.state_dict().items()}
                    if expected is None:
                        expected = actual
                    else:
                        self.assertEqual(actual.keys(), expected.keys())
                        for name in actual:
                            self.assertTrue(
                                torch.equal(actual[name], expected[name]),
                                f"head tensor differs: seed={seed}, condition={unfreeze}, tensor={name}",
                            )
                    del model

    def test_requires_grad_and_optimizer_membership_for_every_condition(self) -> None:
        for unfreeze, expected_modules in CONDITION_MODULES.items():
            with self.subTest(condition=unfreeze):
                model = resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 10)
                training.apply_unfreeze(model, unfreeze)
                actual_modules = {
                    name.split(".", 1)[0]
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                self.assertEqual(actual_modules, expected_modules)
                optimizer = torch.optim.AdamW(
                    [parameter for parameter in model.parameters() if parameter.requires_grad]
                )
                training.assert_optimizer_matches_trainable(model, optimizer)
                expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
                optimizer_ids = [
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ]
                self.assertEqual(set(optimizer_ids), expected_ids)
                self.assertEqual(len(optimizer_ids), len(expected_ids))

                frozen_bn = []
                trainable_bn = []
                for module_name, module in model.named_modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        target = trainable_bn if module_name.split(".", 1)[0] in expected_modules else frozen_bn
                        target.extend(parameter.requires_grad for parameter in module.parameters(recurse=False))
                self.assertTrue(all(trainable_bn))
                self.assertFalse(any(frozen_bn))

    def test_bn_buffers_stay_exactly_fixed_after_training_steps_c1_to_c3(self) -> None:
        torch.manual_seed(123)
        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        loader = [(images, labels)]

        for unfreeze in ("layer4", "layer3_layer4", "all"):
            with self.subTest(condition=unfreeze):
                model = resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 10)
                training.apply_unfreeze(model, unfreeze)
                optimizer = torch.optim.AdamW(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    lr=1e-4,
                    foreach=False,
                    fused=False,
                )
                initial = training.snapshot_bn(model)
                metrics = training.run_epoch(model, loader, torch.device("cpu"), 1, optimizer)
                self.assertGreater(metrics["loss"], 0.0)
                training.assert_bn_unchanged(model, initial)
                self.assertTrue(model.training)
                self.assertTrue(
                    all(
                        not module.training
                        for module in model.modules()
                        if isinstance(module, nn.modules.batchnorm._BatchNorm)
                    )
                )

    def test_completed_run_writes_stable_history_and_identifying_artifacts(self) -> None:
        class TinyClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 10)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.fc(inputs)

        config = training.load_config(training.PROJECT_ROOT / "configs" / "c0.yaml")
        config.update(device="cpu", max_epochs=1, micro_batch_size=2, num_workers=0)
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
        labels = torch.tensor([0, 1])
        loaders = {"train": [(inputs, labels)], "val": [(inputs, labels)], "test": []}

        def build_tiny(seed: int, unfreeze: str) -> nn.Module:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                model = TinyClassifier()
            training.apply_unfreeze(model, unfreeze)
            return model

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.object(training, "get_loaders", return_value=loaders),
                patch.object(training, "build_model", side_effect=build_tiny),
            ):
                runtime = training.train(config, output)

            self.assertEqual(runtime["status"], "completed")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"best.pt", "config.yaml", "history.csv", "metadata.json", "runtime.json"},
            )
            saved_config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved_config, config)
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["condition"], "C0")
            self.assertEqual(metadata["training_seed"], 42)
            self.assertEqual(metadata["split_seed"], 42)
            self.assertEqual(metadata["git"]["commit"], training.subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=training.PROJECT_ROOT, text=True
            ).strip())
            self.assertIs(type(metadata["git"]["dirty"]), bool)
            with (output / "history.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(tuple(rows[0]), training.HISTORY_FIELDS)
            self.assertTrue(all(math.isfinite(float(value)) for value in rows[0].values()))
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"], config)
            self.assertEqual(checkpoint["training_seed"], config["training_seed"])
            self.assertEqual(checkpoint["git"], metadata["git"])

            with self.assertRaises(FileExistsError):
                training.train(config, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
