#!/usr/bin/env python3
"""Verify production ResNet50 head initialization across all conditions."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_model, hash_tensors  # noqa: E402


def main() -> None:
    conditions = ("head", "layer4", "layer3_layer4", "all")
    results: dict[str, dict[str, str]] = {}
    for seed in (42, 43, 44):
        baseline: dict[str, torch.Tensor] | None = None
        hashes: dict[str, str] = {}
        for condition in conditions:
            model = build_model(seed, condition)
            state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.fc.state_dict().items()
            }
            if baseline is None:
                baseline = state
            elif state.keys() != baseline.keys() or any(
                not torch.equal(state[name], baseline[name]) for name in state
            ):
                raise AssertionError(
                    f"Classifier head differs for training_seed={seed}, condition={condition}"
                )
            hashes[condition] = hash_tensors(state)
            del model, state
            gc.collect()
        results[str(seed)] = hashes
    print(json.dumps({"status": "ok", "head_sha256": results}, indent=2))


if __name__ == "__main__":
    main()
