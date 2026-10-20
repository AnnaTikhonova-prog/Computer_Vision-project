#!/usr/bin/env python3
"""Bounded real-step CPU benchmark for C2; never evaluates validation or test."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_loaders  # noqa: E402
from src.train import (  # noqa: E402
    apply_unfreeze, build_model, configure_determinism, load_config, run_epoch,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def config(workers: int) -> dict[str, Any]:
    result = load_config(PROJECT_ROOT / "configs" / "c2_cpu.yaml", training_seed=42, lr=1e-4)
    result["num_workers"] = workers
    return result


def optimizer_for(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,
        foreach=False, fused=False,
    )


def timed_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: Any) -> float:
    started = time.perf_counter()
    run_epoch(model, [batch], torch.device("cpu"), 4, optimizer)
    return time.perf_counter() - started


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def loader_trial(workers: int, batches: int) -> dict[str, Any]:
    started = time.perf_counter()
    iterator = iter(get_loaders(config(workers))["train"])
    first = None
    examples = 0
    for index in range(batches):
        before = time.perf_counter()
        _, labels = next(iterator)
        if index == 0:
            first = time.perf_counter() - before
        examples += labels.numel()
    elapsed = time.perf_counter() - started
    return {
        "num_workers": workers, "batches": batches, "examples": examples,
        "first_batch_seconds": first, "total_seconds_including_startup": elapsed,
        "examples_per_second_including_startup": examples / elapsed,
    }


def reproducibility_check(output: Path, threads: int) -> None:
    configure_determinism(42, threads)
    batch = next(iter(get_loaders(config(0))["train"]))
    model = build_model(42, "layer3_layer4").to("cpu")
    initial_model_hash = state_hash(model)
    optimizer = optimizer_for(model)
    metrics = run_epoch(model, [batch], torch.device("cpu"), 4, optimizer)
    atomic_json(output, {
        "training_seed": 42, "torch_num_threads": torch.get_num_threads(),
        "num_workers": 0, "batch_labels_sha256": hashlib.sha256(batch[1].numpy().tobytes()).hexdigest(),
        "batch_images_sha256": hashlib.sha256(batch[0].numpy().tobytes()).hexdigest(),
        "initial_model_sha256": initial_model_hash,
        "model_state_sha256_after_one_step": state_hash(model),
        "train_loss": metrics["loss"], "train_accuracy": metrics["accuracy"],
    })


def benchmark(output: Path, measured_steps: int, loader_batches: int) -> None:
    started = time.perf_counter()
    configure_determinism(42)
    batches = []
    iterator = iter(get_loaders(config(0))["train"])
    for _ in range(max(2, measured_steps)):
        batches.append(next(iterator))
    base = build_model(42, "layer3_layer4").to("cpu")
    compute = []
    for threads in (1, 2, 4, 6):
        torch.set_num_threads(threads)
        model = copy.deepcopy(base)
        apply_unfreeze(model, "layer3_layer4")
        optimizer = optimizer_for(model)
        warmup = timed_step(model, optimizer, batches[0])
        samples = [timed_step(model, optimizer, batches[index % len(batches)]) for index in range(measured_steps)]
        mean = statistics.fmean(samples)
        compute.append({
            "torch_num_threads": threads, "warmup_seconds": warmup,
            "step_seconds": samples, "mean_step_seconds": mean,
            "median_step_seconds": statistics.median(samples),
            "samples_per_second": 32 / mean,
            "relative_range": (max(samples) - min(samples)) / mean if len(samples) > 1 else None,
        })
        del optimizer, model
        gc.collect()
    loaders = [loader_trial(workers, loader_batches) for workers in (0, 2, 4)]
    stable = [row for row in compute if row["relative_range"] is None or row["relative_range"] <= 0.20]
    fastest = min(stable or compute, key=lambda row: row["median_step_seconds"])
    baseline = next(row for row in compute if row["torch_num_threads"] == 1)
    selected_loader = max(loaders, key=lambda row: row["examples_per_second_including_startup"])
    atomic_json(output, {
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": {
            "condition": "C2", "effective_batch_size": 32, "micro_batch_size": 4,
            "optimizer": "AdamW", "augmentation": True, "precision": "float32",
            "validation_or_test_used": False,
        },
        "compute": compute, "data_loader": loaders,
        "measured_fastest_compute": {
            "torch_num_threads": fastest["torch_num_threads"],
            "speedup_vs_one_thread": baseline["median_step_seconds"] / fastest["median_step_seconds"],
        },
        "production_selection": {
            "torch_num_threads": 6,
            "num_workers": selected_loader["num_workers"],
            "reason": "explicitly authorized local trainer configuration using the measured stable setting",
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--loader-batches", type=int, default=6)
    parser.add_argument("--repro-only", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.repro_only:
        reproducibility_check(args.output.resolve(), args.threads)
    else:
        benchmark(args.output.resolve(), args.measured_steps, args.loader_batches)


if __name__ == "__main__":
    main()
