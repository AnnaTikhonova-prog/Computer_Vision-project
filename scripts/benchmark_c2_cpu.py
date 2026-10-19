#!/usr/bin/env python3
"""Bounded CPU benchmark for C2 compute threads and Windows data loading."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_loaders
from src.train import build_model, load_config, run_epoch


def _config(*, workers: int) -> dict[str, Any]:
    config = load_config(PROJECT_ROOT / "configs" / "c2.yaml", training_seed=42, lr=1e-4)
    config.update(device="cpu", num_workers=workers)
    return config


def _one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> float:
    started = time.perf_counter()
    run_epoch(model, [batch], torch.device("cpu"), 4, optimizer)
    return time.perf_counter() - started


def _loader_trial(workers: int, batches: int, deadline: float) -> dict[str, Any]:
    config = _config(workers=workers)
    started = time.perf_counter()
    loader = get_loaders(config)["train"]
    iterator = iter(loader)
    fetched = 0
    examples = 0
    first_batch_seconds = None
    while fetched < batches and time.perf_counter() < deadline:
        before = time.perf_counter()
        _, labels = next(iterator)
        elapsed = time.perf_counter() - before
        if first_batch_seconds is None:
            first_batch_seconds = elapsed
        fetched += 1
        examples += labels.numel()
    total = time.perf_counter() - started
    return {
        "num_workers": workers,
        "batches": fetched,
        "examples": examples,
        "first_batch_seconds": first_batch_seconds,
        "total_seconds": total,
        "examples_per_second_including_startup": examples / total if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float, default=285.0)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--loader-batches", type=int, default=6)
    args = parser.parse_args()
    if args.budget_seconds <= 0 or args.measured_steps <= 0 or args.loader_batches <= 0:
        raise ValueError("Benchmark limits must be positive")

    benchmark_started = time.perf_counter()
    deadline = benchmark_started + args.budget_seconds
    baseline_config = _config(workers=0)
    batch_iterator = iter(get_loaders(baseline_config)["train"])
    batches = [next(batch_iterator) for _ in range(2)]

    model = build_model(42, "layer3_layer4").to("cpu")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,
        foreach=False, fused=False,
    )

    torch.set_num_threads(1)
    warmup_seconds = _one_step(model, optimizer, batches[0])
    compute = []
    for thread_count in (1, 2, 4, 6):
        if time.perf_counter() >= deadline:
            break
        torch.set_num_threads(thread_count)
        samples = []
        for index in range(args.measured_steps):
            if time.perf_counter() >= deadline:
                break
            samples.append(_one_step(model, optimizer, batches[index % len(batches)]))
        if samples:
            mean = statistics.fmean(samples)
            compute.append({
                "torch_num_threads": thread_count,
                "step_seconds": samples,
                "mean_step_seconds": mean,
                "median_step_seconds": statistics.median(samples),
                "samples_per_second": 32 / mean,
                "relative_range": (max(samples) - min(samples)) / mean if len(samples) > 1 else None,
            })

    loader = []
    for workers in (0, 2, 4):
        if time.perf_counter() >= deadline:
            break
        loader.append(_loader_trial(workers, args.loader_batches, deadline))

    stable = [
        row for row in compute
        if row["relative_range"] is None or row["relative_range"] <= 0.20
    ]
    candidates = stable or compute
    selected_compute = min(candidates, key=lambda row: row["median_step_seconds"])
    selected_loader = max(
        (row for row in loader if row["batches"] == args.loader_batches),
        key=lambda row: row["examples_per_second_including_startup"],
        default=None,
    )
    baseline = next((row for row in compute if row["torch_num_threads"] == 1), None)
    epoch_batches = math.ceil(1399 / 32)
    result = {
        "status": "completed" if time.perf_counter() < deadline else "budget_exhausted",
        "elapsed_seconds": time.perf_counter() - benchmark_started,
        "budget_seconds": args.budget_seconds,
        "protocol": {
            "condition": "C2", "effective_batch_size": 32, "micro_batch_size": 4,
            "training_seed": 42, "lr": 1e-4, "optimizer": "AdamW",
            "augmentation": True, "precision": "float32", "dataset_examples": 1399,
        },
        "warmup_seconds": warmup_seconds,
        "compute": compute,
        "loader": loader,
        "selected": {
            "torch_num_threads": selected_compute["torch_num_threads"],
            "num_workers": selected_loader["num_workers"] if selected_loader else None,
            "measured_speedup_vs_one_thread": (
                baseline["median_step_seconds"] / selected_compute["median_step_seconds"]
                if baseline else None
            ),
            "estimated_epoch_seconds_compute_only": epoch_batches * selected_compute["median_step_seconds"],
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
