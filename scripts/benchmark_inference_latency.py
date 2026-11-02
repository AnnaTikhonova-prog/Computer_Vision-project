"""Rerun inference latency only, without loading data or computing quality metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision

from src.evaluate import LATENCY_REPEATS, LATENCY_WARMUP, _latency_ms, _load_model


CHECKPOINT_GROUPS = {
    "C0_augment_on": [f"deliverables/c0/runs/lr_1e-3/seed_{seed}/best.pt" for seed in (42, 43, 44)],
    "C1_augment_on": [f"deliverables/c1/runs/lr_1e-4/seed_{seed}/best.pt" for seed in (42, 43, 44)],
    "C2_augment_on": [f"deliverables/c2/runs/lr_1e-5/seed_{seed}/best.pt" for seed in (42, 43, 44)],
    "C3_augment_on": [f"deliverables/c3/runs/lr_1e-5/seed_{seed}/best.pt" for seed in (42, 43, 44)],
    "C0_augment_off": [f"artifacts/ablation_noaug/runs/c0_lr1e-3_seed{seed}/best.pt" for seed in (42, 43, 44)],
    "C3_augment_off": [f"artifacts/ablation_noaug/runs/c3_lr1e-5_seed{seed}/best.pt" for seed in (42, 43, 44)],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_lfs_object(checkpoint: Path) -> tuple[Path, str]:
    if checkpoint.stat().st_size > 1024:
        return checkpoint, "working-tree checkpoint"
    pointer = checkpoint.read_text(encoding="ascii")
    match = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Small checkpoint is not a valid Git LFS pointer: {checkpoint}")
    oid = match.group(1)
    object_path = PROJECT_ROOT / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not object_path.is_file():
        raise FileNotFoundError(f"Git LFS object {oid} is unavailable for {checkpoint}")
    if _sha256(object_path) != oid:
        raise ValueError(f"Git LFS object hash mismatch for {checkpoint}")
    return object_path, f"local Git LFS object sha256:{oid}"


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    return platform.processor() or "unknown"


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "tables")
    parser.add_argument("--label", default="2026-11-02")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"latency_rerun_{args.label}.csv"
    runtime_path = output_dir / f"latency_rerun_{args.label}_runtime.json"
    device = torch.device("cpu")

    rows: list[dict[str, object]] = []
    checkpoint_provenance: list[dict[str, str]] = []
    for configuration, relative_paths in CHECKPOINT_GROUPS.items():
        for relative_path in relative_paths:
            checkpoint = PROJECT_ROOT / relative_path
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            resolved_checkpoint, storage = _resolve_lfs_object(checkpoint)
            checkpoint_provenance.append(
                {
                    "configuration": configuration,
                    "path": relative_path,
                    "sha256": _sha256(resolved_checkpoint),
                    "storage": storage,
                }
            )
            model, payload = _load_model(resolved_checkpoint, device)
            for batch_size in (1, 32):
                timings = _latency_ms(model, device, batch_size)
                rows.append(
                    {
                        "configuration": configuration,
                        "training_seed": payload["training_seed"],
                        "checkpoint": relative_path,
                        "checkpoint_load_mode": "weights_only=True via src.evaluate._load_model",
                        "batch_size": batch_size,
                        "warmups": LATENCY_WARMUP,
                        "repeats": LATENCY_REPEATS,
                        "mean_ms": statistics.mean(timings),
                        "std_ms": statistics.stdev(timings),
                        "mean_ms_per_image": statistics.mean(timings) / batch_size,
                    }
                )
            del model

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    runtime = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "purpose": "Latency-only rerun; no dataset loading and no accuracy/F1 evaluation.",
        "method": "src.evaluate._latency_ms: torch.inference_mode, synthetic zero float32 NCHW 224x224 input, time.perf_counter",
        "device": "cpu",
        "cpu": _cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "warmups_per_checkpoint_and_batch_size": LATENCY_WARMUP,
        "timed_repeats_per_checkpoint_and_batch_size": LATENCY_REPEATS,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_at_measurement": _git_output("status", "--short"),
        "benchmark_script": "scripts/benchmark_inference_latency.py",
        "benchmark_source": "src/evaluate.py::_latency_ms",
        "results_csv": str(csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "checkpoints": checkpoint_provenance,
    }
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(csv_path.relative_to(PROJECT_ROOT))
    print(runtime_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
