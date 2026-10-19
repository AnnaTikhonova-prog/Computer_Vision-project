#!/usr/bin/env python3
"""Run nine production experiments for one condition with safe restart."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_c0_grid import (  # noqa: E402
    LRS, SEEDS, expected_config, summarize_batch, validate_completed_run,
)
from src.train import CONDITIONS  # noqa: E402


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prevent_system_sleep() -> None:
    """Keep Windows awake while this long local batch process is alive."""
    if sys.platform == "win32":
        es_continuous = 0x80000000
        es_system_required = 0x00000001
        result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
            es_continuous | es_system_required
        )
        if not result:
            raise ctypes.WinError()


def provenance(config_path: Path) -> dict[str, Any]:
    resolved = expected_config(config_path, LRS[0], SEEDS[0])
    condition = CONDITIONS[resolved["unfreeze"]]
    condition_config = f"configs/{condition.lower()}.yaml"
    relevant = [
        "src/train.py", "src/dataset.py", "configs/data.yaml", "configs/training.yaml",
        condition_config, str(config_path.relative_to(PROJECT_ROOT)),
        "splits/v2/train.csv", "splits/v2/val.csv", "splits/v2/test.csv",
        "splits/v2/class_mapping.json", "splits/v2/split_report.json",
    ]
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip()),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).splitlines(),
        "files_sha256": {name: file_sha256(PROJECT_ROOT / name) for name in relevant},
        "interpreter": sys.executable,
        "grid": {
            "condition": condition,
            "learning_rates": list(LRS),
            "training_seeds": list(SEEDS),
        },
        "tie_rule": "smaller LR on an exact full-precision mean tie",
        "standard_deviation": "sample standard deviation (ddof=1)",
        "test_split_used_for_selection": False,
    }


def compatible_completed(run_dir: Path, expected: dict[str, Any]) -> bool:
    try:
        validate_completed_run(run_dir, expected, verify_checkpoint=True)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    return True


def choose_output(runs_root: Path, stem: str, expected: dict[str, Any]) -> tuple[Path, bool]:
    candidates = [runs_root / stem, *sorted(runs_root.glob(f"{stem}_attempt*"))]
    for candidate in candidates:
        if candidate.is_dir() and compatible_completed(candidate, expected):
            return candidate, True
    base = runs_root / stem
    if not base.exists():
        return base, False
    attempt = 2
    while (runs_root / f"{stem}_attempt{attempt}").exists():
        attempt += 1
    return runs_root / f"{stem}_attempt{attempt}", False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "c0_cpu_grid")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "c0_cpu.yaml")
    args = parser.parse_args()
    prevent_system_sleep()
    batch_dir, config_path = args.batch_dir.resolve(), args.config.resolve()
    condition = CONDITIONS[expected_config(config_path, LRS[0], SEEDS[0])["unfreeze"]]
    condition_slug = condition.lower()
    runs_root, logs_root = batch_dir / "runs", batch_dir / "logs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    current = provenance(config_path)
    manifest_path = batch_dir / "batch_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("provenance") != current:
            raise RuntimeError(
                "Batch provenance differs from the current code/config/data. "
                "Preserve this batch and choose a new --batch-dir."
            )
        for previous in manifest.get("attempts", []):
            if previous.get("status") == "running":
                previous["status"] = "interrupted"
                previous["reconciled_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    else:
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running", "provenance": current, "attempts": [],
        }
        write_json(manifest_path, manifest)

    failures = []
    cells = [(lr, seed) for lr in LRS for seed in SEEDS]
    for index, (lr, seed) in enumerate(cells, start=1):
        expected = expected_config(config_path, lr, seed)
        lr_name = {1e-3: "1e-3", 1e-4: "1e-4", 1e-5: "1e-5"}[lr]
        stem = f"{condition_slug}_lr{lr_name}_seed{seed}"
        output_dir, reuse = choose_output(runs_root, stem, expected)
        print(
            f"[{index}/9] {condition} lr={lr_name} seed={seed} output={output_dir} "
            f"status={'reusing-completed' if reuse else 'starting'}",
            flush=True,
        )
        if reuse:
            continue
        log_path = logs_root / f"{output_dir.name}.log"
        command = [
            sys.executable, "-m", "src.train", "--config", str(config_path),
            "--training-seed", str(seed), "--lr", repr(lr), "--output-dir", str(output_dir),
        ]
        attempt = {
            "grid_index": index, "lr": lr, "training_seed": seed,
            "output_dir": str(output_dir), "log": str(log_path), "command": command,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        }
        manifest["attempts"].append(attempt)
        write_json(manifest_path, manifest)
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        attempt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        attempt["returncode"] = returncode
        if returncode == 0 and compatible_completed(output_dir, expected):
            attempt["status"] = "completed"
            print(f"[{index}/9] completed", flush=True)
        else:
            attempt["status"] = "failed"
            failures.append(stem)
            print(f"[{index}/9] FAILED; retained output and log", flush=True)
        write_json(manifest_path, manifest)

    if failures:
        manifest["status"] = "incomplete"
        manifest["failed_cells"] = failures
        write_json(manifest_path, manifest)
        raise SystemExit(f"Grid incomplete; failed cells: {failures}")

    result = summarize_batch(batch_dir, config_path)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["selected_lr"] = result["selected_lr"]
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": "completed", "runs": len(result["rows"]),
        "selected_lr": result["selected_lr"],
        "summary": str((batch_dir / "summary").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
