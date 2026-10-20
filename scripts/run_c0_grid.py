#!/usr/bin/env python3
"""Run one condition grid sequentially, then validate and prepare delivery."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_c0_grid import (  # noqa: E402
    LRS, SEEDS, expected_config, summarize_batch, validate_completed_run,
)
from src.train import CONDITIONS  # noqa: E402


if sys.platform == "win32":
    import msvcrt
elif os.name == "posix":
    import fcntl


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
    relevant = [
        "src/train.py", "src/dataset.py", "configs/data.yaml", "configs/training.yaml",
        f"configs/{condition.lower()}.yaml", str(config_path.relative_to(PROJECT_ROOT)),
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


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextlib.contextmanager
def pipeline_lock(batch_dir: Path):
    """Hold a platform-appropriate advisory lock for one orchestrator."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    lock_path = batch_dir / "pipeline.lock"
    handle = lock_path.open("a+b")
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if sys.platform == "win32":
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        elif os.name == "posix":
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            raise RuntimeError(f"Pipeline locking is unsupported on platform {sys.platform!r}")
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Another pipeline holds {lock_path}") from exc
    except BaseException:
        handle.close()
        raise
    try:
        yield lock_path
    finally:
        try:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def record_pipeline_failure(batch_dir: Path, exc: BaseException) -> None:
    """Persist failure without discarding completed attempts or authoritative runs."""
    status_path = batch_dir / "status.json"
    manifest_path = batch_dir / "batch_manifest.json"
    failed_at = now_utc()
    error = f"{type(exc).__name__}: {exc}"
    status: dict[str, Any] = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
    failed_stage = status.get("stage", "initialization")
    status.update(
        status="FAILED", stage=failed_stage, orchestrator_pid=os.getpid(),
        trainer_pid=None, error=error, failed_at_utc=failed_at, updated_at_utc=failed_at,
    )
    write_json(status_path, status)

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        manifest.update(
            status="failed", stage=failed_stage, error=error, failed_at_utc=failed_at,
            training_completed="grid_completed_at_utc" in manifest,
        )
        write_json(manifest_path, manifest)


def set_authoritative(manifest: dict[str, Any], lr: float, seed: int, output_dir: Path) -> None:
    records = [
        record for record in manifest.setdefault("authoritative_runs", [])
        if not (float(record["lr"]) == lr and record["training_seed"] == seed)
    ]
    records.append({"lr": lr, "training_seed": seed, "output_dir": str(output_dir.resolve())})
    manifest["authoritative_runs"] = sorted(records, key=lambda row: (LRS.index(float(row["lr"])), row["training_seed"]))


EPOCH_LINE = re.compile(
    r"^(?P<condition>C\d) epoch=(?P<epoch>\d+) train_loss=(?P<loss>[0-9.eE+-]+) "
    r"val_macro_f1=(?P<score>[0-9.eE+-]+)$"
)


def monitor_process(
    process: subprocess.Popen[str], log: Any, *, index: int, lr_name: str, seed: int,
    output_dir: Path, status_path: Path, status: dict[str, Any], heartbeat_seconds: float = 45.0,
) -> int:
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    best_score, best_epoch, bad_epochs = float("-inf"), 0, 0
    last_message = last_epoch_time = time.monotonic()
    stream_closed = False
    actual_threads = None
    try:
        while not stream_closed or process.poll() is None:
            if actual_threads is None:
                metadata_path = output_dir / "metadata.json"
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    actual_threads = metadata.get("environment", {}).get("torch_num_threads")
                    if actual_threads is not None:
                        status["actual_torch_num_threads"] = actual_threads
                        status["updated_at_utc"] = now_utc()
                        write_json(status_path, status)
                        print(
                            f"[{now_local()}] runtime run={index}/9 actual_torch_num_threads={actual_threads}",
                            flush=True,
                        )
            try:
                line = messages.get(timeout=1.0)
            except queue.Empty:
                line = ""
            if line is None:
                stream_closed = True
            elif line:
                log.write(line)
                log.flush()
                match = EPOCH_LINE.match(line.strip())
                if match:
                    epoch = int(match["epoch"])
                    score = float(match["score"])
                    epoch_seconds = time.monotonic() - last_epoch_time
                    last_epoch_time = time.monotonic()
                    if score > best_score:
                        best_score, best_epoch, bad_epochs = score, epoch, 0
                    else:
                        bad_epochs += 1
                    message = (
                        f"[{now_local()}] progress run={index}/9 lr={lr_name} seed={seed} "
                        f"threads={actual_threads} "
                        f"epoch={epoch} val_macro_f1={score:.6f} best_epoch={best_epoch} "
                        f"best_score={best_score:.6f} no_improve={bad_epochs} epoch_seconds={epoch_seconds:.1f}"
                    )
                    print(message, flush=True)
                    status["progress"] = {
                        "epoch": epoch, "val_macro_f1": score, "best_epoch": best_epoch,
                        "best_score": best_score, "no_improve_epochs": bad_epochs,
                        "last_epoch_seconds": epoch_seconds,
                    }
                else:
                    print(f"[{now_local()}] trainer {line.rstrip()}", flush=True)
                last_message = time.monotonic()
                status["updated_at_utc"] = now_utc()
                write_json(status_path, status)
            if time.monotonic() - last_message >= heartbeat_seconds:
                print(
                    f"[{now_local()}] heartbeat run={index}/9 lr={lr_name} seed={seed} "
                    f"threads={actual_threads} trainer_pid={process.pid} "
                    f"alive={process.poll() is None} output={output_dir}",
                    flush=True,
                )
                status.update(updated_at_utc=now_utc(), heartbeat="alive", trainer_pid=process.pid)
                write_json(status_path, status)
                last_message = time.monotonic()
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
        raise


def run_pipeline(args: argparse.Namespace) -> None:
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
                previous["reconciled_at_utc"] = now_utc()
        manifest.update(status="running", stage="training")
        for key in ("error", "failed_at_utc", "failed_stage"):
            manifest.pop(key, None)
        write_json(manifest_path, manifest)
    else:
        manifest = {
            "created_at_utc": now_utc(), "status": "running", "stage": "training",
            "provenance": current,
            "attempts": [], "authoritative_runs": [],
        }
        write_json(manifest_path, manifest)

    status_path = batch_dir / "status.json"
    status: dict[str, Any] = {
        "status": "RUNNING", "stage": "training", "orchestrator_pid": os.getpid(),
        "batch_dir": str(batch_dir), "config": str(config_path), "updated_at_utc": now_utc(),
    }
    write_json(status_path, status)
    failures = []
    cells = [(lr, seed) for lr in LRS for seed in SEEDS]
    for index, (lr, seed) in enumerate(cells, start=1):
        expected = expected_config(config_path, lr, seed)
        lr_name = {1e-3: "1e-3", 1e-4: "1e-4", 1e-5: "1e-5"}[lr]
        stem = f"{condition_slug}_lr{lr_name}_seed{seed}"
        output_dir, reuse = choose_output(runs_root, stem, expected)
        print(
            f"[{now_local()}] run={index}/9 condition={condition} lr={lr_name} seed={seed} "
            f"device={expected['device']} configured_threads={expected.get('torch_num_threads', 1)} "
            f"output={output_dir} "
            f"status={'reusing-completed' if reuse else 'starting'}",
            flush=True,
        )
        if reuse:
            set_authoritative(manifest, lr, seed, output_dir)
            write_json(manifest_path, manifest)
            continue
        log_path = logs_root / f"{output_dir.name}.log"
        command = [
            sys.executable, "-u", "-m", "src.train", "--config", str(config_path),
            "--training-seed", str(seed), "--lr", repr(lr), "--output-dir", str(output_dir),
        ]
        attempt = {
            "grid_index": index, "lr": lr, "training_seed": seed,
            "output_dir": str(output_dir), "log": str(log_path), "command": command,
            "started_at_utc": now_utc(), "status": "running",
        }
        manifest["attempts"].append(attempt)
        write_json(manifest_path, manifest)
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0),
            )
            attempt["pid"] = process.pid
            write_json(manifest_path, manifest)
            status.update(
                current_run=index, lr=lr, training_seed=seed, trainer_pid=process.pid,
                output_dir=str(output_dir), updated_at_utc=now_utc(), progress=None,
            )
            write_json(status_path, status)
            returncode = monitor_process(
                process, log, index=index, lr_name=lr_name, seed=seed, output_dir=output_dir,
                status_path=status_path, status=status,
            )
        attempt["finished_at_utc"] = now_utc()
        attempt["returncode"] = returncode
        if returncode == 0 and compatible_completed(output_dir, expected):
            attempt["status"] = "completed"
            set_authoritative(manifest, lr, seed, output_dir)
            print(f"[{now_local()}] run={index}/9 completed", flush=True)
        else:
            attempt["status"] = "failed"
            failures.append(stem)
            print(f"[{now_local()}] run={index}/9 FAILED; retained output and log", flush=True)
        write_json(manifest_path, manifest)

    if failures:
        manifest["status"] = "failed"
        manifest["stage"] = "training"
        manifest["training_completed"] = False
        manifest["failed_cells"] = failures
        write_json(manifest_path, manifest)
        raise SystemExit(f"Grid incomplete; failed cells: {failures}")

    manifest.update(
        status="training_completed", stage="aggregation", training_completed=True,
        grid_completed_at_utc=now_utc(),
    )
    write_json(manifest_path, manifest)
    status.update(stage="aggregation", updated_at_utc=now_utc())
    write_json(status_path, status)
    result = summarize_batch(batch_dir, config_path)
    manifest.update(
        status="postprocessing", stage="validation_evaluation",
        selected_lr=result["selected_lr"],
    )
    write_json(manifest_path, manifest)
    status.update(stage="validation_evaluation", updated_at_utc=now_utc())
    write_json(status_path, status)
    from scripts.evaluate_c0_validation import evaluate_grid
    evaluation_dir = batch_dir / "validation_evaluation"
    evaluation = evaluate_grid(batch_dir, config_path, evaluation_dir)
    if evaluation["mismatches"]:
        raise RuntimeError(f"Validation score mismatches: {evaluation['mismatches']}")
    manifest.update(status="postprocessing", stage="delivery")
    write_json(manifest_path, manifest)
    status.update(stage="delivery", updated_at_utc=now_utc())
    write_json(status_path, status)
    from scripts.prepare_c0_delivery import build_delivery
    delivery = build_delivery(batch_dir, config_path, evaluation_dir, args.delivery_dir.resolve())
    manifest.update(
        status="completed", stage="completed", training_completed=True,
        completed_at_utc=now_utc(), selected_lr=result["selected_lr"],
    )
    for key in ("error", "failed_at_utc", "failed_stage", "failed_cells"):
        manifest.pop(key, None)
    write_json(manifest_path, manifest)
    status.update(
        status="COMPLETED", stage="completed", selected_lr=result["selected_lr"],
        delivery=delivery, updated_at_utc=now_utc(), trainer_pid=None,
    )
    write_json(status_path, status)
    print(json.dumps({
        "status": "completed", "runs": len(result["rows"]),
        "selected_lr": result["selected_lr"],
        "summary": str((batch_dir / "summary").resolve()),
        "delivery": str(args.delivery_dir.resolve()),
    }, indent=2), flush=True)


def execute_pipeline(args: argparse.Namespace) -> None:
    """Run the pipeline and persist an overall failure state on every exception."""
    try:
        run_pipeline(args)
    except BaseException as exc:
        record_pipeline_failure(args.batch_dir.resolve(), exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "c0_cpu_grid")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "c0_cpu.yaml")
    parser.add_argument("--delivery-dir", type=Path, default=PROJECT_ROOT / "deliverables" / "c0")
    args = parser.parse_args()
    with pipeline_lock(args.batch_dir.resolve()):
        execute_pipeline(args)


if __name__ == "__main__":
    main()
