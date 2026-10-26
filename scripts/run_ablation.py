#!/usr/bin/env python3
"""Run and aggregate the fixed six-cell augmentation-off ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import CONDITIONS, load_config  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "ablation_noaug"
RUNS_ROOT = OUTPUT_ROOT / "runs"
SUMMARY_PATH = OUTPUT_ROOT / "summary" / "ablation.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "ablation.md"
SEEDS = (42, 43, 44)
SPECS = {
    "C0": {
        "lr": 1e-3,
        "config": PROJECT_ROOT / "configs" / "ablation_c0_noaug_cpu.yaml",
        "base_config": PROJECT_ROOT / "configs" / "c0_cpu.yaml",
        "baseline_manifest": PROJECT_ROOT / "deliverables" / "c0" / "chosen_lr_manifest.json",
        "baseline_metrics": PROJECT_ROOT / "deliverables" / "c0" / "evaluation" / "lr_1e-3" / "metrics.csv",
        "slug": "c0_lr1e-3",
    },
    "C3": {
        "lr": 1e-5,
        "config": PROJECT_ROOT / "configs" / "ablation_c3_noaug_cpu.yaml",
        "base_config": PROJECT_ROOT / "configs" / "c3.yaml",
        "baseline_manifest": PROJECT_ROOT / "deliverables" / "c3" / "chosen_lr_manifest.json",
        "baseline_metrics": PROJECT_ROOT / "deliverables" / "c3" / "evaluation" / "lr_1e-5" / "metrics.csv",
        "slug": "c3_lr1e-5",
    },
}
CELLS = tuple((condition, SPECS[condition]["lr"], seed) for condition in ("C0", "C3") for seed in SEEDS)
EPOCH_RE = re.compile(r"^(C[03]) epoch=(\d+) .* val_macro_f1=([0-9.eE+-]+)$")
HISTORICAL_COMMITS = {
    "C0": "155bf1b5325c758455de03431f08008ae310d1ff",
    "C3": "37f1e666390bde90743cc5ec765f286359c04d90",
}
SPLIT_FILES = ("train.csv", "val.csv", "test.csv", "class_mapping.json", "split_report.json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_baselines() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for condition, spec in SPECS.items():
        manifest = json.loads(spec["baseline_manifest"].read_text(encoding="utf-8"))
        if manifest["condition"] != condition or float(manifest["selected_lr"]) != spec["lr"]:
            raise ValueError(f"Authoritative selected LR mismatch for {condition}")
        rows = read_csv(spec["baseline_metrics"])
        by_seed = {int(row["training_seed"]): row for row in rows if float(row["lr"]) == spec["lr"]}
        if set(by_seed) != set(SEEDS):
            raise ValueError(f"Authoritative baseline seeds are incomplete for {condition}")
        scores, epochs = {}, {}
        for seed in SEEDS:
            row = by_seed[seed]
            if row["within_tolerance"] != "True":
                raise ValueError(f"Baseline validation mismatch for {condition} seed {seed}")
            saved, evaluated = float(row["saved_val_macro_f1"]), float(row["evaluated_val_macro_f1"])
            if saved != evaluated:
                raise ValueError(f"Baseline score disagreement for {condition} seed {seed}")
            scores[seed], epochs[seed] = evaluated, int(row["selected_epoch"])
        result[condition] = {
            "lr": spec["lr"], "scores": scores, "epochs": epochs,
            "mean": statistics.fmean(scores.values()), "std": statistics.stdev(scores.values()),
            "manifest": str(spec["baseline_manifest"].relative_to(PROJECT_ROOT)),
            "metrics": str(spec["baseline_metrics"].relative_to(PROJECT_ROOT)),
        }
    return result


def _parsed_csv_bytes(data: bytes) -> list[tuple[str, str, str, str]]:
    rows = csv.DictReader(data.decode("utf-8-sig").splitlines())
    return [(row["image_id"], row["relative_path"], row["label"], row["class_name"]) for row in rows]


def verify_splits() -> dict[str, Any]:
    evidence: dict[str, Any] = {"git_blob_match": {}, "parsed_order_match": {}, "line_endings": {}}
    for name in SPLIT_FILES:
        current_oid = git("rev-parse", f"HEAD:splits/v2/{name}")
        evidence["git_blob_match"][name] = {}
        for condition, revision in HISTORICAL_COMMITS.items():
            old_oid = git("rev-parse", f"{revision}:splits/v2/{name}")
            evidence["git_blob_match"][name][condition] = old_oid == current_oid
            if old_oid != current_oid:
                raise ValueError(f"Tracked split content differs for {condition}: {name}")
        current_bytes = (PROJECT_ROOT / "splits" / "v2" / name).read_bytes()
        evidence["line_endings"][name] = {
            "current_sha256": hashlib.sha256(current_bytes).hexdigest(),
            "lf_sha256": hashlib.sha256(current_bytes.replace(b"\r\n", b"\n")).hexdigest(),
            "has_crlf": b"\r\n" in current_bytes,
        }
        if name.endswith(".csv"):
            current_rows = _parsed_csv_bytes(current_bytes)
            evidence["parsed_order_match"][name] = {}
            for condition, revision in HISTORICAL_COMMITS.items():
                old_bytes = subprocess.check_output(
                    ["git", "show", f"{revision}:splits/v2/{name}"], cwd=PROJECT_ROOT
                )
                matched = _parsed_csv_bytes(old_bytes) == current_rows
                evidence["parsed_order_match"][name][condition] = matched
                if not matched:
                    raise ValueError(f"Parsed split rows differ for {condition}: {name}")
    c0_meta = json.loads((PROJECT_ROOT / "deliverables/c0/runs/lr_1e-3/seed_42/metadata.json").read_text(encoding="utf-8"))
    c3_meta = json.loads((PROJECT_ROOT / "deliverables/c3/runs/lr_1e-5/seed_42/metadata.json").read_text(encoding="utf-8"))
    for name in SPLIT_FILES:
        line = evidence["line_endings"][name]
        line["matches_c0_raw"] = line["current_sha256"] == c0_meta["input_sha256"][name]
        line["matches_c3_after_lf_normalization"] = line["lf_sha256"] == c3_meta["input_sha256"][name]
        if not line["matches_c0_raw"] or not line["matches_c3_after_lf_normalization"]:
            raise ValueError(f"Line endings do not fully explain split hash discrepancy: {name}")
    return evidence


def expected_configs() -> dict[tuple[str, int], dict[str, Any]]:
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    allowed_changes = {"augment", "device", "num_workers", "lr", "training_seed"}
    for condition, spec in SPECS.items():
        base = load_config(spec["base_config"], training_seed=42, lr=spec["lr"])
        for seed in SEEDS:
            config = load_config(spec["config"], training_seed=seed, lr=spec["lr"])
            if CONDITIONS[config["unfreeze"]] != condition:
                raise ValueError(f"Trainable-layer policy mismatch for {condition}")
            if config["augment"] is not False or config["device"] != "cpu" or config["num_workers"] != 0:
                raise ValueError(f"Execution override mismatch for {condition}")
            if config["split_seed"] != 42 or config["batch_size"] != 32 or config["micro_batch_size"] != 4:
                raise ValueError(f"Scientific protocol mismatch for {condition}")
            if config["max_epochs"] != 30 or config["patience"] != 5 or "torch_num_threads" in config:
                raise ValueError(f"Epoch/thread protocol mismatch for {condition}")
            unexpected = {key for key in config if config[key] != base[key] and key not in allowed_changes}
            if unexpected:
                raise ValueError(f"Unexpected changes from main config for {condition}: {sorted(unexpected)}")
            expected[(condition, seed)] = config
    return expected


def current_provenance() -> dict[str, Any]:
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": {name: file_sha256(PROJECT_ROOT / name) for name in ("src/train.py", "src/dataset.py")},
        "split_sha256": {name: file_sha256(PROJECT_ROOT / "splits" / "v2" / name) for name in SPLIT_FILES},
        "python": sys.version.split()[0], "torch": str(torch.__version__),
    }


def preflight() -> dict[str, Any]:
    if CELLS != (
        ("C0", 1e-3, 42), ("C0", 1e-3, 43), ("C0", 1e-3, 44),
        ("C3", 1e-5, 42), ("C3", 1e-5, 43), ("C3", 1e-5, 44),
    ):
        raise RuntimeError("The ablation cell list is not the pre-specified six-cell experiment")
    return {
        "cells": [list(cell) for cell in CELLS],
        "baselines": verify_baselines(),
        "splits": verify_splits(),
        "configs": {f"{c}_seed_{s}": cfg for (c, s), cfg in expected_configs().items()},
        "provenance": current_provenance(),
    }


def first_history_max(rows: list[dict[str, str]]) -> tuple[int, float]:
    if not rows:
        raise ValueError("Empty history")
    best_epoch, best_score = 0, float("-inf")
    for row in rows:
        score = float(row["val_macro_f1"])
        if score > best_score:
            best_epoch, best_score = int(row["epoch"]), score
    return best_epoch, best_score


def validate_run(run_dir: Path, expected: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    if runtime.get("status") != "completed":
        raise ValueError("Run is not completed")
    saved_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if saved_config != expected:
        raise ValueError("Effective config mismatch")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["git"]["commit"] != provenance["git_commit"]:
        raise ValueError("Git commit mismatch")
    if metadata["source_sha256"] != provenance["source_sha256"]:
        raise ValueError("Training source mismatch")
    if metadata["input_sha256"] != provenance["split_sha256"]:
        raise ValueError("Split provenance mismatch")
    environment = metadata["environment"]
    if environment["device"] != "cpu" or environment.get("torch_num_threads") != 1:
        raise ValueError("CPU/thread execution mismatch")
    history = read_csv(run_dir / "history.csv")
    best_epoch, best_score = first_history_max(history)
    checkpoint_path = run_dir / "best.pt"
    if checkpoint_path.stat().st_size < 1_000_000:
        raise ValueError("Checkpoint is absent or only an LFS pointer")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config") != expected or checkpoint.get("training_seed") != expected["training_seed"]:
        raise ValueError("Checkpoint config/seed mismatch")
    if checkpoint.get("epoch") != best_epoch or float(checkpoint.get("val_macro_f1")) != best_score:
        raise ValueError("Checkpoint and best history row disagree")
    if runtime.get("best_epoch") != best_epoch or float(runtime.get("best_val_macro_f1")) != best_score:
        raise ValueError("Runtime and best history row disagree")
    return {
        "best_epoch": best_epoch, "best_val_macro_f1": best_score,
        "epochs_completed": len(history), "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "history": str((run_dir / "history.csv").relative_to(PROJECT_ROOT)),
        "metadata": str((run_dir / "metadata.json").relative_to(PROJECT_ROOT)),
    }


def choose_run_dir(condition: str, seed: int, expected: dict[str, Any], provenance: dict[str, Any]) -> tuple[Path, bool]:
    base = RUNS_ROOT / f"{SPECS[condition]['slug']}_seed{seed}"
    candidates = [base, *sorted(RUNS_ROOT.glob(f"{base.name}_attempt*"))]
    for candidate in candidates:
        if candidate.is_dir():
            try:
                validate_run(candidate, expected, provenance)
                return candidate, True
            except (OSError, ValueError, KeyError, RuntimeError):
                pass
    if not base.exists():
        return base, False
    attempt = 2
    while (RUNS_ROOT / f"{base.name}_attempt{attempt}").exists():
        attempt += 1
    return RUNS_ROOT / f"{base.name}_attempt{attempt}", False


def monitor(process: subprocess.Popen[str], context: dict[str, Any], status: dict[str, Any], status_path: Path) -> int:
    messages: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line.rstrip("\r\n"))
        messages.put(None)

    threading.Thread(target=reader, daemon=True).start()
    closed = False
    while not closed or process.poll() is None:
        try:
            line = messages.get(timeout=30)
        except queue.Empty:
            status.update(heartbeat_at_utc=now(), heartbeat="trainer alive")
            atomic_json(status_path, status)
            print(
                f"HEARTBEAT condition={context['condition']} run={context['index']}/6 "
                f"lr={context['lr']:.0e} seed={context['seed']} pid={process.pid}", flush=True,
            )
            continue
        if line is None:
            closed = True
            continue
        match = EPOCH_RE.match(line)
        if match:
            epoch, score = int(match.group(2)), float(match.group(3))
            status.update(epoch=epoch, val_macro_f1=score, heartbeat_at_utc=now(), heartbeat="epoch completed")
            atomic_json(status_path, status)
            print(
                f"condition={context['condition']} run={context['index']}/6 lr={context['lr']:.0e} "
                f"seed={context['seed']} epoch={epoch} val_macro_f1={score:.6f}", flush=True,
            )
        else:
            print(
                f"condition={context['condition']} run={context['index']}/6 lr={context['lr']:.0e} "
                f"seed={context['seed']} trainer={line}", flush=True,
            )
    return process.wait()


def aggregate_rows(results: Mapping[tuple[str, int], Mapping[str, Any]], baselines: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if set(results) != {(condition, seed) for condition in SPECS for seed in SEEDS}:
        raise ValueError("Refusing to aggregate an incomplete six-cell grid")
    rows: list[dict[str, Any]] = []
    for condition in ("C0", "C3"):
        for augmentation, scores, evidence in (
            ("on", baselines[condition]["scores"], baselines[condition]["metrics"]),
            ("off", {seed: float(results[(condition, seed)]["best_val_macro_f1"]) for seed in SEEDS},
             str((OUTPUT_ROOT / "runs").relative_to(PROJECT_ROOT))),
        ):
            values = [scores[seed] for seed in SEEDS]
            rows.append({
                "condition": condition, "augmentation": augmentation, "lr": SPECS[condition]["lr"],
                **{f"seed_{seed}_val_macro_f1": scores[seed] for seed in SEEDS},
                "mean_val_macro_f1": statistics.fmean(values),
                "sample_std_val_macro_f1": statistics.stdev(values), "evidence": evidence,
            })
    return rows


def write_summary_and_report(rows: list[dict[str, Any]], preflight_data: Mapping[str, Any], results: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "condition", "augmentation", "lr", "seed_42_val_macro_f1", "seed_43_val_macro_f1",
        "seed_44_val_macro_f1", "mean_val_macro_f1", "sample_std_val_macro_f1", "evidence",
    )
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: format(value, ".17g") if isinstance(value, float) else value for key, value in row.items()} for row in rows)
    indexed = {(row["condition"], row["augmentation"]): row for row in rows}
    deltas = {
        condition: indexed[(condition, "off")]["mean_val_macro_f1"] - indexed[(condition, "on")]["mean_val_macro_f1"]
        for condition in ("C0", "C3")
    }
    interpretation = (
        "В этих шести запусках отключение аугментации изменило средний validation macro-F1 "
        f"на {deltas['C0']:+.6f} для C0 и на {deltas['C3']:+.6f} для C3; это описательное "
        "сравнение не доказывает статистическую значимость или причинность."
    )
    lines = [
        "# Ablation аугментации", "",
        "Validation macro-F1: среднее ± выборочное стандартное отклонение по seeds 42, 43, 44 (ddof=1).", "",
        "| Condition | Аугментация on | Аугментация off |", "|---|---:|---:|",
    ]
    for condition in ("C0", "C3"):
        on, off = indexed[(condition, "on")], indexed[(condition, "off")]
        lines.append(
            f"| {condition} | {on['mean_val_macro_f1']:.7f} ± {on['sample_std_val_macro_f1']:.7f} "
            f"| {off['mean_val_macro_f1']:.7f} ± {off['sample_std_val_macro_f1']:.7f} |"
        )
    lines.extend([
        "", "## Протокол", "",
        "- C0: LR `1e-3`; C3: LR `1e-5`; нового LR-поиска не выполнялось.",
        "- Training seeds: `42, 43, 44`; `split_seed=42`.",
        "- Checkpoint выбирался по лучшему validation macro-F1; максимум 30 epochs, early stopping patience 5.",
        "- Аугментация off использует детерминированный `ResNet50_Weights.IMAGENET1K_V2.transforms()`; train shuffle сохранён.",
        "- Новые запуски: CPU, `num_workers=0`, один CPU thread, float32, текущий shared trainer.",
        "", "## Источники", "",
        f"- Полная агрегация: `{SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix()}`.",
        f"- C0-on: `{SPECS['C0']['baseline_metrics'].relative_to(PROJECT_ROOT).as_posix()}`.",
        f"- C3-on: `{SPECS['C3']['baseline_metrics'].relative_to(PROJECT_ROOT).as_posix()}`.",
        f"- C0/C3-off histories, configs, metadata и checkpoints: `{RUNS_ROOT.relative_to(PROJECT_ROOT).as_posix()}/`.",
        "", "## Ограничения сопоставимости", "",
        "C0-on был обучен на CPU с Python 3.14.6 / torch 2.14.1 при историческом dirty commit `155bf1b`; "
        "C3-on — на Tesla T4 с Python 3.13.15 / torch 2.11.0+cu128 при commit `37f1e66`; "
        f"оба off-набора выполнены локально на CPU текущим trainer из commit `{preflight_data['provenance']['git_commit'][:7]}`. "
        "Исторические same-seed head hashes C0/C3 различаются, поэтому сравнение не считается идеально контролируемой "
        "изоляцией аугментации. Split membership, labels и row order совпадают; различие byte hashes C3 объясняется CRLF/LF.",
        "", "## Интерпретация", "", interpretation, "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


class PipelineLock:
    def __init__(self, path: Path):
        self.path, self.handle = path, None

    def __enter__(self) -> "PipelineLock":
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.handle.seek(0, os.SEEK_END) == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError("Another ablation orchestrator holds the pipeline lock") from exc
        return self

    def __exit__(self, *_: Any) -> None:
        import msvcrt
        assert self.handle is not None
        self.handle.seek(0)
        msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        self.handle.close()


def run_pipeline() -> None:
    with PipelineLock(OUTPUT_ROOT / "pipeline.lock"):
        data = preflight()
        expected = expected_configs()
        provenance = data["provenance"]
        atomic_json(OUTPUT_ROOT / "preflight.json", data)
        status_path = OUTPUT_ROOT / "status.json"
        status: dict[str, Any] = {
            "status": "RUNNING", "stage": "training", "orchestrator_pid": os.getpid(),
            "completed_run_count": 0, "total_runs": 6, "started_at_utc": now(),
        }
        atomic_json(status_path, status)
        results: dict[tuple[str, int], dict[str, Any]] = {}
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            for index, (condition, lr, seed) in enumerate(CELLS, start=1):
                config = expected[(condition, seed)]
                run_dir, reuse = choose_run_dir(condition, seed, config, provenance)
                context = {"condition": condition, "lr": lr, "seed": seed, "index": index}
                status.update(
                    current_condition=condition, current_lr=lr, current_seed=seed,
                    current_run=index, output_dir=str(run_dir), epoch=None, val_macro_f1=None,
                    heartbeat="preparing run", heartbeat_at_utc=now(), trainer_pid=None,
                )
                atomic_json(status_path, status)
                print(
                    f"START condition={condition} run={index}/6 lr={lr:.0e} seed={seed} "
                    f"output={run_dir} reuse={reuse}", flush=True,
                )
                if not reuse:
                    command = [
                        sys.executable, "-u", "-m", "src.train", "--config", str(SPECS[condition]["config"]),
                        "--training-seed", str(seed), "--lr", repr(lr), "--output-dir", str(run_dir),
                    ]
                    process = subprocess.Popen(
                        command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=None,
                        text=True, bufsize=1,
                    )
                    status.update(trainer_pid=process.pid, heartbeat="trainer started", heartbeat_at_utc=now())
                    atomic_json(status_path, status)
                    print(f"TRAINER_STARTED pid={process.pid} command={' '.join(command)}", flush=True)
                    returncode = monitor(process, context, status, status_path)
                    if returncode != 0:
                        raise RuntimeError(f"Trainer failed for {condition} seed {seed} with exit code {returncode}")
                results[(condition, seed)] = validate_run(run_dir, config, provenance)
                status.update(
                    completed_run_count=len(results), trainer_pid=None, heartbeat="run verified",
                    heartbeat_at_utc=now(), last_completed_output=str(run_dir),
                )
                atomic_json(status_path, status)
                print(
                    f"VERIFIED condition={condition} run={index}/6 lr={lr:.0e} seed={seed} "
                    f"best_epoch={results[(condition, seed)]['best_epoch']} "
                    f"best_val_macro_f1={results[(condition, seed)]['best_val_macro_f1']:.17g}", flush=True,
                )
            status.update(stage="aggregation", heartbeat="aggregating", heartbeat_at_utc=now())
            atomic_json(status_path, status)
            rows = aggregate_rows(results, data["baselines"])
            write_summary_and_report(rows, data, results)
            atomic_json(OUTPUT_ROOT / "results.json", {
                f"{condition}_seed_{seed}": value for (condition, seed), value in results.items()
            })
            status.update(
                status="COMPLETED", stage="completed", trainer_pid=None,
                summary=str(SUMMARY_PATH), report=str(REPORT_PATH), completed_at_utc=now(),
                heartbeat="pipeline completed", heartbeat_at_utc=now(),
            )
            atomic_json(status_path, status)
            print(f"COMPLETED summary={SUMMARY_PATH} report={REPORT_PATH}", flush=True)
        except BaseException as exc:
            status.update(
                status="FAILED", trainer_pid=None, error=f"{type(exc).__name__}: {exc}",
                failed_at_utc=now(), heartbeat="pipeline failed", heartbeat_at_utc=now(),
            )
            atomic_json(status_path, status)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and fixed cells without training")
    args = parser.parse_args()
    if args.check_only:
        data = preflight()
        print(json.dumps({
            "status": "ok", "cells": data["cells"], "git_commit": data["provenance"]["git_commit"],
            "split_rows_and_order_match": data["splits"]["parsed_order_match"],
        }, indent=2))
    else:
        run_pipeline()


if __name__ == "__main__":
    main()

