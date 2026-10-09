"""Shared, deterministic ResNet50 training. Run with ``python -m src.train``.
Experiment owners change YAML configurations, never condition-specific code.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.dataset import get_loaders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"head": "C0", "layer4": "C1", "layer3_layer4": "C2", "all": "C3"}
TRAINABLE_MODULES = {
    "head": ("fc",),
    "layer4": ("fc", "layer4"),
    "layer3_layer4": ("fc", "layer3", "layer4"),
    "all": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"),
}

FIELDS = {
    "dataset_root", "split_dir", "batch_size", "num_workers", "augment",
    "training_seed", "split_seed", "unfreeze", "lr", "micro_batch_size",
    "max_epochs", "patience", "device", "weight_decay", "betas", "eps",
}

HISTORY_FIELDS = (
    "epoch", "lr", "train_loss", "train_accuracy", "val_loss",
    "val_accuracy", "val_macro_f1",
)

BN_BUFFERS = {"running_mean", "running_var", "num_batches_tracked"}


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete, flat configuration; do not mutate the caller."""
    result = dict(config)
    unknown, missing = set(result) - FIELDS, FIELDS - set(result)
    if unknown or missing:
        raise ValueError(f"Config fields: unknown={sorted(unknown)}, missing={sorted(missing)}")
    
    for key in ("batch_size", "num_workers", "training_seed", "split_seed",
                "micro_batch_size", "max_epochs", "patience"):
        if type(result[key]) is not int:
            raise ValueError(f"{key} must be an integer")
        
    if result["training_seed"] not in (42, 43, 44) or result["split_seed"] != 42:
        raise ValueError("training_seed must be 42/43/44; split_seed must be 42")
    
    if not isinstance(result["unfreeze"], str) or result["unfreeze"] not in CONDITIONS:
        raise ValueError(f"unfreeze must be one of {tuple(CONDITIONS)}")
    
    for key in ("lr", "weight_decay", "eps"):
        if isinstance(result[key], bool):
            raise ValueError(f"{key} must be a finite number")
        try:
            result[key] = float(result[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a finite number") from exc
        
        if not math.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
        
    if result["lr"] not in (1e-3, 1e-4, 1e-5):
        raise ValueError("lr must be 1e-3, 1e-4 or 1e-5")
    
    if result["weight_decay"] != 0.01 or result["eps"] != 1e-8 or result["betas"] != [0.9, 0.999]:
        raise ValueError("Protocol fixes AdamW weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8")
    
    if result["batch_size"] != 32 or not 1 <= result["micro_batch_size"] <= 32:
        raise ValueError("batch_size must be 32; micro_batch_size must be in 1..32")
    
    if result["num_workers"] < 0 or not 1 <= result["max_epochs"] <= 30 or result["patience"] != 5:
        raise ValueError("num_workers must be >=0, max_epochs in 1..30, patience=5")
    
    if type(result["augment"]) is not bool or result["device"] not in ("cpu", "cuda"):
        raise ValueError("augment must be a boolean; device must be cpu or cuda")
    
    for key in ("dataset_root", "split_dir"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a nonempty path string")
    
    return result


def load_config(path: str | Path, *, training_seed: int | None = None,
                lr: float | None = None) -> dict[str, Any]:
    """Resolve single-parent YAML inheritance, then apply supported CLI overrides."""
    def read(current: Path, ancestors: tuple[Path, ...]) -> dict[str, Any]:
        current = current.resolve()
        if current in ancestors:
            raise ValueError(f"Config inheritance cycle: {current}")
        
        payload = yaml.safe_load(current.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not all(isinstance(k, str) for k in payload):
            raise ValueError(f"Config must be a string-keyed mapping: {current}")
        
        unknown = set(payload) - FIELDS - {"extends"}
        if unknown:
            raise ValueError(f"Unknown config fields in {current}: {sorted(unknown)}")
        
        parent = payload.pop("extends", None)
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            raise ValueError(f"extends must name one YAML file: {current}")
        
        base = read(current.parent / parent, (*ancestors, current)) if parent else {}
        
        return base | payload

    config = read(Path(path), ())
    if training_seed is not None:
        config["training_seed"] = training_seed
        
    if lr is not None:
        config["lr"] = lr
    return validate_config(config)


def configure_determinism(seed: int) -> None:
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Configure CUBLAS_WORKSPACE_CONFIG=:4096:8 before initializing CUDA")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)


def apply_unfreeze(model: nn.Module, unfreeze: str) -> None:
    if unfreeze not in TRAINABLE_MODULES:
        raise ValueError(f"Unknown unfreeze: {unfreeze}")
    
    model.requires_grad_(False)
    for name in TRAINABLE_MODULES[unfreeze]:
        model.get_submodule(name).requires_grad_(True)


def build_model(training_seed: int, unfreeze: str) -> nn.Module:
    """Always use ImageNet V2; isolated CPU head RNG is independent of condition."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(training_seed)
        model.fc = nn.Linear(model.fc.in_features, 10)
        
    apply_unfreeze(model, unfreeze)
    return model


def set_training_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def snapshot_bn(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{key}": buffer.detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for key, buffer in module.named_buffers(recurse=False)
        if key in BN_BUFFERS
    }


def assert_bn_unchanged(model: nn.Module, initial: dict[str, torch.Tensor]) -> None:
    current = snapshot_bn(model)
    if current.keys() != initial.keys() or any(not torch.equal(current[k], initial[k]) for k in initial):
        raise RuntimeError("BatchNorm running statistics changed")


def hash_tensors(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_epoch(model: nn.Module, loader: Any, device: torch.device,
              micro_batch_size: int, optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
    """Train when an optimizer is supplied; otherwise validate without gradients."""
    training = optimizer is not None
    if training:
        set_training_mode(model)
    else:
        model.eval()
        
    total_loss, correct, count = 0.0, 0, 0
    targets, predictions = [], []
    
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            batch_size = labels.numel()
            if training:
                optimizer.zero_grad(set_to_none=True)
                
            for start in range(0, batch_size, micro_batch_size):
                x = images[start:start + micro_batch_size].to(device)
                y = labels[start:start + micro_batch_size].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction="sum")
                
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite loss/logits")
                
                if training:
                    (loss / batch_size).backward()
                    
                total_loss += loss.item()
                predicted = logits.detach().argmax(dim=1)
                correct += (predicted == y).sum().item()
                count += y.numel()
                
                if not training:
                    targets.extend(y.cpu().tolist())
                    predictions.extend(predicted.cpu().tolist())
                    
            if training:
                finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not finite or not torch.stack(finite).all().item():
                    raise FloatingPointError("Non-finite or missing gradients")
                
                optimizer.step()
    if count == 0:
        raise ValueError("Cannot compute an epoch on an empty loader")
    
    metrics = {"loss": total_loss / count, "accuracy": correct / count}
    if not training:
        metrics["macro_f1"] = float(f1_score(targets, predictions, labels=list(range(10)),
                                             average="macro", zero_division=0))
    return metrics


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, model: nn.Module, *, epoch: int, score: float,
                    config: dict[str, Any], mapping: Any, git: dict[str, Any]) -> None:
    payload = {
        "format_version": 1, "architecture": "resnet50", "weights": "IMAGENET1K_V2",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": epoch, "val_macro_f1": score, "config": config,
        "class_mapping": mapping, "training_seed": config["training_seed"], "git": git,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Run one condition. Existing output directories are never reused or overwritten."""
    config = validate_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config_text = yaml.safe_dump(config, sort_keys=True)
    (output / "config.yaml").write_text(config_text, encoding="utf-8", newline="\n")
    started = time.perf_counter()
    runtime: dict[str, Any] = {"status": "running", "epochs_completed": 0, "bn_checks_passed": 0}
    write_json(output / "runtime.json", runtime)
    device = torch.device(config["device"])
    cuda_started = False
    
    try:
        configure_determinism(config["training_seed"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Install a CUDA PyTorch build; no CPU fallback.")
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            cuda_started = True
            
        data_config = dict(config)
        for key in ("dataset_root", "split_dir"):
            data_config[key] = str((PROJECT_ROOT / config[key]).resolve())
            
        split_dir = Path(data_config["split_dir"])
        report = json.loads((split_dir / "split_report.json").read_text(encoding="utf-8"))
        if report["seed"] != config["split_seed"]:
            raise ValueError("Frozen split seed does not match split_seed")
        
        mapping = json.loads((split_dir / "class_mapping.json").read_text(encoding="utf-8"))
        git = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()),
        }
        
        metadata = {
            "training_seed": config["training_seed"], "split_seed": config["split_seed"],
            "condition": CONDITIONS[config["unfreeze"]], "git": git,
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "input_sha256": {name: file_sha256(split_dir / name) for name in
                             ("train.csv", "val.csv", "test.csv", "class_mapping.json", "split_report.json")},
            "source_sha256": {name: file_sha256(PROJECT_ROOT / name) for name in ("src/train.py", "src/dataset.py")},
            "environment": {
                "python": sys.version.split()[0], "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "architecture": "resnet50", "weights": "IMAGENET1K_V2",
        }
        
        write_json(output / "metadata.json", metadata)
        loaders = get_loaders(data_config)
        model = build_model(config["training_seed"], config["unfreeze"])
        initial_bn = snapshot_bn(model)
        
        metadata.update({
            "head_init_sha256": hash_tensors(model.fc.state_dict()),
            "bn_initial_sha256": hash_tensors(initial_bn),
            "trainable_parameters": [name for name, p in model.named_parameters() if p.requires_grad],
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        
        write_json(output / "metadata.json", metadata)
        model.to(device)
        
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=config["lr"],
            weight_decay=config["weight_decay"], betas=tuple(config["betas"]),
            eps=config["eps"], foreach=False, fused=False,
        )
        
        best_score, best_epoch, bad_epochs = -math.inf, 0, 0
        
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, lineterminator="\n")
            writer.writeheader()
            handle.flush()
            for epoch in range(1, config["max_epochs"] + 1):
                training = run_epoch(model, loaders["train"], device, config["micro_batch_size"], optimizer)
                validation = run_epoch(model, loaders["val"], device, config["micro_batch_size"])
                assert_bn_unchanged(model, initial_bn)
                
                row = {
                    "epoch": epoch, "lr": config["lr"],
                    "train_loss": training["loss"], "train_accuracy": training["accuracy"],
                    "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                    "val_macro_f1": validation["macro_f1"],
                }
                
                if not all(math.isfinite(v) for v in row.values()):
                    raise FloatingPointError("Non-finite epoch metrics")
                
                if validation["macro_f1"] > best_score:
                    best_score, best_epoch, bad_epochs = validation["macro_f1"], epoch, 0
                    save_checkpoint(output / "best.pt", model, epoch=epoch, score=best_score,
                                    config=config, mapping=mapping, git=git)
                else:
                    bad_epochs += 1
                
                writer.writerow({key: value if key == "epoch" else format(value, ".17g") for key, value in row.items()})
                handle.flush()
                runtime.update(epochs_completed=epoch, bn_checks_passed=epoch)
                
                print(f"{CONDITIONS[config['unfreeze']]} epoch={epoch} "
                      f"train_loss={training['loss']:.6f} val_macro_f1={validation['macro_f1']:.6f}", flush=True)
                
                if bad_epochs >= config["patience"]:
                    break

        runtime.update(status="completed", best_epoch=best_epoch,
                       best_val_macro_f1=best_score, early_stopped=bad_epochs >= config["patience"],
                       bn_final_sha256=hash_tensors(snapshot_bn(model)))
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            
    except BaseException as exc:
        runtime.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    
    finally:
        runtime["elapsed_seconds"] = time.perf_counter() - started
        runtime["peak_gpu_memory_bytes"] = (
            torch.cuda.max_memory_allocated(device) if cuda_started else None
        )
        write_json(output / "runtime.json", runtime)
        
    return runtime


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--lr", type=float, choices=(1e-3, 1e-4, 1e-5))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, training_seed=args.training_seed, lr=args.lr)
    train(config, args.output_dir)


if __name__ == "__main__":
    main()
