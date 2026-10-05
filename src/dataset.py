"""Dataset and deterministic DataLoader construction for frozen Pet manifests.

The public contract is ``get_loaders(config) -> {'train', 'val', 'test'}``.
``config`` may be a mapping, a dataclass/namespace, or an object with a
``data`` mapping/namespace. Required fields are ``dataset_root``, ``split_dir``,
``batch_size``, ``num_workers``, ``augment``, and ``training_seed`` (``seed``
is accepted as a backwards-compatible alias).

This module is deliberately read-only with respect to ``splits/``: manifests
are produced only by ``scripts/prepare_oxford_pets.py``.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision.models import ResNet50_Weights
    from torchvision.transforms import ColorJitter, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
except ImportError as exc:  # pragma: no cover - gives users an actionable message
    raise ImportError(
        "src.dataset requires torchvision. Install dependencies with "
        "`py -3 -m pip install -r requirements.txt`."
    ) from exc


REQUIRED_COLUMNS = {"image_id", "relative_path", "class_name", "label", "content_hash"}
EXPECTED_LABELS = set(range(10))


def _get_value(config: Any, name: str, *, default: Any = None, required: bool = False) -> Any:
    """Read a field from mapping/namespace, accepting an optional ``data`` nest."""
    sources = (config,)
    if isinstance(config, Mapping) and "data" in config:
        sources = (config["data"], config)
    elif hasattr(config, "data"):
        sources = (getattr(config, "data"), config)
    for source in sources:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    if required:
        raise ValueError(f"Missing required data config field: {name}")
    return default


def _read_mapping(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        classes = payload["classes"]
        labels = [entry["label"] for entry in classes]
        names = [entry["class_name"] for entry in classes]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid class mapping at {path}: {exc}") from exc
    if labels != list(range(10)) or len(set(names)) != 10:
        raise ValueError(f"Mapping {path} must contain exactly labels 0..9 once.")
    return dict(zip(labels, names))


class PetManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    """A strict RGB image dataset backed by one frozen CSV manifest."""

    def __init__(self, dataset_root: str | Path, manifest_path: str | Path, mapping_path: str | Path, transform: Callable):
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.mapping = _read_mapping(Path(mapping_path))
        self.transform = transform
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset_root does not exist or is not a directory: {self.dataset_root}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest is missing: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                raise ValueError(f"Manifest {self.manifest_path} must contain {sorted(REQUIRED_COLUMNS)}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        for row_number, row in enumerate(self.rows, start=2):
            try:
                label = int(row["label"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid label in {self.manifest_path}:{row_number}: {row.get('label')!r}") from exc
            if label not in EXPECTED_LABELS or self.mapping[label] != row["class_name"]:
                raise ValueError(
                    f"Mapping mismatch in {self.manifest_path}:{row_number}: "
                    f"label={label}, class_name={row.get('class_name')!r}"
                )
            relative = Path(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe relative_path in {self.manifest_path}:{row_number}: {relative}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = self.dataset_root / row["relative_path"]
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Image listed in {self.manifest_path} is missing: {image_path}") from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Image listed in {self.manifest_path} cannot be decoded as RGB: {image_path}: {exc}") from exc
        return self.transform(image), int(row["label"])


def _train_transform() -> Compose:
    weights = ResNet50_Weights.IMAGENET1K_V2
    return Compose([
        RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.3333)),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ToTensor(),
        Normalize(mean=weights.meta["mean"] if "mean" in weights.meta else (0.485, 0.456, 0.406),
                  std=weights.meta["std"] if "std" in weights.meta else (0.229, 0.224, 0.225)),
    ])


def _eval_transform() -> Callable:
    return ResNet50_Weights.IMAGENET1K_V2.transforms()


def _seed_worker(worker_id: int) -> None:
    # torch seeds each worker from DataLoader.generator; mirror that state in
    # Python and NumPy so future transform additions remain reproducible.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:  # NumPy is optional for this module today.
        pass


def get_loaders(config: Any) -> dict[str, DataLoader]:
    """Return train/validation/test loaders from existing frozen manifests.

    ``augment=False`` selects the exact deterministic ImageNet V2 transform for
    train too. It does not turn off training-set shuffling.
    """
    dataset_root = Path(_get_value(config, "dataset_root", required=True))
    split_dir = Path(_get_value(config, "split_dir", required=True))
    batch_size = int(_get_value(config, "batch_size", required=True))
    num_workers = int(_get_value(config, "num_workers", required=True))
    augment = bool(_get_value(config, "augment", required=True))
    seed = _get_value(config, "training_seed", default=None)
    if seed is None:
        seed = _get_value(config, "seed", required=True)
    seed = int(seed)
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")

    mapping_path = split_dir / "class_mapping.json"
    eval_transform = _eval_transform()
    transforms = {"train": _train_transform() if augment else eval_transform, "val": eval_transform, "test": eval_transform}
    datasets = {
        name: PetManifestDataset(dataset_root, split_dir / f"{name}.csv", mapping_path, transforms[name])
        for name in ("train", "val", "test")
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            # Independent streams prevent validation/test iteration from
            # changing the reproducible training shuffle order.
            generator=torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[name]),
        )
        for name, dataset in datasets.items()
    }
