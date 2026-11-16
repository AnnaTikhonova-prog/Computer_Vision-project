"""Small CPU-only serving path; never trains or writes evaluation artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "deliverables/c1/runs/lr_1e-4/seed_42/best.pt"
CHECKPOINT_SHA256 = "d4cda19030d5ee60eb2c8b7ec79a3444cf366f5922d1a9f09acdac2a61ec1384"
MAX_BYTES = 10 * 1024 * 1024
MAX_SOURCE_PIXELS = 64_000_000
MAX_IMAGE_SIDE = 2048


def read_image(data: bytes) -> Image.Image:
    """Decode camera photos at a bounded size, preserving orientation and aspect ratio."""
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Please choose an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Please choose a JPG, PNG, or WebP image.")
                if image.width * image.height > MAX_SOURCE_PIXELS:
                    raise ValueError("This image is too large to process. Please use a photo under 64 megapixels.")
                # JPEG draft reduces decode memory before thumbnailing. Other
                # formats use the same final bound without changing model transforms.
                image.draft("RGB", (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
                image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as exc:
        raise ValueError("This image could not be opened. Try another JPG, PNG, or WebP.") from exc


def load_predictor():
    """Load one fixed, hash-verified checkpoint without downloading ImageNet weights."""
    if not CHECKPOINT.is_file():
        raise FileNotFoundError("Model weights are missing. Download the demo checkpoint with Git LFS.")
    with CHECKPOINT.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Model weights are incomplete or unexpected. Download the demo checkpoint with Git LFS.")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    mapping = json.loads((ROOT / "splits/v2/class_mapping.json").read_text(encoding="utf-8"))
    if (payload.get("architecture") != "resnet50"
            or payload.get("weights") != "IMAGENET1K_V2"
            or payload.get("class_mapping") != mapping):
        raise ValueError("The model does not match this demo's breed mapping.")
    torch.set_num_threads(1)
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    names = tuple(item["class_name"] for item in sorted(mapping["classes"], key=lambda item: item["label"]))
    return model, ResNet50_Weights.IMAGENET1K_V2.transforms(), names


def predict(image: Image.Image, predictor) -> list[tuple[str, float]]:
    model, transform, names = predictor
    with torch.inference_mode():
        scores = model(transform(image).unsqueeze(0)).softmax(dim=1)[0]
    values, labels = scores.topk(3)
    return [(names[label], float(value)) for label, value in zip(labels.tolist(), values.tolist())]
