#!/usr/bin/env python3
"""Save a Grad-CAM overlay for one image from a frozen evaluation split.

Example::

    python scripts/generate_gradcam.py --checkpoint deliverables/c3/runs/lr_1e-5/seed_42/best.pt \
        --config configs/c3.yaml --image-id pug_123 --output reports/figures/gradcam_pug_123.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import LABELS, _load_model, _resolve_config
from src.dataset import get_loaders


def generate(checkpoint: Path, config_path: Path, image_id: str, output: Path, device_override: str | None) -> Path:
    config = _resolve_config(config_path, device_override)
    device = torch.device(config["device"])
    dataset = get_loaders(config)["test"].dataset
    matches = [index for index, row in enumerate(dataset.rows) if row["image_id"] == image_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one test image with ID {image_id!r}, found {len(matches)}")

    model, _ = _load_model(checkpoint, device)
    activation: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], result: torch.Tensor) -> None:
        result.retain_grad()
        activation.append(result)

    handle = model.layer4.register_forward_hook(capture)
    image, target = dataset[matches[0]]
    image_batch = image.unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    logits = model(image_batch)
    predicted = int(logits.argmax(dim=1).item())
    logits[0, predicted].backward()
    handle.remove()

    feature_map = activation[0]
    gradient = feature_map.grad
    if gradient is None:
        raise RuntimeError("Grad-CAM target layer did not retain gradients")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    heatmap = F.relu((weights * feature_map).sum(dim=1, keepdim=True))
    heatmap = F.interpolate(heatmap, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    rgb = (image.cpu() * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    class_names = [dataset.mapping[index] for index in LABELS]
    confidence = float(logits.softmax(dim=1)[0, predicted].item())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title(f"Input · true: {class_names[target]} · predicted: {class_names[predicted]}")
    axes[1].imshow(rgb)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM · {class_names[predicted]} ({confidence:.3f})")
    for axis in axes:
        axis.axis("off")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"true={class_names[target]} predicted={class_names[predicted]} confidence={confidence:.6f}")
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-id", required=True, help="image_id from the selected test split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    generate(args.checkpoint, args.config, args.image_id, args.output, args.device)


if __name__ == "__main__":
    main()
