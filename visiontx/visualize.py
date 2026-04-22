"""
Visualization utilities for Project 4: Vision Transformer.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor


def plot_training_curves(log_path: str, output_path: str) -> None:
    """
    Plot train loss and val accuracy vs epoch from a JSONL log file.

    Args:
        log_path:    Path to the JSONL experiment log.
        output_path: Path to save the output PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    epochs: list[int] = []
    train_losses: list[float] = []
    val_accs: list[float] = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "val_epoch":
                epoch = entry.get("epoch", len(epochs) + 1)
                epochs.append(epoch)
                train_losses.append(entry.get("avg_train_loss", float("nan")))
                val_accs.append(entry.get("val_accuracy", float("nan")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, marker="o", linewidth=1.5, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accs, marker="o", linewidth=1.5, color="orange", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patch_grid(image: Tensor, patch_size: int, output_path: str) -> None:
    """
    Visualize how an image is divided into patches with a grid overlay.

    Args:
        image:       (C, H, W) image tensor (values in any range; will be normalized for display).
        patch_size:  Size of each patch in pixels.
        output_path: Path to save the output PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib and numpy are required for plotting.")

    # Convert tensor to numpy for display
    img = image.detach().cpu().float()

    # Normalize to [0, 1] for display
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    img = img.clamp(0.0, 1.0)

    # (C, H, W) -> (H, W, C)
    if img.shape[0] == 1:
        img_np = img.squeeze(0).numpy()
        cmap = "gray"
    else:
        img_np = img.permute(1, 2, 0).numpy()
        cmap = None

    H, W = img_np.shape[:2] if img_np.ndim == 3 else img_np.shape
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_np, cmap=cmap, interpolation="nearest")

    # Draw patch grid lines
    for i in range(n_patches_h + 1):
        ax.axhline(y=i * patch_size - 0.5, color="red", linewidth=0.8, alpha=0.8)
    for j in range(n_patches_w + 1):
        ax.axvline(x=j * patch_size - 0.5, color="red", linewidth=0.8, alpha=0.8)

    ax.set_title(
        f"Patch Grid (patch_size={patch_size}, "
        f"{n_patches_h}×{n_patches_w}={n_patches_h * n_patches_w} patches)"
    )
    ax.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
