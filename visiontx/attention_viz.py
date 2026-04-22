"""
Attention visualization utilities for Project 4: Vision Transformer.

References:
    # Ref: Abnar & Zuidema, 2020 — "Quantifying Attention Flow in Transformers"
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


def attention_rollout(
    attention_weights: list[Tensor],
    discard_ratio: float = 0.9,
) -> Tensor:
    """
    Abnar & Zuidema (2020) attention rollout.

    Computes the effective attention from the class token to each patch by
    recursively multiplying attention matrices across layers, accounting for
    residual connections.

    # Ref: Abnar & Zuidema, 2020 — "Quantifying Attention Flow in Transformers"

    Args:
        attention_weights: List of attention weight tensors per layer.
                           Each tensor has shape (B, n_heads, T, T) where
                           T = n_patches + 1 (includes class token at position 0).
        discard_ratio:     Fraction of lowest attention weights to discard
                           (set to zero) before rollout. Default: 0.9.

    Returns:
        (n_patches,) relevance scores for each patch (excluding class token),
        averaged over the batch dimension.
    """
    if not attention_weights:
        raise ValueError("attention_weights list is empty")

    # Use the first item to determine shapes
    B, n_heads, T, _ = attention_weights[0].shape

    # Initialize rollout as identity matrix
    rollout = torch.eye(T, device=attention_weights[0].device)  # (T, T)
    rollout = rollout.unsqueeze(0).expand(B, -1, -1)             # (B, T, T)

    for attn in attention_weights:
        # attn: (B, n_heads, T, T)
        # Average over heads
        attn_avg = attn.mean(dim=1)  # (B, T, T)

        # Discard lowest attention weights
        if discard_ratio > 0.0:
            flat = attn_avg.view(B, -1)
            threshold_idx = int(discard_ratio * flat.shape[-1])
            if threshold_idx > 0:
                # Find threshold value per batch item
                sorted_vals, _ = flat.sort(dim=-1)
                threshold = sorted_vals[:, threshold_idx - 1].unsqueeze(-1).unsqueeze(-1)
                attn_avg = torch.where(attn_avg < threshold, torch.zeros_like(attn_avg), attn_avg)

        # Add residual connection: A_hat = 0.5 * A + 0.5 * I
        identity = torch.eye(T, device=attn_avg.device).unsqueeze(0).expand(B, -1, -1)
        attn_hat = 0.5 * attn_avg + 0.5 * identity

        # Normalize rows to sum to 1
        row_sums = attn_hat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_hat = attn_hat / row_sums

        # Multiply rollout: rollout = attn_hat @ rollout
        rollout = torch.bmm(attn_hat, rollout)  # (B, T, T)

    # Extract class token row (position 0) -> relevance for all tokens
    # Shape: (B, T) — relevance of each token from the class token's perspective
    cls_relevance = rollout[:, 0, :]  # (B, T)

    # Exclude class token itself (position 0), keep patch tokens
    patch_relevance = cls_relevance[:, 1:]  # (B, n_patches)

    # Average over batch
    patch_relevance = patch_relevance.mean(dim=0)  # (n_patches,)

    # Normalize to [0, 1]
    min_val = patch_relevance.min()
    max_val = patch_relevance.max()
    if max_val > min_val:
        patch_relevance = (patch_relevance - min_val) / (max_val - min_val)

    return patch_relevance


def overlay_attention_on_image(
    image: Tensor,
    rollout: Tensor,
    patch_size: int,
    save_path: str,
) -> None:
    """
    Overlay attention heatmap on original image and save as PNG.

    Args:
        image:      (C, H, W) image tensor (values in any range).
        rollout:    (n_patches,) relevance scores from attention_rollout().
        patch_size: Patch size used in the ViT model.
        save_path:  Path to save the output PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib and numpy are required for visualization.")

    img = image.detach().cpu().float()

    # Normalize image to [0, 1] for display
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    img = img.clamp(0.0, 1.0)

    # (C, H, W) -> (H, W, C) or (H, W)
    if img.shape[0] == 1:
        img_np = img.squeeze(0).numpy()
    else:
        img_np = img.permute(1, 2, 0).numpy()

    H = img_np.shape[0]
    W = img_np.shape[1]
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size

    # Reshape rollout to 2D grid
    rollout_np = rollout.detach().cpu().float().numpy()
    rollout_2d = rollout_np.reshape(n_patches_h, n_patches_w)

    # Upsample to image size using nearest-neighbor
    import torch.nn.functional as F_nn
    rollout_tensor = torch.from_numpy(rollout_2d).unsqueeze(0).unsqueeze(0)  # (1, 1, H/P, W/P)
    rollout_upsampled = F_nn.interpolate(
        rollout_tensor, size=(H, W), mode="bilinear", align_corners=False
    ).squeeze().numpy()  # (H, W)

    # Create figure with original image and overlay
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    if img_np.ndim == 2:
        axes[0].imshow(img_np, cmap="gray")
    else:
        axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention heatmap
    axes[1].imshow(rollout_upsampled, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Attention Rollout")
    axes[1].axis("off")

    # Overlay: blend image with heatmap
    heatmap = cm.hot(rollout_upsampled)[:, :, :3]  # (H, W, 3) RGB
    if img_np.ndim == 2:
        img_rgb = np.stack([img_np] * 3, axis=-1)
    else:
        img_rgb = img_np

    overlay = 0.6 * img_rgb + 0.4 * heatmap
    overlay = overlay.clip(0.0, 1.0)
    axes[2].imshow(overlay)
    axes[2].set_title("Attention Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
