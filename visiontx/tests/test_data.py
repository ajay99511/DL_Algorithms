"""
Tests for visiontx data loading.

Includes:
    - Property test: CIFAR-10 normalization range
    - Example test: data loader output shapes
"""

from __future__ import annotations

import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from visiontx.config import ViTConfig
from visiontx.data import load_cifar10


def _cifar10_config(batch_size: int = 64) -> ViTConfig:
    """Return a ViTConfig for CIFAR-10 tests."""
    return ViTConfig(
        dataset="cifar10",
        image_size=32,
        patch_size=4,
        n_channels=3,
        n_classes=10,
        batch_size=batch_size,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Property: Normalization range
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Normalization Range Property
@given(batch_size=st.integers(1, 16))
@settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_cifar10_normalization(batch_size: int) -> None:
    """
    **Validates: Requirements 4.4**

    Normalized CIFAR-10 images SHALL have per-channel mean ≈ 0 and std ≈ 1
    (within tolerance 0.5) across a batch.
    """
    config = _cifar10_config(batch_size=max(batch_size, 64))
    _, val_loader, _ = load_cifar10(config)

    # Collect a few batches to get a reasonable sample
    all_images = []
    for images, _ in val_loader:
        all_images.append(images)
        if len(all_images) >= 3:
            break

    if not all_images:
        return

    images = torch.cat(all_images, dim=0)  # (N, 3, 32, 32)

    # Per-channel statistics across all spatial positions and batch items
    # images shape: (N, C, H, W) -> compute stats over (N, H, W) for each C
    for c in range(images.shape[1]):
        channel = images[:, c, :, :]  # (N, H, W)
        mean = channel.mean().item()
        std = channel.std().item()
        assert abs(mean) < 0.5, (
            f"Channel {c} mean={mean:.4f} is not close to 0 (tolerance 0.5)"
        )
        assert abs(std - 1.0) < 0.5, (
            f"Channel {c} std={std:.4f} is not close to 1 (tolerance 0.5)"
        )


# ---------------------------------------------------------------------------
# Example test: data loader output shapes
# Validates: Requirements 4.12
# ---------------------------------------------------------------------------

def test_cifar10_dataloader_shapes() -> None:
    """
    Assert that CIFAR-10 data loaders return correctly shaped tensors.

    images.shape == (batch_size, 3, 32, 32)
    labels.shape == (batch_size,)
    """
    batch_size = 32
    config = _cifar10_config(batch_size=batch_size)
    train_loader, val_loader, test_loader = load_cifar10(config)

    for loader_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        images, labels = next(iter(loader))
        assert images.shape[1:] == (3, 32, 32), (
            f"{loader_name}: expected image shape (B, 3, 32, 32), got {images.shape}"
        )
        assert labels.ndim == 1, (
            f"{loader_name}: expected labels shape (B,), got {labels.shape}"
        )
        assert images.shape[0] == labels.shape[0], (
            f"{loader_name}: batch size mismatch: images={images.shape[0]}, labels={labels.shape[0]}"
        )
        # Labels should be integers in [0, 9]
        assert labels.min().item() >= 0
        assert labels.max().item() <= 9
