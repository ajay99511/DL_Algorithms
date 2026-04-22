"""
Tests for visiontx model components.

Includes property-based tests for:
    - Property 12: Patch Embedding Shape Invariant
    - Property 5 (adapted): ViT Output Shape Invariant
    - Property 3 (ViT): Checkpoint Round-Trip Fidelity
"""

from __future__ import annotations

import tempfile
import os

import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from visiontx.config import ViTConfig
from visiontx.model import PatchEmbedding, ViT
from shared.checkpointing import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Tiny config factory for fast tests
# ---------------------------------------------------------------------------

def _tiny_config(patch_size: int = 4, image_size: int = 32) -> ViTConfig:
    """Return a minimal ViTConfig for fast CPU tests."""
    return ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        n_channels=3,
        n_classes=10,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Property 12: Patch Embedding Shape Invariant
# Validates: Requirements 4.1, 4.12
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 12: Patch Embedding Shape Invariant
@given(
    batch_size=st.integers(1, 16),
    patch_size=st.sampled_from([4, 8]),
    image_size=st.sampled_from([32, 64]),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_patch_embedding_shape(batch_size: int, patch_size: int, image_size: int) -> None:
    """
    **Validates: Requirements 4.1, 4.12**

    For any batch of images (B, C, H, W) where H and W are divisible by patch_size,
    PatchEmbedding SHALL produce output of shape (B, (H/P)*(W/P), d_model).
    """
    d_model = 16
    n_channels = 3
    embed = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        n_channels=n_channels,
        d_model=d_model,
    )
    x = torch.randn(batch_size, n_channels, image_size, image_size)
    out = embed(x)

    expected_n_patches = (image_size // patch_size) ** 2
    assert out.shape == (batch_size, expected_n_patches, d_model), (
        f"Expected ({batch_size}, {expected_n_patches}, {d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Property 5 (adapted): ViT Output Shape Invariant
# Validates: Requirements 4.1, 4.12
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 5 (adapted): ViT Output Shape Invariant
@given(batch_size=st.integers(1, 8))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_vit_output_shape(batch_size: int) -> None:
    """
    **Validates: Requirements 4.1, 4.12**

    For any batch size B, ViT forward pass SHALL produce logits of shape (B, n_classes).
    """
    config = _tiny_config(patch_size=4, image_size=32)
    vit = ViT(config)
    vit.eval()

    x = torch.randn(batch_size, config.n_channels, config.image_size, config.image_size)
    with torch.no_grad():
        logits = vit(x)

    assert logits.shape == (batch_size, config.n_classes), (
        f"Expected ({batch_size}, {config.n_classes}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# Property 3 (ViT): Checkpoint Round-Trip Fidelity
# Validates: Requirements 4.9
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 3: Checkpoint Round-Trip Fidelity (ViT)
@given(patch_size=st.sampled_from([4, 8]))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_vit_checkpoint_round_trip(patch_size: int) -> None:
    """
    **Validates: Requirements 4.9**

    Saving and reloading a ViT checkpoint SHALL produce a model whose
    state_dict is element-wise identical to the original.
    """
    config = _tiny_config(patch_size=patch_size, image_size=32)
    model = ViT(config)
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_vit.pt")

        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            step=10,
            best_metric=0.85,
        )

        # Create a fresh model and load the checkpoint
        model2 = ViT(config)
        load_checkpoint(ckpt_path, model2)

        # All state_dict tensors must be element-wise equal
        sd1 = model.state_dict()
        sd2 = model2.state_dict()

        assert set(sd1.keys()) == set(sd2.keys()), "State dict keys differ after round-trip"
        for key in sd1:
            assert torch.allclose(sd1[key].float(), sd2[key].float(), atol=1e-6), (
                f"Tensor '{key}' differs after checkpoint round-trip"
            )
