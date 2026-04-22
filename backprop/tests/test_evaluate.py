"""
Tests for backprop/evaluate.py

Covers:
  - evaluate() returns EvalResult with finite rmse, mae, r2 (11.1)
  - r2 == 0.0 when all targets are identical / SS_tot == 0 (11.2)
  - load_model_from_checkpoint raises RuntimeError on non-existent path (11.3)
  - Property 2: rmse >= 0, mae >= 0, r2 <= 1.0 for random predictions/targets (11.4)
  - Property 1: checkpoint round-trip preserves weights (11.5)
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from backprop.config import MLPConfig
from backprop.evaluate import EvalResult, evaluate, load_model_from_checkpoint
from backprop.model import MLP
from shared.checkpointing import save_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int = 16) -> DataLoader:
    """Wrap tensors in a TensorDataset and return a DataLoader."""
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def make_mlp(hidden_dims: list[int] | None = None) -> MLP:
    hidden_dims = hidden_dims or [16, 8]
    return MLP(input_dim=8, hidden_dims=hidden_dims, dropout=0.0)


def make_config(hidden_dims: list[int] | None = None) -> MLPConfig:
    cfg = MLPConfig()
    cfg.hidden_dims = hidden_dims or [16, 8]
    cfg.dropout = 0.0
    return cfg


# ---------------------------------------------------------------------------
# 11.1 — evaluate() returns EvalResult with finite rmse, mae, r2
# ---------------------------------------------------------------------------

def test_evaluate_returns_finite_metrics():
    """evaluate() should return an EvalResult with finite rmse, mae, and r2."""
    model = make_mlp()
    x = torch.randn(64, 8)
    y = torch.randn(64, 1)
    loader = make_loader(x, y)

    result = evaluate(model, loader)

    assert isinstance(result, EvalResult)
    assert math.isfinite(result.rmse), f"rmse is not finite: {result.rmse}"
    assert math.isfinite(result.mae), f"mae is not finite: {result.mae}"
    assert math.isfinite(result.r2), f"r2 is not finite: {result.r2}"


# ---------------------------------------------------------------------------
# 11.2 — r2 == 0.0 when all targets are identical (SS_tot == 0)
# ---------------------------------------------------------------------------

def test_evaluate_r2_zero_when_all_targets_identical():
    """When all y values are identical, SS_tot == 0 and r2 should be 0.0."""
    model = make_mlp()
    x = torch.randn(32, 8)
    y = torch.ones(32, 1)  # all targets identical
    loader = make_loader(x, y)

    result = evaluate(model, loader)

    assert result.r2 == 0.0, f"Expected r2=0.0 when SS_tot==0, got {result.r2}"


# ---------------------------------------------------------------------------
# 11.3 — load_model_from_checkpoint raises RuntimeError on non-existent path
# ---------------------------------------------------------------------------

def test_load_model_from_checkpoint_raises_on_missing_file():
    """load_model_from_checkpoint should raise RuntimeError for a non-existent path."""
    config = make_config()
    with pytest.raises(RuntimeError):
        load_model_from_checkpoint("/nonexistent/path/model.pt", config)


# ---------------------------------------------------------------------------
# 11.4 — Property 2: rmse >= 0, mae >= 0, r2 <= 1.0 for random inputs
# Feature: backprop-improvements, Property 2
# Validates: Requirements 4.1, 4.2, 9.3
# ---------------------------------------------------------------------------

@given(
    preds=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    ),
    targets=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_eval_metric_bounds(preds: list[float], targets: list[float]):
    """
    Feature: backprop-improvements, Property 2
    For randomly generated predictions and targets, rmse >= 0, mae >= 0, r2 <= 1.0.
    Validates: Requirements 4.1, 4.2, 9.3
    """
    n = min(len(preds), len(targets))
    if n == 0:
        return

    preds_t = torch.tensor(preds[:n], dtype=torch.float32).unsqueeze(1)
    targets_t = torch.tensor(targets[:n], dtype=torch.float32).unsqueeze(1)

    # Build a model whose output matches preds_t exactly by using a fixed bias trick.
    # Simpler: use a 1-layer MLP and override its output via a custom loader approach.
    # Instead, we build a synthetic loader and a model that we override with a hook.
    model = make_mlp([8])

    # Patch the model to return our desired predictions regardless of input
    original_forward = model.forward

    def patched_forward(x: torch.Tensor) -> torch.Tensor:
        # Return the stored predictions slice matching the batch
        return preds_t[: x.shape[0]]

    model.forward = patched_forward  # type: ignore[method-assign]

    loader = make_loader(torch.randn(n, 8), targets_t)

    result = evaluate(model, loader)

    assert result.rmse >= 0.0, f"rmse should be >= 0, got {result.rmse}"
    assert result.mae >= 0.0, f"mae should be >= 0, got {result.mae}"
    assert result.r2 <= 1.0, f"r2 should be <= 1.0, got {result.r2}"


# ---------------------------------------------------------------------------
# 11.5 — Property 1: checkpoint round-trip preserves weights
# Feature: backprop-improvements, Property 1
# Validates: Requirements 3.1, 3.2
# ---------------------------------------------------------------------------

@given(
    hidden_dims=st.lists(
        st.integers(min_value=4, max_value=32),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_checkpoint_roundtrip(hidden_dims: list[int]):
    """
    Feature: backprop-improvements, Property 1
    Save MLP weights to a checkpoint, load via load_model_from_checkpoint,
    and verify all weight tensors are identical.
    Validates: Requirements 3.1, 3.2
    """
    model = MLP(input_dim=8, hidden_dims=hidden_dims, dropout=0.0)
    config = make_config(hidden_dims=hidden_dims)

    # Need a minimal optimizer and scheduler to satisfy save_checkpoint signature
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=0,
            step=0,
            best_metric=float("inf"),
        )

        loaded_model = load_model_from_checkpoint(ckpt_path, config)

        # Compare all weight tensors
        original_params = dict(model.named_parameters())
        loaded_params = dict(loaded_model.named_parameters())

        assert set(original_params.keys()) == set(loaded_params.keys()), (
            "Parameter keys differ after round-trip"
        )
        for name, orig_tensor in original_params.items():
            loaded_tensor = loaded_params[name]
            assert torch.allclose(orig_tensor, loaded_tensor), (
                f"Tensor '{name}' differs after checkpoint round-trip"
            )
    finally:
        os.unlink(ckpt_path)
