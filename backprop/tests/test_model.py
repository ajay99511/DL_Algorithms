"""
Tests for backprop/model.py

Covers:
  - MLP forward pass shape invariant (Property 2)
  - Activation statistics validity (Property 18)
  - Init strategies produce different weights (Req 1.7)
"""

import torch
import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from backprop.model import MLP, initialize_weights, activation_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mlp(input_dim: int = 8, hidden_dims=None, dropout: float = 0.0) -> MLP:
    if hidden_dims is None:
        hidden_dims = [16, 8]
    return MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


# ---------------------------------------------------------------------------
# Property 2: MLP Forward Pass Shape Invariant
# Validates: Requirements 1.1, 1.10
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 2: MLP Forward Pass Shape Invariant
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    input_dim=st.integers(min_value=1, max_value=32),
    hidden_dims=st.lists(st.integers(min_value=4, max_value=64), min_size=1, max_size=4),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_mlp_output_shape(batch_size, input_dim, hidden_dims):
    """For any batch size B and input_dim, MLP output shape must be (B, 1)."""
    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0)
    x = torch.randn(batch_size, input_dim)
    out = model(x)
    assert out.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# Property 18: Activation Statistics Validity
# Validates: Requirements 1.8
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 18: Activation Statistics Validity
@given(
    batch_size=st.integers(min_value=1, max_value=32),
    input_dim=st.integers(min_value=2, max_value=16),
    hidden_dims=st.lists(st.integers(min_value=4, max_value=32), min_size=1, max_size=3),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_activation_stats_validity(batch_size, input_dim, hidden_dims):
    """
    For any valid input, activation_stats must return finite mean/std and
    dead_fraction in [0.0, 1.0] for every layer.
    """
    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0)
    x = torch.randn(batch_size, input_dim)
    stats = activation_stats(model, x)

    # Should have one entry per hidden layer (one ReLU per hidden layer)
    assert len(stats) == len(hidden_dims), (
        f"Expected {len(hidden_dims)} layers, got {len(stats)}"
    )

    for layer_name, layer_stats in stats.items():
        assert "mean" in layer_stats
        assert "std" in layer_stats
        assert "dead_fraction" in layer_stats

        mean = layer_stats["mean"]
        std = layer_stats["std"]
        dead = layer_stats["dead_fraction"]

        assert torch.isfinite(torch.tensor(mean)), f"{layer_name} mean is not finite: {mean}"
        assert torch.isfinite(torch.tensor(std)), f"{layer_name} std is not finite: {std}"
        assert 0.0 <= dead <= 1.0, f"{layer_name} dead_fraction out of range: {dead}"


# ---------------------------------------------------------------------------
# Example tests: init strategies (Req 1.7)
# ---------------------------------------------------------------------------

def test_init_strategies_differ():
    """Different init strategies should produce different weight distributions."""
    results = {}
    for strategy in ("normal", "xavier", "kaiming"):
        model = make_mlp()
        initialize_weights(model, strategy)
        # Collect all linear weights
        weights = torch.cat([p.data.flatten() for p in model.parameters() if p.dim() > 1])
        results[strategy] = weights.clone()

    # Each strategy should produce distinct weights
    assert not torch.allclose(results["normal"], results["xavier"]), \
        "normal and xavier produced identical weights"
    assert not torch.allclose(results["normal"], results["kaiming"]), \
        "normal and kaiming produced identical weights"
    assert not torch.allclose(results["xavier"], results["kaiming"]), \
        "xavier and kaiming produced identical weights"


def test_initialize_weights_biases_zero():
    """All init strategies should zero out biases."""
    for strategy in ("normal", "xavier", "kaiming"):
        model = make_mlp()
        initialize_weights(model, strategy)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert torch.all(module.bias == 0), \
                    f"Strategy '{strategy}': bias not zeroed in {module}"


def test_initialize_weights_invalid_strategy():
    """Unknown strategy should raise ValueError."""
    model = make_mlp()
    with pytest.raises(ValueError, match="Unknown init strategy"):
        initialize_weights(model, "unknown")


def test_activation_stats_keys():
    """activation_stats keys should be relu_0, relu_1, ... in order."""
    model = MLP(input_dim=4, hidden_dims=[8, 4, 2], dropout=0.0)
    x = torch.randn(5, 4)
    stats = activation_stats(model, x)
    assert list(stats.keys()) == ["relu_0", "relu_1", "relu_2"]


def test_activation_stats_dead_fraction_all_negative():
    """With very negative inputs, dead_fraction should be 1.0."""
    model = MLP(input_dim=4, hidden_dims=[8], dropout=0.0)
    # Force all pre-activation values to be very negative
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.constant_(module.weight, -100.0)
            torch.nn.init.constant_(module.bias, -100.0)
    x = torch.ones(3, 4)
    stats = activation_stats(model, x)
    assert stats["relu_0"]["dead_fraction"] == 1.0
