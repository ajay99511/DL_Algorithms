"""
Tests for pretrain/model.py and pretrain/generate.py

Covers:
  - Property 5:  Transformer Output Shape Invariant
  - Property 7:  Causal Attention Mask Correctness
  - Property 8:  Inference Reproducibility Under Fixed Seed
  - Property 17: Config Round-Trip Serialization (TransformerConfig)
"""

from __future__ import annotations

import tempfile

import torch
import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from pretrain.config import TransformerConfig
from pretrain.generate import nucleus_sample
from pretrain.model import CausalSelfAttention, GPTModel
from shared.config import load_config, save_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_small_model(
    vocab_size: int = 64,
    context_length: int = 32,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 2,
    d_ff: int = 64,
    dropout: float = 0.0,
) -> GPTModel:
    """Build a tiny GPTModel for fast testing."""
    return GPTModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Property 5: Transformer Output Shape Invariant
# Validates: Requirements 2.1, 2.10
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 5: Transformer Output Shape Invariant
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=32),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_transformer_output_shape(batch_size: int, seq_len: int) -> None:
    """
    For any batch_size and seq_len <= context_length,
    logits.shape must be (batch_size, seq_len, vocab_size).
    """
    vocab_size = 64
    model = make_small_model(vocab_size=vocab_size, context_length=32)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        logits, loss = model(input_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size), (
        f"Expected ({batch_size}, {seq_len}, {vocab_size}), got {logits.shape}"
    )
    assert loss is None  # no targets provided


# ---------------------------------------------------------------------------
# Property 7: Causal Attention Mask Correctness
# Validates: Requirements 2.12
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 7: Causal Attention Mask Correctness
@given(seq_len=st.integers(min_value=2, max_value=32))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_causal_attention_mask_correctness(seq_len: int) -> None:
    """
    The upper triangle (above diagonal) of attention weights must be zero
    after softmax for all heads and layers.
    """
    vocab_size = 64
    model = make_small_model(vocab_size=vocab_size, context_length=32, n_layers=2, n_heads=2)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    with torch.no_grad():
        model(input_ids)

    # Check each block's stored attention weights
    for layer_idx, block in enumerate(model.blocks):
        attn_weights = block.attn._last_attn_weights  # (1, n_heads, T, T)
        assert attn_weights is not None, f"Layer {layer_idx}: no attention weights captured"

        # Upper triangle (above diagonal) should be zero
        # triu with diagonal=1 gives positions where j > i
        upper = torch.triu(attn_weights, diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), (
            f"Layer {layer_idx}: upper triangle of attention weights is not zero. "
            f"Max value: {upper.abs().max().item():.6f}"
        )


# ---------------------------------------------------------------------------
# Property 8: Inference Reproducibility Under Fixed Seed
# Validates: Requirements 2.8, 5.8
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 8: Inference Reproducibility Under Fixed Seed
@given(
    seed=st.integers(min_value=0, max_value=2**31),
    prompt_len=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_inference_reproducibility(seed: int, prompt_len: int) -> None:
    """
    Running nucleus_sample twice with the same seed must produce byte-identical outputs.
    """
    vocab_size = 64
    model = make_small_model(vocab_size=vocab_size, context_length=32)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (1, prompt_len))

    out1 = nucleus_sample(model, input_ids.clone(), max_new_tokens=5, top_p=0.9, temperature=1.0, seed=seed)
    out2 = nucleus_sample(model, input_ids.clone(), max_new_tokens=5, top_p=0.9, temperature=1.0, seed=seed)

    assert torch.equal(out1, out2), (
        f"nucleus_sample with seed={seed} produced different outputs:\n"
        f"  out1={out1.tolist()}\n  out2={out2.tolist()}"
    )


# ---------------------------------------------------------------------------
# Property 17: Config Round-Trip Serialization (TransformerConfig)
# Validates: Requirements 7.2, 7.3
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 17: Config Round-Trip Serialization (TransformerConfig)
@given(
    n_layers=st.integers(min_value=1, max_value=8),
    d_model=st.integers(min_value=32, max_value=256),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_transformer_config_round_trip(n_layers: int, d_model: int) -> None:
    """
    Serializing TransformerConfig to YAML and deserializing must produce
    an object with identical field values.
    """
    config = TransformerConfig(n_layers=n_layers, d_model=d_model)

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp_path = f.name

    try:
        save_config(config, tmp_path)
        loaded = load_config(tmp_path, TransformerConfig)
    finally:
        import os
        os.unlink(tmp_path)

    assert loaded.n_layers == config.n_layers, (
        f"n_layers mismatch: {loaded.n_layers} != {config.n_layers}"
    )
    assert loaded.d_model == config.d_model, (
        f"d_model mismatch: {loaded.d_model} != {config.d_model}"
    )
    assert loaded.vocab_size == config.vocab_size
    assert loaded.n_heads == config.n_heads
    assert loaded.d_ff == config.d_ff
    assert loaded.dropout == config.dropout
    assert loaded.batch_size == config.batch_size
    assert loaded.max_steps == config.max_steps
    assert loaded.learning_rate == config.learning_rate
    assert loaded.context_length == config.context_length


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------

def test_gpt_model_with_targets_returns_loss() -> None:
    """When targets are provided, model should return a scalar loss."""
    model = make_small_model()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        logits, loss = model(input_ids, targets)
    assert loss is not None
    assert loss.shape == ()  # scalar
    assert torch.isfinite(loss)


def test_gpt_model_count_parameters() -> None:
    """count_parameters should return a positive integer."""
    model = make_small_model()
    n_params = model.count_parameters()
    assert isinstance(n_params, int)
    assert n_params > 0


def test_greedy_decode_output_length() -> None:
    """greedy_decode should extend input by exactly max_new_tokens."""
    from pretrain.generate import greedy_decode
    model = make_small_model(vocab_size=64, context_length=32)
    model.eval()
    input_ids = torch.randint(0, 64, (1, 5))
    out = greedy_decode(model, input_ids, max_new_tokens=10)
    assert out.shape == (1, 15), f"Expected (1, 15), got {out.shape}"


def test_weight_tying() -> None:
    """LM head weight should be the same object as token embedding weight."""
    model = make_small_model()
    assert model.lm_head.weight is model.token_emb.weight, "Weight tying not applied"
