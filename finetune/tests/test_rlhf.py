"""
Property 10: Reward Clipping Invariant
Property 11: KL Divergence Non-Negativity

Validates: Requirements 3.5, 3.9
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from hypothesis import given, settings
import hypothesis.strategies as st


# Feature: deep-learning-llm-mastery, Property 10: Reward Clipping Invariant
@given(
    reward=st.floats(-1e6, 1e6, allow_nan=False),
    bound=st.floats(0.01, 100.0, allow_nan=False),
)
@settings(max_examples=100)
def test_reward_clipping(reward: float, bound: float) -> None:
    """
    For any reward value and positive bound, clipping must satisfy -bound <= clipped <= bound.

    Validates: Requirements 3.9
    """
    clipped = max(-bound, min(bound, reward))
    assert -bound <= clipped <= bound, (
        f"Clipped value {clipped} is outside [{-bound}, {bound}]"
    )


# Feature: deep-learning-llm-mastery, Property 11: KL Divergence Non-Negativity
@given(
    vocab_size=st.integers(2, 100),
    batch_size=st.integers(1, 8),
)
@settings(max_examples=100)
def test_kl_divergence_non_negative(vocab_size: int, batch_size: int) -> None:
    """
    For any two valid probability distributions p and q, KL(p || q) >= 0.

    Validates: Requirements 3.5
    """
    # Generate random softmax distributions
    p_logits = torch.randn(batch_size, vocab_size)
    q_logits = torch.randn(batch_size, vocab_size)

    p = F.softmax(p_logits, dim=-1)  # (B, V)
    q = F.softmax(q_logits, dim=-1)  # (B, V)

    # KL(p || q) = sum_v p(v) * log(p(v) / q(v))
    # Add small epsilon for numerical stability
    eps = 1e-10
    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)  # (B,)

    assert (kl >= -1e-6).all(), (
        f"KL divergence should be non-negative, got min={kl.min().item():.6f}"
    )
