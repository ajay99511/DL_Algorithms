"""
Property 13: Reward Model Output Shape Invariant

Validates: Requirements 3.4, 3.12
"""
from __future__ import annotations

import torch
import torch.nn as nn
from hypothesis import given, settings
import hypothesis.strategies as st

from finetune.reward_model import RewardModel


class _TinyBackbone(nn.Module):
    """Minimal backbone for testing: Linear layer that returns hidden states."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        # Simulate hidden states: embed input_ids as one-hot-like floats
        B, T = input_ids.shape
        # Use a simple embedding: map token ids to floats mod d_model
        x = (input_ids.float() % self.d_model).unsqueeze(-1).expand(B, T, self.d_model)
        x = self.linear(x)  # (B, T, d_model)

        class _Output:
            def __init__(self, hidden):
                self.hidden_states = [hidden]

        return _Output(x)


# Feature: deep-learning-llm-mastery, Property 13: Reward Model Output Shape Invariant
@given(
    batch_size=st.integers(1, 16),
    seq_len=st.integers(1, 64),
)
@settings(max_examples=100)
def test_reward_model_output_shape(batch_size: int, seq_len: int) -> None:
    """
    For any batch_size and seq_len, the RewardModel must return a tensor of shape (batch_size,).

    Validates: Requirements 3.4, 3.12
    """
    d_model = 16
    backbone = _TinyBackbone(d_model)
    reward_model = RewardModel(backbone, d_model)
    reward_model.eval()

    # Use small vocab size (100) to keep test fast
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        rewards = reward_model(input_ids, attention_mask)

    assert rewards.shape == (batch_size,), (
        f"Expected shape ({batch_size},), got {rewards.shape}"
    )
    assert rewards.dtype in (torch.float32, torch.float64), (
        f"Expected float tensor, got {rewards.dtype}"
    )
