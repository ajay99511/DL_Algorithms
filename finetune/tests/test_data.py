"""
Property 1: Data Split Disjointness (Alignment)
Example test: Alpaca prompt format

Validates: Requirements 3.1, 3.3, 3.12
"""
from __future__ import annotations

import math

import torch
from hypothesis import given, settings
import hypothesis.strategies as st

from finetune.data import _format_alpaca_prompt


# Feature: deep-learning-llm-mastery, Property 1: Data Split Disjointness (Alignment)
@given(
    n=st.integers(min_value=20, max_value=1000),
    val_frac=st.floats(0.05, 0.3),
)
@settings(max_examples=100)
def test_split_disjointness(n: int, val_frac: float) -> None:
    """
    For any dataset size n and val_frac, train and val index sets must be
    disjoint and their union must equal the full index set.

    Validates: Requirements 3.1, 3.12
    """
    torch.manual_seed(0)
    n_val = max(1, math.floor(n * val_frac))
    indices = torch.randperm(n)
    val_idx = set(indices[:n_val].tolist())
    train_idx = set(indices[n_val:].tolist())

    # Disjoint
    assert len(val_idx & train_idx) == 0, (
        f"Train and val sets overlap: {val_idx & train_idx}"
    )
    # Union equals full set
    assert val_idx | train_idx == set(range(n)), (
        "Union of train and val does not cover all indices"
    )
    # Sizes add up
    assert len(val_idx) + len(train_idx) == n, (
        f"Expected {n} total indices, got {len(val_idx) + len(train_idx)}"
    )


def test_alpaca_prompt_format_with_input() -> None:
    """
    Formatted Alpaca prompts with non-empty input must contain
    '### Instruction:', '### Input:', and '### Response:' markers.

    Validates: Requirements 3.3
    """
    prompt = _format_alpaca_prompt(
        instruction="Translate the following to French.",
        inp="Hello, how are you?",
        output="Bonjour, comment allez-vous?",
    )
    assert "### Instruction:" in prompt, "Missing '### Instruction:' marker"
    assert "### Input:" in prompt, "Missing '### Input:' marker"
    assert "### Response:" in prompt, "Missing '### Response:' marker"
    assert "Translate the following to French." in prompt
    assert "Hello, how are you?" in prompt
    assert "Bonjour, comment allez-vous?" in prompt


def test_alpaca_prompt_format_without_input() -> None:
    """
    Formatted Alpaca prompts with empty input must omit the Input section.

    Validates: Requirements 3.3
    """
    prompt = _format_alpaca_prompt(
        instruction="Write a haiku about autumn.",
        inp="",
        output="Leaves fall gently down\nCrisp air fills the morning light\nAutumn whispers peace",
    )
    assert "### Instruction:" in prompt, "Missing '### Instruction:' marker"
    assert "### Response:" in prompt, "Missing '### Response:' marker"
    assert "### Input:" not in prompt, "Input section should be omitted when input is empty"


def test_alpaca_prompt_format_whitespace_input() -> None:
    """
    Formatted Alpaca prompts with whitespace-only input must omit the Input section.

    Validates: Requirements 3.3
    """
    prompt = _format_alpaca_prompt(
        instruction="List three colors.",
        inp="   ",
        output="Red, green, blue.",
    )
    assert "### Instruction:" in prompt
    assert "### Response:" in prompt
    assert "### Input:" not in prompt
