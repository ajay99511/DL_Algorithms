"""
Tests for pretrain/tokenizer.py

Covers:
  - Property 6: Tokenizer Round-Trip
"""

from __future__ import annotations

import os
import tempfile

import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from pretrain.tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# Fixture: small pre-trained tokenizer
# ---------------------------------------------------------------------------

def _make_small_tokenizer() -> BPETokenizer:
    """
    Train a tiny BPE tokenizer on a small fixed corpus for testing.
    Does NOT require downloading TinyStories.
    """
    corpus = [
        "Once upon a time there was a little girl named Alice.",
        "She lived in a small house in the forest.",
        "Every day she would go for a walk in the woods.",
        "One day she met a friendly fox who could talk.",
        "The fox said hello and Alice was very surprised.",
        "They became good friends and played together every day.",
        "Alice loved to read books about animals and nature.",
        "The fox taught her many things about the forest.",
        "Together they explored the woods and found many treasures.",
        "At the end of the day Alice went home happy.",
        "Her mother was waiting with a warm meal and a hug.",
        "Alice told her mother about the talking fox.",
        "Her mother smiled and said that was a wonderful story.",
        "Alice fell asleep dreaming of her next adventure.",
        "The fox waited patiently in the forest for his friend.",
    ] * 20  # repeat to give BPE enough data

    tok = BPETokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        tok.train(corpus, vocab_size=500, save_dir=tmpdir)
        # Reload from disk to test save/load
        tok = BPETokenizer.load(tmpdir)
    return tok


# Module-level fixture (created once per test session)
_TOKENIZER: BPETokenizer | None = None


def get_tokenizer() -> BPETokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _make_small_tokenizer()
    return _TOKENIZER


# ---------------------------------------------------------------------------
# Property 6: Tokenizer Round-Trip
# Validates: Requirements 2.3, 2.12
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 6: Tokenizer Round-Trip
@given(
    text=st.text(
        min_size=1,
        max_size=200,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_tokenizer_round_trip(text: str) -> None:
    """
    For any non-empty string of letters/digits/spaces,
    tokenizer.decode(tokenizer.encode(text)) == text.

    Validates: Requirements 2.3, 2.12
    """
    tokenizer = get_tokenizer()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text, (
        f"Round-trip failed:\n  input:   {repr(text)}\n  decoded: {repr(decoded)}"
    )


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------

def test_tokenizer_encode_returns_list_of_ints() -> None:
    """encode() should return a list of integers."""
    tokenizer = get_tokenizer()
    ids = tokenizer.encode("hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_tokenizer_decode_returns_string() -> None:
    """decode() should return a string."""
    tokenizer = get_tokenizer()
    ids = tokenizer.encode("hello")
    result = tokenizer.decode(ids)
    assert isinstance(result, str)


def test_tokenizer_vocab_size() -> None:
    """vocab_size property should return a positive integer."""
    tokenizer = get_tokenizer()
    assert tokenizer.vocab_size > 0


def test_tokenizer_save_load_round_trip() -> None:
    """Saving and loading a tokenizer should preserve encode/decode behavior."""
    tokenizer = get_tokenizer()
    original_ids = tokenizer.encode("hello world")

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save(tmpdir)
        loaded = BPETokenizer.load(tmpdir)

    loaded_ids = loaded.encode("hello world")
    assert original_ids == loaded_ids, (
        f"IDs differ after save/load: {original_ids} vs {loaded_ids}"
    )


def test_tokenizer_not_trained_raises() -> None:
    """Calling encode on an untrained tokenizer should raise RuntimeError."""
    tok = BPETokenizer()
    with pytest.raises(RuntimeError, match="not trained or loaded"):
        tok.encode("hello")
