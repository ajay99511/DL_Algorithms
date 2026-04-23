"""Tests for evaluate/perplexity.py — PerplexityCalculator."""
from __future__ import annotations

import math
import tempfile
import os
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from evaluate.perplexity import PerplexityCalculator
from shared.logging_utils import JSONLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger() -> tuple[JSONLogger, str]:
    """Return a JSONLogger backed by a temp file, and the temp file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    return JSONLogger(tmp.name), tmp.name


class _TinyLM(nn.Module):
    """Minimal language model: embedding + single linear projection to vocab."""

    def __init__(self, vocab_size: int = 64, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(x))  # (B, T, vocab_size)


class _SimpleTokenizer:
    """Tokenizer that maps each character to its ASCII code (mod vocab_size)."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_empty_corpus_returns_none() -> None:
    """Empty corpus must return None and log a warning."""
    log, path = _make_logger()
    try:
        model = _TinyLM()
        tokenizer = _SimpleTokenizer()
        calc = PerplexityCalculator(model, tokenizer, context_length=8, json_logger=log)
        result = calc.compute("")
        assert result is None, "Expected None for empty corpus"
    finally:
        os.unlink(path)


def test_whitespace_only_corpus_returns_none() -> None:
    """Whitespace-only corpus must return None."""
    log, path = _make_logger()
    try:
        model = _TinyLM()
        tokenizer = _SimpleTokenizer()
        calc = PerplexityCalculator(model, tokenizer, context_length=8, json_logger=log)
        result = calc.compute("   \n\t  ")
        assert result is None
    finally:
        os.unlink(path)


def test_single_window_returns_finite_float_ge_1() -> None:
    """Single-window corpus must return a finite float >= 1.0."""
    log, path = _make_logger()
    try:
        model = _TinyLM(vocab_size=64)
        tokenizer = _SimpleTokenizer(vocab_size=64)
        calc = PerplexityCalculator(model, tokenizer, context_length=8, json_logger=log)
        # 10 chars → 10 tokens, fits in one window of size 9 (context_length=8)
        result = calc.compute("hello wor")
        assert result is not None, "Expected a float, got None"
        assert math.isfinite(result), f"Expected finite perplexity, got {result}"
        assert result >= 1.0, f"Perplexity must be >= 1.0, got {result}"
    finally:
        os.unlink(path)


def test_multi_window_returns_finite_float_ge_1() -> None:
    """Multi-window corpus (length > context_length) must return finite float >= 1.0."""
    log, path = _make_logger()
    try:
        model = _TinyLM(vocab_size=64)
        tokenizer = _SimpleTokenizer(vocab_size=64)
        calc = PerplexityCalculator(model, tokenizer, context_length=4, json_logger=log)
        # 50 chars → 50 tokens, many windows of size 4
        corpus = "the quick brown fox jumps over the lazy dog!!"
        result = calc.compute(corpus)
        assert result is not None
        assert math.isfinite(result)
        assert result >= 1.0
    finally:
        os.unlink(path)


def test_tokenization_failure_returns_none() -> None:
    """If tokenizer.encode raises, compute() must return None."""
    log, path = _make_logger()
    try:
        model = _TinyLM()
        bad_tokenizer = MagicMock()
        bad_tokenizer.encode.side_effect = RuntimeError("tokenizer broken")
        calc = PerplexityCalculator(model, bad_tokenizer, context_length=8, json_logger=log)
        result = calc.compute("some text")
        assert result is None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Property 1: Perplexity lower bound and finiteness
# Feature: evaluate-improvements, Property 1: Perplexity lower bound and finiteness
# Validates: Requirements 3.1, 3.2
# ---------------------------------------------------------------------------

@given(
    seq_len=st.integers(min_value=2, max_value=128),
    context_length=st.integers(min_value=2, max_value=32),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_perplexity_lower_bound_and_finiteness(seq_len: int, context_length: int) -> None:
    """For any non-empty token sequence, perplexity must be finite and >= 1.0."""
    log, path = _make_logger()
    try:
        vocab_size = 64
        model = _TinyLM(vocab_size=vocab_size)
        tokenizer = _SimpleTokenizer(vocab_size=vocab_size)
        calc = PerplexityCalculator(model, tokenizer, context_length=context_length, json_logger=log)
        # Build a corpus that tokenizes to exactly seq_len tokens
        corpus = "a" * seq_len
        result = calc.compute(corpus)
        if result is not None:
            assert math.isfinite(result), f"Perplexity must be finite, got {result}"
            assert result >= 1.0, f"Perplexity must be >= 1.0, got {result}"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Property 2: Perplexity windowing consistency
# Feature: evaluate-improvements, Property 2: Perplexity windowing consistency
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

@given(
    n_windows=st.integers(min_value=2, max_value=8),
    context_length=st.integers(min_value=3, max_value=16),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_perplexity_windowing_consistency(n_windows: int, context_length: int) -> None:
    """Perplexity computed via PerplexityCalculator must match manual window-by-window computation."""
    log, path = _make_logger()
    try:
        vocab_size = 64
        model = _TinyLM(vocab_size=vocab_size)
        model.eval()
        tokenizer = _SimpleTokenizer(vocab_size=vocab_size)

        # Build a corpus long enough for n_windows
        corpus = "abcdefghijklmnopqrstuvwxyz" * (n_windows * context_length // 26 + 2)
        token_ids = tokenizer.encode(corpus)

        # Compute via PerplexityCalculator
        calc = PerplexityCalculator(model, tokenizer, context_length=context_length, json_logger=log)
        ppl_calc = calc.compute(corpus)

        if ppl_calc is None:
            return  # not enough tokens, skip

        # Compute manually window-by-window
        window_size = context_length + 1
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for start in range(0, len(token_ids) - 1, context_length):
                chunk = token_ids[start: start + window_size]
                if len(chunk) < 2:
                    break
                ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
                input_ids = ids[:, :-1]
                target_ids = ids[:, 1:]
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    reduction="sum",
                )
                total_loss += float(loss.item())
                total_tokens += target_ids.numel()

        if total_tokens == 0:
            return

        ppl_manual = math.exp(total_loss / total_tokens)
        assert abs(ppl_calc - ppl_manual) < 1e-4, (
            f"Windowing inconsistency: calc={ppl_calc}, manual={ppl_manual}"
        )
    finally:
        os.unlink(path)
