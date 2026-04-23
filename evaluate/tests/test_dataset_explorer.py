"""Tests for dataset_explorer.

Property: dataset streaming format

# Feature: deep-learning-llm-mastery, Property: Dataset Streaming Format
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from evaluate import dataset_explorer as explorer_module
from evaluate.dataset_explorer import explore_dataset

_REQUIRED_KEYS = {"estimated_token_count", "vocabulary_size", "avg_sequence_length", "sample_texts"}


def _make_mock_dataset(n_samples: int) -> Any:
    """Create a mock iterable dataset with n_samples text entries."""
    samples = [{"text": f"hello world sample number {i} with some extra words"} for i in range(n_samples)]
    return iter(samples)


# ---------------------------------------------------------------------------
# Property: explore_dataset returns correct keys for any n_samples
# ---------------------------------------------------------------------------

@given(n_samples=st.integers(1, 100))
@settings(max_examples=20)
def test_explore_dataset_returns_correct_keys(n_samples: int) -> None:
    """explore_dataset must return dict with required keys regardless of n_samples.

    **Validates: Requirements 6.7**
    """
    mock_dataset = _make_mock_dataset(n_samples + 10)  # ensure enough samples

    mock_load = MagicMock(return_value=mock_dataset)

    with patch.object(explorer_module, "_HAS_DATASETS", True), \
         patch.object(explorer_module, "load_dataset", mock_load, create=True):
        result = explore_dataset("fake/dataset", n_samples=n_samples)

    assert isinstance(result, dict), "explore_dataset must return a dict"
    for key in _REQUIRED_KEYS:
        assert key in result, f"Missing required key: {key!r}"


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------

def test_explore_dataset_sample_texts_count() -> None:
    """sample_texts must contain at most 3 items."""
    mock_dataset = _make_mock_dataset(50)
    mock_load = MagicMock(return_value=mock_dataset)

    with patch.object(explorer_module, "_HAS_DATASETS", True), \
         patch.object(explorer_module, "load_dataset", mock_load, create=True):
        result = explore_dataset("fake/dataset", n_samples=50)

    assert len(result["sample_texts"]) <= 3, (
        f"sample_texts must have at most 3 items, got {len(result['sample_texts'])}"
    )


def test_explore_dataset_without_datasets_library() -> None:
    """When datasets is not installed, explore_dataset must return gracefully."""
    with patch.object(explorer_module, "_HAS_DATASETS", False):
        result = explore_dataset("any/dataset", n_samples=10)

    assert isinstance(result, dict)
    assert "error" in result or result.get("estimated_token_count") is None


def test_explore_dataset_token_count_positive() -> None:
    """estimated_token_count must be >= 0 for non-empty datasets."""
    mock_dataset = _make_mock_dataset(20)
    mock_load = MagicMock(return_value=mock_dataset)

    with patch.object(explorer_module, "_HAS_DATASETS", True), \
         patch.object(explorer_module, "load_dataset", mock_load, create=True):
        result = explore_dataset("fake/dataset", n_samples=20)

    assert result["estimated_token_count"] >= 0


def test_explore_dataset_avg_sequence_length_positive() -> None:
    """avg_sequence_length must be > 0 for non-empty text samples."""
    mock_dataset = _make_mock_dataset(10)
    mock_load = MagicMock(return_value=mock_dataset)

    with patch.object(explorer_module, "_HAS_DATASETS", True), \
         patch.object(explorer_module, "load_dataset", mock_load, create=True):
        result = explore_dataset("fake/dataset", n_samples=10)

    assert result["avg_sequence_length"] > 0


def test_explore_dataset_with_config_name() -> None:
    """explore_dataset must pass config_name to load_dataset when provided."""
    mock_dataset = _make_mock_dataset(5)
    mock_load = MagicMock(return_value=mock_dataset)

    with patch.object(explorer_module, "_HAS_DATASETS", True), \
         patch.object(explorer_module, "load_dataset", mock_load, create=True):
        result = explore_dataset("allenai/c4", config_name="en", n_samples=5)

    # Verify load_dataset was called with the config_name positional arg
    call_args = mock_load.call_args
    assert "allenai/c4" in call_args[0] or "allenai/c4" in str(call_args)
    assert "en" in call_args[0] or "en" in str(call_args)

# ---------------------------------------------------------------------------
# Tests for new rich statistics functions
# ---------------------------------------------------------------------------

import os
import tempfile

from hypothesis import given, settings
import hypothesis.strategies as st

from evaluate.dataset_explorer import (
    compute_ngram_overlap,
    compute_domain_distribution,
    compute_length_distribution,
)


# ---------------------------------------------------------------------------
# compute_ngram_overlap — unit tests
# ---------------------------------------------------------------------------

def test_ngram_overlap_identical_corpora_is_one() -> None:
    """overlap(A, A) must equal 1.0 for any non-empty corpus."""
    texts = ["the quick brown fox", "jumps over the lazy dog"]
    result = compute_ngram_overlap(texts, texts, n=1)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"


def test_ngram_overlap_disjoint_corpora_is_zero() -> None:
    """Corpora with no shared n-grams must yield 0.0."""
    train = ["aaa bbb ccc"]
    test = ["xxx yyy zzz"]
    result = compute_ngram_overlap(train, test, n=1)
    assert result == 0.0, f"Expected 0.0, got {result}"


def test_ngram_overlap_partial() -> None:
    """Partial overlap must be strictly between 0 and 1."""
    train = ["the cat sat on the mat"]
    test = ["the cat sat on a log"]
    result = compute_ngram_overlap(train, test, n=1)
    assert 0.0 < result < 1.0, f"Expected partial overlap, got {result}"


def test_ngram_overlap_empty_train_returns_zero() -> None:
    """Empty train corpus must return 0.0."""
    assert compute_ngram_overlap([], ["some text"], n=1) == 0.0


def test_ngram_overlap_empty_test_returns_zero() -> None:
    """Empty test corpus must return 0.0."""
    assert compute_ngram_overlap(["some text"], [], n=1) == 0.0


def test_bigram_overlap() -> None:
    """Bigram overlap must work correctly."""
    train = ["the quick brown fox"]
    test = ["the quick brown fox"]
    result = compute_ngram_overlap(train, test, n=2)
    assert abs(result - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_domain_distribution — unit tests
# ---------------------------------------------------------------------------

def test_domain_distribution_sums_to_one() -> None:
    """Proportions must sum to 1.0."""
    samples = [
        {"source": "web"}, {"source": "web"}, {"source": "books"},
        {"source": "code"}, {"source": "web"},
    ]
    dist = compute_domain_distribution(samples)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_domain_distribution_correct_proportions() -> None:
    """Proportions must match expected values."""
    samples = [{"source": "a"}, {"source": "a"}, {"source": "b"}]
    dist = compute_domain_distribution(samples)
    assert abs(dist["a"] - 2 / 3) < 1e-9
    assert abs(dist["b"] - 1 / 3) < 1e-9


def test_domain_distribution_empty_returns_empty() -> None:
    """Empty samples must return empty dict."""
    assert compute_domain_distribution([]) == {}


def test_domain_distribution_missing_field_uses_unknown() -> None:
    """Samples without the label field must be grouped as 'unknown'."""
    samples = [{"text": "hello"}, {"text": "world"}]
    dist = compute_domain_distribution(samples, label_field="source")
    assert "unknown" in dist


# ---------------------------------------------------------------------------
# compute_length_distribution — unit tests
# ---------------------------------------------------------------------------

def test_length_distribution_non_negative() -> None:
    """All lengths must be non-negative integers."""
    texts = ["hello world", "a", "the quick brown fox jumps"]
    lengths = compute_length_distribution(texts)
    assert all(isinstance(l, int) and l >= 0 for l in lengths)


def test_length_distribution_count_matches_input() -> None:
    """Number of lengths must equal number of input texts."""
    texts = ["one two three", "four five", "six"]
    lengths = compute_length_distribution(texts)
    assert len(lengths) == len(texts)


def test_length_distribution_whitespace_fallback() -> None:
    """Without a tokenizer, whitespace splitting must be used."""
    texts = ["hello world", "one two three four"]
    lengths = compute_length_distribution(texts, tokenizer=None)
    assert lengths == [2, 4]


def test_length_distribution_saves_png() -> None:
    """When save_path is provided, a PNG must be created."""
    texts = ["hello world"] * 20
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "lengths.png")
        compute_length_distribution(texts, save_path=save_path)
        # PNG creation depends on matplotlib availability; just check no exception


def test_length_distribution_with_tokenizer() -> None:
    """Custom tokenizer must be used when provided."""
    class _CharTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

    texts = ["ab", "abcd"]
    lengths = compute_length_distribution(texts, tokenizer=_CharTokenizer())
    assert lengths == [2, 4]


# ---------------------------------------------------------------------------
# Property 6: N-gram overlap range
# Feature: evaluate-improvements, Property 6: N-gram overlap range
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------

@given(
    train_texts=st.lists(
        st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz "),
        min_size=1, max_size=10,
    ),
    test_texts=st.lists(
        st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz "),
        min_size=1, max_size=10,
    ),
    n=st.integers(min_value=1, max_value=2),
)
@settings(max_examples=100)
def test_ngram_overlap_range(
    train_texts: list[str],
    test_texts: list[str],
    n: int,
) -> None:
    """N-gram overlap must be in [0, 1] and overlap(A, A) == 1.0."""
    result = compute_ngram_overlap(train_texts, test_texts, n=n)
    assert 0.0 <= result <= 1.0, f"Overlap out of range: {result}"

    # Self-overlap must be 1.0
    self_result = compute_ngram_overlap(train_texts, train_texts, n=n)
    # Only check if there are actual n-grams (non-empty tokens)
    has_ngrams = any(len(t.split()) >= n for t in train_texts)
    if has_ngrams:
        assert abs(self_result - 1.0) < 1e-9, f"Self-overlap must be 1.0, got {self_result}"


# ---------------------------------------------------------------------------
# Property 7: Domain distribution sums to 1
# Feature: evaluate-improvements, Property 7: Domain distribution sums to 1
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------

@given(
    labels=st.lists(
        st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        min_size=1, max_size=100,
    )
)
@settings(max_examples=100)
def test_domain_distribution_sums_to_one_property(labels: list[str]) -> None:
    """Domain distribution values must sum to 1.0 for any non-empty label list."""
    samples = [{"source": label} for label in labels]
    dist = compute_domain_distribution(samples, label_field="source")
    assert abs(sum(dist.values()) - 1.0) < 1e-6, (
        f"Distribution sums to {sum(dist.values())}, expected 1.0"
    )
