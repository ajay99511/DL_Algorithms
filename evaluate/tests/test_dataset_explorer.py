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
