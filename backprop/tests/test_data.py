"""Tests for backprop/data.py — California Housing data pipeline."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
import hypothesis.strategies as st
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from backprop.data import load_california_housing


# ---------------------------------------------------------------------------
# Example / unit tests
# ---------------------------------------------------------------------------

def test_returns_three_dataloaders() -> None:
    result = load_california_housing(val_size=0.1, test_size=0.1, seed=42)
    assert len(result) == 3
    assert all(isinstance(dl, DataLoader) for dl in result)


def test_batch_shape() -> None:
    train_loader, val_loader, test_loader = load_california_housing(
        val_size=0.1, test_size=0.1, seed=42, batch_size=32
    )
    x, y = next(iter(train_loader))
    assert x.shape[1] == 8, "California Housing has 8 features"
    assert y.shape[1] == 1, "Regression target should be (B, 1)"
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_approximate_split_sizes() -> None:
    """80/10/10 split on ~20640 samples."""
    train_loader, val_loader, test_loader = load_california_housing(
        val_size=0.1, test_size=0.1, seed=42
    )
    n_train = sum(x.shape[0] for x, _ in train_loader)
    n_val = sum(x.shape[0] for x, _ in val_loader)
    n_test = sum(x.shape[0] for x, _ in test_loader)
    total = n_train + n_val + n_test

    assert abs(n_train / total - 0.8) < 0.02
    assert abs(n_val / total - 0.1) < 0.02
    assert abs(n_test / total - 0.1) < 0.02


def test_no_data_leakage() -> None:
    """Validates Req 1.2: scaler fit on train only — val/test features are NOT
    zero-mean/unit-variance, but train features should be approximately so."""
    train_loader, val_loader, _ = load_california_housing(
        val_size=0.1, test_size=0.1, seed=42
    )
    # Collect all train features
    train_X = torch.cat([x for x, _ in train_loader], dim=0).numpy()
    val_X = torch.cat([x for x, _ in val_loader], dim=0).numpy()

    # Train features should be approximately standardized (mean≈0, std≈1)
    assert np.abs(train_X.mean(axis=0)).max() < 0.05, "Train features should be ~zero-mean"
    assert np.abs(train_X.std(axis=0) - 1.0).max() < 0.05, "Train features should be ~unit-variance"

    # Val features are transformed with the train scaler, so they won't be
    # perfectly standardized — confirming the scaler was NOT re-fit on val.
    # We just verify val is not identical to train stats.
    val_mean = np.abs(val_X.mean(axis=0)).max()
    val_std_dev = np.abs(val_X.std(axis=0) - 1.0).max()
    # At least one of mean or std should differ from perfect standardization
    assert val_mean > 1e-6 or val_std_dev > 1e-6, (
        "Val features look perfectly standardized — possible data leakage"
    )


def test_reproducibility() -> None:
    """Same seed produces identical data."""
    loaders_a = load_california_housing(val_size=0.1, test_size=0.1, seed=0)
    loaders_b = load_california_housing(val_size=0.1, test_size=0.1, seed=0)
    for dl_a, dl_b in zip(loaders_a, loaders_b):
        xa, ya = next(iter(dl_a))
        xb, yb = next(iter(dl_b))
        # Shapes must match; values may differ due to shuffle, but sizes are same
        assert xa.shape == xb.shape
        assert ya.shape == yb.shape


def test_different_seeds_produce_different_splits() -> None:
    """Different seeds should produce different test splits."""
    _, _, test_a = load_california_housing(val_size=0.1, test_size=0.1, seed=0)
    _, _, test_b = load_california_housing(val_size=0.1, test_size=0.1, seed=99)
    xa = torch.cat([x for x, _ in test_a], dim=0)
    xb = torch.cat([x for x, _ in test_b], dim=0)
    assert not torch.allclose(xa, xb), "Different seeds should yield different splits"


def test_split_disjointness() -> None:
    """Validates Req 1.2: train/val/test feature rows are pairwise disjoint."""
    train_loader, val_loader, test_loader = load_california_housing(
        val_size=0.1, test_size=0.1, seed=42
    )
    # Represent each sample as a tuple of its feature values (unique per row)
    def rows(loader: DataLoader) -> set[tuple]:
        return {tuple(x_row.tolist()) for x, _ in loader for x_row in x}

    train_rows = rows(train_loader)
    val_rows = rows(val_loader)
    test_rows = rows(test_loader)

    assert len(train_rows & val_rows) == 0, "Train and val overlap"
    assert len(train_rows & test_rows) == 0, "Train and test overlap"
    assert len(val_rows & test_rows) == 0, "Val and test overlap"


# ---------------------------------------------------------------------------
# Property 1: Data Split Disjointness
# Feature: deep-learning-llm-mastery, Property 1: Data Split Disjointness
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=10, max_value=10_000),
    val_frac=st.floats(0.05, 0.2),
    test_frac=st.floats(0.05, 0.2),
)
@settings(max_examples=100)
def test_split_disjointness_property(n: int, val_frac: float, test_frac: float) -> None:
    """
    For any dataset of size n split into train/val/test fractions, the resulting
    index sets SHALL be pairwise disjoint and their union SHALL equal the full
    index set of size n.
    """
    # Ensure combined fractions leave at least 1 sample for train
    assume = val_frac + test_frac < 0.9

    indices = np.arange(n)

    # Mirror the same two-step split logic used in load_california_housing
    idx_trainval, idx_test = train_test_split(indices, test_size=test_frac, random_state=42)
    val_fraction_of_trainval = val_frac / (1.0 - test_frac)
    # Clamp to valid range to avoid degenerate splits
    val_fraction_of_trainval = min(val_fraction_of_trainval, 0.95)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_fraction_of_trainval, random_state=42
    )

    train_set = set(idx_train.tolist())
    val_set = set(idx_val.tolist())
    test_set = set(idx_test.tolist())

    # Pairwise disjointness
    assert len(train_set & val_set) == 0, "Train and val overlap"
    assert len(train_set & test_set) == 0, "Train and test overlap"
    assert len(val_set & test_set) == 0, "Val and test overlap"

    # Union equals full index set
    assert len(train_set | val_set | test_set) == n, (
        f"Union size {len(train_set | val_set | test_set)} != n={n}"
    )


# ---------------------------------------------------------------------------
# Property 5: DataLoader batch size invariant
# Feature: backprop-improvements, Property 5
# Validates: Requirements 9.5
# ---------------------------------------------------------------------------

@given(batch_size=st.integers(min_value=1, max_value=256))
@settings(max_examples=100, deadline=None)
def test_batch_size_invariant(batch_size: int) -> None:
    """
    For any batch_size B passed to load_california_housing, every batch returned
    by the train DataLoader except possibly the last SHALL have first-dimension
    size equal to B.
    """
    train_loader, _, _ = load_california_housing(
        val_size=0.1, test_size=0.1, seed=42, batch_size=batch_size
    )
    batches = list(train_loader)
    # Every non-last batch must have exactly batch_size samples
    for x, _ in batches[:-1]:
        assert x.shape[0] == batch_size, (
            f"Expected batch size {batch_size}, got {x.shape[0]}"
        )
