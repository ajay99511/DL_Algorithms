"""Tests for evaluate/calibration.py — CalibrationAnalyzer."""
from __future__ import annotations

import os
import tempfile

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from evaluate.calibration import CalibrationAnalyzer


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_empty_confidences_returns_none() -> None:
    """Empty confidences must return None."""
    analyzer = CalibrationAnalyzer()
    result = analyzer.compute_ece([], [])
    assert result is None


def test_perfect_calibration_ece_is_zero() -> None:
    """Perfectly calibrated predictions must yield ECE = 0.0."""
    # 10 bins, each with confidence == accuracy
    # Bin [0.0, 0.1): confidence=0.05, all wrong → accuracy=0.0 → |0.05-0.0|=0.05
    # To get ECE=0 we need mean_conf == accuracy in every bin.
    # Simplest: all predictions have confidence=1.0 and all labels=1
    confidences = [1.0] * 100
    labels = [1] * 100
    analyzer = CalibrationAnalyzer(n_bins=10)
    ece = analyzer.compute_ece(confidences, labels)
    assert ece is not None
    assert abs(ece) < 1e-6, f"Expected ECE=0.0 for perfect calibration, got {ece}"


def test_ece_in_range() -> None:
    """ECE must be in [0, 1] for typical inputs."""
    confidences = [0.1, 0.5, 0.9, 0.3, 0.7]
    labels = [0, 1, 1, 0, 1]
    analyzer = CalibrationAnalyzer(n_bins=5)
    ece = analyzer.compute_ece(confidences, labels)
    assert ece is not None
    assert 0.0 <= ece <= 1.0


def test_plot_reliability_diagram_creates_png() -> None:
    """plot_reliability_diagram must create a PNG file at the expected path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        confidences = [0.1, 0.4, 0.6, 0.9, 0.8, 0.2]
        labels = [0, 0, 1, 1, 1, 0]
        analyzer = CalibrationAnalyzer(n_bins=5)
        path = analyzer.plot_reliability_diagram(
            confidences, labels, model_name="test-model", save_dir=tmpdir
        )
        if path is not None:  # matplotlib may not be available in CI
            assert os.path.exists(path), f"Expected PNG at {path}"
            assert path.endswith(".png")


def test_plot_empty_confidences_returns_none() -> None:
    """plot_reliability_diagram with empty confidences must return None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = CalibrationAnalyzer()
        result = analyzer.plot_reliability_diagram([], [], "model", tmpdir)
        assert result is None


# ---------------------------------------------------------------------------
# Property 4: ECE range
# Feature: evaluate-improvements, Property 4: ECE range
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

@given(
    pairs=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.integers(min_value=0, max_value=1),
        ),
        min_size=1,
        max_size=200,
    )
)
@settings(max_examples=100)
def test_ece_range(pairs: list[tuple[float, int]]) -> None:
    """ECE must be in [0, 1] for any valid (confidence, label) pairs."""
    confidences = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    analyzer = CalibrationAnalyzer(n_bins=10)
    ece = analyzer.compute_ece(confidences, labels)
    assert ece is not None
    assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"


# ---------------------------------------------------------------------------
# Property 5: ECE perfect calibration
# Feature: evaluate-improvements, Property 5: ECE perfect calibration
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

@given(
    n_bins=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=50)
def test_ece_perfect_calibration(n_bins: int) -> None:
    """When mean confidence equals accuracy in every bin, ECE must be 0.0.

    We construct perfectly calibrated data by placing samples at each bin
    center with exactly the right fraction of correct labels so that
    mean_confidence == empirical_accuracy in every bin.
    """
    # Use a large enough n_per_bin so that rounding doesn't introduce error.
    # We pick n_per_bin = 100 so that round(center * 100) / 100 ≈ center.
    n_per_bin = 100
    bin_width = 1.0 / n_bins
    confidences: list[float] = []
    labels: list[int] = []

    for b in range(n_bins):
        center = (b + 0.5) * bin_width
        # n_correct / n_per_bin == center exactly when center * n_per_bin is integer.
        # With n_per_bin=100 and centers at multiples of 0.5/n_bins this is close enough.
        n_correct = round(center * n_per_bin)
        n_correct = max(0, min(n_per_bin, n_correct))
        for i in range(n_per_bin):
            confidences.append(center)
            labels.append(1 if i < n_correct else 0)

    analyzer = CalibrationAnalyzer(n_bins=n_bins)
    ece = analyzer.compute_ece(confidences, labels)
    assert ece is not None
    # With n_per_bin=100 the rounding error per bin is at most 0.005,
    # so ECE should be very small (< 0.01).
    assert ece < 0.01, f"Expected near-zero ECE for near-perfect calibration, got {ece}"
