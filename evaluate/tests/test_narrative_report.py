"""Tests for narrative report functions in evaluate/report.py."""
from __future__ import annotations

import os
import tempfile

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from evaluate.report import (
    find_best_models,
    wilson_ci,
    two_proportion_ztest,
    generate_narrative_report,
)


# ---------------------------------------------------------------------------
# find_best_models — unit tests
# ---------------------------------------------------------------------------

def test_find_best_models_correct_winner() -> None:
    """find_best_models must return the model with the highest score per task."""
    results = {
        "gpt2": {"arc_challenge": 0.4, "hellaswag": 0.7},
        "pythia": {"arc_challenge": 0.6, "hellaswag": 0.5},
    }
    best = find_best_models(results)
    assert best["arc_challenge"] == "pythia"
    assert best["hellaswag"] == "gpt2"


def test_find_best_models_skips_none() -> None:
    """Tasks where all models have None scores must be omitted."""
    results = {
        "gpt2": {"arc_challenge": None, "hellaswag": 0.7},
        "pythia": {"arc_challenge": None, "hellaswag": 0.5},
    }
    best = find_best_models(results)
    assert "arc_challenge" not in best
    assert "hellaswag" in best


def test_find_best_models_single_model() -> None:
    """Single model must be identified as best for all tasks with non-None scores."""
    results = {"gpt2": {"arc_challenge": 0.5, "mmlu": 0.4}}
    best = find_best_models(results)
    assert best["arc_challenge"] == "gpt2"
    assert best["mmlu"] == "gpt2"


# ---------------------------------------------------------------------------
# wilson_ci — unit tests
# ---------------------------------------------------------------------------

def test_wilson_ci_bounds_contain_p() -> None:
    """Wilson CI must satisfy lower <= p <= upper (within float tolerance)."""
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lo, hi = wilson_ci(p, n=100)
        assert lo <= p + 1e-9, f"lower={lo} > p={p}"
        assert p <= hi + 1e-9, f"p={p} > upper={hi}"


def test_wilson_ci_bounds_in_unit_interval() -> None:
    """Wilson CI bounds must be in [0, 1]."""
    lo, hi = wilson_ci(0.5, n=10)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_wilson_ci_zero_n_returns_full_interval() -> None:
    """n=0 must return (0.0, 1.0) without error."""
    lo, hi = wilson_ci(0.5, n=0)
    assert lo == 0.0
    assert hi == 1.0


# ---------------------------------------------------------------------------
# two_proportion_ztest — unit tests
# ---------------------------------------------------------------------------

def test_ztest_identical_proportions_high_pvalue() -> None:
    """Identical proportions must yield a high p-value (no significant difference)."""
    p_val = two_proportion_ztest(100, 50, 100, 50)
    assert p_val > 0.05, f"Expected p > 0.05 for identical proportions, got {p_val}"


def test_ztest_very_different_proportions_low_pvalue() -> None:
    """Very different proportions must yield a low p-value."""
    p_val = two_proportion_ztest(1000, 900, 1000, 100)
    assert p_val < 0.05, f"Expected p < 0.05 for very different proportions, got {p_val}"


def test_ztest_pvalue_in_range() -> None:
    """p-value must always be in [0, 1]."""
    for k1, k2 in [(0, 0), (50, 50), (100, 0), (0, 100)]:
        p_val = two_proportion_ztest(100, k1, 100, k2)
        assert 0.0 <= p_val <= 1.0, f"p-value out of range: {p_val}"


# ---------------------------------------------------------------------------
# generate_narrative_report — unit tests
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = {
    "gpt2": {"arc_challenge": 0.4, "hellaswag": 0.7, "mmlu": 0.3, "truthfulqa_mc": 0.5},
    "pythia": {"arc_challenge": 0.6, "hellaswag": 0.5, "mmlu": 0.4, "truthfulqa_mc": 0.4},
}
_SAMPLE_PPL = {"gpt2": 45.2, "pythia": 38.7}
_SAMPLE_CAL = {
    "gpt2": {"ece": 0.08, "reliability_diagram_path": None},
    "pythia": {"ece": 0.12, "reliability_diagram_path": None},
}
_SAMPLE_FEW_SHOT: dict = {}


def test_narrative_report_creates_file() -> None:
    """generate_narrative_report must create a Markdown file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "report.md")
        generate_narrative_report(
            _SAMPLE_RESULTS, _SAMPLE_PPL, _SAMPLE_CAL, _SAMPLE_FEW_SHOT, out
        )
        assert os.path.exists(out), "Report file was not created"


def test_narrative_report_contains_model_names() -> None:
    """Report must contain all model names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "report.md")
        generate_narrative_report(
            _SAMPLE_RESULTS, _SAMPLE_PPL, _SAMPLE_CAL, _SAMPLE_FEW_SHOT, out
        )
        content = open(out, encoding="utf-8").read()
        for model in _SAMPLE_RESULTS:
            assert model in content, f"Model '{model}' not found in report"


def test_narrative_report_contains_star_annotation() -> None:
    """Best-model ★ annotation must appear in the report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "report.md")
        generate_narrative_report(
            _SAMPLE_RESULTS, _SAMPLE_PPL, _SAMPLE_CAL, _SAMPLE_FEW_SHOT, out
        )
        content = open(out, encoding="utf-8").read()
        assert "★" in content, "Best-model ★ annotation not found in report"


def test_narrative_report_has_summary_section() -> None:
    """Report must contain a Summary section with >= 50 words."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "report.md")
        generate_narrative_report(
            _SAMPLE_RESULTS, _SAMPLE_PPL, _SAMPLE_CAL, _SAMPLE_FEW_SHOT, out
        )
        content = open(out, encoding="utf-8").read()
        assert "## Summary" in content
        # Extract text after ## Summary
        summary_text = content.split("## Summary")[-1]
        word_count = len(summary_text.split())
        assert word_count >= 50, f"Summary has only {word_count} words, expected >= 50"


def test_narrative_report_single_model_no_comparison() -> None:
    """Single-model report must note that no comparison is possible."""
    single_results = {"gpt2": {"arc_challenge": 0.4, "hellaswag": 0.7}}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "report.md")
        generate_narrative_report(
            single_results,
            {"gpt2": 45.2},
            {"gpt2": {"ece": 0.05}},
            {},
            out,
        )
        content = open(out, encoding="utf-8").read()
        assert "no pairwise comparison" in content.lower() or "not possible" in content.lower(), (
            "Single-model report must note that no comparison is possible"
        )


# ---------------------------------------------------------------------------
# Property 9: Best model correctness
# Feature: evaluate-improvements, Property 9: Best model correctness
# Validates: Requirements 8.1
# ---------------------------------------------------------------------------

@given(
    model_scores=st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_find_best_models_correctness(model_scores: dict[str, float]) -> None:
    """find_best_models must identify the model with the maximum score per task."""
    results = {model: {"task_a": score} for model, score in model_scores.items()}
    best = find_best_models(results)
    if "task_a" in best:
        expected_best = max(model_scores, key=model_scores.__getitem__)
        # Handle ties: best model must have the maximum score
        best_score = model_scores[best["task_a"]]
        max_score = max(model_scores.values())
        assert abs(best_score - max_score) < 1e-9, (
            f"Best model score {best_score} != max score {max_score}"
        )


# ---------------------------------------------------------------------------
# Property 10: Wilson CI validity
# Feature: evaluate-improvements, Property 10: Wilson CI validity
# Validates: Requirements 8.2
# ---------------------------------------------------------------------------

@given(
    p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=1, max_value=10000),
)
@settings(max_examples=100)
def test_wilson_ci_validity(p: float, n: int) -> None:
    """Wilson CI must satisfy 0 <= lower <= p <= upper <= 1."""
    lo, hi = wilson_ci(p, n)
    assert 0.0 <= lo, f"lower={lo} < 0"
    assert lo <= p + 1e-9, f"lower={lo} > p={p}"
    assert p <= hi + 1e-9, f"p={p} > upper={hi}"
    assert hi <= 1.0, f"upper={hi} > 1"


# ---------------------------------------------------------------------------
# Property 11: Z-test p-value range
# Feature: evaluate-improvements, Property 11: Z-test p-value range
# Validates: Requirements 8.3
# ---------------------------------------------------------------------------

@given(
    n1=st.integers(min_value=1, max_value=1000),
    k1=st.integers(min_value=0, max_value=1000),
    n2=st.integers(min_value=1, max_value=1000),
    k2=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_ztest_pvalue_range(n1: int, k1: int, n2: int, k2: int) -> None:
    """Z-test p-value must be in [0.0, 1.0] for any valid inputs."""
    k1 = min(k1, n1)
    k2 = min(k2, n2)
    p_val = two_proportion_ztest(n1, k1, n2, k2)
    assert 0.0 <= p_val <= 1.0, f"p-value out of range: {p_val}"
