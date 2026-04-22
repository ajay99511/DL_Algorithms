"""Tests for report generation.

Property 14: Evaluation Report Column Completeness

# Feature: deep-learning-llm-mastery, Property 14: Evaluation Report Column Completeness
"""

from __future__ import annotations

import csv
import io
import os
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from evaluate.report import generate_csv_report, generate_markdown_report

_EXPECTED_COLUMNS = ["Model", "ARC-Challenge", "HellaSwag", "MMLU", "TruthfulQA", "Average"]

_ALL_TASKS = {
    "arc_challenge": 0.5,
    "hellaswag": 0.6,
    "mmlu": 0.4,
    "truthfulqa_mc": 0.3,
}


# ---------------------------------------------------------------------------
# Property 14: Evaluation Report Column Completeness
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 14: Evaluation Report Column Completeness
@given(
    models=st.lists(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_report_column_completeness(models: list[str]) -> None:
    """CSV must contain exactly: Model,ARC-Challenge,HellaSwag,MMLU,TruthfulQA,Average

    **Validates: Requirements 6.3, 6.12**
    """
    results = {m: dict(_ALL_TASKS) for m in models}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_csv_report(results, tmp_path)

        with open(tmp_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            actual_columns = reader.fieldnames or []

        assert list(actual_columns) == _EXPECTED_COLUMNS, (
            f"Expected columns {_EXPECTED_COLUMNS}, got {actual_columns}"
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------

def test_report_handles_none_values() -> None:
    """Missing task scores (None) must be written as empty string in CSV."""
    results = {
        "gpt2": {
            "arc_challenge": 0.5,
            "hellaswag": None,
            "mmlu": 0.4,
            "truthfulqa_mc": None,
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_csv_report(results, tmp_path)

        with open(tmp_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["HellaSwag"] == "", f"Expected empty string for None, got {row['HellaSwag']!r}"
        assert row["TruthfulQA"] == "", f"Expected empty string for None, got {row['TruthfulQA']!r}"
        # Non-None values should be formatted
        assert row["ARC-Challenge"] != ""
        assert row["MMLU"] != ""
    finally:
        os.unlink(tmp_path)


def test_markdown_report_contains_model_names() -> None:
    """Markdown report must contain all model names."""
    model_names = ["gpt2", "EleutherAI/pythia-160m"]
    results = {m: dict(_ALL_TASKS) for m in model_names}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_markdown_report(results, tmp_path)

        with open(tmp_path, encoding="utf-8") as f:
            content = f.read()

        for name in model_names:
            assert name in content, f"Model name '{name}' not found in markdown report"
    finally:
        os.unlink(tmp_path)


def test_csv_column_order_is_exact() -> None:
    """Column order must be exactly: Model,ARC-Challenge,HellaSwag,MMLU,TruthfulQA,Average."""
    results = {"test-model": dict(_ALL_TASKS)}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_csv_report(results, tmp_path)

        with open(tmp_path, newline="", encoding="utf-8") as f:
            first_line = f.readline().strip()

        assert first_line == ",".join(_EXPECTED_COLUMNS), (
            f"Header line mismatch: {first_line!r}"
        )
    finally:
        os.unlink(tmp_path)


def test_average_computed_correctly() -> None:
    """Average column must be the mean of non-None task scores."""
    results = {
        "model-a": {
            "arc_challenge": 0.4,
            "hellaswag": 0.6,
            "mmlu": 0.5,
            "truthfulqa_mc": 0.5,
        }
    }
    expected_avg = (0.4 + 0.6 + 0.5 + 0.5) / 4  # 0.5

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_csv_report(results, tmp_path)

        with open(tmp_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        actual_avg = float(rows[0]["Average"])
        assert abs(actual_avg - expected_avg) < 1e-4, (
            f"Expected average {expected_avg}, got {actual_avg}"
        )
    finally:
        os.unlink(tmp_path)
