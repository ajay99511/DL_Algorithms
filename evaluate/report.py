"""Report generation: CSV and Markdown tables from evaluation results."""

from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical column order (task_name -> display name)
_TASK_COLUMNS: dict[str, str] = {
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU",
    "truthfulqa_mc": "TruthfulQA",
}

_CSV_COLUMNS = ["Model", "ARC-Challenge", "HellaSwag", "MMLU", "TruthfulQA", "Average"]


def _compute_average(scores: dict[str, float | None]) -> str:
    """Return the mean of non-None task scores as a formatted string, or '' if none."""
    values = [v for v in scores.values() if v is not None]
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def _format_score(value: float | None) -> str:
    """Format a score as a 4-decimal string, or empty string for None."""
    if value is None:
        return ""
    return f"{value:.4f}"


def generate_csv_report(
    results: dict[str, dict[str, float | None]],
    output_path: str,
) -> None:
    """Write evaluation results to a CSV file.

    Columns (exact order): Model, ARC-Challenge, HellaSwag, MMLU, TruthfulQA, Average

    Missing task scores are written as empty strings. A warning is logged for
    each missing task.

    Parameters
    ----------
    results:
        Mapping of model_name -> {task_name: score_or_None}.
    output_path:
        Destination CSV file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for model_name, task_scores in results.items():
            # Warn about missing tasks
            for task_key in _TASK_COLUMNS:
                if task_key not in task_scores:
                    warnings.warn(
                        f"Task '{task_key}' missing for model '{model_name}'; filling with empty string.",
                        stacklevel=2,
                    )

            row: dict[str, str] = {"Model": model_name}
            for task_key, col_name in _TASK_COLUMNS.items():
                row[col_name] = _format_score(task_scores.get(task_key))

            row["Average"] = _compute_average(task_scores)
            writer.writerow(row)


def generate_markdown_report(
    results: dict[str, dict[str, float | None]],
    output_path: str,
) -> None:
    """Write evaluation results to a Markdown table.

    Same data and column order as :func:`generate_csv_report`.

    Parameters
    ----------
    results:
        Mapping of model_name -> {task_name: score_or_None}.
    output_path:
        Destination Markdown file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    header = "| " + " | ".join(_CSV_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(_CSV_COLUMNS)) + " |"

    lines = [header, separator]

    for model_name, task_scores in results.items():
        for task_key in _TASK_COLUMNS:
            if task_key not in task_scores:
                warnings.warn(
                    f"Task '{task_key}' missing for model '{model_name}'; filling with empty string.",
                    stacklevel=2,
                )

        cells = [model_name]
        for task_key in _TASK_COLUMNS:
            cells.append(_format_score(task_scores.get(task_key)))
        cells.append(_compute_average(task_scores))

        lines.append("| " + " | ".join(cells) + " |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
