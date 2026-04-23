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


# ---------------------------------------------------------------------------
# Narrative report: best-model annotation, Wilson CIs, z-test, narrative text
# ---------------------------------------------------------------------------

import math as _math
from typing import Any as _Any


def find_best_models(
    results: dict[str, dict[str, float | None]],
) -> dict[str, str]:
    """Return the model with the highest non-None score for each task.

    Args:
        results: Mapping of model_name -> {task_name: score_or_None}.

    Returns:
        Dict mapping task_name -> model_name with the highest score.
        Tasks where all models have None scores are omitted.
    """
    best: dict[str, str] = {}
    # Collect all task names
    all_tasks: set[str] = set()
    for task_scores in results.values():
        all_tasks.update(task_scores.keys())

    for task in all_tasks:
        best_model: str | None = None
        best_score: float = float("-inf")
        for model_name, task_scores in results.items():
            score = task_scores.get(task)
            if score is not None and score > best_score:
                best_score = score
                best_model = model_name
        if best_model is not None:
            best[task] = best_model

    return best


def wilson_ci(
    p: float,
    n: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a proportion.

    Args:
        p: Observed proportion in [0, 1].
        n: Sample size (>= 1).
        z: Z-score for the desired confidence level (default 1.96 for 95%).

    Returns:
        (lower, upper) satisfying 0 <= lower <= p <= upper <= 1.
    """
    if n <= 0:
        return (0.0, 1.0)

    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * _math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return lower, upper


def two_proportion_ztest(
    n1: int,
    k1: int,
    n2: int,
    k2: int,
) -> float:
    """Two-proportion z-test; returns a p-value in [0.0, 1.0].

    Tests H0: p1 == p2 against H1: p1 != p2 (two-tailed).

    Args:
        n1: Total observations for model 1.
        k1: Successes (correct predictions) for model 1.
        n2: Total observations for model 2.
        k2: Successes for model 2.

    Returns:
        Two-tailed p-value in [0.0, 1.0].
    """
    import math as m

    if n1 <= 0 or n2 <= 0:
        return 1.0

    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)

    denom = _math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0.0:
        return 1.0  # identical proportions → no difference

    z = (p1 - p2) / denom

    # Two-tailed p-value using normal CDF approximation
    # P(|Z| > |z|) = 2 * (1 - Phi(|z|))
    # Use math.erfc for the complementary error function
    p_value = _math.erfc(abs(z) / _math.sqrt(2))
    return max(0.0, min(1.0, p_value))


def generate_narrative_report(
    results: dict[str, dict[str, float | None]],
    perplexity: dict[str, float | None],
    calibration: dict[str, dict[str, _Any] | None],
    few_shot: dict[tuple[str, str, int], float | None],
    output_path: str,
    n_samples: int = 1000,
) -> None:
    """Write a Markdown narrative report with statistical analysis.

    Includes:
    - Results table with Perplexity and ECE columns
    - Best-model ★ annotations per task
    - Wilson 95% CI bounds per cell
    - Pairwise z-test p-values (flagged with * when p < 0.05)
    - Narrative paragraph (>= 50 words)

    Args:
        results:     model_name -> {task_name: score_or_None}
        perplexity:  model_name -> perplexity_or_None
        calibration: model_name -> {"ece": float|None, ...} or None
        few_shot:    (model, task, n_shots) -> accuracy_or_None
        output_path: Destination Markdown file path.
        n_samples:   Assumed sample count per task for CI/z-test computation.
    """
    from pathlib import Path as _Path

    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    best_models = find_best_models(results)

    # Collect all task names in canonical order
    task_keys = list(_TASK_COLUMNS.keys())

    lines: list[str] = ["# Evaluation Narrative Report", ""]

    # ------------------------------------------------------------------ #
    # Results table
    # ------------------------------------------------------------------ #
    header_cols = ["Model"] + [_TASK_COLUMNS.get(t, t) for t in task_keys] + ["Perplexity", "ECE", "Average"]
    lines.append("## Results")
    lines.append("")
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for model_name in model_names:
        task_scores = results[model_name]
        cells = [model_name]
        for task_key in task_keys:
            score = task_scores.get(task_key)
            if score is None:
                cells.append("—")
            else:
                lo, hi = wilson_ci(score, n_samples)
                star = " ★" if best_models.get(task_key) == model_name else ""
                cells.append(f"{score:.4f}{star} [{lo:.3f}, {hi:.3f}]")

        # Perplexity
        ppl = perplexity.get(model_name)
        cells.append(f"{ppl:.2f}" if ppl is not None else "—")

        # ECE
        cal = calibration.get(model_name)
        ece = cal.get("ece") if isinstance(cal, dict) else None
        cells.append(f"{ece:.4f}" if ece is not None else "—")

        # Average
        valid_scores = [v for v in task_scores.values() if v is not None]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else None
        cells.append(f"{avg:.4f}" if avg is not None else "—")

        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # ------------------------------------------------------------------ #
    # Pairwise significance table (skip if only one model)
    # ------------------------------------------------------------------ #
    if len(model_names) >= 2:
        lines.append("## Pairwise Significance (two-proportion z-test)")
        lines.append("")
        lines.append("p < 0.05 flagged with *")
        lines.append("")
        sig_header = ["Task", "Model A", "Model B", "p-value", "Significant?"]
        lines.append("| " + " | ".join(sig_header) + " |")
        lines.append("| " + " | ".join(["---"] * len(sig_header)) + " |")

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m_a = model_names[i]
                m_b = model_names[j]
                for task_key in task_keys:
                    s_a = results[m_a].get(task_key)
                    s_b = results[m_b].get(task_key)
                    if s_a is None or s_b is None:
                        continue
                    k_a = round(s_a * n_samples)
                    k_b = round(s_b * n_samples)
                    p_val = two_proportion_ztest(n_samples, k_a, n_samples, k_b)
                    sig = "Yes *" if p_val < 0.05 else "No"
                    task_display = _TASK_COLUMNS.get(task_key, task_key)
                    lines.append(f"| {task_display} | {m_a} | {m_b} | {p_val:.4f} | {sig} |")

        lines.append("")
    else:
        lines.append("## Pairwise Significance")
        lines.append("")
        lines.append(
            "_Only one model evaluated. No pairwise comparison is possible._"
        )
        lines.append("")

    # ------------------------------------------------------------------ #
    # Narrative paragraph
    # ------------------------------------------------------------------ #
    lines.append("## Summary")
    lines.append("")

    # Identify overall best model by average score
    avg_scores: dict[str, float] = {}
    for model_name in model_names:
        valid = [v for v in results[model_name].values() if v is not None]
        if valid:
            avg_scores[model_name] = sum(valid) / len(valid)

    overall_best = max(avg_scores, key=avg_scores.__getitem__) if avg_scores else None

    # Identify tasks with significant differences
    sig_tasks: list[str] = []
    if len(model_names) >= 2:
        for task_key in task_keys:
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    s_a = results[model_names[i]].get(task_key)
                    s_b = results[model_names[j]].get(task_key)
                    if s_a is not None and s_b is not None:
                        k_a = round(s_a * n_samples)
                        k_b = round(s_b * n_samples)
                        if two_proportion_ztest(n_samples, k_a, n_samples, k_b) < 0.05:
                            task_display = _TASK_COLUMNS.get(task_key, task_key)
                            if task_display not in sig_tasks:
                                sig_tasks.append(task_display)

    # Calibration concerns
    cal_concerns: list[str] = []
    for model_name in model_names:
        cal = calibration.get(model_name)
        ece = cal.get("ece") if isinstance(cal, dict) else None
        if ece is not None and ece > 0.1:
            cal_concerns.append(f"{model_name} (ECE={ece:.3f})")

    # Build narrative
    narrative_parts: list[str] = []
    if overall_best:
        narrative_parts.append(
            f"Across all evaluated benchmarks, **{overall_best}** achieves the highest "
            f"average accuracy ({avg_scores[overall_best]:.4f}), making it the strongest "
            f"overall performer in this evaluation."
        )
    if sig_tasks:
        narrative_parts.append(
            f"Statistically significant differences (p < 0.05) between models were "
            f"observed on the following tasks: {', '.join(sig_tasks)}. "
            f"These results suggest that model choice meaningfully impacts performance "
            f"on these benchmarks."
        )
    else:
        narrative_parts.append(
            "No statistically significant differences were detected between models at "
            "the p < 0.05 level, suggesting that performance gaps may be within the "
            "margin of sampling variability given the evaluation set size."
        )
    if cal_concerns:
        narrative_parts.append(
            f"Calibration analysis reveals potential concerns for: "
            f"{', '.join(cal_concerns)}. High ECE indicates that confidence scores "
            f"do not reliably reflect empirical accuracy, which is important to "
            f"consider when using model outputs for downstream decision-making."
        )
    else:
        narrative_parts.append(
            "Calibration metrics are within acceptable bounds for all evaluated models, "
            "indicating that confidence scores are reasonably well-aligned with "
            "empirical accuracy across the evaluated benchmarks."
        )

    lines.append(" ".join(narrative_parts))
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
