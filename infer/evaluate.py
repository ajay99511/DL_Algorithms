"""Evaluation metrics for reasoning and generation quality.

Implements exact match accuracy and distinct-n diversity metrics.
"""

from __future__ import annotations

from collections import Counter


def exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute fraction of predictions that exactly match references.

    Comparison is case-insensitive and strips leading/trailing whitespace.

    Args:
        predictions: List of predicted answer strings.
        references: List of ground-truth answer strings.

    Returns:
        Float in [0.0, 1.0] — fraction of exact matches.

    Raises:
        ValueError: If predictions and references have different lengths.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length, "
            f"got {len(predictions)} and {len(references)}"
        )
    if not predictions:
        return 0.0

    matches = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return matches / len(predictions)


def distinct_n(texts: list[str], n: int) -> float:
    """Compute distinct-n diversity metric.

    Measures lexical diversity as the ratio of unique n-grams to total n-grams
    across all texts.

    distinct-n = |unique n-grams| / |total n-grams|

    Args:
        texts: List of generated text strings.
        n: N-gram size (e.g., 1 for unigrams, 2 for bigrams).

    Returns:
        Float in [0.0, 1.0] — higher means more diverse.
        Returns 0.0 if there are no n-grams.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    all_ngrams: list[tuple[str, ...]] = []

    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    return unique_ngrams / total_ngrams
