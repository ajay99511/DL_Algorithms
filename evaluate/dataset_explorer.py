"""Dataset exploration utilities — streaming, no full download required."""

from __future__ import annotations

from typing import Any

try:
    from datasets import load_dataset  # type: ignore
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

# Pre-configured streaming datasets
_STREAMING_DATASETS: dict[str, dict[str, Any]] = {
    "EleutherAI/pile": {"streaming": True},
    "allenai/c4": {"streaming": True},
    "Skylion007/openwebtext": {"streaming": True},
}

# Text field names per dataset (fallback order)
_TEXT_FIELDS = ("text", "content", "story", "passage")


def _get_text(sample: dict[str, Any]) -> str:
    """Extract text from a dataset sample using known field names."""
    for field in _TEXT_FIELDS:
        if field in sample and isinstance(sample[field], str):
            return sample[field]
    # Fallback: first string value
    for v in sample.values():
        if isinstance(v, str):
            return v
    return ""


def explore_dataset(
    dataset_name: str,
    config_name: str | None = None,
    n_samples: int = 1000,
) -> dict[str, Any]:
    """Stream *n_samples* from a dataset and compute basic statistics.

    No full download is required — uses HuggingFace streaming mode.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier (e.g. "EleutherAI/pile").
    config_name:
        Optional dataset config / subset name (e.g. "en" for allenai/c4).
    n_samples:
        Number of samples to stream and analyse.

    Returns
    -------
    dict with keys:
        - ``estimated_token_count``: rough token count (whitespace split)
        - ``vocabulary_size``: number of unique whitespace-split tokens
        - ``avg_sequence_length``: mean token count per sample
        - ``sample_texts``: list of the first 3 sample texts
    """
    if not _HAS_DATASETS:
        return {
            "estimated_token_count": None,
            "vocabulary_size": None,
            "avg_sequence_length": None,
            "sample_texts": [],
            "error": "datasets library not installed",
        }

    # Build load_dataset kwargs
    load_kwargs: dict[str, Any] = {"streaming": True, "split": "train", "trust_remote_code": True}
    if config_name is not None:
        dataset = load_dataset(dataset_name, config_name, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, **load_kwargs)

    texts: list[str] = []
    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
        texts.append(_get_text(sample))

    # Compute statistics
    all_tokens: list[str] = []
    sequence_lengths: list[int] = []
    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)
        sequence_lengths.append(len(tokens))

    estimated_token_count = len(all_tokens)
    vocabulary_size = len(set(all_tokens))
    avg_sequence_length = (
        sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0.0
    )
    sample_texts = texts[:3]

    return {
        "estimated_token_count": estimated_token_count,
        "vocabulary_size": vocabulary_size,
        "avg_sequence_length": avg_sequence_length,
        "sample_texts": sample_texts,
    }


# ---------------------------------------------------------------------------
# Rich statistics: n-gram overlap, domain distribution, length distribution
# ---------------------------------------------------------------------------

import logging as _logging
import warnings as _warnings

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_ds_logger = _logging.getLogger(__name__)


def _build_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Return the set of n-grams from a token list."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def compute_ngram_overlap(
    train_texts: list[str],
    test_texts: list[str],
    n: int = 1,
) -> float:
    """Compute the fraction of test n-grams that appear in the training set.

    Used to detect benchmark contamination: a high overlap suggests the test
    set may have been seen during training.

    Args:
        train_texts: List of training corpus strings.
        test_texts:  List of test corpus strings.
        n:           N-gram order (1 = unigram, 2 = bigram).

    Returns:
        Overlap ratio in [0.0, 1.0]. Returns 0.0 if either corpus is empty.
    """
    if not train_texts or not test_texts:
        return 0.0

    train_ngrams: set[tuple[str, ...]] = set()
    for text in train_texts:
        train_ngrams |= _build_ngrams(text.split(), n)

    test_ngrams: set[tuple[str, ...]] = set()
    for text in test_texts:
        test_ngrams |= _build_ngrams(text.split(), n)

    if not test_ngrams:
        return 0.0

    overlap = len(test_ngrams & train_ngrams) / len(test_ngrams)
    return overlap


def compute_domain_distribution(
    samples: list[dict[str, Any]],
    label_field: str = "source",
) -> dict[str, float]:
    """Compute the proportion of samples belonging to each domain/source label.

    Args:
        samples:     List of sample dicts, each expected to have ``label_field``.
        label_field: Key in each sample dict that identifies the domain/source.

    Returns:
        Dict mapping label → proportion, summing to 1.0.
        Returns an empty dict if ``samples`` is empty.
    """
    if not samples:
        return {}

    counts: dict[str, int] = {}
    for sample in samples:
        label = str(sample.get(label_field, "unknown"))
        counts[label] = counts.get(label, 0) + 1

    total = len(samples)
    return {label: count / total for label, count in counts.items()}


def compute_length_distribution(
    texts: list[str],
    tokenizer: Any = None,
    save_path: str | None = None,
) -> list[int]:
    """Compute sequence lengths for a list of texts.

    Uses ``tokenizer.encode(text)`` if a tokenizer is provided; falls back to
    whitespace splitting with a logged warning.

    Args:
        texts:      List of text strings to measure.
        tokenizer:  Optional tokenizer with an ``encode(text) -> list[int]`` method.
                    If None or if loading fails, whitespace splitting is used.
        save_path:  If provided, saves a length distribution histogram PNG here.

    Returns:
        List of non-negative integer sequence lengths, one per input text.
    """
    lengths: list[int] = []
    use_whitespace = tokenizer is None

    if tokenizer is not None:
        # Validate tokenizer has encode method
        if not hasattr(tokenizer, "encode"):
            _ds_logger.warning(
                "Provided tokenizer has no 'encode' method; falling back to whitespace split."
            )
            use_whitespace = True

    for text in texts:
        if use_whitespace:
            lengths.append(len(text.split()))
        else:
            try:
                lengths.append(len(tokenizer.encode(text)))
            except Exception as exc:
                _ds_logger.warning(
                    "Tokenizer encode failed ('%s'); falling back to whitespace split.", exc
                )
                use_whitespace = True
                lengths.append(len(text.split()))

    if save_path is not None and lengths:
        if _HAS_MPL:
            from pathlib import Path as _Path
            _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig, ax = _plt.subplots(figsize=(7, 4))
            ax.hist(lengths, bins=min(50, len(set(lengths))), color="steelblue",
                    edgecolor="none", alpha=0.8)
            ax.set_xlabel("Sequence Length (tokens)")
            ax.set_ylabel("Count")
            ax.set_title("Sequence Length Distribution")
            fig.tight_layout()
            fig.savefig(save_path, dpi=100)
            _plt.close(fig)
        else:
            _ds_logger.warning("matplotlib not available; skipping length distribution plot.")

    return lengths
