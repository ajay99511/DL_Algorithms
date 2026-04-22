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
