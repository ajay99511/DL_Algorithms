"""Data loading utilities for GSM8K and BIG-Bench Hard benchmarks.

References:
    # Ref: Cobbe et al., 2021 — "Training Verifiers to Solve Math Word Problems" (GSM8K)
    # Ref: Srivastava et al., 2022 — "Beyond the Imitation Game: Quantifying and Extrapolating
    #      the Capabilities of Language Models" (BIG-Bench)
"""

from __future__ import annotations


def load_gsm8k(subset_size: int) -> list[dict]:
    """Load gsm8k test split via HF Datasets, return first subset_size problems.

    Each dict has 'question' and 'answer' keys.

    Args:
        subset_size: Number of problems to return.

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from e

    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        problems = []
        for i, item in enumerate(dataset):
            if i >= subset_size:
                break
            problems.append({
                "question": item["question"],
                "answer": item["answer"],
            })
        return problems
    except Exception as e:
        raise RuntimeError(f"Failed to load GSM8K dataset: {e}") from e


def load_bigbench_hard(subset_size: int) -> list[dict]:
    """Load maveriq/bigbenchhard via HF Datasets, return first subset_size problems.

    Each dict has 'input' and 'target' keys.

    Args:
        subset_size: Number of problems to return.

    Returns:
        List of dicts with 'input' and 'target' keys.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from e

    try:
        # maveriq/bigbenchhard has multiple tasks; load the first available config
        dataset = load_dataset("maveriq/bigbenchhard", "boolean_expressions", split="train")
        problems = []
        for i, item in enumerate(dataset):
            if i >= subset_size:
                break
            problems.append({
                "input": item["input"],
                "target": item["target"],
            })
        return problems
    except Exception as e:
        raise RuntimeError(f"Failed to load BIG-Bench Hard dataset: {e}") from e
