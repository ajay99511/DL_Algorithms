"""Benchmark runner for comparing inference strategies on GSM8K.

Evaluates greedy, beam search, top-k, and nucleus sampling on accuracy,
diversity, throughput, and response length.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from infer.config import ReasoningConfig
from infer.data import load_gsm8k
from infer.evaluate import distinct_n, exact_match_accuracy
from infer.inference import (
    beam_search,
    greedy_decode,
    nucleus_sample,
    top_k_sample,
)


def run_benchmark(config: ReasoningConfig) -> list[dict]:
    """Run all inference strategies on GSM8K subset and collect metrics.

    For each strategy: collects exact_match_accuracy, distinct_1, distinct_2,
    tokens_per_sec, and avg_response_length. Saves results to config.benchmark_file.

    Args:
        config: ReasoningConfig with model, inference, and path settings.

    Returns:
        List of result dicts, one per strategy.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        ) from e

    device = torch.device("cpu")

    # Load model and tokenizer
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model = model.to(device)
    model.eval()

    # Load dataset
    print(f"Loading GSM8K subset ({config.gsm8k_subset_size} problems)...")
    problems = load_gsm8k(config.gsm8k_subset_size)

    strategies = [
        ("greedy", _run_greedy),
        ("beam_search", _run_beam_search),
        ("top_k", _run_top_k),
        ("nucleus", _run_nucleus),
    ]

    results: list[dict] = []

    for strategy_name, strategy_fn in strategies:
        print(f"Running strategy: {strategy_name}")
        predictions: list[str] = []
        references: list[str] = []
        response_lengths: list[int] = []

        start_time = time.perf_counter()
        total_new_tokens = 0

        for problem in problems:
            question = problem["question"]
            reference = problem["answer"]

            # Tokenize prompt
            prompt = f"Q: {question}\nA:"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(device)
            prompt_len = input_ids.shape[1]

            # Run strategy
            output_ids = strategy_fn(model, input_ids, tokenizer, config)

            # Decode generated tokens only
            new_token_ids = output_ids[0, prompt_len:]
            generated_text = tokenizer.decode(new_token_ids.tolist(), skip_special_tokens=True)

            predictions.append(generated_text.strip())
            references.append(reference.strip())
            response_lengths.append(len(new_token_ids))
            total_new_tokens += len(new_token_ids)

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_new_tokens / max(elapsed, 1e-6)

        result = {
            "strategy": strategy_name,
            "exact_match_accuracy": exact_match_accuracy(predictions, references),
            "distinct_1": distinct_n(predictions, 1),
            "distinct_2": distinct_n(predictions, 2),
            "tokens_per_sec": tokens_per_sec,
            "avg_response_length": sum(response_lengths) / max(len(response_lengths), 1),
        }
        results.append(result)
        print(f"  {strategy_name}: accuracy={result['exact_match_accuracy']:.3f}, "
              f"tokens/sec={tokens_per_sec:.1f}")

    # Save results
    output_path = Path(config.benchmark_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {config.benchmark_file}")

    return results


def _run_greedy(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    config: ReasoningConfig,
) -> torch.Tensor:
    """Run greedy decoding and return output token IDs."""
    output_ids, _ = greedy_decode(model, input_ids, config.max_new_tokens, tokenizer)
    return output_ids


def _run_beam_search(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    config: ReasoningConfig,
) -> torch.Tensor:
    """Run beam search and return best output token IDs."""
    output_ids, _ = beam_search(
        model, input_ids, config.max_new_tokens, config.beam_width, tokenizer
    )
    return output_ids


def _run_top_k(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    config: ReasoningConfig,
) -> torch.Tensor:
    """Run top-k sampling and return output token IDs."""
    return top_k_sample(
        model,
        input_ids,
        config.max_new_tokens,
        k=config.top_k,
        temperature=config.temperature,
        seed=config.seed,
    )


def _run_nucleus(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    config: ReasoningConfig,
) -> torch.Tensor:
    """Run nucleus sampling and return output token IDs."""
    return nucleus_sample(
        model,
        input_ids,
        config.max_new_tokens,
        top_p=config.top_p,
        temperature=config.temperature,
        seed=config.seed,
    )
