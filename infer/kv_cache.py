"""KV Cache implementation for efficient autoregressive inference.

References:
    # Ref: Pope et al., 2022 — "Efficiently Scaling Transformer Inference"
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from infer.config import ReasoningConfig


class KVCache:
    """Stores past key/value tensors to avoid recomputation during autoregressive generation.

    Each layer maintains a growing cache of shape (1, n_heads, seq_len, d_head).
    On each call to update(), the new k/v slice is appended and the full accumulated
    tensors are returned.

    # Ref: Pope et al., 2022 — "Efficiently Scaling Transformer Inference"
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_head: int,
        max_seq_len: int,
    ) -> None:
        """Initialize empty cache tensors for each layer.

        Args:
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads per layer.
            d_head: Dimension of each attention head.
            max_seq_len: Maximum sequence length the cache can hold.
        """
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len

        # Cache storage: list of (k_cache, v_cache) per layer
        # Each starts as an empty tensor with 0 in the seq_len dimension
        self._k_cache: list[Tensor] = [
            torch.empty(1, n_heads, 0, d_head) for _ in range(n_layers)
        ]
        self._v_cache: list[Tensor] = [
            torch.empty(1, n_heads, 0, d_head) for _ in range(n_layers)
        ]

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append new k/v and return full cached k/v for this layer.

        Args:
            layer_idx: Index of the transformer layer (0-indexed).
            k: New key tensor of shape (1, n_heads, new_tokens, d_head).
            v: New value tensor of shape (1, n_heads, new_tokens, d_head).

        Returns:
            (full_k, full_v): Accumulated key/value tensors of shape
                (1, n_heads, current_seq_len, d_head).
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for cache with {self.n_layers} layers"
            )

        # Concatenate along the sequence dimension (dim=2)
        self._k_cache[layer_idx] = torch.cat([self._k_cache[layer_idx], k], dim=2)
        self._v_cache[layer_idx] = torch.cat([self._v_cache[layer_idx], v], dim=2)

        return self._k_cache[layer_idx], self._v_cache[layer_idx]

    def clear(self) -> None:
        """Reset all cached tensors."""
        self._k_cache = [
            torch.empty(1, self.n_heads, 0, self.d_head) for _ in range(self.n_layers)
        ]
        self._v_cache = [
            torch.empty(1, self.n_heads, 0, self.d_head) for _ in range(self.n_layers)
        ]

    @property
    def current_len(self) -> int:
        """Current sequence length in cache (from layer 0)."""
        return self._k_cache[0].shape[2]


def benchmark_kv_cache(
    model: nn.Module,
    prompts: list[str],
    tokenizer: Any,
    config: ReasoningConfig,
) -> dict[str, float]:
    """Measure tokens/sec with and without KV cache using HuggingFace model.generate().

    For the cached version: use model.generate() with use_cache=True.
    For uncached: use model.generate() with use_cache=False.

    Args:
        model: A HuggingFace AutoModelForCausalLM model.
        prompts: List of text prompts to benchmark.
        tokenizer: HuggingFace tokenizer.
        config: ReasoningConfig with max_new_tokens.

    Returns:
        Dict with keys:
            'tokens_per_sec_cached': throughput with KV cache enabled.
            'tokens_per_sec_uncached': throughput without KV cache.
            'speedup_ratio': cached / uncached throughput ratio.
    """
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)

    total_tokens = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        total_tokens += inputs["input_ids"].shape[1] + config.max_new_tokens

    def _run(use_cache: bool) -> float:
        """Run generation and return tokens per second."""
        start = time.perf_counter()
        generated_tokens = 0
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                input_ids = inputs["input_ids"].to(device)
                output = model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    use_cache=use_cache,
                    do_sample=False,  # greedy for fair comparison
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated_tokens += output.shape[1] - input_ids.shape[1]
        elapsed = time.perf_counter() - start
        return generated_tokens / max(elapsed, 1e-6)

    tokens_per_sec_cached = _run(use_cache=True)
    tokens_per_sec_uncached = _run(use_cache=False)
    speedup_ratio = tokens_per_sec_cached / max(tokens_per_sec_uncached, 1e-6)

    return {
        "tokens_per_sec_cached": tokens_per_sec_cached,
        "tokens_per_sec_uncached": tokens_per_sec_uncached,
        "speedup_ratio": speedup_ratio,
    }
