"""
Text generation strategies for Project 2: Transformer Pre-training.

Provides greedy decoding and nucleus (top-p) sampling.
Both are deterministic under a fixed seed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from pretrain.model import GPTModel
from shared.seed import fix_all_seeds


def greedy_decode(
    model: GPTModel,
    input_ids: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """
    Greedy autoregressive decoding: always pick the highest-probability next token.

    Args:
        model:          GPTModel in eval mode.
        input_ids:      (1, T) or (B, T) LongTensor of prompt token IDs.
        max_new_tokens: Number of new tokens to generate.

    Returns:
        (B, T + max_new_tokens) LongTensor — prompt + generated tokens.
    """
    model.eval()
    device = torch.device("cpu")
    ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context_length if needed
            context = ids[:, -model.context_length:]
            logits, _ = model(context)          # (B, T', vocab_size)
            next_logits = logits[:, -1, :]      # (B, vocab_size)
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            ids = torch.cat([ids, next_token], dim=1)

    return ids


def nucleus_sample(
    model: GPTModel,
    input_ids: Tensor,
    max_new_tokens: int,
    top_p: float = 0.9,
    temperature: float = 1.0,
    seed: int | None = None,
) -> Tensor:
    """
    Nucleus (top-p) sampling: sample from the smallest set of tokens whose
    cumulative probability exceeds top_p.

    Deterministic under a fixed seed.

    Args:
        model:          GPTModel in eval mode.
        input_ids:      (1, T) or (B, T) LongTensor of prompt token IDs.
        max_new_tokens: Number of new tokens to generate.
        top_p:          Cumulative probability threshold (0 < top_p <= 1.0).
        temperature:    Softmax temperature. Higher = more random.
        seed:           Optional integer seed for reproducibility.
                        When provided, fix_all_seeds(seed) is called before generation.

    Returns:
        (B, T + max_new_tokens) LongTensor — prompt + generated tokens.
    """
    if seed is not None:
        fix_all_seeds(seed)

    model.eval()
    device = torch.device("cpu")
    ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context_length if needed
            context = ids[:, -model.context_length:]
            logits, _ = model(context)          # (B, T', vocab_size)
            next_logits = logits[:, -1, :]      # (B, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Convert to probabilities
            probs = F.softmax(next_logits, dim=-1)  # (B, vocab_size)

            # Sort descending
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above top_p
            # Shift right so the first token above threshold is kept
            sorted_mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_mask] = 0.0

            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Sample from the filtered distribution
            sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1)
            next_token = sorted_indices.gather(dim=-1, index=sampled_sorted_idx)  # (B, 1)

            ids = torch.cat([ids, next_token], dim=1)

    return ids
