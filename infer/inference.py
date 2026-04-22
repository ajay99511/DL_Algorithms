"""Inference strategies for autoregressive language models.

Implements greedy decoding, beam search, top-k sampling, and nucleus (top-p) sampling.

References:
    # Ref: Radford et al., 2019 — "Language Models are Unsupervised Multitask Learners" (GPT-2)
    # Ref: Holtzman et al., 2020 — "The Curious Case of Neural Text Degeneration" (nucleus sampling)
    # Ref: Fan et al., 2018 — "Hierarchical Neural Story Generation" (top-k sampling)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from shared.seed import fix_all_seeds


def greedy_decode(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    tokenizer: Any,
) -> tuple[Tensor, list[Tensor]]:
    """Greedy decoding: always pick the highest-probability next token.

    Args:
        model: A causal language model (HF AutoModelForCausalLM or compatible).
        input_ids: (1, T) LongTensor of prompt token IDs.
        max_new_tokens: Number of new tokens to generate.
        tokenizer: Tokenizer with eos_token_id attribute.

    Returns:
        (output_ids, per_step_log_probs):
            output_ids: (1, T + max_new_tokens) generated token IDs.
            per_step_log_probs: list of (vocab_size,) log-prob tensors, one per step.
    """
    model.eval()
    device = torch.device("cpu")
    current_ids = input_ids.to(device)
    per_step_log_probs: list[Tensor] = []

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits from model
            outputs = _get_logits(model, current_ids)
            # Take logits for the last position
            next_token_logits = outputs[:, -1, :]  # (1, vocab_size)
            log_probs = F.log_softmax(next_token_logits, dim=-1)  # (1, vocab_size)
            per_step_log_probs.append(log_probs.squeeze(0))  # (vocab_size,)

            # Greedy: pick argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Stop at EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return current_ids, per_step_log_probs


def beam_search(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    beam_width: int,
    tokenizer: Any,
) -> tuple[Tensor, list[dict]]:
    """Beam search decoding: maintain beam_width candidate sequences.

    Args:
        model: A causal language model.
        input_ids: (1, T) LongTensor of prompt token IDs.
        max_new_tokens: Number of new tokens to generate.
        beam_width: Number of beams to maintain.
        tokenizer: Tokenizer with eos_token_id attribute.

    Returns:
        (best_output_ids, beam_log):
            best_output_ids: (1, T + generated_len) best sequence.
            beam_log: list of dicts per step, each with:
                'candidates': list of token ids (top beam_width)
                'log_probs': list of cumulative log-probs (floats)
    """
    model.eval()
    device = torch.device("cpu")
    prompt = input_ids.to(device)
    prompt_len = prompt.shape[1]

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # Initialize beams: list of (cumulative_log_prob, sequence_ids)
    beams: list[tuple[float, Tensor]] = [(0.0, prompt)]
    beam_log: list[dict] = []
    completed: list[tuple[float, Tensor]] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if not beams:
                break

            all_candidates: list[tuple[float, Tensor]] = []

            for cum_log_prob, seq in beams:
                # Get logits for this beam
                outputs = _get_logits(model, seq)
                next_token_logits = outputs[:, -1, :]  # (1, vocab_size)
                log_probs = F.log_softmax(next_token_logits, dim=-1).squeeze(0)  # (vocab_size,)

                # Get top beam_width tokens
                top_log_probs, top_tokens = torch.topk(log_probs, beam_width)

                for lp, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                    new_seq = torch.cat([seq, torch.tensor([[tok]], device=device)], dim=1)
                    all_candidates.append((cum_log_prob + lp, new_seq))

            # Sort by cumulative log prob (descending) and keep top beam_width
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = all_candidates[:beam_width]

            # Log this step
            step_tokens = [int(cand[1][0, -1].item()) for cand in top_candidates]
            step_log_probs = [float(cand[0]) for cand in top_candidates]
            beam_log.append({"candidates": step_tokens, "log_probs": step_log_probs})

            # Filter completed vs active beams
            beams = []
            for cum_lp, seq in top_candidates:
                last_tok = int(seq[0, -1].item())
                if eos_token_id is not None and last_tok == eos_token_id:
                    completed.append((cum_lp, seq))
                else:
                    beams.append((cum_lp, seq))

            # If all beams completed, stop
            if not beams:
                break

    # Pick best: from completed if any, else from remaining beams
    all_final = completed + beams
    if all_final:
        best_log_prob, best_seq = max(all_final, key=lambda x: x[0])
    else:
        best_seq = prompt

    return best_seq, beam_log


def top_k_sample(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    k: int,
    temperature: float,
    seed: int | None = None,
) -> Tensor:
    """Top-k sampling: sample from the k highest-probability tokens.

    Args:
        model: A causal language model.
        input_ids: (1, T) LongTensor of prompt token IDs.
        max_new_tokens: Number of new tokens to generate.
        k: Number of top tokens to sample from.
        temperature: Softmax temperature (higher = more random).
        seed: Optional random seed for reproducibility.

    Returns:
        (1, T + max_new_tokens) generated token IDs.
    """
    if seed is not None:
        fix_all_seeds(seed)

    model.eval()
    device = torch.device("cpu")
    current_ids = input_ids.to(device)

    eos_token_id = None  # top_k_sample doesn't require tokenizer

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = _get_logits(model, current_ids)
            next_token_logits = outputs[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            scaled_logits = next_token_logits / max(temperature, 1e-8)

            # Top-k filtering: zero out all but top-k
            top_k_logits, top_k_indices = torch.topk(scaled_logits, min(k, scaled_logits.shape[-1]))
            # Build filtered logits
            filtered_logits = torch.full_like(scaled_logits, float("-inf"))
            filtered_logits.scatter_(1, top_k_indices, top_k_logits)

            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)

    return current_ids


def nucleus_sample(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    seed: int | None = None,
) -> Tensor:
    """Nucleus (top-p) sampling: sample from the smallest set of tokens whose
    cumulative probability exceeds top_p.

    # Ref: Holtzman et al., 2020 — "The Curious Case of Neural Text Degeneration"

    Args:
        model: A causal language model.
        input_ids: (1, T) LongTensor of prompt token IDs.
        max_new_tokens: Number of new tokens to generate.
        top_p: Cumulative probability threshold (0 < top_p <= 1).
        temperature: Softmax temperature.
        seed: Optional random seed for reproducibility.

    Returns:
        (1, T + max_new_tokens) generated token IDs.
    """
    if seed is not None:
        fix_all_seeds(seed)

    model.eval()
    device = torch.device("cpu")
    current_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = _get_logits(model, current_ids)
            next_token_logits = outputs[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            scaled_logits = next_token_logits / max(temperature, 1e-8)
            probs = F.softmax(scaled_logits, dim=-1)  # (1, vocab_size)

            # Sort probabilities descending
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative prob above top_p
            # Shift right so the first token above threshold is kept
            sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0.0

            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Sample from filtered distribution
            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)  # (1, 1)
            next_token = sorted_indices.gather(1, sampled_idx)  # (1, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)

    return current_ids


def _get_logits(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Helper: run model forward pass and return logits tensor (B, T, vocab_size).

    Handles both HuggingFace CausalLM models (which return an object with .logits)
    and custom models that return (logits, loss) tuples or plain tensors.
    """
    output = model(input_ids)

    # HuggingFace CausalLM output object
    if hasattr(output, "logits"):
        return output.logits

    # Tuple output (logits, loss) — as in project2 GPTModel
    if isinstance(output, tuple):
        return output[0]

    # Plain tensor
    return output
