"""Tests for inference strategies.

Property 8: Inference Reproducibility Under Fixed Seed
Property: Beam search width invariant
Example: Greedy decode output shapes

# Feature: deep-learning-llm-mastery, Property 8: Inference Reproducibility Under Fixed Seed
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from infer.inference import (
    beam_search,
    greedy_decode,
    nucleus_sample,
    top_k_sample,
)


# ---------------------------------------------------------------------------
# Minimal mock model for tests — avoids downloading GPT-2
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    """Tiny language model that returns deterministic random logits.

    Uses a fixed embedding + linear head so the output is deterministic
    given the same input_ids (no randomness in the model itself).
    """

    def __init__(self, vocab_size: int = 100, d_model: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, None]:
        """Returns (logits, None) — compatible with project2 GPTModel interface."""
        x = self.embed(input_ids)  # (B, T, d_model)
        logits = self.head(x)      # (B, T, vocab_size)
        return logits, None


class _MockTokenizer:
    """Minimal tokenizer stub for tests."""
    eos_token_id = 2
    pad_token_id = 0


VOCAB_SIZE = 100
_MODEL = _TinyLM(vocab_size=VOCAB_SIZE)
_TOKENIZER = _MockTokenizer()


def _make_input(seq_len: int = 4) -> Tensor:
    """Create a (1, seq_len) input_ids tensor."""
    return torch.randint(0, VOCAB_SIZE, (1, seq_len))


# ---------------------------------------------------------------------------
# Property 8: Inference Reproducibility Under Fixed Seed
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 8: Inference Reproducibility Under Fixed Seed
@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20)
def test_nucleus_sample_reproducibility(seed: int) -> None:
    """Same seed must produce identical outputs for nucleus sampling.

    **Validates: Requirements 2.8, 5.8**
    """
    model = _TinyLM(vocab_size=VOCAB_SIZE)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    out1 = nucleus_sample(model, input_ids.clone(), max_new_tokens=5, top_p=0.9, temperature=1.0, seed=seed)
    out2 = nucleus_sample(model, input_ids.clone(), max_new_tokens=5, top_p=0.9, temperature=1.0, seed=seed)

    assert torch.equal(out1, out2), (
        f"nucleus_sample with seed={seed} produced different outputs: {out1} vs {out2}"
    )


@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20)
def test_top_k_sample_reproducibility(seed: int) -> None:
    """Same seed must produce identical outputs for top-k sampling.

    **Validates: Requirements 2.8, 5.8**
    """
    model = _TinyLM(vocab_size=VOCAB_SIZE)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    out1 = top_k_sample(model, input_ids.clone(), max_new_tokens=5, k=10, temperature=1.0, seed=seed)
    out2 = top_k_sample(model, input_ids.clone(), max_new_tokens=5, k=10, temperature=1.0, seed=seed)

    assert torch.equal(out1, out2), (
        f"top_k_sample with seed={seed} produced different outputs: {out1} vs {out2}"
    )


# ---------------------------------------------------------------------------
# Property: Beam search returns tensor for any beam_width >= 1
# ---------------------------------------------------------------------------

@given(beam_width=st.integers(1, 4))
@settings(max_examples=20)
def test_beam_search_returns_tensor(beam_width: int) -> None:
    """beam_search must return a tensor without error for any beam_width >= 1.

    _Requirements: 5.11_
    """
    model = _TinyLM(vocab_size=VOCAB_SIZE)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids, beam_log = beam_search(
        model, input_ids, max_new_tokens=3, beam_width=beam_width, tokenizer=_TOKENIZER
    )

    assert isinstance(output_ids, Tensor), "beam_search must return a Tensor"
    assert output_ids.shape[0] == 1, "Output must have batch size 1"
    assert output_ids.shape[1] >= input_ids.shape[1], "Output must be at least as long as input"
    assert isinstance(beam_log, list), "beam_log must be a list"


# ---------------------------------------------------------------------------
# Example test: greedy decode output shape
# ---------------------------------------------------------------------------

class _MockTokenizerNoEOS:
    """Tokenizer stub with no EOS token — ensures greedy decode runs full max_new_tokens."""
    eos_token_id = None
    pad_token_id = 0


def test_greedy_decode_output_shape() -> None:
    """greedy_decode output shape must be (1, prompt_len + max_new_tokens).

    Uses a tokenizer with no EOS token to ensure all max_new_tokens are generated.

    _Requirements: 5.1_
    """
    model = _TinyLM(vocab_size=VOCAB_SIZE)
    model.eval()
    prompt_len = 4
    max_new_tokens = 6
    input_ids = torch.randint(0, VOCAB_SIZE, (1, prompt_len))
    tokenizer_no_eos = _MockTokenizerNoEOS()

    output_ids, per_step_log_probs = greedy_decode(
        model, input_ids, max_new_tokens=max_new_tokens, tokenizer=tokenizer_no_eos
    )

    assert output_ids.shape == (1, prompt_len + max_new_tokens), (
        f"Expected shape (1, {prompt_len + max_new_tokens}), got {output_ids.shape}"
    )
    assert len(per_step_log_probs) == max_new_tokens, (
        f"Expected {max_new_tokens} log prob tensors, got {len(per_step_log_probs)}"
    )
    assert per_step_log_probs[0].shape == (VOCAB_SIZE,), (
        f"Expected log prob shape ({VOCAB_SIZE},), got {per_step_log_probs[0].shape}"
    )


def test_greedy_decode_is_deterministic() -> None:
    """greedy_decode must produce identical outputs on repeated calls (no randomness)."""
    model = _TinyLM(vocab_size=VOCAB_SIZE)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    out1, _ = greedy_decode(model, input_ids.clone(), max_new_tokens=5, tokenizer=_TOKENIZER)
    out2, _ = greedy_decode(model, input_ids.clone(), max_new_tokens=5, tokenizer=_TOKENIZER)

    assert torch.equal(out1, out2), "greedy_decode must be deterministic"


def test_top_k_sample_output_shape() -> None:
    """top_k_sample must return tensor of shape (1, prompt_len + max_new_tokens)."""
    model = _TinyLM(vocab_size=VOCAB_SIZE)
    model.eval()
    prompt_len = 3
    max_new_tokens = 4
    input_ids = torch.randint(0, VOCAB_SIZE, (1, prompt_len))

    output = top_k_sample(model, input_ids, max_new_tokens=max_new_tokens, k=10, temperature=1.0, seed=42)
    assert output.shape == (1, prompt_len + max_new_tokens)


def test_nucleus_sample_output_shape() -> None:
    """nucleus_sample must return tensor of shape (1, prompt_len + max_new_tokens)."""
    model = _TinyLM(vocab_size=VOCAB_SIZE)
    model.eval()
    prompt_len = 3
    max_new_tokens = 4
    input_ids = torch.randint(0, VOCAB_SIZE, (1, prompt_len))

    output = nucleus_sample(model, input_ids, max_new_tokens=max_new_tokens, top_p=0.9, temperature=1.0, seed=42)
    assert output.shape == (1, prompt_len + max_new_tokens)
