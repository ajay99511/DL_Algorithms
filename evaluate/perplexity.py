"""Perplexity computation for language models.

Perplexity = exp(mean cross-entropy loss) over non-overlapping context windows.
Always >= 1.0 for valid inputs.
"""

from __future__ import annotations

import math
import logging
from typing import Any

import torch
import torch.nn as nn

from shared.logging_utils import JSONLogger

logger = logging.getLogger(__name__)


class PerplexityCalculator:
    """Compute perplexity of a language model on a plain-text corpus.

    Args:
        model:          A PyTorch nn.Module that accepts (B, T) LongTensor input
                        and returns logits of shape (B, T, vocab_size).
        tokenizer:      Any tokenizer with an ``encode(text) -> list[int]`` method.
        context_length: Window size in tokens. Sequences are split into
                        non-overlapping windows of this length.
        json_logger:    JSONLogger for structured warning/error logging.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        context_length: int,
        json_logger: JSONLogger,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.json_logger = json_logger

    def compute(self, corpus: str) -> float | None:
        """Compute perplexity over *corpus*.

        Tokenizes the corpus, splits into non-overlapping windows of
        ``context_length`` tokens, runs a forward pass on each window,
        accumulates cross-entropy loss, and returns ``exp(mean_loss)``.

        Returns:
            Perplexity as a finite float >= 1.0, or None if the corpus is
            empty or tokenization fails.
        """
        if not corpus or not corpus.strip():
            self.json_logger.log({
                "type": "warning",
                "component": "PerplexityCalculator",
                "message": "Empty corpus provided; skipping perplexity computation.",
            })
            return None

        # Tokenize
        try:
            token_ids: list[int] = self.tokenizer.encode(corpus)
        except Exception as exc:
            self.json_logger.log({
                "type": "warning",
                "component": "PerplexityCalculator",
                "message": f"Tokenization failed: {exc}",
            })
            return None

        if len(token_ids) < 2:
            self.json_logger.log({
                "type": "warning",
                "component": "PerplexityCalculator",
                "message": "Corpus tokenizes to fewer than 2 tokens; cannot compute perplexity.",
            })
            return None

        # Split into non-overlapping windows of context_length
        # Each window needs at least 2 tokens (input + target)
        win = self.context_length
        # We need windows of size win+1 so that input=[:win] and target=[1:win+1]
        # but if context_length already accounts for that, use win tokens as input
        # and shift by 1 for targets.  Use windows of size win+1 when possible.
        window_size = win + 1
        windows: list[list[int]] = []
        for start in range(0, len(token_ids) - 1, win):
            chunk = token_ids[start: start + window_size]
            if len(chunk) < 2:
                break
            windows.append(chunk)

        if not windows:
            self.json_logger.log({
                "type": "warning",
                "component": "PerplexityCalculator",
                "message": "No valid windows after chunking; skipping perplexity.",
            })
            return None

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for window in windows:
                ids = torch.tensor(window, dtype=torch.long).unsqueeze(0)  # (1, T)
                input_ids = ids[:, :-1]   # (1, T-1)
                target_ids = ids[:, 1:]   # (1, T-1)

                try:
                    logits = self.model(input_ids)  # (1, T-1, vocab_size)
                except Exception as exc:
                    self.json_logger.log({
                        "type": "warning",
                        "component": "PerplexityCalculator",
                        "message": f"Forward pass failed on window: {exc}",
                    })
                    continue

                # logits may be a tuple (transformer returns (logits, cache))
                if isinstance(logits, tuple):
                    logits = logits[0]

                vocab_size = logits.size(-1)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    reduction="sum",
                )
                total_loss += float(loss.item())
                total_tokens += target_ids.numel()

        if total_tokens == 0:
            self.json_logger.log({
                "type": "warning",
                "component": "PerplexityCalculator",
                "message": "No tokens processed; cannot compute perplexity.",
            })
            return None

        mean_loss = total_loss / total_tokens
        perplexity = math.exp(mean_loss)
        return perplexity
