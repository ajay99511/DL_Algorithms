"""
Evaluation module for Project 2: Transformer Pre-training.

Computes perplexity on a validation DataLoader.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from pretrain.model import GPTModel


def compute_perplexity(model: GPTModel, val_loader: DataLoader) -> float:
    """
    Compute perplexity on a validation DataLoader.

    Perplexity = exp(mean cross-entropy loss over all tokens).

    Args:
        model:      A GPTModel in eval mode (or will be set to eval internally).
        val_loader: DataLoader yielding (B, T) LongTensor batches.

    Returns:
        Perplexity as a float. Returns inf if the loader is empty.
    """
    device = torch.device("cpu")
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)  # (B, T)
            if batch.shape[1] < 2:
                # Need at least 2 tokens to form input/target pair
                continue

            input_ids = batch[:, :-1]   # (B, T-1)
            targets = batch[:, 1:]      # (B, T-1)

            # CPU-only: on a CUDA-enabled machine with BF16 you would use:
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, targets)

            if loss is not None and not torch.isnan(loss):
                total_loss += float(loss.item())
                total_batches += 1

    if total_batches == 0:
        return float("inf")

    mean_loss = total_loss / total_batches
    return math.exp(min(mean_loss, 20.0))  # cap to avoid overflow
