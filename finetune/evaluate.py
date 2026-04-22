"""
Evaluation utilities for Project 3: gradient magnitude logging and reward accuracy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.logging_utils import JSONLogger


def log_gradient_magnitudes(
    model: nn.Module,
    logger: JSONLogger,
    step: int,
) -> None:
    """
    Compute per-layer gradient L2 norms and log them to the JSONLogger.

    Only layers with gradients are included. Layers with no gradient
    (e.g., frozen layers) are skipped.

    Args:
        model: The model whose gradients to inspect.
        logger: JSONLogger instance to write to.
        step: Current training step (for log entry identification).
    """
    grad_norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.detach().norm(2).item()

    logger.log({
        "type": "grad_magnitudes",
        "step": step,
        "grad_norms": grad_norms,
    })


def evaluate_reward_accuracy(
    reward_model: nn.Module,
    chosen_loader: DataLoader,
    rejected_loader: DataLoader,
) -> float:
    """
    Compute the fraction of pairs where the chosen reward > rejected reward.

    Args:
        reward_model: RewardModel that returns scalar rewards of shape (B,).
        chosen_loader: DataLoader yielding (input_ids, attention_mask) for chosen sequences.
        rejected_loader: DataLoader yielding (input_ids, attention_mask) for rejected sequences.

    Returns:
        Accuracy in [0.0, 1.0] — fraction of pairs correctly ranked.
    """
    reward_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (chosen_ids, chosen_mask), (rejected_ids, rejected_mask) in zip(
            chosen_loader, rejected_loader
        ):
            r_chosen = reward_model(chosen_ids, chosen_mask)      # (B,)
            r_rejected = reward_model(rejected_ids, rejected_mask)  # (B,)
            correct += (r_chosen > r_rejected).sum().item()
            total += chosen_ids.shape[0]

    return correct / total if total > 0 else 0.0
