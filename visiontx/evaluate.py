"""
Evaluation utilities for Project 4: Vision Transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_top1(model: nn.Module, loader: DataLoader) -> float:
    """
    Compute top-1 accuracy on a DataLoader.

    Args:
        model: A classification model that returns (B, n_classes) logits.
        loader: DataLoader yielding (images, labels) batches.

    Returns:
        Top-1 accuracy as a float in [0.0, 1.0].
    """
    device = torch.device("cpu")
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)                          # (B, n_classes)
            preds = logits.argmax(dim=-1)                   # (B,)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0
    return correct / total
