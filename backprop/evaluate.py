"""
Evaluation module for Project 1: MLP on California Housing.

Computes RMSE and MAE on a test DataLoader and prints a formatted results table.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from backprop.model import MLP


def evaluate(model: MLP, test_loader: DataLoader) -> tuple[float, float]:
    """
    Evaluate the model on a DataLoader.

    Returns:
        (rmse, mae) — Root Mean Squared Error and Mean Absolute Error.
    """
    model.eval()
    total_se = 0.0
    total_ae = 0.0
    n = 0

    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            diff = preds - y
            total_se += float((diff ** 2).sum().item())
            total_ae += float(diff.abs().sum().item())
            n += y.numel()

    rmse = math.sqrt(total_se / n) if n > 0 else float("inf")
    mae = total_ae / n if n > 0 else float("inf")

    _print_results_table(rmse, mae)

    return rmse, mae


def _print_results_table(rmse: float, mae: float) -> None:
    """Print a formatted Unicode box table with RMSE and MAE."""
    rmse_str = f"{rmse:.4f}"
    mae_str = f"{mae:.4f}"

    print("┌─────────────────────────────────┐")
    print("│  Test Set Evaluation Results    │")
    print("├──────────────┬──────────────────┤")
    print(f"│  RMSE        │  {rmse_str:<16}│")
    print(f"│  MAE         │  {mae_str:<16}│")
    print("└──────────────┴──────────────────┘")
