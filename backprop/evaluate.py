"""
Evaluation module for Project 1: MLP on California Housing.

Computes RMSE, MAE, and R² on a DataLoader and prints a formatted results table.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backprop.config import MLPConfig
from backprop.model import MLP


def load_model_from_checkpoint(checkpoint_path: str, config: MLPConfig) -> MLP:
    """
    Reconstruct an MLP from a saved checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        config: MLPConfig defining the model architecture.

    Returns:
        MLP in eval mode with weights loaded from the checkpoint.

    Raises:
        RuntimeError: If the file is not found or cannot be loaded (corrupt file).
    """
    model = MLP(input_dim=8, hidden_dims=config.hidden_dims, dropout=config.dropout)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except FileNotFoundError:
        raise RuntimeError(f"Checkpoint file not found: '{checkpoint_path}'")
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {exc}") from exc
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@dataclass
class EvalResult:
    """Evaluation metrics returned by evaluate()."""
    rmse: float
    mae: float
    r2: float


def evaluate(model: MLP, loader: DataLoader) -> EvalResult:
    """
    Evaluate the model on a DataLoader.

    Returns:
        EvalResult with rmse, mae, and r2 fields.
    """
    model.eval()
    ss_res = 0.0
    total_ae = 0.0
    all_targets: list[float] = []
    n = 0

    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            diff = preds - y
            ss_res += float((diff ** 2).sum().item())
            total_ae += float(diff.abs().sum().item())
            all_targets.extend(y.view(-1).tolist())
            n += y.numel()

    rmse = math.sqrt(ss_res / n) if n > 0 else float("inf")
    mae = total_ae / n if n > 0 else float("inf")

    # Compute R²
    if n > 0:
        y_mean = sum(all_targets) / n
        ss_tot = sum((t - y_mean) ** 2 for t in all_targets)
        r2 = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    else:
        r2 = 0.0

    result = EvalResult(rmse=rmse, mae=mae, r2=r2)
    _print_results_table(result)
    return result


def _print_results_table(result: EvalResult) -> None:
    """Print a formatted Unicode box table with RMSE, MAE, and R²."""
    rmse_str = f"{result.rmse:.4f}"
    mae_str = f"{result.mae:.4f}"
    r2_str = f"{result.r2:.4f}"

    print("┌─────────────────────────────────┐")
    print("│  Test Set Evaluation Results    │")
    print("├──────────────┬──────────────────┤")
    print(f"│  RMSE        │  {rmse_str:<16}│")
    print(f"│  MAE         │  {mae_str:<16}│")
    print(f"│  R²          │  {r2_str:<16}│")
    print("└──────────────┴──────────────────┘")


if __name__ == "__main__":
    from backprop.data import load_california_housing
    from shared.config import load_config

    parser = argparse.ArgumentParser(description="Evaluate a saved MLP checkpoint on California Housing.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint file.")
    parser.add_argument("--config", default="backprop/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Which split to evaluate on.")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: checkpoint file not found: '{args.checkpoint}'", file=sys.stderr)
        sys.exit(1)

    config: MLPConfig = load_config(args.config, MLPConfig)
    model = load_model_from_checkpoint(args.checkpoint, config)
    train_loader, val_loader, test_loader = load_california_housing(
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
        batch_size=config.batch_size,
    )
    loader = val_loader if args.split == "val" else test_loader
    evaluate(model, loader)
