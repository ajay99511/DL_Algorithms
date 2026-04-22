"""
Visualization module for Project 1: MLP on California Housing.

Provides:
    plot_loss_curves   — train loss and val RMSE vs epoch from a JSONL log
    plot_init_comparison — overlaid val RMSE curves for multiple init strategies
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_jsonl(log_path: str) -> list[dict]:
    """Read all JSON lines from a JSONL file."""
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def plot_loss_curves(log_path: str, output_path: str) -> None:
    """
    Read a JSONL experiment log and plot train loss and val RMSE vs epoch.

    - Train loss is read from entries with type=="train_step" (averaged per epoch).
    - Val RMSE is read from entries with type=="val_epoch".

    Saves the figure as a PNG to output_path.
    """
    entries = _read_jsonl(log_path)

    # Collect val RMSE per epoch
    val_epochs: list[int] = []
    val_rmse: list[float] = []
    for e in entries:
        if e.get("type") == "val_epoch":
            val_epochs.append(int(e["epoch"]))
            val_rmse.append(float(e["val_rmse"]))

    # Collect train loss per epoch (average over steps within each epoch)
    train_loss_by_epoch: dict[int, list[float]] = {}
    for e in entries:
        if e.get("type") == "train_step" and "loss" in e and "epoch" in e:
            ep = int(e["epoch"])
            train_loss_by_epoch.setdefault(ep, []).append(float(e["loss"]))

    train_epochs = sorted(train_loss_by_epoch.keys())
    train_loss = [
        sum(train_loss_by_epoch[ep]) / len(train_loss_by_epoch[ep])
        for ep in train_epochs
    ]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_train = "#1f77b4"
    color_val = "#d62728"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss (MSE)", color=color_train)
    if train_epochs:
        ax1.plot(train_epochs, train_loss, color=color_train, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color_train)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val RMSE", color=color_val)
    if val_epochs:
        ax2.plot(val_epochs, val_rmse, color=color_val, linestyle="--", label="Val RMSE")
    ax2.tick_params(axis="y", labelcolor=color_val)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Training Loss and Validation RMSE vs Epoch")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_init_comparison(logs: list[str], labels: list[str], output_path: str) -> None:
    """
    Overlay val RMSE curves from multiple experiment logs on a single figure.

    Each log corresponds to a different weight initialization strategy.
    Reads entries with type=="val_epoch" for val_rmse.

    Args:
        logs:        List of paths to JSONL log files.
        labels:      Display labels for each log (e.g. ["normal", "xavier", "kaiming"]).
        output_path: Path to save the output PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for log_path, label in zip(logs, labels):
        entries = _read_jsonl(log_path)
        epochs = []
        rmse_vals = []
        for e in entries:
            if e.get("type") == "val_epoch":
                epochs.append(int(e["epoch"]))
                rmse_vals.append(float(e["val_rmse"]))
        ax.plot(epochs, rmse_vals, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val RMSE")
    ax.set_title("Validation RMSE by Weight Initialization Strategy")
    ax.legend()
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
