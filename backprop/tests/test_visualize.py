"""
Tests for backprop/visualize.py

Covers:
  - plot_loss_curves writes a PNG file given synthetic JSONL data (Req 9.4)
  - plot_init_comparison writes a PNG file given synthetic JSONL data (Req 9.4)
"""

import json
import pytest

from backprop.visualize import plot_loss_curves, plot_init_comparison


def _write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


TRAIN_ENTRIES = [
    {"type": "train_step", "epoch": 0, "step": 1, "loss": 0.5, "lr": 0.001, "grad_norm": 1.0},
    {"type": "train_step", "epoch": 0, "step": 2, "loss": 0.4, "lr": 0.001, "grad_norm": 0.9},
    {"type": "val_epoch", "epoch": 0, "step": 10, "val_rmse": 0.8, "val_mae": 0.6, "val_r2": 0.5, "train_loss": 0.5},
    {"type": "train_step", "epoch": 1, "step": 3, "loss": 0.3, "lr": 0.001, "grad_norm": 0.8},
    {"type": "val_epoch", "epoch": 1, "step": 20, "val_rmse": 0.7, "val_mae": 0.5, "val_r2": 0.6, "train_loss": 0.4},
]

VAL_ENTRIES_A = [
    {"type": "val_epoch", "epoch": 0, "step": 10, "val_rmse": 0.9, "val_mae": 0.7, "val_r2": 0.4, "train_loss": 0.6},
    {"type": "val_epoch", "epoch": 1, "step": 20, "val_rmse": 0.8, "val_mae": 0.6, "val_r2": 0.5, "train_loss": 0.5},
]

VAL_ENTRIES_B = [
    {"type": "val_epoch", "epoch": 0, "step": 10, "val_rmse": 0.7, "val_mae": 0.5, "val_r2": 0.6, "train_loss": 0.5},
    {"type": "val_epoch", "epoch": 1, "step": 20, "val_rmse": 0.6, "val_mae": 0.4, "val_r2": 0.7, "train_loss": 0.4},
]


def test_plot_loss_curves_writes_png(tmp_path):
    """plot_loss_curves with synthetic JSONL data writes a PNG file."""
    log_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "loss_curves.png"

    _write_jsonl(log_path, TRAIN_ENTRIES)
    plot_loss_curves(log_path=str(log_path), output_path=str(output_path))

    assert output_path.exists(), "Expected PNG file to be created by plot_loss_curves"
    assert output_path.stat().st_size > 0, "Expected non-empty PNG file"


def test_plot_init_comparison_writes_png(tmp_path):
    """plot_init_comparison with two synthetic JSONL files writes a PNG file."""
    log_a = tmp_path / "run_a.jsonl"
    log_b = tmp_path / "run_b.jsonl"
    output_path = tmp_path / "init_comparison.png"

    _write_jsonl(log_a, VAL_ENTRIES_A)
    _write_jsonl(log_b, VAL_ENTRIES_B)

    plot_init_comparison(
        logs=[str(log_a), str(log_b)],
        labels=["xavier", "kaiming"],
        output_path=str(output_path),
    )

    assert output_path.exists(), "Expected PNG file to be created by plot_init_comparison"
    assert output_path.stat().st_size > 0, "Expected non-empty PNG file"
