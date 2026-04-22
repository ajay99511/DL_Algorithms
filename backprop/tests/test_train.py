"""
Tests for backprop/train.py

Covers:
  - Smoke test: train() completes and writes best.pt (Req 9.1)
"""

import tempfile
from pathlib import Path

from backprop.train import train
from backprop.config import MLPConfig


def test_train_smoke(tmp_path):
    """
    Smoke test: train() with minimal config completes without exception
    and writes best.pt to checkpoint_dir.
    """
    config = MLPConfig(
        max_epochs=2,
        hidden_dims=[8],
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "train.jsonl"),
    )

    train(config)

    assert (Path(config.checkpoint_dir) / "best.pt").exists(), (
        "best.pt was not created after training"
    )
