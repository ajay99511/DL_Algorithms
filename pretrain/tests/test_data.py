"""
Tests for pretrain/data.py

Covers:
  - Example test: DataLoader output shapes (Req 2.12)
"""

from __future__ import annotations

import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from pretrain.data import _ChunkedTokenDataset


# ---------------------------------------------------------------------------
# Example test: DataLoader output shapes
# Validates: Requirements 2.12
# ---------------------------------------------------------------------------

def test_chunked_token_dataset_shape() -> None:
    """
    _ChunkedTokenDataset should return LongTensors of the correct context_length.
    """
    context_length = 32
    n_chunks = 20
    chunks = [list(range(context_length)) for _ in range(n_chunks)]

    dataset = _ChunkedTokenDataset(chunks)
    assert len(dataset) == n_chunks

    item = dataset[0]
    assert item.shape == (context_length,), f"Expected ({context_length},), got {item.shape}"
    assert item.dtype == torch.long


def test_dataloader_batch_shape() -> None:
    """
    A DataLoader wrapping _ChunkedTokenDataset should yield batches of shape
    (batch_size, context_length).
    """
    context_length = 16
    batch_size = 4
    n_chunks = 20

    # Create synthetic chunks
    chunks = [list(range(i, i + context_length)) for i in range(n_chunks)]
    dataset = _ChunkedTokenDataset(chunks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    batch = next(iter(loader))
    assert batch.shape == (batch_size, context_length), (
        f"Expected ({batch_size}, {context_length}), got {batch.shape}"
    )
    assert batch.dtype == torch.long


def test_dataloader_all_batches_correct_shape() -> None:
    """All batches from the DataLoader should have the correct shape."""
    context_length = 8
    batch_size = 3
    n_chunks = 15

    chunks = [list(range(context_length)) for _ in range(n_chunks)]
    dataset = _ChunkedTokenDataset(chunks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for batch in loader:
        assert batch.shape == (batch_size, context_length), (
            f"Expected ({batch_size}, {context_length}), got {batch.shape}"
        )
        assert batch.dtype == torch.long


def test_chunked_token_dataset_values() -> None:
    """Dataset items should contain the exact token IDs from the chunks."""
    chunks = [[1, 2, 3, 4], [5, 6, 7, 8]]
    dataset = _ChunkedTokenDataset(chunks)

    item0 = dataset[0]
    assert item0.tolist() == [1, 2, 3, 4]

    item1 = dataset[1]
    assert item1.tolist() == [5, 6, 7, 8]
