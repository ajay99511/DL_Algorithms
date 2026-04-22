"""
Data loading for Project 2: Transformer Pre-training on TinyStories.

Streams roneneldan/TinyStories, tokenizes with BPETokenizer, chunks into
context_length windows, and returns train/val DataLoaders.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from pretrain.config import TransformerConfig
from pretrain.tokenizer import BPETokenizer


class _ChunkedTokenDataset(Dataset):
    """
    A Dataset of fixed-length token ID chunks.

    Each item is a 1-D LongTensor of length context_length.
    """

    def __init__(self, chunks: list[list[int]]) -> None:
        self._chunks = chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor(self._chunks[idx], dtype=torch.long)


def load_tinystories(config: TransformerConfig) -> tuple[DataLoader, DataLoader]:
    """
    Stream TinyStories, tokenize, chunk, split, and return DataLoaders.

    Steps:
        1. Stream roneneldan/TinyStories via datasets.load_dataset(..., streaming=True).
        2. Take the first config.max_stories stories.
        3. Train BPETokenizer on those stories (or load from tokenizer_dir if it exists).
        4. Concatenate all token IDs into one long sequence.
        5. Chunk into non-overlapping windows of config.context_length tokens.
        6. 95/5 train/val split with fixed seed.
        7. Return (train_loader, val_loader).

    Args:
        config: TransformerConfig with all hyperparameters.

    Returns:
        (train_loader, val_loader) — DataLoaders of shape (batch_size, context_length).
    """
    import datasets as hf_datasets  # lazy import to avoid hard dep at module level

    # ------------------------------------------------------------------ #
    # 1. Stream and collect stories
    # ------------------------------------------------------------------ #
    dataset = hf_datasets.load_dataset(
        config.dataset_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    texts: list[str] = []
    for example in dataset:
        texts.append(example["text"])
        if len(texts) >= config.max_stories:
            break

    # ------------------------------------------------------------------ #
    # 2. Tokenizer: load if exists, else train and save
    # ------------------------------------------------------------------ #
    tokenizer_dir = Path(config.tokenizer_dir)
    vocab_file = tokenizer_dir / "vocab.json"
    merges_file = tokenizer_dir / "merges.txt"

    if vocab_file.exists() and merges_file.exists():
        tokenizer = BPETokenizer.load(str(tokenizer_dir))
    else:
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=config.vocab_size, save_dir=str(tokenizer_dir))

    # ------------------------------------------------------------------ #
    # 3. Tokenize all stories and concatenate
    # ------------------------------------------------------------------ #
    all_ids: list[int] = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text))

    # ------------------------------------------------------------------ #
    # 4. Chunk into context_length windows
    # ------------------------------------------------------------------ #
    context_length = config.context_length
    n_chunks = len(all_ids) // context_length
    chunks: list[list[int]] = [
        all_ids[i * context_length : (i + 1) * context_length]
        for i in range(n_chunks)
    ]

    # ------------------------------------------------------------------ #
    # 5. Train/val split with fixed seed
    # ------------------------------------------------------------------ #
    rng = random.Random(config.seed)
    indices = list(range(len(chunks)))
    rng.shuffle(indices)

    n_val = max(1, int(len(chunks) * config.val_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_chunks = [chunks[i] for i in train_indices]
    val_chunks = [chunks[i] for i in val_indices]

    train_dataset = _ChunkedTokenDataset(train_chunks)
    val_dataset = _ChunkedTokenDataset(val_chunks)

    # ------------------------------------------------------------------ #
    # 6. DataLoaders
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader
