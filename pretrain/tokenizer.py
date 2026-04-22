"""
BPE Tokenizer for Project 2: Transformer Pre-training.

Wraps tokenizers.ByteLevelBPETokenizer to provide train/encode/decode/save/load.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import ByteLevelBPETokenizer


class BPETokenizer:
    """
    Byte-level BPE tokenizer wrapping HuggingFace tokenizers.ByteLevelBPETokenizer.

    Usage:
        # Train from scratch
        tok = BPETokenizer()
        tok.train(texts, vocab_size=8000, save_dir="outputs/project2/tokenizer")

        # Load pre-trained
        tok = BPETokenizer.load("outputs/project2/tokenizer")

        ids = tok.encode("Hello world")
        text = tok.decode(ids)
    """

    _VOCAB_FILE = "vocab.json"
    _MERGES_FILE = "merges.txt"

    def __init__(self) -> None:
        self._tokenizer: ByteLevelBPETokenizer | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts: list[str], vocab_size: int, save_dir: str) -> None:
        """
        Train a Byte-Level BPE tokenizer on the provided texts.

        Args:
            texts:      List of raw text strings to train on.
            vocab_size: Target vocabulary size (e.g. 8000).
            save_dir:   Directory to save vocab.json and merges.txt.
        """
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            texts,
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        )
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(save_dir)
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Call train() or load() first.")
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Call train() or load() first.")
        return self._tokenizer.decode(ids)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, save_dir: str) -> None:
        """
        Save the tokenizer vocab and merges to save_dir.

        Args:
            save_dir: Directory to write vocab.json and merges.txt.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Nothing to save.")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._tokenizer.save_model(save_dir)

    @classmethod
    def load(cls, save_dir: str) -> "BPETokenizer":
        """
        Load a pre-trained tokenizer from save_dir.

        Args:
            save_dir: Directory containing vocab.json and merges.txt.

        Returns:
            A BPETokenizer instance ready for encode/decode.
        """
        vocab_file = os.path.join(save_dir, cls._VOCAB_FILE)
        merges_file = os.path.join(save_dir, cls._MERGES_FILE)
        tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        instance = cls()
        instance._tokenizer = tokenizer
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the trained tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded.")
        return self._tokenizer.get_vocab_size()
