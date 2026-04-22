"""
Data loaders for Project 3: Alpaca (SFT) and Anthropic HH-RLHF (reward model).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, TensorDataset

from shared.seed import fix_all_seeds

if TYPE_CHECKING:
    from finetune.config import AlignmentConfig

try:
    from datasets import load_dataset  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


def _get_tokenizer(model_name: str = "gpt2"):
    """Load GPT-2 tokenizer with padding token set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _format_alpaca_prompt(instruction: str, inp: str, output: str) -> str:
    """Format an Alpaca example using the standard prompt template."""
    if inp and inp.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def _tokenize_texts(
    texts: list[str],
    tokenizer,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a list of texts, returning (input_ids, attention_mask) tensors."""
    encoding = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


def _train_val_split(
    *tensors: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split tensors into train/val sets with a fixed seed."""
    fix_all_seeds(seed)
    n = tensors[0].shape[0]
    n_val = max(1, math.floor(n * val_fraction))
    indices = torch.randperm(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_tensors = [t[train_idx] for t in tensors]
    val_tensors = [t[val_idx] for t in tensors]
    return train_tensors, val_tensors


def load_alpaca(
    config: "AlignmentConfig",
) -> tuple[DataLoader, DataLoader]:
    """
    Load tatsu-lab/alpaca, format with Alpaca prompt template, tokenize,
    and return (train_loader, val_loader).

    Args:
        config: AlignmentConfig with sft_* fields.

    Returns:
        (train_loader, val_loader) — each yields (input_ids, attention_mask) batches.
    """
    if not _HAS_HF:
        raise ImportError("datasets and transformers are required for load_alpaca.")

    dataset = load_dataset(config.sft_dataset, split="train")
    tokenizer = _get_tokenizer(config.base_model_name)

    texts = [
        _format_alpaca_prompt(
            ex["instruction"],
            ex.get("input", ""),
            ex["output"],
        )
        for ex in dataset
    ]

    input_ids, attention_mask = _tokenize_texts(texts, tokenizer, max_length=512)

    train_tensors, val_tensors = _train_val_split(
        input_ids, attention_mask,
        val_fraction=config.sft_val_fraction,
        seed=config.seed,
    )

    train_ds = TensorDataset(*train_tensors)
    val_ds = TensorDataset(*val_tensors)

    train_loader = DataLoader(train_ds, batch_size=config.sft_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.sft_batch_size, shuffle=False)

    return train_loader, val_loader


def load_hh_rlhf(
    config: "AlignmentConfig",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Anthropic/hh-rlhf, tokenize chosen and rejected sequences,
    and return (chosen_loader, rejected_loader, val_chosen_loader).

    The chosen and rejected loaders are paired by index (same ordering).

    Args:
        config: AlignmentConfig with rm_* fields.

    Returns:
        (chosen_loader, rejected_loader, val_chosen_loader)
    """
    if not _HAS_HF:
        raise ImportError("datasets and transformers are required for load_hh_rlhf.")

    dataset = load_dataset(config.rm_dataset, split="train")
    tokenizer = _get_tokenizer(config.base_model_name)

    chosen_texts = [ex["chosen"] for ex in dataset]
    rejected_texts = [ex["rejected"] for ex in dataset]

    chosen_ids, chosen_mask = _tokenize_texts(chosen_texts, tokenizer, max_length=512)
    rejected_ids, rejected_mask = _tokenize_texts(rejected_texts, tokenizer, max_length=512)

    # Split both chosen and rejected with the same indices
    fix_all_seeds(config.seed)
    n = chosen_ids.shape[0]
    n_val = max(1, math.floor(n * config.sft_val_fraction))
    indices = torch.randperm(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_chosen_ids = chosen_ids[train_idx]
    train_chosen_mask = chosen_mask[train_idx]
    train_rejected_ids = rejected_ids[train_idx]
    train_rejected_mask = rejected_mask[train_idx]
    val_chosen_ids = chosen_ids[val_idx]
    val_chosen_mask = chosen_mask[val_idx]

    batch_size = config.sft_batch_size

    chosen_ds = TensorDataset(train_chosen_ids, train_chosen_mask)
    rejected_ds = TensorDataset(train_rejected_ids, train_rejected_mask)
    val_chosen_ds = TensorDataset(val_chosen_ids, val_chosen_mask)

    chosen_loader = DataLoader(chosen_ds, batch_size=batch_size, shuffle=False)
    rejected_loader = DataLoader(rejected_ds, batch_size=batch_size, shuffle=False)
    val_chosen_loader = DataLoader(val_chosen_ds, batch_size=batch_size, shuffle=False)

    return chosen_loader, rejected_loader, val_chosen_loader
