"""
Pre-training loop for Project 2: GPT-style Transformer on TinyStories.

References:
    # Ref: Loshchilov & Hutter, 2019 — "Decoupled Weight Decay Regularization" — AdamW
    # Ref: Loshchilov & Hutter, 2017 — "SGDR: Stochastic Gradient Descent with Warm Restarts"
    # Ref: Radford et al., 2019 — "Language Models are Unsupervised Multitask Learners" (GPT-2)
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from pretrain.config import TransformerConfig
from pretrain.data import load_tinystories
from pretrain.evaluate import compute_perplexity
from pretrain.model import GPTModel
from pretrain.tokenizer import BPETokenizer
from shared.checkpointing import load_checkpoint, save_checkpoint
from shared.config import load_config
from shared.logging_utils import JSONLogger
from shared.lr_schedule import cosine_with_warmup
from shared.seed import fix_all_seeds

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_logger = logging.getLogger(__name__)


def train(config: TransformerConfig, resume: bool = False) -> None:
    """
    Full pre-training loop for the GPT model on TinyStories.

    Args:
        config: TransformerConfig with all hyperparameters.
        resume: If True, resume from the best checkpoint in config.checkpoint_dir.
    """
    fix_all_seeds(config.seed)

    json_logger = JSONLogger(config.log_path)
    json_logger.log_config(config)

    # ------------------------------------------------------------------ #
    # Tokenizer: train or load
    # ------------------------------------------------------------------ #
    tokenizer_dir = Path(config.tokenizer_dir)
    vocab_file = tokenizer_dir / "vocab.json"
    merges_file = tokenizer_dir / "merges.txt"

    if vocab_file.exists() and merges_file.exists():
        _logger.info("Loading tokenizer from %s", config.tokenizer_dir)
        tokenizer = BPETokenizer.load(str(tokenizer_dir))
    else:
        _logger.info("Tokenizer not found — will be trained during data loading.")
        tokenizer = None  # data loading will train it

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    _logger.info("Loading TinyStories data...")
    train_loader, val_loader = load_tinystories(config)
    _logger.info(
        "Data loaded: %d train batches, %d val batches",
        len(train_loader),
        len(val_loader),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    device = torch.device("cpu")
    model = GPTModel(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)

    _logger.info("Model parameters: %d", model.count_parameters())

    # ------------------------------------------------------------------ #
    # Optimizer and scheduler
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    scheduler = cosine_with_warmup(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
    )

    # ------------------------------------------------------------------ #
    # Resume from checkpoint
    # ------------------------------------------------------------------ #
    start_step = 0
    best_val_ppl = float("inf")
    checkpoint_path = str(Path(config.checkpoint_dir) / "best.pt")

    if resume:
        state = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_step = state["step"]
        best_val_ppl = state["best_metric"]
        _logger.info(
            "Resumed from checkpoint: step=%d, best_val_ppl=%.4f",
            start_step,
            best_val_ppl,
        )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    model.train()
    global_step = start_step
    optimizer.zero_grad()

    # Infinite iterator over train_loader
    def _cycle(loader):
        while True:
            yield from loader

    train_iter = _cycle(train_loader)

    if HAS_TQDM:
        pbar = tqdm(
            range(start_step, config.max_steps),
            desc="Pre-training",
            initial=start_step,
            total=config.max_steps,
        )
    else:
        pbar = range(start_step, config.max_steps)

    accum_loss = 0.0

    for step in pbar:
        # Gradient accumulation over grad_accum_steps micro-batches
        for micro_step in range(config.grad_accum_steps):
            batch = next(train_iter).to(device)  # (B, T)
            # Shift: input = batch[:, :-1], targets = batch[:, 1:]
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            # CPU-only: on a CUDA-enabled machine with BF16 you would use:
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, targets)
            loss = loss / config.grad_accum_steps

            # NaN guard
            if torch.isnan(loss):
                warnings.warn(
                    f"NaN loss at step={step}, micro_step={micro_step}. Skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                json_logger.log({
                    "type": "warning",
                    "message": "NaN loss",
                    "step": step,
                    "micro_step": micro_step,
                })
                optimizer.zero_grad()
                break

            loss.backward()
            accum_loss += float(loss.item()) * config.grad_accum_steps

        # Optimizer step
        grad_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm).item()
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        current_lr = float(scheduler.get_last_lr()[0])
        avg_loss = accum_loss / config.grad_accum_steps
        accum_loss = 0.0

        # Logging
        if global_step % config.log_every_n_steps == 0:
            json_logger.log({
                "type": "train_step",
                "step": global_step,
                "loss": avg_loss,
                "perplexity": math.exp(min(avg_loss, 20)),
                "lr": current_lr,
                "grad_norm": grad_norm,
            })

        if HAS_TQDM:
            pbar.set_postfix(  # type: ignore[union-attr]
                loss=f"{avg_loss:.4f}",
                ppl=f"{math.exp(min(avg_loss, 20)):.2f}",
                lr=f"{current_lr:.2e}",
            )

        # Validation every checkpoint_every_n_epochs steps (reused as checkpoint interval)
        val_interval = max(1, config.max_steps // 20)  # ~20 val checkpoints
        if global_step % val_interval == 0 or global_step == config.max_steps:
            model.eval()
            val_ppl = compute_perplexity(model, val_loader)
            model.train()

            json_logger.log({
                "type": "val_epoch",
                "step": global_step,
                "val_perplexity": val_ppl,
            })
            _logger.info("Step %d — val_perplexity=%.4f", global_step, val_ppl)

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=0,
                    step=global_step,
                    best_metric=best_val_ppl,
                )
                json_logger.log({
                    "type": "checkpoint",
                    "step": global_step,
                    "path": checkpoint_path,
                    "val_perplexity": val_ppl,
                })
                _logger.info("  ✓ New best val_ppl=%.4f — checkpoint saved.", best_val_ppl)

    _logger.info("Training complete. Best val_perplexity=%.4f", best_val_ppl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train GPT on TinyStories")
    parser.add_argument(
        "--config",
        type=str,
        default="pretrain/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, TransformerConfig)
    train(cfg, resume=args.resume)
