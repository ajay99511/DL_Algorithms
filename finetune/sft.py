"""
Supervised Fine-Tuning (SFT) for Project 3.

Fine-tunes GPT-2 on the Alpaca instruction-following dataset.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from shared.seed import fix_all_seeds
from shared.logging_utils import JSONLogger
from shared.lr_schedule import cosine_with_warmup
from shared.checkpointing import save_checkpoint
from finetune.config import AlignmentConfig
from finetune.data import load_alpaca
from finetune.evaluate import log_gradient_magnitudes

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM  # type: ignore
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


def run_sft(config: AlignmentConfig) -> None:
    """
    Fine-tune GPT-2 on Alpaca with AdamW + cosine_with_warmup.

    # CPU-only: on a CUDA-enabled machine with BF16 you would use:
    # model = model.to(torch.bfloat16)

    Args:
        config: AlignmentConfig with sft_* fields.
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers is required for run_sft.")

    fix_all_seeds(config.seed)
    device = torch.device("cpu")

    json_logger = JSONLogger(config.log_path)
    json_logger.log_config(config)

    # Load model and data
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    model = model.to(device)

    train_loader, val_loader = load_alpaca(config)

    total_steps = config.sft_max_epochs * (len(train_loader) // config.sft_grad_accum_steps + 1)
    warmup_steps = max(1, total_steps // 10)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft_lr)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0

    for epoch in range(config.sft_max_epochs):
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"SFT Epoch {epoch + 1}/{config.sft_max_epochs}",
        )

        for micro_step, batch in pbar:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Causal LM: targets are the same as inputs shifted by 1
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss / config.sft_grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (micro_step + 1) % config.sft_grad_accum_steps == 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every_n_steps == 0:
                    log_gradient_magnitudes(model, json_logger, global_step)
                    json_logger.log({
                        "type": "train_step",
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": accum_loss * config.sft_grad_accum_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                    })

                pbar.set_postfix(loss=f"{accum_loss * config.sft_grad_accum_steps:.4f}")
                accum_loss = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                val_loss += outputs.loss.item()
                n_val += 1

        avg_val_loss = val_loss / max(1, n_val)
        logger.info("SFT epoch %d/%d — val_loss: %.4f", epoch + 1, config.sft_max_epochs, avg_val_loss)
        json_logger.log({
            "type": "val_epoch",
            "epoch": epoch,
            "val_loss": avg_val_loss,
        })

        # Save checkpoint
        ckpt_path = f"{config.checkpoint_dir}/sft_epoch_{epoch + 1}.pt"
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch=epoch + 1,
            step=global_step,
            best_metric=avg_val_loss,
        )
        logger.info("Saved SFT checkpoint: %s", ckpt_path)
