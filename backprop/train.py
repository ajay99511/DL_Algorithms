"""
Training loop for Project 1: MLP on California Housing (regression).

References:
    # Ref: Loshchilov & Hutter, 2019 — "Decoupled Weight Decay Regularization" — AdamW
    # Ref: Loshchilov & Hutter, 2017 — "SGDR: Stochastic Gradient Descent with Warm Restarts" — cosine schedule
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from backprop.config import MLPConfig
from backprop.data import load_california_housing
from backprop.evaluate import evaluate as eval_fn
from backprop.model import MLP, activation_stats, initialize_weights
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
logger = logging.getLogger(__name__)


def train(config: MLPConfig, resume: bool = False) -> None:
    """Full training loop for the MLP on California Housing."""
    fix_all_seeds(config.seed)

    json_logger = JSONLogger(config.log_path)
    json_logger.log_config(config)

    # Data
    train_loader, val_loader, test_loader = load_california_housing(
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
        batch_size=config.batch_size,
    )

    # Model
    input_dim = 8  # California Housing has 8 features
    model = MLP(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    initialize_weights(model, config.init_strategy)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    # steps_per_epoch = number of optimizer steps per epoch
    # = ceil(len(train_dataset) / (batch_size * grad_accum_steps))
    train_dataset_size = len(train_loader.dataset)  # type: ignore[arg-type]
    steps_per_epoch = math.ceil(
        train_dataset_size / (config.batch_size * config.grad_accum_steps)
    )
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.max_epochs * steps_per_epoch

    scheduler = cosine_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    global_step = 0
    best_val_rmse = float("inf")
    patience_counter = 0

    checkpoint_path = str(Path(config.checkpoint_dir) / "best.pt")
    if resume:
        state = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch = state["epoch"] + 1
        global_step = state["step"]
        best_val_rmse = state["best_metric"]
        logger.info(
            "Resumed from checkpoint: epoch=%d, step=%d, best_val_rmse=%.4f",
            state["epoch"],
            global_step,
            best_val_rmse,
        )

    criterion = nn.MSELoss()
    activation_stats_buffer: collections.deque = collections.deque(maxlen=3)

    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        optimizer.zero_grad()

        epoch_loss_sum = 0.0
        epoch_batches = 0

        # tqdm progress bar over micro-batches
        if HAS_TQDM:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{config.max_epochs}",
                leave=False,
            )
        else:
            pbar = enumerate(train_loader)

        for micro_step, (x, y) in pbar:
            # CPU-only: on GPU with BF16 you would use torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            preds = model(x)
            loss = criterion(preds, y) / config.grad_accum_steps

            # NaN loss guard
            if torch.isnan(loss):
                warnings.warn(
                    f"NaN loss detected at epoch={epoch}, micro_step={micro_step}, "
                    f"global_step={global_step}. Skipping optimizer step.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                json_logger.log({
                    "type": "warning",
                    "message": "NaN loss",
                    "epoch": epoch,
                    "micro_step": micro_step,
                    "step": global_step,
                })
                optimizer.zero_grad()
                continue

            loss.backward()
            epoch_loss_sum += float(loss.item()) * config.grad_accum_steps
            epoch_batches += 1

            if (micro_step + 1) % config.grad_accum_steps == 0:
                grad_norm = float(
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm).item()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = float(scheduler.get_last_lr()[0])
                scaled_loss = float(loss.item()) * config.grad_accum_steps

                if global_step % config.log_every_n_steps == 0:
                    json_logger.log({
                        "type": "train_step",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": scaled_loss,
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                    })

                    with torch.no_grad():
                        stats = activation_stats(model, x)
                    json_logger.log({
                        "type": "activation_stats",
                        "epoch": epoch,
                        "step": global_step,
                        "layers": stats,
                    })
                    activation_stats_buffer.append(stats)
                    if (
                        len(activation_stats_buffer) == 3
                        and all(
                            all(layer["dead_fraction"] == 1.0 for layer in entry.values())
                            for entry in activation_stats_buffer
                        )
                    ):
                        logger.warning(
                            "All neurons appear to be dead for the last 3 activation-stats "
                            "log entries (epoch=%d, step=%d).",
                            epoch,
                            global_step,
                        )

                if HAS_TQDM:
                    pbar.set_postfix(  # type: ignore[union-attr]
                        loss=f"{scaled_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                    )

        # Handle any remaining micro-batches that didn't complete a full accumulation cycle
        remaining = len(train_loader) % config.grad_accum_steps
        if remaining != 0:
            grad_norm = float(
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm).item()
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            current_lr = float(scheduler.get_last_lr()[0])
            if global_step % config.log_every_n_steps == 0:
                json_logger.log({
                    "type": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": epoch_loss_sum / max(epoch_batches, 1),
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                })

        # Validation
        val_result = eval_fn(model, val_loader)
        val_rmse = val_result.rmse
        val_mae = val_result.mae
        avg_train_loss = epoch_loss_sum / max(epoch_batches, 1)

        json_logger.log({
            "type": "val_epoch",
            "epoch": epoch,
            "step": global_step,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_result.r2,
            "train_loss": avg_train_loss,
        })

        logger.info(
            "Epoch %d/%d — train_loss=%.4f  val_rmse=%.4f  val_mae=%.4f",
            epoch + 1,
            config.max_epochs,
            avg_train_loss,
            val_rmse,
            val_mae,
        )

        # Save best checkpoint
        if val_rmse < best_val_rmse - config.early_stopping_delta:
            best_val_rmse = val_rmse
            patience_counter = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                best_metric=best_val_rmse,
            )
            json_logger.log({
                "type": "checkpoint",
                "epoch": epoch,
                "step": global_step,
                "path": checkpoint_path,
                "val_rmse": val_rmse,
            })
            logger.info("  ✓ New best val_rmse=%.4f — checkpoint saved.", best_val_rmse)
        else:
            if config.early_stopping_patience > 0:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    json_logger.log({
                        "type": "early_stop",
                        "epoch": epoch,
                        "step": global_step,
                        "best_val_rmse": best_val_rmse,
                    })
                    logger.info(
                        "Early stopping at epoch %d. Best val_rmse=%.4f",
                        epoch + 1,
                        best_val_rmse,
                    )
                    break

        # Periodic checkpointing
        if config.checkpoint_every_n_epochs > 0 and (epoch + 1) % config.checkpoint_every_n_epochs == 0:
            periodic_path = str(Path(config.checkpoint_dir) / f"epoch_{epoch:04d}.pt")
            save_checkpoint(
                periodic_path,
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                step=global_step,
                best_metric=best_val_rmse,
            )
            json_logger.log({
                "type": "periodic_checkpoint",
                "epoch": epoch,
                "step": global_step,
                "path": periodic_path,
            })

    logger.info("Training complete. Best val_rmse=%.4f", best_val_rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on California Housing")
    parser.add_argument(
        "--config",
        type=str,
        default="backprop/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, MLPConfig)
    train(cfg, resume=args.resume)
