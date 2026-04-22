"""
Training loop for Project 4: Vision Transformer on CIFAR-10 / ImageNette.

References:
    # Ref: Dosovitskiy et al., 2020 — "An Image is Worth 16x16 Words"
    # Ref: Loshchilov & Hutter, 2019 — "Decoupled Weight Decay Regularization" — AdamW
    # Ref: Loshchilov & Hutter, 2017 — "SGDR: Stochastic Gradient Descent with Warm Restarts"
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from visiontx.config import ViTConfig
from visiontx.data import get_data_loaders
from visiontx.evaluate import evaluate_top1
from visiontx.model import ViT
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


def train(
    model: nn.Module,
    config: ViTConfig,
    model_name: str = "vit",
    resume: bool = False,
) -> None:
    """
    Train a classification model (ViT or SmallResNet) on the configured dataset.

    Args:
        model:      The model to train (ViT or SmallResNet).
        config:     ViTConfig with all hyperparameters.
        model_name: Name tag used for checkpoint filenames ("vit" or "resnet").
        resume:     If True, resume from the best checkpoint.

    # CPU-only: on a CUDA-enabled machine with BF16 you would use:
    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    """
    fix_all_seeds(config.seed)

    json_logger = JSONLogger(config.log_path)
    json_logger.log_config(config)

    device = torch.device("cpu")
    model = model.to(device)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    _logger.info("Loading %s data...", config.dataset)
    train_loader, val_loader, _ = get_data_loaders(config)
    steps_per_epoch = len(train_loader)
    total_steps = config.max_epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    _logger.info(
        "Data loaded: %d train batches, %d val batches, %d total steps",
        steps_per_epoch, len(val_loader), total_steps,
    )

    # ------------------------------------------------------------------ #
    # Optimizer and scheduler
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = cosine_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # ------------------------------------------------------------------ #
    # Resume from checkpoint
    # ------------------------------------------------------------------ #
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = str(Path(config.checkpoint_dir) / f"best_{model_name}.pt")

    if resume:
        state = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch = state["epoch"]
        best_val_acc = state.get("best_metric", 0.0)
        # best_metric was stored as val_acc (higher is better)
        # load_checkpoint stores inf for fresh start; treat inf as 0
        if best_val_acc == float("inf"):
            best_val_acc = 0.0
        _logger.info(
            "Resumed from checkpoint: epoch=%d, best_val_acc=%.4f",
            start_epoch, best_val_acc,
        )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if HAS_TQDM:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}", leave=False)
        else:
            pbar = train_loader

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # CPU-only: on a CUDA-enabled machine with BF16 you would use:
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            # NaN guard
            if torch.isnan(loss):
                warnings.warn(
                    f"NaN loss at epoch={epoch+1}, step={global_step}. Skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                json_logger.log({
                    "type": "warning",
                    "message": "NaN loss",
                    "epoch": epoch + 1,
                    "step": global_step,
                })
                optimizer.zero_grad()
                global_step += 1
                continue

            optimizer.zero_grad()
            loss.backward()

            grad_norm = float(
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm).item()
            )
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1
            current_lr = float(scheduler.get_last_lr()[0])

            if global_step % config.log_every_n_steps == 0:
                json_logger.log({
                    "type": "train_step",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": loss.item(),
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                })

            if HAS_TQDM:
                pbar.set_postfix(  # type: ignore[union-attr]
                    loss=f"{loss.item():.4f}",
                    lr=f"{current_lr:.2e}",
                )

        avg_epoch_loss = epoch_loss / max(n_batches, 1)

        # ---------------------------------------------------------------- #
        # Validation
        # ---------------------------------------------------------------- #
        val_acc = evaluate_top1(model, val_loader)
        model.train()

        json_logger.log({
            "type": "val_epoch",
            "epoch": epoch + 1,
            "avg_train_loss": avg_epoch_loss,
            "val_accuracy": val_acc,
        })
        _logger.info(
            "Epoch %d/%d — avg_loss=%.4f, val_acc=%.4f",
            epoch + 1, config.max_epochs, avg_epoch_loss, val_acc,
        )

        # ---------------------------------------------------------------- #
        # Checkpoint on improvement
        # ---------------------------------------------------------------- #
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
                best_metric=best_val_acc,
            )
            json_logger.log({
                "type": "checkpoint",
                "epoch": epoch + 1,
                "path": checkpoint_path,
                "val_accuracy": val_acc,
            })
            _logger.info("  ✓ New best val_acc=%.4f — checkpoint saved.", best_val_acc)

    _logger.info("Training complete. Best val_accuracy=%.4f", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT or ResNet on CIFAR-10")
    parser.add_argument(
        "--config",
        type=str,
        default="visiontx/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "resnet"],
        help="Model to train: 'vit' or 'resnet'",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, ViTConfig)

    if args.model == "vit":
        from visiontx.model import ViT
        net = ViT(cfg)
    else:
        from visiontx.baseline import SmallResNet
        net = SmallResNet(n_classes=cfg.n_classes)

    train(net, cfg, model_name=args.model, resume=args.resume)
