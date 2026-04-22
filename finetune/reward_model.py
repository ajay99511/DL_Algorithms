"""
Reward model for Project 3: transformer backbone + scalar reward head.

References:
    # Ref: Ouyang et al., 2022 — "Training language models to follow instructions with human feedback"
    # Ref: Bai et al., 2022 — "Training a Helpful and Harmless Assistant with RLHF"
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from shared.lr_schedule import cosine_with_warmup
from shared.checkpointing import save_checkpoint
from finetune.evaluate import evaluate_reward_accuracy, log_gradient_magnitudes

if TYPE_CHECKING:
    from finetune.config import AlignmentConfig

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Transformer backbone + scalar reward head.

    The backbone produces hidden states; we take the last token's hidden state
    and project it to a scalar reward via a linear head.

    # Ref: Ouyang et al., 2022 — "Training language models to follow instructions with human feedback"
    # Ref: Bai et al., 2022 — "Training a Helpful and Harmless Assistant with RLHF"
    """

    def __init__(self, backbone: nn.Module, d_model: int) -> None:
        """
        Args:
            backbone: A transformer model (e.g., GPT-2) that returns hidden states.
            d_model: Hidden dimension of the backbone's output.
        """
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(d_model, 1, bias=True)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Returns scalar reward per sequence. Shape: (B,)

        # CPU-only: on a CUDA-enabled machine with BF16 you would wrap this in:
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

        Args:
            input_ids: (B, T) token IDs.
            attention_mask: (B, T) binary mask (1 = real token, 0 = padding).

        Returns:
            rewards: (B,) scalar reward per sequence.
        """
        # Get backbone hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use the last hidden state of the last real token
        hidden_states = outputs.hidden_states[-1]  # (B, T, d_model)

        # Find the last non-padding token for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,) — index of last real token
        seq_lengths = seq_lengths.clamp(min=0)

        batch_size = input_ids.shape[0]
        last_hidden = hidden_states[
            torch.arange(batch_size, device=input_ids.device),
            seq_lengths,
        ]  # (B, d_model)

        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return rewards


def train_reward_model(
    model: RewardModel,
    chosen_loader: DataLoader,
    rejected_loader: DataLoader,
    config: "AlignmentConfig",
) -> None:
    """
    Train the reward model using Bradley-Terry pairwise ranking loss.

    Loss: -log_sigmoid(r_chosen - r_rejected)

    # Ref: Ouyang et al., 2022 — "Training language models to follow instructions with human feedback"

    Args:
        model: RewardModel to train.
        chosen_loader: DataLoader for chosen (preferred) sequences.
        rejected_loader: DataLoader for rejected sequences.
        config: AlignmentConfig with rm_* fields.
    """
    try:
        from shared.logging_utils import JSONLogger
    except ImportError:
        JSONLogger = None  # type: ignore

    device = torch.device("cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.rm_lr)
    total_steps = config.rm_max_epochs * len(chosen_loader)
    warmup_steps = max(1, total_steps // 10)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    log_path = config.log_path
    rm_logger = None
    if JSONLogger is not None:
        rm_logger = JSONLogger(log_path)

    global_step = 0

    for epoch in range(config.rm_max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (chosen_ids, chosen_mask), (rejected_ids, rejected_mask) in zip(
            chosen_loader, rejected_loader
        ):
            chosen_ids = chosen_ids.to(device)
            chosen_mask = chosen_mask.to(device)
            rejected_ids = rejected_ids.to(device)
            rejected_mask = rejected_mask.to(device)

            r_chosen = model(chosen_ids, chosen_mask)      # (B,)
            r_rejected = model(rejected_ids, rejected_mask)  # (B,)

            # Bradley-Terry loss
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if rm_logger and global_step % config.log_every_n_steps == 0:
                log_gradient_magnitudes(model, rm_logger, global_step)
                rm_logger.log({
                    "type": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "rm_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                })

        # Evaluate reward accuracy on val set (reuse chosen_loader as proxy)
        val_acc = evaluate_reward_accuracy(model, chosen_loader, rejected_loader)
        avg_loss = epoch_loss / max(1, n_batches)

        logger.info(
            "Reward model epoch %d/%d — loss: %.4f, val_acc: %.4f",
            epoch + 1, config.rm_max_epochs, avg_loss, val_acc,
        )

        if rm_logger:
            rm_logger.log({
                "type": "val_epoch",
                "epoch": epoch,
                "rm_loss": avg_loss,
                "reward_accuracy": val_acc,
            })

        # Save checkpoint
        ckpt_path = f"{config.checkpoint_dir}/reward_model_epoch_{epoch + 1}.pt"
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch=epoch + 1,
            step=global_step,
            best_metric=val_acc,
        )
