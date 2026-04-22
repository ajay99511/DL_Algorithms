"""
AlignmentConfig for Project 3: SFT, Reward Model, RLHF, and RLAIF.
"""
from __future__ import annotations

from dataclasses import dataclass

from shared.config import BaseConfig


@dataclass
class AlignmentConfig(BaseConfig):
    # Base model
    base_model_name: str = "gpt2"

    # SFT
    sft_dataset: str = "tatsu-lab/alpaca"
    sft_val_fraction: float = 0.1
    sft_max_epochs: int = 3
    sft_lr: float = 2e-5
    sft_batch_size: int = 8
    sft_grad_accum_steps: int = 4

    # Reward model
    rm_dataset: str = "Anthropic/hh-rlhf"
    rm_max_epochs: int = 2
    rm_lr: float = 1e-5

    # RLHF / PPO
    ppo_steps: int = 500
    ppo_lr: float = 1e-6
    kl_coeff: float = 0.1
    reward_clip_bound: float = 5.0

    # RLAIF
    rlaif_model: str = "google/flan-t5-small"

    # Paths
    checkpoint_dir: str = "outputs/project3/checkpoints"
    log_path: str = "outputs/project3/experiment_log.jsonl"
    comparison_file: str = "outputs/project3/stage_comparison.json"
