"""
Configuration for Project 4: Vision Transformer (ViT) on CIFAR-10 / ImageNette.

References:
    # Ref: Dosovitskiy et al., 2020 — "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.config import BaseConfig


@dataclass
class ViTConfig(BaseConfig):
    # Data
    dataset: str = "cifar10"   # "cifar10" | "imagenette"
    # Model
    image_size: int = 32
    patch_size: int = 4        # 4 or 8
    n_channels: int = 3
    n_classes: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    # Training
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    # Paths
    checkpoint_dir: str = "outputs/project4/checkpoints"
    log_path: str = "outputs/project4/experiment_log.jsonl"
    plot_dir: str = "outputs/project4/plots"
