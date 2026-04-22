from dataclasses import dataclass, field
from shared.config import BaseConfig


@dataclass
class MLPConfig(BaseConfig):
    # Data
    test_size: float = 0.1
    val_size: float = 0.1
    # Model
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.1
    init_strategy: str = "kaiming"  # "normal" | "xavier" | "kaiming"
    # Training
    batch_size: int = 32
    grad_accum_steps: int = 4
    max_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 5
    # Paths
    checkpoint_dir: str = "outputs/project1/checkpoints"
    log_path: str = "outputs/project1/experiment_log.jsonl"
    plot_dir: str = "outputs/project1/plots"
