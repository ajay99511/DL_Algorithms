from dataclasses import dataclass
from shared.config import BaseConfig


@dataclass
class TransformerConfig(BaseConfig):
    # Data
    dataset_name: str = "roneneldan/TinyStories"
    max_stories: int = 50_000
    val_fraction: float = 0.05
    context_length: int = 256
    vocab_size: int = 8_000
    # Model
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512          # 4 * d_model
    dropout: float = 0.1
    # Training
    batch_size: int = 16
    grad_accum_steps: int = 16   # effective batch = 256
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    # Paths
    tokenizer_dir: str = "outputs/project2/tokenizer"
    checkpoint_dir: str = "outputs/project2/checkpoints"
    log_path: str = "outputs/project2/experiment_log.jsonl"
