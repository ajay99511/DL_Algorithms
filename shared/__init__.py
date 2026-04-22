"""Shared utilities for the deep-learning-llm-mastery curriculum.

Exports all cross-project utilities so any project can import from `shared` directly:

    from shared import BaseConfig, JSONLogger, save_checkpoint, load_checkpoint
    from shared import fix_all_seeds, cosine_with_warmup, load_config, save_config
"""

from shared.config import BaseConfig, load_config, save_config
from shared.logging_utils import JSONLogger
from shared.checkpointing import save_checkpoint, load_checkpoint
from shared.seed import fix_all_seeds
from shared.lr_schedule import cosine_with_warmup

__all__ = [
    "BaseConfig",
    "load_config",
    "save_config",
    "JSONLogger",
    "save_checkpoint",
    "load_checkpoint",
    "fix_all_seeds",
    "cosine_with_warmup",
]
