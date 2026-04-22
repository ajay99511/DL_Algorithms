import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup from 0 to base_lr over warmup_steps,
    then cosine decay to min_lr_ratio * base_lr.
    # Ref: Loshchilov & Hutter, 2017 — "SGDR: Stochastic Gradient Descent with Warm Restarts"
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: scale from 0 to 1
            return step / max(1, warmup_steps)
        # Cosine decay from 1 to min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
