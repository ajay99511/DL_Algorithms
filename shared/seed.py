import random
import numpy as np
import torch


def fix_all_seeds(seed: int) -> None:
    """Fix Python random, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # CPU-only: on a CUDA-enabled machine you would also call:
    # torch.cuda.manual_seed_all(seed)
