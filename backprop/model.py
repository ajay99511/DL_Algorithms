"""
MLP model for regression on California Housing dataset.

References:
    # Ref: He et al., 2015 — "Delving Deep into Rectifiers: Surpassing Human-Level
    #      Performance on ImageNet Classification" — Kaiming initialization
    # Ref: Glorot & Bengio, 2010 — "Understanding the difficulty of training deep
    #      neural networks" — Xavier initialization
"""

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """
    Multi-layer perceptron for regression.

    Architecture per hidden layer: Linear → ReLU → Dropout
    Final layer: Linear(hidden[-1], 1)

    # Ref: He et al., 2015 — "Delving Deep into Rectifiers" — Kaiming init
    # Ref: Glorot & Bengio, 2010 — "Understanding the difficulty of training deep NNs" — Xavier init
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, input_dim) -> (B, 1)"""
        return self.network(x)


def initialize_weights(model: MLP, strategy: str) -> None:
    """
    Apply weight initialization to all Linear layers in the MLP.

    Strategies:
        "normal"  — N(0, 0.01) for weights, zeros for biases
        "xavier"  — Glorot uniform initialization (Glorot & Bengio, 2010)
        "kaiming" — He/Kaiming normal initialization (He et al., 2015)

    # Ref: He et al., 2015 — "Delving Deep into Rectifiers" — Kaiming init
    # Ref: Glorot & Bengio, 2010 — "Understanding the difficulty of training deep NNs" — Xavier init
    """
    if strategy not in ("normal", "xavier", "kaiming"):
        raise ValueError(f"Unknown init strategy '{strategy}'. Choose from: normal, xavier, kaiming")

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if strategy == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif strategy == "xavier":
                # Ref: Glorot & Bengio, 2010
                nn.init.xavier_uniform_(module.weight)
            elif strategy == "kaiming":
                # Ref: He et al., 2015
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def activation_stats(model: MLP, x: torch.Tensor) -> dict[str, dict[str, float]]:
    """
    Run a forward pass with hooks on ReLU layers; return per-layer activation statistics.

    Returns a dict keyed by layer name (e.g. "relu_0", "relu_1", ...) where each value is:
        {
            "mean": float,          — mean activation value
            "std": float,           — std of activation values
            "dead_fraction": float, — fraction of activations <= 0
        }

    All returned values are guaranteed to be finite (not NaN, not Inf).
    """
    stats: dict[str, dict[str, float]] = {}
    hooks = []
    relu_index = 0

    def make_hook(name: str):
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            flat = output.detach().float()
            mean_val = float(flat.mean().item())
            std_val = float(flat.std().item())
            dead_val = float((flat <= 0).float().mean().item())

            # Ensure finite values
            if not (mean_val == mean_val and abs(mean_val) != float("inf")):
                mean_val = 0.0
            if not (std_val == std_val and abs(std_val) != float("inf")):
                std_val = 0.0
            dead_val = max(0.0, min(1.0, dead_val))

            stats[name] = {"mean": mean_val, "std": std_val, "dead_fraction": dead_val}
        return hook

    # Register hooks on all ReLU modules
    for module in model.network:
        if isinstance(module, nn.ReLU):
            name = f"relu_{relu_index}"
            hooks.append(module.register_forward_hook(make_hook(name)))
            relu_index += 1

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()

    return stats
