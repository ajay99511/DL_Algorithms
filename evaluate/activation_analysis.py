"""Activation analysis: capture and visualise layer activations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def record_activations(
    model: nn.Module,
    dataloader: DataLoader,
    n_batches: int = 5,
) -> dict[str, Tensor]:
    """Capture activation tensors from all Linear layers via forward hooks.

    Activations are averaged over *n_batches* batches to give a representative
    sample of the activation distribution.

    Parameters
    ----------
    model:
        The model to probe (CPU).
    dataloader:
        DataLoader providing input batches.
    n_batches:
        Number of batches to average over.

    Returns
    -------
    dict mapping layer name -> averaged activation tensor (flattened to 1-D).
    """
    accumulated: dict[str, list[Tensor]] = {}
    hooks: list[Any] = []

    def _make_hook(layer_name: str):
        def _hook(module: nn.Module, input: Any, output: Tensor) -> None:
            # Detach and flatten to 1-D for distribution analysis
            accumulated.setdefault(layer_name, []).append(
                output.detach().cpu().view(-1)
            )
        return _hook

    # Register hooks on all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(torch.device("cpu"))
            model(inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average activations across batches
    result: dict[str, Tensor] = {}
    for layer_name, tensors in accumulated.items():
        # Concatenate all collected activations for this layer
        result[layer_name] = torch.cat(tensors, dim=0)

    return result


def plot_activation_distributions(
    activations: dict[str, Tensor],
    save_dir: str,
) -> None:
    """Plot a histogram of activation values per layer, saved as PNG files.

    Parameters
    ----------
    activations:
        Output of :func:`record_activations`.
    save_dir:
        Directory where PNG files are written.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if not _HAS_MPL:
        print("matplotlib not available — skipping activation distribution plots.")
        return

    for layer_name, act_tensor in activations.items():
        values = act_tensor.numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Count")
        safe_name = layer_name.replace("/", "_").replace(".", "_")
        ax.set_title(f"Activation Distribution — {layer_name}")
        fig.tight_layout()
        fig.savefig(str(Path(save_dir) / f"activation_{safe_name}.png"), dpi=100)
        plt.close(fig)
