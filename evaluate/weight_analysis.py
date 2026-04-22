"""Weight analysis utilities: norms, singular values, dead neuron detection."""

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


def compute_weight_norms(model: nn.Module) -> dict[str, float]:
    """Compute the Frobenius norm for each named weight matrix in the model.

    Returns
    -------
    dict mapping parameter name -> Frobenius norm (float).
    """
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            norms[name] = float(torch.linalg.norm(param.data, ord="fro"))
    return norms


def compute_singular_values(model: nn.Module) -> dict[str, Tensor]:
    """Compute singular values (via SVD) for each 2-D weight matrix.

    Only 2-D parameters (weight matrices) are processed; biases and
    higher-dimensional tensors are skipped.

    Returns
    -------
    dict mapping parameter name -> 1-D tensor of singular values (descending).
    """
    svd_map: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.dim() == 2:
            # torch.linalg.svd returns (U, S, Vh); we only need S
            _, singular_values, _ = torch.linalg.svd(param.data, full_matrices=False)
            svd_map[name] = singular_values.detach().cpu()
    return svd_map


def compute_dead_neuron_ratio(
    model: nn.Module,
    dataloader: DataLoader,
    threshold: float = 1e-6,
) -> dict[str, float]:
    """Compute the fraction of neurons with mean activation below *threshold*.

    Uses forward hooks on ReLU and GELU activation layers.

    Parameters
    ----------
    model:
        The model to analyse (CPU).
    dataloader:
        DataLoader providing input batches.
    threshold:
        Neurons whose mean absolute activation across the batch is below this
        value are considered "dead".

    Returns
    -------
    dict mapping layer name -> dead neuron fraction in [0, 1].
    """
    activation_sums: dict[str, Tensor] = {}
    activation_counts: dict[str, int] = {}
    hooks: list[Any] = []

    def _make_hook(layer_name: str):
        def _hook(module: nn.Module, input: Any, output: Tensor) -> None:
            # output shape: (B, ...) — flatten all but batch dim
            flat = output.detach().abs().mean(dim=0).view(-1)
            if layer_name not in activation_sums:
                activation_sums[layer_name] = flat.clone()
                activation_counts[layer_name] = 1
            else:
                activation_sums[layer_name] += flat
                activation_counts[layer_name] += 1
        return _hook

    # Register hooks on ReLU and GELU layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.GELU)):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Support (inputs,) tuples or plain tensors
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(torch.device("cpu"))
            model(inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    dead_ratios: dict[str, float] = {}
    for layer_name, total in activation_sums.items():
        mean_act = total / activation_counts[layer_name]
        dead_fraction = float((mean_act < threshold).float().mean())
        dead_ratios[layer_name] = dead_fraction

    return dead_ratios


def plot_weight_analysis(
    model: nn.Module,
    save_dir: str,
) -> None:
    """Save weight norm bar chart and singular value distribution plots.

    Parameters
    ----------
    model:
        The model to visualise.
    save_dir:
        Directory where PNG files are written.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    norms = compute_weight_norms(model)
    svd_map = compute_singular_values(model)

    if not _HAS_MPL:
        print("matplotlib not available — skipping weight analysis plots.")
        return

    # --- Weight norm bar chart ---
    fig, ax = plt.subplots(figsize=(max(6, len(norms) * 0.4), 5))
    names = list(norms.keys())
    values = [norms[n] for n in names]
    ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Frobenius Norm")
    ax.set_title("Weight Norms per Layer")
    fig.tight_layout()
    fig.savefig(str(Path(save_dir) / "weight_norms.png"), dpi=100)
    plt.close(fig)

    # --- Singular value distribution ---
    if svd_map:
        fig, ax = plt.subplots(figsize=(8, 5))
        for layer_name, sv in list(svd_map.items())[:10]:  # cap at 10 layers
            ax.plot(sv.numpy(), label=layer_name, alpha=0.7)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value")
        ax.set_title("Singular Value Spectra (top 10 layers)")
        ax.legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        fig.savefig(str(Path(save_dir) / "singular_values.png"), dpi=100)
        plt.close(fig)
