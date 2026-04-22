"""
Visualization module for Project 2: Transformer Pre-training.

Provides:
    plot_attention_heatmaps   — hook-based attention weight capture per head/layer
    plot_weight_distributions — histograms and spectral norms per layer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from pretrain.model import CausalSelfAttention, GPTModel
from pretrain.tokenizer import BPETokenizer


def plot_attention_heatmaps(
    model: GPTModel,
    input_ids: Tensor,
    tokenizer: BPETokenizer,
    save_dir: str,
) -> None:
    """
    Capture attention weights for each head in each layer via forward hooks,
    then save one heatmap PNG per (layer, head) pair.

    Args:
        model:     GPTModel in eval mode.
        input_ids: (1, T) LongTensor of token IDs.
        tokenizer: BPETokenizer used to decode token labels.
        save_dir:  Directory to save heatmap PNGs.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval()

    # Decode token labels for axis ticks
    ids_list = input_ids[0].tolist()
    token_labels = [tokenizer.decode([tok_id]) for tok_id in ids_list]

    # Collect attention weights from each CausalSelfAttention block
    captured: list[dict[str, Any]] = []
    hooks = []

    for layer_idx, block in enumerate(model.blocks):
        def make_hook(lidx: int):
            def hook(module: CausalSelfAttention, inp: tuple, out: Tensor) -> None:
                if module._last_attn_weights is not None:
                    captured.append({
                        "layer": lidx,
                        "weights": module._last_attn_weights.cpu(),  # (B, n_heads, T, T)
                    })
            return hook
        hooks.append(block.attn.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    # Plot one heatmap per (layer, head)
    for entry in captured:
        layer_idx = entry["layer"]
        weights = entry["weights"][0]  # (n_heads, T, T) — take first batch item
        n_heads = weights.shape[0]

        for head_idx in range(n_heads):
            attn_matrix = weights[head_idx].numpy()  # (T, T)
            T = attn_matrix.shape[0]

            fig, ax = plt.subplots(figsize=(max(4, T * 0.4), max(4, T * 0.4)))
            im = ax.imshow(attn_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax)

            tick_labels = token_labels[:T]
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
            ax.set_yticklabels(tick_labels, fontsize=6)
            ax.set_title(f"Layer {layer_idx} — Head {head_idx}")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")

            fig.tight_layout()
            out_path = Path(save_dir) / f"attn_layer{layer_idx}_head{head_idx}.png"
            plt.savefig(str(out_path), dpi=100)
            plt.close(fig)


def plot_weight_distributions(model: GPTModel, save_dir: str) -> None:
    """
    For each named parameter in the model, plot a histogram of weight values
    and compute the spectral norm. Saves one PNG per layer.

    Args:
        model:    GPTModel (any mode).
        save_dir: Directory to save distribution PNGs.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for name, param in model.named_parameters():
        if param.dim() < 2:
            # Skip 1-D parameters (biases, layer norm scales)
            continue

        data = param.detach().cpu().float()
        flat = data.view(-1).numpy()

        # Spectral norm: largest singular value of the 2-D weight matrix
        mat = data.view(data.shape[0], -1)
        try:
            _, s, _ = torch.linalg.svd(mat, full_matrices=False)
            spectral_norm = float(s[0].item())
        except Exception:
            spectral_norm = float("nan")

        # Plot histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(flat, bins=50, color="#1f77b4", edgecolor="none", alpha=0.8)
        ax.set_title(f"{name}\nSpectral norm: {spectral_norm:.4f}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        fig.tight_layout()

        # Sanitize filename
        safe_name = name.replace(".", "_").replace("/", "_")
        out_path = Path(save_dir) / f"weights_{safe_name}.png"
        plt.savefig(str(out_path), dpi=100)
        plt.close(fig)
