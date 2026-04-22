"""
Vision Transformer (ViT) for image classification.

References:
    # Ref: Dosovitskiy et al., 2020 — "An Image is Worth 16x16 Words:
    #      Transformers for Image Recognition at Scale"
    # Ref: Vaswani et al., 2017 — "Attention Is All You Need"
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visiontx.config import ViTConfig


class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects to d_model.

    Uses a single Conv2d with kernel_size=patch_size and stride=patch_size
    to efficiently extract and project all patches in one operation.

    # Ref: Dosovitskiy et al., 2020 — "An Image is Worth 16x16 Words"
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        n_channels: int,
        d_model: int,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # Conv2d as patch extractor + linear projection
        self.projection = nn.Conv2d(
            in_channels=n_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor

        Returns:
            (B, n_patches, d_model) patch embeddings
        """
        # (B, C, H, W) -> (B, d_model, H/P, W/P)
        x = self.projection(x)
        # (B, d_model, H/P, W/P) -> (B, n_patches, d_model)
        B, d_model, h, w = x.shape
        x = x.flatten(2)          # (B, d_model, n_patches)
        x = x.transpose(1, 2)     # (B, n_patches, d_model)
        return x


class ViTEncoderBlock(nn.Module):
    """
    Pre-norm transformer encoder block.

    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # Pre-norm layers
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Multi-head self-attention (bidirectional — no causal mask for ViT)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Store attention weights for visualization
        self._last_attn_weights: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model) where T = n_patches + 1 (includes class token)

        Returns:
            (B, T, d_model)
        """
        B, T, d_model = x.shape

        # --- Multi-head self-attention (pre-norm) ---
        normed = self.ln1(x)
        qkv = self.qkv_proj(normed)                          # (B, T, 3*d_model)
        q, k, v = qkv.split(d_model, dim=-1)                 # each (B, T, d_model)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, n_heads, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)                     # (B, n_heads, T, T)

        # Store for get_attention_weights
        self._last_attn_weights = attn_weights.detach()

        attn_weights_dropped = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights_dropped, v)          # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        out = self.resid_dropout(self.out_proj(out))

        x = x + out

        # --- Feed-forward (pre-norm) ---
        x = x + self.ffn(self.ln2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Architecture:
        1. PatchEmbedding: (B, C, H, W) -> (B, n_patches, d_model)
        2. Prepend learnable class token: (B, n_patches+1, d_model)
        3. Add learned 1D positional embeddings
        4. N ViTEncoderBlocks
        5. Final LayerNorm
        6. MLP classification head on class token output

    # Ref: Dosovitskiy et al., 2020 — "An Image is Worth 16x16 Words"
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        n_patches = (config.image_size // config.patch_size) ** 2
        self.n_patches = n_patches
        seq_len = n_patches + 1  # +1 for class token

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            n_channels=config.n_channels,
            d_model=config.d_model,
        )

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # Learned 1D positional embeddings for (n_patches + 1) positions
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, config.d_model))

        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            ViTEncoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # MLP classification head on class token
        self.head = nn.Linear(config.d_model, config.n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following ViT paper conventions."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor

        Returns:
            (B, n_classes) logits

        # CPU-only: on a CUDA-enabled machine with BF16 you would wrap this in:
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        """
        B = x.shape[0]

        # Patch embedding: (B, n_patches, d_model)
        x = self.patch_embed(x)

        # Prepend class token: (B, n_patches+1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.emb_dropout(x)

        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Classification head on class token (position 0)
        cls_out = x[:, 0]          # (B, d_model)
        logits = self.head(cls_out)  # (B, n_classes)
        return logits

    def get_attention_weights(self, x: Tensor) -> list[Tensor]:
        """
        Run a forward pass and return attention weights from each encoder block.

        Args:
            x: (B, C, H, W) image tensor

        Returns:
            List of attention weight tensors, one per layer.
            Each tensor has shape (B, n_heads, T, T) where T = n_patches + 1.
        """
        # Run forward pass to populate _last_attn_weights in each block
        with torch.no_grad():
            self.forward(x)

        weights = []
        for block in self.blocks:
            if block._last_attn_weights is not None:
                weights.append(block._last_attn_weights)
        return weights

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
