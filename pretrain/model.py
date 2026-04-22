"""
GPT-style decoder-only transformer for Project 2: Language Model Pre-training.

References:
    # Ref: Vaswani et al., 2017 — "Attention Is All You Need"
    # Ref: Radford et al., 2019 — "Language Models are Unsupervised Multitask Learners" (GPT-2)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with pre-norm.

    The causal mask is registered as a buffer so it moves with the module
    to any device without manual management.

    # Ref: Vaswani et al., 2017 — "Attention Is All You Need"
    # Ref: Radford et al., 2019 — "Language Models are Unsupervised Multitask Learners" (GPT-2)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        context_length: int,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # Fused QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: upper-triangular (above diagonal) = -inf, rest = 0
        # Shape: (1, 1, context_length, context_length)
        mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", mask)

        # Store last attention weights for visualization (set during forward)
        self._last_attn_weights: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) — attended output.
        """
        B, T, d_model = x.shape

        # QKV projection and split
        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.split(d_model, dim=-1)  # each (B, T, d_model)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, n_heads, T, T)

        # Apply causal mask: mask out future positions
        # causal_mask shape: (context_length, context_length) — True where masked
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf")
        )

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, T, T)
        # Store for visualization
        self._last_attn_weights = attn_weights.detach()

        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)  # (B, T, d_model)
        out = self.resid_dropout(self.out_proj(out))
        return out


class TransformerBlock(nn.Module):
    """
    A single GPT-style transformer block with pre-LayerNorm.

    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    FFN uses GELU activation (Hendrycks & Gimpel, 2016).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        context_length: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, context_length)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        # Pre-norm attention residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm FFN residual
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """
    GPT-style decoder-only language model.

    Architecture:
        - Token embeddings (vocab_size, d_model)
        - Learned positional encodings (context_length, d_model)
        - N TransformerBlocks
        - Final LayerNorm
        - LM head (d_model -> vocab_size) with weight tying to token embeddings

    # Ref: Vaswani et al., 2017 — "Attention Is All You Need"
    # Ref: Radford et al., 2019 — "Language Models are Unsupervised Multitask Learners" (GPT-2)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, context_length)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # LM head — weight tied to token embeddings (GPT-2 style)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply GPT-2-style weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass.

        # CPU-only: on a CUDA-enabled machine with BF16 you would wrap this in:
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

        Args:
            input_ids: (B, T) LongTensor of token IDs.
            targets:   (B, T) LongTensor of target token IDs (optional).
                       If provided, cross-entropy loss is computed and returned.

        Returns:
            (logits, loss) where:
                logits: (B, T, vocab_size)
                loss:   scalar cross-entropy loss if targets provided, else None.
        """
        B, T = input_ids.shape
        assert T <= self.context_length, (
            f"Sequence length {T} exceeds context_length {self.context_length}"
        )

        # Token + positional embeddings
        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # LM head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss: Tensor | None = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
