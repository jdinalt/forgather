"""
Canon Pre-LN Transformer Layer.

Extends the standard Pre-LN transformer layer with Canon layers at
positions A (before attention) and C (before FFN).

From "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers" (Allen-Zhu, 2025).
"""

from typing import Callable, Optional

import torch
from torch import FloatTensor, nn


class _CanonLayer(nn.Module):
    """Depthwise causal 1D convolution with optional residual connection."""

    def __init__(self, dim: int, kernel_size: int = 4, residual: bool = True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.residual = residual
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=kernel_size,
            groups=dim, bias=False, padding=kernel_size - 1,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x_conv = x.transpose(1, 2)
        out = self.conv(x_conv)
        out = out[..., : x.shape[1]]
        out = out.transpose(1, 2)
        if self.residual:
            return x + out
        return out


class CanonPreLNLayer(nn.Module):
    """Pre-LN transformer layer with Canon-A (pre-attention) and Canon-C (pre-FFN)."""

    def __init__(
        self,
        *,
        feedforward_factory: Callable,
        attention_factory: Callable,
        norm_factory: Callable,
        dropout: Optional[float] = 0.1,
        residual_dropout: Optional[float] = 0.0,
        canon_kernel: int = 4,
        canon_residual: bool = True,
        d_model: int,
        **kwargs,
    ):
        super().__init__()
        self.feedforward = feedforward_factory(**kwargs)
        self.attention = attention_factory(**kwargs)
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()
        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)
        if residual_dropout == 0.0:
            self.residual_dropout = nn.Identity()
        else:
            self.residual_dropout = nn.Dropout(residual_dropout)

        # Canon-A: after norm1, before attention
        self.canon_a = _CanonLayer(dim=d_model, kernel_size=canon_kernel, residual=canon_residual)
        # Canon-C: after norm2, before feedforward
        self.canon_c = _CanonLayer(dim=d_model, kernel_size=canon_kernel, residual=canon_residual)

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        residual = self.residual_dropout(x)
        x = self.norm1(x)
        x = self.canon_a(x)
        x = self.attention(x, **kwargs)
        x = residual + self.dropout(x)
        residual = self.residual_dropout(x)
        x = self.norm2(x)
        x = self.canon_c(x)
        x = self.feedforward(x, **kwargs)
        x = residual + self.dropout(x)
        return x
