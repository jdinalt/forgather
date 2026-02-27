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
        d_model: int,
        canon_factory: Callable,
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
        self.canon_a = canon_factory(dim=d_model)
        # Canon-C: after norm2, before feedforward
        self.canon_c = canon_factory(dim=d_model)

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
