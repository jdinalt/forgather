"""
Canon Layer: Depthwise causal 1D convolution for local token mixing.

From "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers" (Allen-Zhu, 2025). Canon layers compute a causal
weighted sum of neighboring token representations using a depthwise
Conv1d with a small kernel (default K=4).

Each channel independently computes:
    h'_t = w_0 * h_t + w_1 * h_{t-1} + ... + w_{K-1} * h_{t-K+1}

With residual connection (default):
    output = h_t + h'_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonLayer(nn.Module):
    """Depthwise causal 1D convolution with optional residual connection."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        bias: bool = False,
        residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.residual = residual

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=bias,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            [batch_size, seq_len, dim]
        """
        # Conv1d expects [B, C, T]
        x_conv = x.transpose(1, 2)
        out = self.conv(x_conv)
        # Remove right padding to maintain causality
        out = out[..., : x.shape[1]]
        out = out.transpose(1, 2)

        if self.residual:
            return x + out
        return out
