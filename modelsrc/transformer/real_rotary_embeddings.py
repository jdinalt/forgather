import math
from typing import Tuple

import torch
from torch import Tensor

""" Real-valued RoPE implementation (vs. complex) """


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_cos_sin(
    dim: int, end: int, theta: float = 10000.0
) -> Tuple[Tensor, Tensor]:
    """
    Precompute cosine and sine tensors for RoPE.

    Args:
        dim: Dimension of the embedding (typically d_head)
        end: Maximum sequence length
        theta: Base for the geometric progression (default: 10000.0)

    Returns:
        Tuple of (cos, sin) tensors of shape (end, dim)
    """
    # Compute inverse frequencies - identical to HF
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Create position indices
    t = torch.arange(end, device=inv_freq.device, dtype=torch.float32)

    # Compute frequencies for each position
    freqs = torch.outer(t, inv_freq)

    # Duplicate frequencies to match full dimension
    emb = torch.cat((freqs, freqs), dim=-1)

    # Compute cos and sin
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary position embeddings using HuggingFace-compatible cos/sin method.

    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, d_head)
        k: Key tensor of shape (batch_size, seq_len, num_heads, d_head)
        cos: Cosine tensor of shape (seq_len, d_head)
        sin: Sine tensor of shape (seq_len, d_head)

    Returns:
        Tuple of (rotated_q, rotated_k) tensors with same shapes as input
    """
    # Reshape cos/sin for broadcasting to match input tensor dimensions
    # Input q/k: [batch, seq_len, num_heads, d_head]
    # Need cos/sin: [1, seq_len, 1, d_head] for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_head]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_head]

    # Apply rotary embeddings.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RealRotaryPE(torch.nn.Module):
    """
    Real-valued RoPE positional encoder
    """

    def __init__(
        self,
        d_head: int,
        max_sequence_length: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # Precompute cos/sin tensors once for the entire model
        cos, sin = precompute_cos_sin(d_head, max_sequence_length, rope_theta)
        self.register_buffer("cos_cached", cos, persistent=True)
        self.register_buffer("sin_cached", sin, persistent=True)

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """
        Return cos and sin tensors for the given sequence length.

        Args:
            seq_len: Sequence length to return embeddings for

        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, d_head)
        """
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    q: Tensor, k: Tensor, pos_emb: Tuple[Tensor, Tensor]
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, d_head)
        k: Key tensor of shape (batch_size, seq_len, num_heads, d_head)
        pos_emb: Tuple of (cos, sin) tensors from HFRotaryPE.forward()

    Returns:
        Tuple of (rotated_q, rotated_k) tensors with same shapes as input
    """
    cos, sin = pos_emb
    return apply_rotary_pos_emb(q, k, cos, sin)
