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
    default_dtype = torch.get_default_dtype()
    cos = emb.cos().to(dtype=default_dtype)
    sin = emb.sin().to(dtype=default_dtype)
    return cos, sin

class RealRotaryPE(torch.nn.Module):
    """
    Real-valued RoPE positional encoder module
    """

    def __init__(
        self,
        d_head: int,
        max_sequence_length: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.max_sequence_length = max_sequence_length
        self.rope_theta = rope_theta

        # Precompute cos/sin tensors once for the entire model
        cos, sin = precompute_cos_sin(d_head, max_sequence_length, rope_theta)

        # Note: Use nn.Buffer for buffers, rather than register_buffer(). The later does
        # not work properly with model splitting in torch.distributed.pipelining
        self.cos_cached = torch.nn.Buffer(cos)
        self.sin_cached = torch.nn.Buffer(sin)

    def extra_repr(self):
        return f"d_head={self.d_head}, max_sequence_length={self.max_sequence_length}, rope_theta={self.rope_theta}"

    def forward(self, q: Tensor, k: Tensor, position_ids: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE embedding to query and key

        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, d_head)
            k: Key tensor of shape (batch_size, seq_len, num_heads, d_head)
            position_ids: Position indices tensor of shape (1, seq_len).
                         If None, uses sequential positions [0, 1, 2, ..., seq_len-1]

        Returns:
            Tuple of (rotated_q, rotated_k) tensors with same shapes as input
        """
        seq_len = q.shape[1]
        assert seq_len == k.shape[1]

        if position_ids is None:
            # Default behavior: use sequential positions
            assert (
                seq_len <= self.cos_cached.shape[0]
            ), f"seq_len {seq_len} > max_seq_len {self.cos_cached.shape[0]}"
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
            # Reshape cos/sin for broadcasting to match input tensor dimensions
            # Need cos/sin: [1, seq_len, 1, d_head] for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_head]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_head]
        else:
            cos = self.cos_cached[position_ids].unsqueeze(2)
            sin = self.sin_cached[position_ids].unsqueeze(2)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
