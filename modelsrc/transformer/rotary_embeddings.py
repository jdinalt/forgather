from typing import Tuple

import torch
from torch import Tensor


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates the rotary position embedding frequencies used in RoPE.

    Args:
        dim: Dimension of the embedding (typically d_head)
        end: Maximum sequence length
        theta: Base for the geometric progression (default: 10000.0)

    Returns:
        Tensor of shape (end, dim//2) containing complex exponentials
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    """
    Reshape frequency tensor to be broadcastable with input tensor.

    Args:
        freqs_cis: Frequency tensor of shape (seq_len, dim//2)
        x: Input tensor of shape (batch_size, seq_len, num_heads, dim)

    Returns:
        Reshaped freqs_cis tensor for broadcasting
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q: Tensor, k: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, d_head)
        k: Key tensor of shape (batch_size, seq_len, num_heads, d_head)
        freqs_cis: Frequency tensor of shape (seq_len, d_head//2)

    Returns:
        Tuple of (rotated_q, rotated_k) tensors with same shapes as input
    """
    # Reshape q and k to complex representation
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, q_complex)

    # Apply rotation
    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(3)

    return q_out.type_as(q), k_out.type_as(k)


class RotaryPE(torch.nn.Module):
    """
    Complex-valued RoPE positional encoder module
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

        freqs_cis = precompute_freqs_cis(d_head, max_sequence_length, rope_theta)
        # Note: Use nn.Buffer for buffers, rather than register_buffer(). The later does
        # not work properly with model splitting in torch.distributed.pipelining
        self.torch.nn.Buffer = torch.nn.Buffer(freqs_cis)

    def extra_repr(self):
        return f"d_head={self.d_head}, max_sequence_length={self.max_sequence_length}, rope_theta={self.rope_theta}"

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE embedding to query and key

        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, d_head)
            k: Key tensor of shape (batch_size, seq_len, num_heads, d_head)

        Returns:
            Tuple of (rotated_q, rotated_k) tensors with same shapes as input
        """
        seq_len = q.shape[1]
        assert seq_len == k.shape[1]
        assert (
            seq_len <= self.freqs_cis.shape[0]
        ), f"seq_len {seq_len} > max_seq_len {self.freqs_cis.shape[0]}"
        return apply_rotary_emb(q, k, self.freqs_cis[:seq_len])
