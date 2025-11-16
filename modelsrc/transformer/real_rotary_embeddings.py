from typing import Tuple, Optional, Dict, Any
import math

import torch
from torch import Tensor

""" Real-valued RoPE implementation (vs. complex) """


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_llama3_scaling(inv_freq: Tensor, rope_scaling: Dict[str, Any]) -> Tensor:
    """
    Apply Llama3-style frequency scaling with wavelength-based bands.

    Args:
        inv_freq: Base inverse frequencies tensor
        rope_scaling: Dict containing:
            - factor: Overall scaling factor (e.g., 8.0)
            - low_freq_factor: Scaling factor for low frequency band (e.g., 1.0)
            - high_freq_factor: Scaling factor for high frequency band (e.g., 4.0)
            - original_max_position_embeddings: Original context length (e.g., 8192)

    Returns:
        Scaled inverse frequencies tensor
    """
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    # Compute wavelength boundaries
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    # Convert inverse frequencies to wavelengths
    wavelen = 2 * math.pi / inv_freq

    # Three-band scaling:
    # - High freq (wavelen < high_freq_wavelen): unchanged
    # - Low freq (wavelen > low_freq_wavelen): divide by factor
    # - Medium freq: smooth interpolation between the two

    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )

    # Compute smooth interpolation factor for medium frequencies
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama

    # Apply smooth interpolation only to medium frequency band
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama


def precompute_cos_sin(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_scaling: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Precompute cosine and sine tensors for RoPE.

    Args:
        dim: Dimension of the embedding (typically d_head)
        end: Maximum sequence length
        theta: Base for the geometric progression (default: 10000.0)
        rope_scaling: Optional dict with scaling parameters for extended context.
                     Supports 'llama3' type scaling with factor-based wavelength bands.

    Returns:
        Tuple of (cos, sin) tensors of shape (end, dim)
    """
    # Compute base inverse frequencies - identical to HF
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Apply scaling if specified
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type == "llama3":
            inv_freq = apply_llama3_scaling(inv_freq, rope_scaling)
        elif rope_type != "default":
            raise ValueError(
                f"Unsupported rope_type: {rope_type}. Only 'llama3' is currently supported."
            )

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
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.d_head = d_head
        self.max_sequence_length = max_sequence_length
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # Precompute cos/sin tensors once for the entire model
        cos, sin = precompute_cos_sin(
            d_head, max_sequence_length, rope_theta, rope_scaling
        )

        # Note: Use nn.Buffer for buffers, rather than register_buffer(). The later does
        # not work properly with model splitting in torch.distributed.pipelining
        self.cos_cached = torch.nn.Buffer(cos)
        self.sin_cached = torch.nn.Buffer(sin)

    def extra_repr(self):
        rope_type = "default"
        if self.rope_scaling:
            rope_type = self.rope_scaling.get(
                "rope_type", self.rope_scaling.get("type", "default")
            )
        return f"d_head={self.d_head}, max_sequence_length={self.max_sequence_length}, rope_theta={self.rope_theta}, rope_type={rope_type}"

    def forward(
        self, q: Tensor, k: Tensor, position_ids: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
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
