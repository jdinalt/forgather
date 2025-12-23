import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

"""
Complex-valued RoPE implementation for academic/experimental purposes.

This implementation uses complex number arithmetic (e^(iθ)) instead of
separate cos/sin computations. While theoretically more elegant, current
PyTorch compiler support for complex operations is limited, resulting in
slower performance than the real-valued approach.

Benchmark results (Llama 1B, dynamic KV cache):
- Real-valued compiled: 114.87 tok/s
- Complex-valued compiled: 102.27 tok/s (11% slower)

Warning: TorchInductor does not generate efficient code for complex operators.

This implementation is preserved for:
- Future experiments if compiler support improves
- Academic reference and comparison
- Verification of mathematical equivalence
"""


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


class RotaryPE:
    """
    Complex-valued RoPE positional encoder (experimental/academic).

    Uses complex exponentials (e^(iθ)) for rotation instead of separate
    cos/sin computations. While mathematically equivalent and more elegant,
    current PyTorch compiler limitations make this slower than real-valued RoPE.

    This implementation is simplified:
    - No caching (always computes on-demand)
    - No compilation (doesn't help with complex ops)
    - Interface-compatible with real-valued RotaryPE for easy swapping

    Requirements:
    - d_head must be even (pairs of dimensions for complex representation)

    Note: This class intentionally uses the same name as the real-valued version
    to allow easy model configuration swapping via module import path.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_liger: bool = False,
        # Ignored parameters for interface compatibility
        cache_embeddings: bool = None,
        compile_on_demand: bool = None,
        use_complex_rope: bool = None,
    ):
        self.d_head = hidden_size // num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.dtype = torch.get_default_dtype()

        # Validate requirements
        if self.d_head % 2 != 0:
            raise ValueError(
                f"Complex RoPE requires d_head to be even, got {self.d_head}"
            )

        # Ignore cache/compile parameters but log if they're set
        if cache_embeddings is True:
            logger.warning(
                "Complex RoPE does not support caching. Ignoring cache_embeddings=True"
            )

        self.liger_kernel = None
        if use_liger:
            logger.warning(
                "Liger kernel not supported with complex RoPE. Using eager implementation."
            )

    def __repr__(self):
        rope_type = "default"
        if self.rope_scaling:
            rope_type = self.rope_scaling.get(
                "rope_type", self.rope_scaling.get("type", "default")
            )
        return (
            f"ComplexRotaryPE(d_head={self.d_head}, max_sequence_length={self.max_sequence_length}, "
            f"rope_theta={self.rope_theta}, rope_type={rope_type})"
        )

    def _compute_embeddings(
        self, position_ids: Tensor, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute cos/sin embeddings using complex exponentials.

        Args:
            position_ids: Position indices tensor of shape (batch_size, seq_len) or (1, seq_len)
            device: Device to compute on
            dtype: Data type for output tensors

        Returns:
            Tuple of (cos, sin) tensors of shape (batch_size, seq_len, d_head)
        """
        # Compute base inverse frequencies - shape: (d_head // 2,)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.d_head, 2, device=device, dtype=torch.float32)
                / self.d_head
            )
        )

        # Apply scaling if specified
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get(
                "rope_type", self.rope_scaling.get("type", "default")
            )
            if rope_type == "llama3":
                inv_freq = apply_llama3_scaling(inv_freq, self.rope_scaling)
            elif rope_type != "default":
                raise ValueError(
                    f"Unsupported rope_type: {rope_type}. Only 'llama3' is currently supported."
                )

        # Ensure position_ids is 2D: (batch_size, seq_len)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        batch_size, seq_len = position_ids.shape
        position_ids_float = position_ids.float()
        positions_flat = position_ids_float.reshape(-1)

        # Compute frequencies for all positions: (batch_size * seq_len, d_head // 2)
        freqs = torch.outer(positions_flat, inv_freq)

        # Create complex exponential: e^(i*theta) = cos(theta) + i*sin(theta)
        # torch.polar(abs, angle) creates complex tensor: abs * e^(i*angle)
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # Shape: (batch_size * seq_len, d_head // 2)

        # Reshape to (batch_size, seq_len, d_head // 2)
        freqs_cis = freqs_cis.reshape(batch_size, seq_len, self.d_head // 2)

        # Extract real and imaginary parts (cos and sin)
        # For applying rotation: q_rotated = q * cos + rotate_half(q) * sin
        cos = freqs_cis.real  # Shape: (batch_size, seq_len, d_head // 2)
        sin = freqs_cis.imag  # Shape: (batch_size, seq_len, d_head // 2)

        # Duplicate to full d_head dimension to match real-valued interface
        cos = torch.cat([cos, cos], dim=-1).to(dtype=dtype)
        sin = torch.cat([sin, sin], dim=-1).to(dtype=dtype)

        return cos, sin

    def __call__(
        self, q: Tensor, k: Tensor, position_ids: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE embedding to query and key using complex exponentials.

        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, d_head)
            k: Key tensor of shape (batch_size, seq_len, num_heads, d_head)
            position_ids: Position indices tensor of shape (1, seq_len) or (batch_size, seq_len).
                         If None, uses sequential positions [0, 1, 2, ..., seq_len-1]

        Returns:
            Tuple of (rotated_q, rotated_k) tensors with same shapes as input
        """
        seq_len = q.shape[1]
        batch_size = q.shape[0]
        assert seq_len == k.shape[1]

        # Create position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=q.device, dtype=torch.long
            ).unsqueeze(0)

        # Compute cos/sin using complex exponentials
        cos, sin = self._compute_embeddings(position_ids, q.device, q.dtype)

        # Reshape for broadcasting: [batch_size, seq_len, 1, d_head]
        cos = cos.unsqueeze(2)  # [batch_size, seq_len, 1, d_head]
        sin = sin.unsqueeze(2)  # [batch_size, seq_len, 1, d_head]

        # Apply rotation
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
