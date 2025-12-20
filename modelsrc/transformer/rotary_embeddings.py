from typing import Tuple, Optional, Dict, Any, Iterable
import math

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

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
    dtype: torch.dtype,
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
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


class RotaryPE:
    """
    Real-valued RoPE positional encoder module

    Supports two computation strategies:
    - Cached (cache_embeddings=True): Precompute cos/sin for all positions once
    - On-demand (cache_embeddings=False): Compute cos/sin on each forward pass
      The on-demand strategy is better for torch.compile() and torch.export()
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_liger: bool = False,
        cache_embeddings: bool = False,
    ):
        self.d_head = hidden_size // num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.dtype = torch.get_default_dtype()
        self.cache_embeddings = cache_embeddings

        # OLD STRATEGY: Precompute cos/sin tensors once for the entire model
        # Currently disabled by default (cache_embeddings=False)
        # This can cause issues with torch.compile() and torch.export()
        if self.cache_embeddings:
            self.cos_cached, self.sin_cached = precompute_cos_sin(
                self.d_head, self.max_sequence_length, self.dtype, self.rope_theta, self.rope_scaling,
            )
        else:
            # Store None to indicate we're using on-demand computation
            self.cos_cached = None
            self.sin_cached = None

        self.liger_kernel = None

        if use_liger:
            try:
                from liger_kernel.ops.rope import LigerRopeFunction

                self.liger_kernel = LigerRopeFunction.apply

            except ImportError as e:
                logger.info(
                    "liger-kernel not installed. Install with:\n"
                    "  pip install liger-kernel\n"
                    "Using eager RoPE implementation as fallback"
                )

    def __repr__(self):
        rope_type = "default"
        if self.rope_scaling:
            rope_type = self.rope_scaling.get(
                "rope_type", self.rope_scaling.get("type", "default")
            )
        return (
            f"d_head={self.d_head}, max_sequence_length={self.max_sequence_length}, "
            f"rope_theta={self.rope_theta}, rope_type={rope_type}, "
            f"cache_embeddings={self.cache_embeddings}, liger_kernel={self.liger_kernel is not None}"
        )

    def compute_embeddings_on_demand(
        self, position_ids: Tensor, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        """
        NEW STRATEGY: Compute cos/sin embeddings on-demand for specific positions.

        This avoids caching large tensors and allows torch.compile() to potentially:
        1. Recognize constant computations across layers
        2. Optimize or cache the computation automatically
        3. Better support torch.export() which struggles with cached state

        This implementation is optimized for torch.compile() by:
        - Using only tensor operations (no Python dicts or list comprehensions)
        - Computing embeddings directly for requested positions
        - Allowing the compiler to recognize and optimize repeated constant computations

        For inference with KV caching, position_ids often contains a single index
        (the current token position), making computation very efficient.

        Args:
            position_ids: Position indices tensor of shape (batch_size, seq_len) or (1, seq_len)
            device: Device to compute on
            dtype: Data type for the output tensors

        Returns:
            Tuple of (cos, sin) tensors of shape (batch_size, seq_len, d_head)
        """
        # Compute base inverse frequencies - shape: (d_head // 2,)
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.d_head, 2, device=device, dtype=torch.float32) / self.d_head)
        )

        # Apply scaling if specified
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
            if rope_type == "llama3":
                inv_freq = apply_llama3_scaling(inv_freq, self.rope_scaling)
            elif rope_type != "default":
                raise ValueError(
                    f"Unsupported rope_type: {rope_type}. Only 'llama3' is currently supported."
                )

        # Ensure position_ids is 2D: (batch_size, seq_len)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        # Convert position_ids to float for outer product
        # Shape: (batch_size, seq_len)
        position_ids_float = position_ids.float()

        # For each position in position_ids, compute frequencies
        # We need to handle batched position_ids properly
        # Reshape for batch processing: (batch_size * seq_len,)
        batch_size, seq_len = position_ids.shape
        positions_flat = position_ids_float.reshape(-1)

        # Compute frequencies for all positions: (batch_size * seq_len, d_head // 2)
        freqs = torch.outer(positions_flat, inv_freq)

        # Duplicate frequencies to match full dimension: (batch_size * seq_len, d_head)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Compute cos and sin, then reshape back to (batch_size, seq_len, d_head)
        cos = emb.cos().to(dtype=dtype).reshape(batch_size, seq_len, self.d_head)
        sin = emb.sin().to(dtype=dtype).reshape(batch_size, seq_len, self.d_head)

        return cos, sin

    def embeddings(self, device: torch.device):
        """
        OLD STRATEGY: Get cached embeddings, handling device movement.
        Only used when cache_embeddings=True.
        """
        if not self.cache_embeddings:
            raise RuntimeError(
                "embeddings() method should not be called when cache_embeddings=False. "
                "Use compute_embeddings_on_demand() instead."
            )

        if device == self.cos_cached.device:
            return self.cos_cached, self.sin_cached

        # If we were on "meta" and have moved to a real device, we need to initialize the embeddings
        if self.cos_cached.device == torch.device("meta"):
            with torch.device(device):
                self.cos_cached, self.sin_cached = precompute_cos_sin(
                    self.d_head, self.max_sequence_length, self.dtype, self.rope_theta, self.rope_scaling
                )
        else:
            # This can happen when using HF device_map = "auto"
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)

        return self.cos_cached, self.sin_cached 

    def __call__(
        self, q: Tensor, k: Tensor, position_ids: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE embedding to query and key

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

        # NEW STRATEGY: Compute embeddings on-demand
        if not self.cache_embeddings:
            # Create position_ids if not provided
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=q.device, dtype=torch.long).unsqueeze(0)

            # Compute cos/sin for the specific positions needed
            cos, sin = self.compute_embeddings_on_demand(position_ids, q.device, q.dtype)

            # Reshape for broadcasting: [batch_size, seq_len, 1, d_head]
            # cos/sin from compute_embeddings_on_demand are [batch_size, seq_len, d_head]
            cos = cos.unsqueeze(2)  # [batch_size, seq_len, 1, d_head]
            sin = sin.unsqueeze(2)  # [batch_size, seq_len, 1, d_head]

            # Apply rotation
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        # OLD STRATEGY: Use cached embeddings
        cos_cached, sin_cached = self.embeddings(k.device)

        if self.liger_kernel and q.is_cuda:
            return self.liger_kernel(
                q, k, cos_cached, sin_cached, position_ids
            )

        if position_ids is None:
            # Default behavior: use sequential positions
            assert (
                seq_len <= cos_cached.shape[0]
            ), f"seq_len {seq_len} > max_seq_len {cos_cached.shape[0]}"
            cos = cos_cached[:seq_len]
            sin = sin_cached[:seq_len]
            # Reshape cos/sin for broadcasting to match input tensor dimensions
            # Need cos/sin: [1, seq_len, 1, d_head] for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_head]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_head]
        else:
            cos = cos_cached[position_ids].unsqueeze(2)
            sin = sin_cached[position_ids].unsqueeze(2)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
