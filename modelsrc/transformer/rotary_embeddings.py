import logging
import math
from typing import Any, Dict, Iterable, Optional, Tuple

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


class RotaryPE:
    """
    Real-valued RoPE positional encoder module

    Supports two computation strategies:
    - Cached (cache_embeddings=True, default): Precompute cos/sin for all positions once.
      Best performance for dynamic KV cache scenarios. Caches inverse frequencies
      to avoid recomputation on device movements.
    - On-demand (cache_embeddings=False): Compute cos/sin on each forward pass.
      Supports torch.export() which struggles with cached state. When combined with
      compile_on_demand=True (default), provides reasonable performance through
      kernel fusion, though still ~15% slower than cached for dynamic KV cache.

    Performance (Llama 1B, dynamic KV cache, 512 token generation):
    - cache_embeddings=True: ~135 tok/s (recommended default)
    - cache_embeddings=False, compile_on_demand=True: ~115 tok/s
    - cache_embeddings=False, compile_on_demand=False: ~92 tok/s

    Note: With static KV cache, both strategies perform equally (~160 tok/s).

    For complex-valued RoPE (experimental), see complex_rotary_embeddings.py
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 2048,
        rope_parameters: Optional[Dict[str, Any]] = None,
        rope_theta: Optional[float] = None,  # Deprecated: use rope_parameters
        rope_scaling: Optional[
            Dict[str, Any]
        ] = None,  # Deprecated: use rope_parameters
        use_liger: bool = False,
        cache_embeddings: bool = True,
        compile_on_demand: bool = True,
    ):
        """Initialize RoPE embeddings.

        Args:
            rope_parameters: Dictionary with rope_theta, rope_type, and scaling params.
                            Example: {'rope_theta': 10000.0, 'rope_type': 'llama3',
                                     'factor': 32.0, 'low_freq_factor': 1.0, ...}
            rope_theta: (Deprecated) Use rope_parameters instead
            rope_scaling: (Deprecated) Use rope_parameters instead
        """
        # Extract parameters from rope_parameters dict (v5.0 format)
        # Or fall back to legacy rope_theta/rope_scaling params
        if rope_parameters is not None:
            extracted_theta = rope_parameters.get("rope_theta", 10000.0)

            # Build rope_scaling dict from rope_parameters
            # (Internal implementation still uses separate rope_scaling)
            extracted_scaling = {}
            if "rope_type" in rope_parameters:
                extracted_scaling["rope_type"] = rope_parameters["rope_type"]
            for key in [
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
                "attention_factor",
                "beta_fast",
                "beta_slow",
                "short_factor",
                "long_factor",
            ]:
                if key in rope_parameters:
                    extracted_scaling[key] = rope_parameters[key]

            if not extracted_scaling:
                extracted_scaling = None
        else:
            # Legacy mode: use separate rope_theta/rope_scaling params
            extracted_theta = rope_theta if rope_theta is not None else 10000.0
            extracted_scaling = rope_scaling

        self.d_head = hidden_size // num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rope_theta = extracted_theta
        self.rope_scaling = extracted_scaling
        self.dtype = torch.get_default_dtype()
        self.cache_embeddings = cache_embeddings
        self.compile_on_demand = compile_on_demand

        # Cache inv_freq to avoid recomputation on device movements
        # Stored as instance variable (not buffer) to avoid library issues
        self._inv_freq_cached = None
        if self.cache_embeddings:
            self._inv_freq_cached = self._compute_inv_freq()

        # Precompute cos/sin tensors once for the entire model
        if self.cache_embeddings:
            self.cos_cached, self.sin_cached = self._precompute_with_cached_inv_freq(
                self.max_sequence_length, self.dtype
            )
        else:
            # Store None to indicate we're using on-demand computation
            self.cos_cached = None
            self.sin_cached = None

        # Compile the on-demand computation for performance
        # This provides kernel fusion benefits without requiring full model compilation
        self._compute_fn = None
        if not self.cache_embeddings and self.compile_on_demand:
            # Compile the computation function for better performance
            # This will fuse operations and eliminate Python overhead
            self._compute_fn = torch.compile(self._compute_embeddings_core)

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
            f"cache_embeddings={self.cache_embeddings}, compile_on_demand={self.compile_on_demand}, "
            f"liger_kernel={self.liger_kernel is not None}"
        )

    def _compute_inv_freq(self) -> Tensor:
        """
        Compute inverse frequencies once and cache them.
        This avoids recomputing inv_freq on device movements.

        Returns:
            Inverse frequencies tensor of shape (d_head // 2,)
        """
        # Compute base inverse frequencies
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.d_head, 2).float() / self.d_head)
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

        return inv_freq

    def _precompute_with_cached_inv_freq(
        self, max_len: int, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        """
        Precompute cos/sin using cached inv_freq.
        This reuses the inv_freq computation across device movements.

        Args:
            max_len: Maximum sequence length
            dtype: Data type for output tensors

        Returns:
            Tuple of (cos, sin) tensors of shape (max_len, d_head)
        """
        # Get cached inv_freq, moving to the correct device if needed
        inv_freq = self._inv_freq_cached
        if inv_freq.device == torch.device("meta"):
            # Need to initialize on a real device
            inv_freq = self._compute_inv_freq()
            self._inv_freq_cached = inv_freq

        # Create position indices
        t = torch.arange(max_len, device=inv_freq.device, dtype=torch.float32)

        # Compute frequencies for each position
        freqs = torch.outer(t, inv_freq)

        # Duplicate frequencies to match full dimension
        emb = torch.cat((freqs, freqs), dim=-1)

        # Compute cos and sin
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)

        return cos, sin

    def _compute_embeddings_core(
        self, position_ids: Tensor, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        """
        Core computation logic for on-demand RoPE embeddings.
        This method can be compiled with torch.compile() for better performance.

        Using only tensor operations (no Python dicts or list comprehensions)
        to allow the compiler to fuse operations and eliminate overhead.

        Args:
            position_ids: Position indices tensor of shape (batch_size, seq_len) or (1, seq_len)
            device: Device to compute on
            dtype: Data type for the output tensors

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

    def compute_embeddings_on_demand(
        self, position_ids: Tensor, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute cos/sin embeddings on-demand for specific positions.

        This avoids caching large tensors and allows torch.compile() to potentially:
        1. Recognize constant computations across layers
        2. Optimize or cache the computation automatically
        3. Better support torch.export() which struggles with cached state

        The core computation can be compiled (via compile_on_demand=True) to provide:
        - Kernel fusion for all tensor operations
        - Elimination of Python interpreter overhead
        - Optimized memory access patterns

        For inference with KV caching, position_ids often contains a single index
        (the current token position), making this very efficient when compiled.

        Args:
            position_ids: Position indices tensor of shape (batch_size, seq_len) or (1, seq_len)
            device: Device to compute on
            dtype: Data type for the output tensors

        Returns:
            Tuple of (cos, sin) tensors of shape (batch_size, seq_len, d_head)
        """
        # Use compiled version if available, otherwise use eager execution
        if self._compute_fn is not None:
            return self._compute_fn(position_ids, device, dtype)
        else:
            return self._compute_embeddings_core(position_ids, device, dtype)

    def embeddings(self, device: torch.device):
        """
        Get cached embeddings, handling device movement.
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
                # Use cached inv_freq to avoid recomputing
                self.cos_cached, self.sin_cached = (
                    self._precompute_with_cached_inv_freq(
                        self.max_sequence_length, self.dtype
                    )
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

        # Compute embeddings on-demand
        if not self.cache_embeddings:
            # Create position_ids if not provided
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, device=q.device, dtype=torch.long
                ).unsqueeze(0)

            # Compute cos/sin for the specific positions needed
            cos, sin = self.compute_embeddings_on_demand(
                position_ids, q.device, q.dtype
            )

            # Reshape for broadcasting: [batch_size, seq_len, 1, d_head]
            # cos/sin from compute_embeddings_on_demand are [batch_size, seq_len, d_head]
            cos = cos.unsqueeze(2)  # [batch_size, seq_len, 1, d_head]
            sin = sin.unsqueeze(2)  # [batch_size, seq_len, 1, d_head]

            # Apply rotation
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        # Use cached embeddings
        cos_cached, sin_cached = self.embeddings(k.device)

        if self.liger_kernel and q.is_cuda:
            return self.liger_kernel(q, k, cos_cached, sin_cached, position_ids)

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

        return q_embed.to(dtype=q.dtype), k_embed.to(dtype=q.dtype)
