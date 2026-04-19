import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

# YaRN default bounds from arxiv.org/abs/2309.00071
_YARN_DEFAULT_BETA_FAST = 32
_YARN_DEFAULT_BETA_SLOW = 1

logger = logging.getLogger(__name__)


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


def apply_yarn_scaling(
    inv_freq: Tensor,
    rope_scaling: Dict[str, Any],
    d_head: int,
) -> Tuple[Tensor, float]:
    """
    Apply YaRN (Yet another RoPE extensioN) scaling. See arxiv.org/abs/2309.00071.

    Blends extrapolated (unscaled) and interpolated (divided by ``factor``)
    inverse frequencies using a linear ramp over the frequency index. The ramp
    bounds come from ``beta_fast`` / ``beta_slow`` via the "correction dim"
    formula (paper eq. 16): dim(r) = d * log(L / (2*pi*r)) / (2 * log(base)).

    Args:
        inv_freq: Base (unscaled) inverse frequencies of shape [d_head // 2].
        rope_scaling: Dict containing:
            - factor (required): Context-extension scaling factor.
            - original_max_position_embeddings (required): Pretraining context length.
            - beta_fast (default 32): Extrapolation-only frequency boundary.
            - beta_slow (default 1): Interpolation-only frequency boundary.
            - attention_factor (optional): Explicit override for the returned
              attention scaling; otherwise inferred from ``factor``.
            - mscale, mscale_all_dim (optional): HF-compatible ratio form for
              inferring ``attention_factor`` when both are provided.
        d_head: Head dimension (inv_freq length is d_head // 2).

    Returns:
        Tuple of (scaled inv_freq, attention_factor).
    """
    if "original_max_position_embeddings" not in rope_scaling:
        raise ValueError(
            "YaRN scaling requires 'original_max_position_embeddings' in rope_scaling."
        )
    factor = rope_scaling["factor"]
    original_max_pos = rope_scaling["original_max_position_embeddings"]
    beta_fast = rope_scaling.get("beta_fast") or _YARN_DEFAULT_BETA_FAST
    beta_slow = rope_scaling.get("beta_slow") or _YARN_DEFAULT_BETA_SLOW
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")

    # Recover the base used to build inv_freq: inv_freq[0] = 1.0, inv_freq[1] = base^(-2/d_head)
    # so base = (inv_freq[0] / inv_freq[-1]) ** (d_head / (d_head - 2)) for d_head > 2.
    # Simpler: use rope_scaling if present, else infer from inv_freq[1].
    if inv_freq.numel() >= 2:
        base = float(inv_freq[0] / inv_freq[1]) ** (d_head / 2.0)
    else:
        base = 10000.0

    def get_mscale(scale: float, m: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * m * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)

    def correction_dim(num_rotations: float) -> float:
        return (d_head * math.log(original_max_pos / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), d_head - 1)
    if low == high:
        high = low + 0.001

    ramp = torch.clamp(
        (torch.arange(d_head // 2, dtype=torch.float32, device=inv_freq.device) - low)
        / (high - low),
        0.0,
        1.0,
    )
    # extrapolation_factor=1 keeps high-frequency (fast) dims unscaled; =0 scales by 1/factor.
    extrapolation_factor = 1.0 - ramp
    inv_freq_interp = inv_freq / factor
    inv_freq_yarn = (
        inv_freq_interp * (1.0 - extrapolation_factor) + inv_freq * extrapolation_factor
    )
    return inv_freq_yarn, float(attention_factor)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.

    Computes (cos, sin) position embeddings from position_ids. Designed to be
    instantiated once on the model (in CasualLM) and called once per forward pass.
    The resulting (cos, sin) tuple is passed to attention layers via kwargs as
    ``position_embeddings``.

    Precision
    ---------
    Frequency computation (inv_freq, cos, sin) is always performed in float32
    with autocast disabled, as bf16 rounding in periodic functions produces
    large absolute errors at moderate-to-large positions.

    The ``float32_output`` flag controls whether cos/sin are returned in float32
    or cast to the input dtype. ``apply_rotary_pos_emb`` casts q/k to match,
    so this flag effectively controls the rotation precision:

    - **float32_output=True**: Rotation in float32. Recommended for long-context
      training (>4K positions) and when maximum precision is needed. Matches
      Flash Attention's Triton kernel behavior.
    - **float32_output=False** (default): Rotation in model dtype. Matches
      HF Transformers, torchtitan, and vLLM. Sufficient for most training
      scenarios and uses less memory.

    See ``docs/model-architecture.md`` for a detailed discussion of RoPE
    precision tradeoffs.

    Initialization
    --------------
    Follows the two-phase pattern for meta device support:

    - ``__init__`` allocates an empty ``inv_freq`` buffer (non-persistent).
    - ``reset_parameters()`` computes the actual frequency values.

    The weight initialization system calls ``reset_parameters()`` as its
    fallback for modules without ``init_prefix``. On meta device,
    ``reset_parameters()`` is a no-op; values are computed after the buffer
    is moved to a real device.

    See ``docs/configuration/model-initialization.md`` for the full
    initialization call chain.
    """

    inv_freq: Tensor

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int = 2048,
        rope_parameters: Optional[Dict[str, Any]] = None,
        rope_theta: Optional[float] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        float32_output: bool = False,
        device=None,
    ):
        """
        Args:
            hidden_size: Model hidden dimension.
            num_attention_heads: Number of attention heads (used to compute d_head).
            max_position_embeddings: Maximum sequence length (informational only,
                does not limit forward pass).
            rope_parameters: Unified dict with rope_theta, rope_type, and scaling
                params (HF v5 format). Takes priority over legacy args.
            rope_theta: (Legacy) Base frequency. Use rope_parameters instead.
            rope_scaling: (Legacy) Scaling dict. Use rope_parameters instead.
            float32_output: If True (default), return cos/sin in float32 so that
                the rotation in ``apply_rotary_pos_emb`` is computed in float32.
                If False, cast cos/sin to the input dtype before returning, so
                the rotation stays in the model's native precision (matching HF
                Transformers' default). Float32 is recommended for training
                stability, especially with long sequences and bf16/fp16.
            device: Device for initial buffer allocation.
        """
        super().__init__()

        self.d_head = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings

        # Extract parameters from rope_parameters dict (v5 format)
        # or fall back to legacy rope_theta/rope_scaling args
        if rope_parameters is not None:
            self.rope_theta = rope_parameters.get("rope_theta", 10000.0)
            self.rope_type = rope_parameters.get("rope_type", "default")
            self.rope_scaling = rope_parameters
        else:
            self.rope_theta = rope_theta if rope_theta is not None else 10000.0
            self.rope_type = "default"
            if rope_scaling is not None:
                self.rope_type = rope_scaling.get(
                    "rope_type", rope_scaling.get("type", "default")
                )
            self.rope_scaling = rope_scaling

        # Attention scaling factor (defaults 1.0; used by YaRN/LongRoPE)
        self.attention_scaling = 1.0
        self.float32_output = float32_output

        # Allocate buffer; actual values are filled by reset_parameters().
        self.register_buffer(
            "inv_freq",
            torch.empty(self.d_head // 2, device=device),
            persistent=False,
        )

    def extra_repr(self) -> str:
        return (
            f"d_head={self.d_head}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"rope_theta={self.rope_theta}, "
            f"rope_type={self.rope_type!r}, "
            f"float32_output={self.float32_output}"
        )

    def reset_parameters(self):
        """Compute inverse frequencies and fill the inv_freq buffer.

        Called by the weight initialization system (``init_weights_by_regex``
        falls back to ``reset_parameters()`` for modules without an
        ``init_prefix``). Also handles meta-device construction: HF's loading
        pipeline moves non-persistent buffers from meta to the target device,
        then calls ``_init_weights`` -> ``reset_parameters()`` to populate them.
        """
        if self.inv_freq.is_meta:
            return

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.d_head,
                    2,
                    dtype=torch.float32,
                    device=self.inv_freq.device,
                )
                / self.d_head
            )
        )

        if self.rope_type == "llama3" and self.rope_scaling is not None:
            inv_freq = apply_llama3_scaling(inv_freq, self.rope_scaling)
        elif self.rope_type == "yarn" and self.rope_scaling is not None:
            inv_freq, attention_factor = apply_yarn_scaling(
                inv_freq, self.rope_scaling, self.d_head
            )
            self.attention_scaling = attention_factor
        elif self.rope_type not in ("default", "llama3", "yarn"):
            raise ValueError(
                f"Unsupported rope_type: {self.rope_type}. "
                "Supported: 'default', 'llama3', 'yarn'."
            )

        self.inv_freq.copy_(inv_freq)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute rotary position embeddings.

        Args:
            x: Input tensor, used for device and (when ``float32_output=False``) dtype.
            position_ids: Position indices of shape ``[batch_size, seq_len]``.

        Returns:
            Tuple of ``(cos, sin)``, each of shape ``[batch_size, seq_len, d_head]``.
            Dtype is float32 when ``float32_output=True``, otherwise ``x.dtype``.
        """
        # [1, d_head//2, 1] -> [batch, d_head//2, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 for numerical stability (disable autocast).
        # Note: torch.set_float32_matmul_precision() does NOT affect this
        # matmul because the inner (contraction) dimension is 1 -- each
        # output is a single multiply with no accumulation, so TF32/medium
        # produce identical results to full float32.
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # [batch, d_head//2, seq] -> [batch, seq, d_head//2]
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        if not self.float32_output:
            cos = cos.to(dtype=x.dtype)
            sin = sin.to(dtype=x.dtype)

        return cos, sin


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    position_embeddings: Tuple[Tensor, Tensor] = None,  # type: ignore[assignment]
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Designed to be used as a ``pos_encoder`` callable in
    :class:`CausalMultiheadAttn`. Receives ``position_embeddings`` via kwargs
    from the attention forward pass (originally computed by
    :class:`RotaryEmbedding` in ``CasualLM``).

    The rotation precision is controlled by :class:`RotaryEmbedding`'s
    ``float32_output`` flag:

    - **float32_output=True** (default): cos/sin arrive in float32, q/k are
      upcast to float32 for the rotation, then cast back. Best precision,
      important for long sequences and bf16/fp16 training.
    - **float32_output=False**: cos/sin arrive in the model dtype, q/k stay
      in their native dtype. Matches HF Transformers' default behavior.
      Faster, less memory, but may lose precision.

    In both cases, q/k are cast to match the dtype of cos/sin, and the
    result is cast back to the original q/k dtype. When dtypes already
    match, the casts are no-ops.

    Args:
        q: Query tensor of shape ``[batch, seq, heads, d_head]``.
        k: Key tensor of shape ``[batch, seq, heads, d_head]``.
        position_embeddings: Tuple of ``(cos, sin)``, each
            ``[batch, seq, d_head]``. Computed by :class:`RotaryEmbedding`.
        **kwargs: Ignored (absorbs other forward kwargs).

    Returns:
        Tuple of ``(rotated_q, rotated_k)`` with same shapes as input.
    """
    if position_embeddings is None:
        raise ValueError(
            "position_embeddings is None. When using apply_rotary_pos_emb as "
            "pos_encoder, ensure that CasualLM is configured with a rotary_emb "
            "module (RotaryEmbedding) so that position_embeddings are computed "
            "and passed through kwargs."
        )
    cos, sin = position_embeddings
    # Broadcast over heads: [batch, seq, 1, d_head]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    # Cast q/k to match cos/sin dtype. When RotaryEmbedding.float32_output
    # is True, this upcasts to float32. When False, this is a no-op.
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
