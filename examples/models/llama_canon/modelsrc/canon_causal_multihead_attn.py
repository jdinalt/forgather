"""
Canon Causal Multi-Head Attention.

Extends standard causal multi-head attention with Canon-B: a depthwise
causal convolution applied to concatenated Q, K, V projections before
attention computation.

From "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers" (Allen-Zhu, 2025).
"""

import math
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch import FloatTensor, nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Triton kernels for causal depthwise conv1d (self-contained copy)
# Streaming approach: processes all T steps sequentially per program.
# Weight layout matches nn.Conv1d: [C, K] where w[c, K-1] = current token.
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _causal_dconv1d_fwd_kernel(
        x_ptr, w_ptr, out_ptr,
        T, C, K: tl.constexpr,
        stride_xb, stride_xt, stride_xc,
        stride_ob, stride_ot, stride_oc,
        RESIDUAL: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        w0 = tl.load(w_ptr + c_offs * K + (K - 1), mask=c_mask, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + c_offs * K + (K - 2), mask=c_mask, other=0.0).to(tl.float32) if K > 1 else tl.zeros([BLOCK_C], dtype=tl.float32)
        w2 = tl.load(w_ptr + c_offs * K + (K - 3), mask=c_mask, other=0.0).to(tl.float32) if K > 2 else tl.zeros([BLOCK_C], dtype=tl.float32)
        w3 = tl.load(w_ptr + c_offs * K + (K - 4), mask=c_mask, other=0.0).to(tl.float32) if K > 3 else tl.zeros([BLOCK_C], dtype=tl.float32)
        buf0 = tl.zeros([BLOCK_C], dtype=tl.float32)
        buf1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        buf2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        buf3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        base_x = x_ptr + pid_b * stride_xb
        base_o = out_ptr + pid_b * stride_ob
        for t in range(T):
            buf3 = buf2
            buf2 = buf1
            buf1 = buf0
            buf0 = tl.load(base_x + t * stride_xt + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float32)
            acc = w0 * buf0 + w1 * buf1
            if K > 2:
                acc += w2 * buf2
            if K > 3:
                acc += w3 * buf3
            if RESIDUAL:
                acc += buf0
            tl.store(base_o + t * stride_ot + c_offs * stride_oc, acc, mask=c_mask)

    @triton.jit
    def _causal_dconv1d_bwd_kernel(
        dout_ptr, x_ptr, w_ptr, dx_ptr, dw_ptr,
        T, C, K: tl.constexpr,
        stride_db, stride_dt, stride_dc,
        stride_xb, stride_xt, stride_xc,
        RESIDUAL: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        w0 = tl.load(w_ptr + c_offs * K + (K - 1), mask=c_mask, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + c_offs * K + (K - 2), mask=c_mask, other=0.0).to(tl.float32) if K > 1 else tl.zeros([BLOCK_C], dtype=tl.float32)
        w2 = tl.load(w_ptr + c_offs * K + (K - 3), mask=c_mask, other=0.0).to(tl.float32) if K > 2 else tl.zeros([BLOCK_C], dtype=tl.float32)
        w3 = tl.load(w_ptr + c_offs * K + (K - 4), mask=c_mask, other=0.0).to(tl.float32) if K > 3 else tl.zeros([BLOCK_C], dtype=tl.float32)
        dw0 = tl.zeros([BLOCK_C], dtype=tl.float32)
        dw1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        dw2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        dw3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        x_buf0 = tl.zeros([BLOCK_C], dtype=tl.float32)
        x_buf1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        x_buf2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        x_buf3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        base_x = x_ptr + pid_b * stride_xb
        base_d = dout_ptr + pid_b * stride_db
        for t in range(T):
            x_buf3 = x_buf2
            x_buf2 = x_buf1
            x_buf1 = x_buf0
            x_buf0 = tl.load(base_x + t * stride_xt + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float32)
            dout_val = tl.load(base_d + t * stride_dt + c_offs * stride_dc, mask=c_mask, other=0.0).to(tl.float32)
            dw0 += dout_val * x_buf0
            dw1 += dout_val * x_buf1
            if K > 2:
                dw2 += dout_val * x_buf2
            if K > 3:
                dw3 += dout_val * x_buf3
        tl.atomic_add(dw_ptr + c_offs * K + (K - 1), dw0, mask=c_mask)
        if K > 1:
            tl.atomic_add(dw_ptr + c_offs * K + (K - 2), dw1, mask=c_mask)
        if K > 2:
            tl.atomic_add(dw_ptr + c_offs * K + (K - 3), dw2, mask=c_mask)
        if K > 3:
            tl.atomic_add(dw_ptr + c_offs * K + (K - 4), dw3, mask=c_mask)
        d_buf0 = tl.zeros([BLOCK_C], dtype=tl.float32)
        d_buf1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        d_buf2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        d_buf3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        base_dx = dx_ptr + pid_b * stride_xb
        for t_rev in range(T):
            t = T - 1 - t_rev
            d_buf3 = d_buf2
            d_buf2 = d_buf1
            d_buf1 = d_buf0
            d_buf0 = tl.load(base_d + t * stride_dt + c_offs * stride_dc, mask=c_mask, other=0.0).to(tl.float32)
            acc = w0 * d_buf0 + w1 * d_buf1
            if K > 2:
                acc += w2 * d_buf2
            if K > 3:
                acc += w3 * d_buf3
            if RESIDUAL:
                acc += d_buf0
            tl.store(base_dx + t * stride_xt + c_offs * stride_xc, acc, mask=c_mask)


class _CausalDConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, residual):
        B, T, C = x.shape
        out = torch.empty_like(x)
        BLOCK_C = min(triton.next_power_of_2(C), 256)
        grid = (B, triton.cdiv(C, BLOCK_C))
        _causal_dconv1d_fwd_kernel[grid](
            x, weight, out, T, C, weight.shape[1],
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            RESIDUAL=residual, BLOCK_C=BLOCK_C,
        )
        ctx.save_for_backward(x, weight)
        ctx.residual = residual
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        B, T, C = x.shape
        K = weight.shape[1]
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros(C, K, device=x.device, dtype=torch.float32)
        BLOCK_C = min(triton.next_power_of_2(C), 256)
        grid = (B, triton.cdiv(C, BLOCK_C))
        _causal_dconv1d_bwd_kernel[grid](
            grad_out, x, weight, grad_x, grad_w, T, C, K,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            RESIDUAL=ctx.residual, BLOCK_C=BLOCK_C,
        )
        return grad_x, grad_w.to(weight.dtype), None


def _apply_canon_conv(x, weight, K, residual):
    """Apply causal depthwise conv to [B, T, C] tensor.

    Uses Triton kernel on CUDA, Conv1d fallback on CPU.
    weight: [C, 1, K] slice from the canon layer's Conv1d weight.
    """
    C = x.shape[-1]
    if _HAS_TRITON and x.is_cuda and C >= 384:
        return _CausalDConv1dFn.apply(
            x.contiguous(), weight.squeeze(1).contiguous(), residual,
        )
    # Conv1d fallback
    out = F.conv1d(x.transpose(1, 2), weight, groups=C, padding=K - 1)
    out = out[..., : x.shape[1]].transpose(1, 2)
    if residual:
        return x + out
    return out


class _CanonLayer(nn.Module):
    """Depthwise causal 1D convolution with optional residual connection."""

    def __init__(self, dim: int, kernel_size: int = 4, residual: bool = True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.residual = residual
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=kernel_size,
            groups=dim, bias=False, padding=kernel_size - 1,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if _HAS_TRITON and x.is_cuda and x.shape[-1] >= 384:
            return _CausalDConv1dFn.apply(
                x.contiguous(), self.conv.weight.squeeze(1).contiguous(), self.residual,
            )
        x_conv = x.transpose(1, 2)
        out = self.conv(x_conv)
        out = out[..., : x.shape[1]]
        out = out.transpose(1, 2)
        if self.residual:
            return x + out
        return out


class CanonCausalMultiheadAttn(nn.Module):
    """Causal multi-head attention with Canon-B on QKV projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        attn_implementation: str,
        attn_functions: Optional[dict[str, Callable]] = None,
        config: Any = None,
        num_kv_heads: Optional[int] = None,
        pos_encoder: Optional[Callable] = None,
        bias: bool = False,
        dropout: float = 0.0,
        qk_norm_factory: Optional[Callable] = None,
        layer_idx: int,
        sliding_window: Optional[int] = None,
        canon_kernel: int = 4,
        canon_residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.pos_encoder = pos_encoder

        if attn_implementation == "flash_attention_2":
            self.is_causal = True
        self.config = config
        self.sliding_window = sliding_window
        self.layer_idx = layer_idx
        self.attn_implementation = attn_implementation

        if attn_functions and attn_implementation in attn_functions:
            self.attn_fn = attn_functions[attn_implementation]
        else:
            self.attn_fn = ALL_ATTENTION_FUNCTIONS[attn_implementation]

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"
        assert (
            num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.query_linear = nn.Linear(d_model, self.num_heads * self.d_head, bias=bias)
        setattr(self.query_linear, "init_prefix", "attn.query")

        self.key_linear = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=bias)
        setattr(self.key_linear, "init_prefix", "attn.key")

        self.value_linear = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=bias)
        setattr(self.value_linear, "init_prefix", "attn.value")

        self.output_linear = nn.Linear(self.num_heads * self.d_head, d_model, bias=bias)
        setattr(self.output_linear, "init_prefix", "attn.output")

        self.dropout_p = dropout

        if qk_norm_factory:
            self.q_norm = qk_norm_factory(normalized_shape=self.d_head)
            self.k_norm = qk_norm_factory(normalized_shape=self.d_head)
        else:
            self.q_norm = None
            self.k_norm = None

        # Canon-B: applied to concatenated Q, K, V
        total_qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.d_head
        self.canon_b = _CanonLayer(
            dim=total_qkv_dim, kernel_size=canon_kernel, residual=canon_residual,
        )

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"attn_implementation={self.attn_implementation}, "
            f"dropout_p={self.dropout_p}"
        )

    def forward(
        self,
        hidden_states: FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> FloatTensor:
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.d_head)

        query = self.query_linear(hidden_states)
        key = self.key_linear(hidden_states)
        value = self.value_linear(hidden_states)

        # Canon-B: apply depthwise causal conv to Q, K, V separately
        # (depthwise conv is channel-independent, so this is mathematically
        # equivalent to cat+conv+split but avoids the concatenation overhead)
        q_dim = self.num_heads * self.d_head
        k_dim = self.num_kv_heads * self.d_head
        w = self.canon_b.conv.weight  # [total_qkv_dim, 1, K]
        K = w.shape[2]
        residual = self.canon_b.residual
        query = _apply_canon_conv(query, w[:q_dim], K, residual)
        key = _apply_canon_conv(key, w[q_dim:q_dim + k_dim], K, residual)
        value = _apply_canon_conv(value, w[q_dim + k_dim:], K, residual)

        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        if self.pos_encoder:
            query, key = self.pos_encoder(query, key, position_ids)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs
            )

        attended_values, attn_weights = self.attn_fn(
            module=self,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=(self.dropout_p if self.training else 0.0),
            scaling=self.scale,
            config=self.config,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attended_values = attended_values.reshape(batch_size, seq_len, -1).contiguous()
        return self.output_linear(attended_values)
