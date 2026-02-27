"""
Canon Pre-LN Transformer Layer.

Extends the standard Pre-LN transformer layer with Canon layers at
positions A (before attention) and C (before FFN).

From "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers" (Allen-Zhu, 2025).
"""

from typing import Callable, Optional

import torch
from torch import FloatTensor, nn

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


class CanonPreLNLayer(nn.Module):
    """Pre-LN transformer layer with Canon-A (pre-attention) and Canon-C (pre-FFN)."""

    def __init__(
        self,
        *,
        feedforward_factory: Callable,
        attention_factory: Callable,
        norm_factory: Callable,
        dropout: Optional[float] = 0.1,
        residual_dropout: Optional[float] = 0.0,
        canon_kernel: int = 4,
        canon_residual: bool = True,
        d_model: int,
        **kwargs,
    ):
        super().__init__()
        self.feedforward = feedforward_factory(**kwargs)
        self.attention = attention_factory(**kwargs)
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()
        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)
        if residual_dropout == 0.0:
            self.residual_dropout = nn.Identity()
        else:
            self.residual_dropout = nn.Dropout(residual_dropout)

        # Canon-A: after norm1, before attention
        self.canon_a = _CanonLayer(dim=d_model, kernel_size=canon_kernel, residual=canon_residual)
        # Canon-C: after norm2, before feedforward
        self.canon_c = _CanonLayer(dim=d_model, kernel_size=canon_kernel, residual=canon_residual)

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        residual = self.residual_dropout(x)
        x = self.norm1(x)
        x = self.canon_a(x)
        x = self.attention(x, **kwargs)
        x = residual + self.dropout(x)
        residual = self.residual_dropout(x)
        x = self.norm2(x)
        x = self.canon_c(x)
        x = self.feedforward(x, **kwargs)
        x = residual + self.dropout(x)
        return x
