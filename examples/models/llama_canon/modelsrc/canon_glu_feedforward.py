"""
Canon GLU Feedforward Layer.

Extends the standard GLU feedforward with Canon-D: a depthwise causal
convolution applied to concatenated gate and up projections before the
gating activation.

From "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers" (Allen-Zhu, 2025).
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import FloatTensor, Tensor, nn

logger = logging.getLogger(__name__)

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Triton fused SiLU * up kernel
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _silu_mul_fwd_kernel(
        gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        out = up * gate * sig
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _silu_mul_bwd_kernel(
        gate_ptr, up_ptr, grad_out_ptr, grad_gate_ptr, grad_up_ptr,
        n_elements, BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        g_up = grad_out * silu_gate
        g_gate = grad_out * up * sig * (1.0 + gate * (1.0 - sig))
        tl.store(grad_up_ptr + offs, g_up, mask=mask)
        tl.store(grad_gate_ptr + offs, g_gate, mask=mask)


class _FusedSiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        assert gate.is_contiguous() and up.is_contiguous()
        out = torch.empty_like(gate)
        n_elements = gate.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _silu_mul_fwd_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple:
        gate, up = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        n_elements = gate.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _silu_mul_bwd_kernel[grid](
            gate, up, grad_out, grad_gate, grad_up, n_elements, BLOCK_SIZE
        )
        return grad_gate, grad_up


_TRITON_FUSED_OPS = {}
if _HAS_TRITON:
    _TRITON_FUSED_OPS = {nn.SiLU: _FusedSiLUMul}


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


class CanonGLUFeedforwardLayer(nn.Module):
    """GLU feedforward with Canon-D on gate/up projections."""

    def __init__(
        self,
        d_model: int,
        d_feedforward: int,
        *,
        activation_factory: Optional[Callable] = lambda: nn.SiLU(),
        dropout: Optional[float] = 0.1,
        use_triton: bool = True,
        canon_kernel: int = 4,
        canon_residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not _HAS_TRITON:
            use_triton = False
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        self.up_proj = nn.Linear(self.d_model, self.d_feedforward, bias=False)
        setattr(self.up_proj, "init_prefix", "ff.up_proj")
        self.gate_proj = nn.Linear(self.d_model, self.d_feedforward, bias=False)
        setattr(self.gate_proj, "init_prefix", "ff.gate_proj")
        self.down_proj = nn.Linear(self.d_feedforward, self.d_model, bias=False)
        setattr(self.down_proj, "init_prefix", "ff.down_proj")
        self.activation = activation_factory()
        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

        self._fused_op = None
        if use_triton:
            activation_type = type(self.activation)
            if _HAS_TRITON and activation_type in _TRITON_FUSED_OPS:
                self._fused_op = _TRITON_FUSED_OPS[activation_type]

        # Canon-D: applied to concatenated [gate, up] projections
        self.canon_d = _CanonLayer(
            dim=2 * d_feedforward, kernel_size=canon_kernel, residual=canon_residual,
        )

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, d_feedforward={self.d_feedforward}, "
            f"use_triton={self._fused_op is not None}"
        )

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Canon-D: apply depthwise causal conv to gate and up separately
        # (depthwise conv is channel-independent, so this is mathematically
        # equivalent to cat+conv+split but avoids the concatenation overhead)
        w = self.canon_d.conv.weight  # [2 * d_feedforward, 1, K]
        K = w.shape[2]
        residual = self.canon_d.residual
        d_ff = self.d_feedforward
        gate = _apply_canon_conv(gate, w[:d_ff], K, residual)
        up = _apply_canon_conv(up, w[d_ff:], K, residual)

        if self._fused_op is not None and gate.is_cuda:
            x = self._fused_op.apply(gate.contiguous(), up.contiguous())
        else:
            x = up * self.activation(gate)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
