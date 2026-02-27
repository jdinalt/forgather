"""
Canon Layer: Depthwise causal 1D convolution for local token mixing.

From "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers" (Allen-Zhu, 2025). Canon layers compute a causal
weighted sum of neighboring token representations using a depthwise
Conv1d with a small kernel (default K=4).

Each channel independently computes:
    h'_t = w_0 * h_t + w_1 * h_{t-1} + ... + w_{K-1} * h_{t-K+1}

With residual connection (default):
    output = h_t + h'_t

Includes an optimized Triton kernel that operates directly on [B, T, C]
layout with a streaming approach -- each program processes all time steps
sequentially with a register-based sliding window, eliminating transposes
and minimizing memory traffic.
"""

import torch
import torch.nn as nn

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Triton kernels for causal depthwise conv1d
#
# Streaming approach: each program handles one batch element and a block
# of channels, processing all T time steps sequentially. A K-element
# sliding window is maintained in registers for each channel.
#
# Weight layout matches nn.Conv1d: [C, 1, K] squeezed to [C, K], where
# w[c, K-1] is the weight for the current token and w[c, 0] for t-(K-1).
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
            buf0 = tl.load(
                base_x + t * stride_xt + c_offs * stride_xc,
                mask=c_mask, other=0.0,
            ).to(tl.float32)

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
        """Fused backward: computes both dx and partial dw in a single pass.

        dx uses reverse-time streaming with a sliding window.
        dw accumulates partial sums per (batch, channel_block), then uses
        atomic adds to sum across batch elements.
        """
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)

        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        # Load weights for dx computation
        w0 = tl.load(w_ptr + c_offs * K + (K - 1), mask=c_mask, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + c_offs * K + (K - 2), mask=c_mask, other=0.0).to(tl.float32) if K > 1 else tl.zeros([BLOCK_C], dtype=tl.float32)
        w2 = tl.load(w_ptr + c_offs * K + (K - 3), mask=c_mask, other=0.0).to(tl.float32) if K > 2 else tl.zeros([BLOCK_C], dtype=tl.float32)
        w3 = tl.load(w_ptr + c_offs * K + (K - 4), mask=c_mask, other=0.0).to(tl.float32) if K > 3 else tl.zeros([BLOCK_C], dtype=tl.float32)

        # dw accumulators (per-channel partial sums for this batch element)
        dw0 = tl.zeros([BLOCK_C], dtype=tl.float32)
        dw1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        dw2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        dw3 = tl.zeros([BLOCK_C], dtype=tl.float32)

        # Forward pass: read x and dout together, accumulate dw
        # We need x sliding window for dw, so process forward
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
            x_buf0 = tl.load(
                base_x + t * stride_xt + c_offs * stride_xc,
                mask=c_mask, other=0.0,
            ).to(tl.float32)

            dout_val = tl.load(
                base_d + t * stride_dt + c_offs * stride_dc,
                mask=c_mask, other=0.0,
            ).to(tl.float32)

            # dw[c, K-1-k] += dout[b,t,c] * x[b,t-k,c]
            dw0 += dout_val * x_buf0  # k=0, current token -> w[:, K-1]
            dw1 += dout_val * x_buf1  # k=1, prev token -> w[:, K-2]
            if K > 2:
                dw2 += dout_val * x_buf2
            if K > 3:
                dw3 += dout_val * x_buf3

        # Atomic-add partial dw sums (summing across batch elements)
        tl.atomic_add(dw_ptr + c_offs * K + (K - 1), dw0, mask=c_mask)
        if K > 1:
            tl.atomic_add(dw_ptr + c_offs * K + (K - 2), dw1, mask=c_mask)
        if K > 2:
            tl.atomic_add(dw_ptr + c_offs * K + (K - 3), dw2, mask=c_mask)
        if K > 3:
            tl.atomic_add(dw_ptr + c_offs * K + (K - 4), dw3, mask=c_mask)

        # dx: reverse-time streaming
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
            d_buf0 = tl.load(
                base_d + t * stride_dt + c_offs * stride_dc,
                mask=c_mask, other=0.0,
            ).to(tl.float32)

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
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, residual: bool):
        B, T, C = x.shape
        out = torch.empty_like(x)

        BLOCK_C = min(triton.next_power_of_2(C), 256)
        grid = (B, triton.cdiv(C, BLOCK_C))

        _causal_dconv1d_fwd_kernel[grid](
            x, weight, out,
            T, C, weight.shape[1],
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            RESIDUAL=residual,
            BLOCK_C=BLOCK_C,
        )
        ctx.save_for_backward(x, weight)
        ctx.residual = residual
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        B, T, C = x.shape
        K = weight.shape[1]

        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros(C, K, device=x.device, dtype=torch.float32)

        BLOCK_C = min(triton.next_power_of_2(C), 256)
        grid = (B, triton.cdiv(C, BLOCK_C))

        _causal_dconv1d_bwd_kernel[grid](
            grad_out, x, weight, grad_x, grad_w,
            T, C, K,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            RESIDUAL=ctx.residual,
            BLOCK_C=BLOCK_C,
        )

        return grad_x, grad_w.to(weight.dtype), None


def _causal_dconv1d(
    x: torch.Tensor, weight: torch.Tensor, residual: bool = True
) -> torch.Tensor:
    """Causal depthwise conv1d operating on [B, T, C] layout.

    Args:
        x: [B, T, C] input
        weight: [C, K] depthwise conv weights (Conv1d order)
        residual: add input as residual connection
    """
    return _CausalDConv1dFn.apply(x.contiguous(), weight.contiguous(), residual)


class CanonLayer(nn.Module):
    """Depthwise causal 1D convolution with optional residual connection."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        bias: bool = False,
        residual: bool = True,
        use_triton: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.residual = residual
        self.use_triton = use_triton

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=bias,
            padding=kernel_size - 1,
        )

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"kernel_size={self.kernel_size}, "
            f"bias={self.bias}, "
            f"residual={self.residual}, "
            f"use_triton={self.use_triton}"
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            [batch_size, seq_len, dim]
        """
        if self.use_triton and _HAS_TRITON and x.is_cuda and x.shape[-1] >= 384:
            return _causal_dconv1d(x, self.conv.weight.squeeze(1), self.residual)

        # Fallback: Conv1d expects [B, C, T]
        x_conv = x.transpose(1, 2)
        out = self.conv(x_conv)
        out = out[..., : x.shape[1]]
        out = out.transpose(1, 2)

        if self.residual:
            return x + out
        return out
