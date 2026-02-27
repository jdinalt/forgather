# pyright: reportPossiblyUnboundVariable=false
import logging
from typing import Callable, Optional

import torch
from torch import FloatTensor, Tensor, nn

logger = logging.getLogger(__name__)

# ============================================================
# Triton kernels for fused activation * gate operations
# ============================================================

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    pass  # Triton kernels guarded by _HAS_TRITON checks


if _HAS_TRITON:

    @triton.jit
    def _silu_mul_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SiLU(gate) * up forward kernel."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # silu(gate) = gate * sigmoid(gate)
        sig = tl.sigmoid(gate)
        out = up * gate * sig

        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _silu_mul_bwd_kernel(
        gate_ptr,
        up_ptr,
        grad_out_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused backward for SiLU(gate) * up.

        d(out)/d(up) = silu(gate) = gate * sigmoid(gate)
        d(out)/d(gate) = up * d(silu)/d(gate)
                       = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        """
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

    @triton.jit
    def _relu_mul_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused ReLU(gate) * up forward kernel."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # relu(gate) = max(0, gate)
        relu_gate = tl.maximum(gate, 0.0)
        out = up * relu_gate

        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _relu_mul_bwd_kernel(
        gate_ptr,
        up_ptr,
        grad_out_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused backward for ReLU(gate) * up.

        d(out)/d(up) = relu(gate) = max(0, gate)
        d(out)/d(gate) = up * d(relu)/d(gate) = up * (gate > 0)
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        relu_gate = tl.maximum(gate, 0.0)
        gate_positive = (gate > 0.0).to(tl.float32)

        g_up = grad_out * relu_gate
        g_gate = grad_out * up * gate_positive

        tl.store(grad_up_ptr + offs, g_up, mask=mask)
        tl.store(grad_gate_ptr + offs, g_gate, mask=mask)

    @triton.jit
    def _gelu_mul_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused GELU(gate) * up forward kernel.

        GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # GELU(gate) = 0.5 * gate * (1 + erf(gate / sqrt(2)))
        SQRT_2_INV: tl.constexpr = 0.7071067811865476
        erf_val = tl.math.erf(gate * SQRT_2_INV)
        gelu_gate = 0.5 * gate * (1.0 + erf_val)
        out = up * gelu_gate

        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _gelu_mul_bwd_kernel(
        gate_ptr,
        up_ptr,
        grad_out_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused backward for GELU(gate) * up.

        d(GELU)/d(gate) = 0.5 * (1 + erf(g/sqrt2)) + g * exp(-g^2/2) / sqrt(2*pi)
        d(out)/d(up) = GELU(gate)
        d(out)/d(gate) = up * d(GELU)/d(gate)
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        SQRT_2_INV: tl.constexpr = 0.7071067811865476
        INV_SQRT_2PI: tl.constexpr = 0.3989422804014327

        erf_val = tl.math.erf(gate * SQRT_2_INV)
        cdf = 0.5 * (1.0 + erf_val)
        pdf = INV_SQRT_2PI * tl.exp(-0.5 * gate * gate)
        gelu_gate = gate * cdf

        g_up = grad_out * gelu_gate
        g_gate = grad_out * up * (cdf + gate * pdf)

        tl.store(grad_up_ptr + offs, g_up, mask=mask)
        tl.store(grad_gate_ptr + offs, g_gate, mask=mask)


# Map activation types to their fused implementations
_TRITON_FUSED_OPS: dict = {}

if _HAS_TRITON:

    class _FusedSiLUMul(torch.autograd.Function):
        """Fused SiLU(gate) * up with custom backward."""

        @staticmethod
        def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:  # type: ignore[override]
            assert gate.is_contiguous() and up.is_contiguous()
            out = torch.empty_like(gate)
            n_elements = gate.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # type: ignore[union-attr]
            _silu_mul_fwd_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE)
            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx, grad_out: Tensor) -> tuple:  # type: ignore[override]
            gate, up = ctx.saved_tensors
            grad_out = grad_out.contiguous()
            grad_gate = torch.empty_like(gate)
            grad_up = torch.empty_like(up)
            n_elements = gate.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # type: ignore[union-attr]
            _silu_mul_bwd_kernel[grid](
                gate, up, grad_out, grad_gate, grad_up, n_elements, BLOCK_SIZE
            )
            return grad_gate, grad_up

    class _FusedReLUMul(torch.autograd.Function):
        """Fused ReLU(gate) * up with custom backward."""

        @staticmethod
        def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:  # type: ignore[override]
            assert gate.is_contiguous() and up.is_contiguous()
            out = torch.empty_like(gate)
            n_elements = gate.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # type: ignore[union-attr]
            _relu_mul_fwd_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE)
            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx, grad_out: Tensor) -> tuple:  # type: ignore[override]
            gate, up = ctx.saved_tensors
            grad_out = grad_out.contiguous()
            grad_gate = torch.empty_like(gate)
            grad_up = torch.empty_like(up)
            n_elements = gate.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # type: ignore[union-attr]
            _relu_mul_bwd_kernel[grid](
                gate, up, grad_out, grad_gate, grad_up, n_elements, BLOCK_SIZE
            )
            return grad_gate, grad_up

    class _FusedGELUMul(torch.autograd.Function):
        """Fused GELU(gate) * up with custom backward."""

        @staticmethod
        def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:  # type: ignore[override]
            assert gate.is_contiguous() and up.is_contiguous()
            out = torch.empty_like(gate)
            n_elements = gate.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # type: ignore[union-attr]
            _gelu_mul_fwd_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE)
            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx, grad_out: Tensor) -> tuple:  # type: ignore[override]
            gate, up = ctx.saved_tensors
            grad_out = grad_out.contiguous()
            grad_gate = torch.empty_like(gate)
            grad_up = torch.empty_like(up)
            n_elements = gate.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # type: ignore[union-attr]
            _gelu_mul_bwd_kernel[grid](
                gate, up, grad_out, grad_gate, grad_up, n_elements, BLOCK_SIZE
            )
            return grad_gate, grad_up

    _TRITON_FUSED_OPS = {
        nn.SiLU: _FusedSiLUMul,
        nn.ReLU: _FusedReLUMul,
        nn.GELU: _FusedGELUMul,
    }


# GLU Variants Improve Transformer
# https://arxiv.org/pdf/2002.05202v1.pdf
class GLUFeedforwardLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feedforward: int,
        *,
        activation_factory: Callable = lambda: nn.SiLU(),
        dropout: Optional[float] = 0.0,
        use_triton: bool = True,
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
        self.dropout_p = dropout
        if not dropout:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

        # Triton kernel dispatch
        self._fused_op = None
        if use_triton:
            activation_type = type(self.activation)
            if not _HAS_TRITON:
                logger.info(
                    "Triton not installed. Using PyTorch fallback for GLU activation. "
                    "Install with: pip install triton"
                )
            elif activation_type in _TRITON_FUSED_OPS:
                self._fused_op = _TRITON_FUSED_OPS[activation_type]
                logger.debug(
                    f"Using Triton fused kernel for {activation_type.__name__}"
                )
            else:
                logger.info(
                    f"No Triton kernel for activation {activation_type.__name__}. "
                    f"Supported: {[c.__name__ for c in _TRITON_FUSED_OPS]}. "
                    f"Using PyTorch fallback."
                )

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, d_feedforward={self.d_feedforward}, "
            f"dropout={self.dropout_p}, "
            f"use_triton={self._fused_op is not None}"
        )

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if (
            self._fused_op is not None
            and gate.is_cuda
            and not torch.compiler.is_compiling()
        ):
            x = self._fused_op.apply(gate.contiguous(), up.contiguous())
        else:
            x = up * self.activation(gate)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
