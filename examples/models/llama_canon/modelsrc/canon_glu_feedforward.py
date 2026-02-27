"""
Canon GLU Feedforward Layer.

Extends the standard GLU feedforward with Canon-D: a depthwise causal
convolution applied to gate and up projections before the gating activation.

From "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers" (Allen-Zhu, 2025).
"""

import logging
from typing import Callable, Optional

import torch
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
        canon_factory: Callable,
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

        # Canon-D: applied to gate and up projections separately
        # (depthwise conv is channel-independent, so separate layers are
        # mathematically equivalent to a single layer on concatenated gate+up)
        self.canon_d_gate = canon_factory(dim=d_feedforward)
        self.canon_d_up = canon_factory(dim=d_feedforward)

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, d_feedforward={self.d_feedforward}, "
            f"use_triton={self._fused_op is not None}"
        )

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Canon-D: apply depthwise causal conv to gate and up
        gate = self.canon_d_gate(gate)
        up = self.canon_d_up(up)

        if self._fused_op is not None and gate.is_cuda:
            x = self._fused_op.apply(gate.contiguous(), up.contiguous())
        else:
            x = up * self.activation(gate)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
