"""
Memory-optimized Triton kernels for Adafactor optimizer.

This implementation focuses on reducing peak memory utilization by:
1. Avoiding materialization of intermediate tensors (grad**2, update, etc.)
2. Computing factored preconditioner element-wise without outer product
3. In-place parameter updates
4. Fused operations (weight decay, clipping, parameter update)

Target configuration:
- relative_step=False (use fixed lr)
- bf16_stochastic_round=False (standard conversion)
- weight_decay > 0 (typically 0.001)
- clip_threshold=1.0 (enabled)
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def factored_row_reduction_kernel(
    grad_ptr,
    row_ptr,
    row_sums_ptr,  # Temporary buffer for row sums
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps1,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Compute row sums of grad**2 + eps1 without materializing full tensor.

    Each program computes one row sum.
    """
    row_idx = tl.program_id(0)
    if row_idx < n_rows:
        row_sum = 0.0

        # Iterate over columns in blocks
        for col_start in range(0, n_cols, BLOCK_SIZE_COL):
            col_offs = col_start + tl.arange(0, BLOCK_SIZE_COL)
            mask = col_offs < n_cols

            # Load gradient values
            grad_offs = row_idx * n_cols + col_offs
            grad_vals = tl.load(grad_ptr + grad_offs, mask=mask, other=0.0)

            # Accumulate grad**2 + eps1
            grad_sq = grad_vals * grad_vals + eps1
            row_sum += tl.sum(grad_sq)

        # Store row sum to temporary buffer
        tl.store(row_sums_ptr + row_idx, row_sum)


@triton.jit
def factored_col_reduction_kernel(
    grad_ptr,
    col_ptr,
    col_sums_ptr,  # Temporary buffer for column sums
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps1,
    BLOCK_SIZE_ROW: tl.constexpr,
):
    """
    Compute column sums of grad**2 + eps1 without materializing full tensor.

    Each program computes one column sum.
    """
    col_idx = tl.program_id(0)

    if col_idx < n_cols:
        col_sum = 0.0

        # Iterate over rows in blocks
        for row_start in range(0, n_rows, BLOCK_SIZE_ROW):
            row_offs = row_start + tl.arange(0, BLOCK_SIZE_ROW)
            mask = row_offs < n_rows

            # Load gradient values
            grad_offs = row_offs * n_cols + col_idx
            grad_vals = tl.load(grad_ptr + grad_offs, mask=mask, other=0.0)

            # Accumulate grad**2 + eps1
            grad_sq = grad_vals * grad_vals + eps1
            col_sum += tl.sum(grad_sq)

        # Store column sum to temporary buffer
        tl.store(col_sums_ptr + col_idx, col_sum)


@triton.jit
def row_sum_kernel(
    row_ptr,
    row_sum_ptr,
    n_rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute sum of row state for normalization.
    Simple reduction kernel.
    """
    row_sum_acc = 0.0

    for row_start in range(0, n_rows, BLOCK_SIZE):
        row_offs = row_start + tl.arange(0, BLOCK_SIZE)
        mask = row_offs < n_rows
        row_vals = tl.load(row_ptr + row_offs, mask=mask, other=0.0)
        row_sum_acc += tl.sum(row_vals)

    # Store result (only program 0 does this)
    if tl.program_id(0) == 0:
        tl.store(row_sum_ptr, row_sum_acc)


@triton.jit
def compute_update_rms_kernel(
    grad_ptr,
    row_ptr,
    col_ptr,
    update_sq_sum_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    row_sum,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute sum of squared updates for RMS calculation.

    This kernel computes update^2 element-wise and accumulates the sum
    without materializing the update tensor.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    # Convert 1D offset to 2D indices
    row_idx = offs // n_cols
    col_idx = offs % n_cols
    mask = offs < (n_rows * n_cols)

    # Load gradient
    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)

    # Load row and col preconditioner values
    row_val = tl.load(row_ptr + row_idx, mask=mask, other=1.0)
    col_val = tl.load(col_ptr + col_idx, mask=mask, other=1.0)

    # Compute preconditioned update element-wise
    row_scale = tl.rsqrt(row_val / row_sum)
    col_scale = tl.rsqrt(col_val)
    update = grad * row_scale * col_scale

    # Accumulate update**2
    update_sq_sum = tl.sum(tl.where(mask, update * update, 0.0))

    # Atomic add to global sum
    tl.atomic_add(update_sq_sum_ptr, update_sq_sum)


@triton.jit
def apply_update_with_clipping_kernel(
    param_ptr,
    grad_ptr,
    row_ptr,
    col_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    row_sum,
    lr,
    weight_decay,
    clip_scale,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply preconditioned update to parameters with clipping.

    This kernel computes update element-wise and applies it with the
    pre-computed clipping scale, along with weight decay.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    # Convert 1D offset to 2D indices
    row_idx = offs // n_cols
    col_idx = offs % n_cols
    mask = offs < (n_rows * n_cols)

    # Load gradient
    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)

    # Load row and col preconditioner values
    row_val = tl.load(row_ptr + row_idx, mask=mask, other=1.0)
    col_val = tl.load(col_ptr + col_idx, mask=mask, other=1.0)

    # Compute preconditioned update element-wise
    row_scale = tl.rsqrt(row_val / row_sum)
    col_scale = tl.rsqrt(col_val)
    update = grad * row_scale * col_scale

    # Apply clipping
    update = update / clip_scale

    # Load parameter
    param = tl.load(param_ptr + offs, mask=mask, other=0.0)

    # Apply weight decay
    if weight_decay > 0.0:
        param = param * (1.0 - lr * weight_decay)

    # Apply update
    param = param - lr * update

    # Store updated parameter
    tl.store(param_ptr + offs, param, mask=mask)


@triton.jit
def vector_moment_update_kernel(
    grad_ptr,
    state_ptr,
    n_elements: tl.constexpr,
    beta2t,
    eps1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Update second moment state for 1D tensors (vectors/biases).

    Performs:
    - state.lerp_(grad**2 + eps1, 1-beta2t)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load data
    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    state = tl.load(state_ptr + offs, mask=mask, other=0.0)

    # Update second moment state
    grad_sq = grad * grad + eps1
    new_state = state * beta2t + grad_sq * (1.0 - beta2t)
    tl.store(state_ptr + offs, new_state, mask=mask)


@triton.jit
def vector_compute_update_rms_kernel(
    grad_ptr,
    state_ptr,
    update_sq_sum_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute sum of squared updates for RMS calculation (1D case).
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load data
    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    state = tl.load(state_ptr + offs, mask=mask, other=0.0)

    # Compute update
    update = grad / tl.sqrt(state)

    # Accumulate update**2
    update_sq_sum = tl.sum(tl.where(mask, update * update, 0.0))
    tl.atomic_add(update_sq_sum_ptr, update_sq_sum)


@triton.jit
def vector_apply_update_kernel(
    param_ptr,
    grad_ptr,
    state_ptr,
    n_elements: tl.constexpr,
    lr,
    weight_decay,
    clip_scale,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply preconditioned update to parameters (1D case).
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load data
    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    state = tl.load(state_ptr + offs, mask=mask, other=0.0)
    param = tl.load(param_ptr + offs, mask=mask, other=0.0)

    # Compute update
    update = grad / tl.sqrt(state)

    # Apply clipping
    update = update / clip_scale

    # Apply weight decay
    if weight_decay > 0.0:
        param = param * (1.0 - lr * weight_decay)

    # Apply update
    param = param - lr * update
    tl.store(param_ptr + offs, param, mask=mask)


# Helper functions for launching kernels


def adafactor_step_2d_triton(
    param: torch.Tensor,
    grad: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    beta2t: float,
    eps1: float,
    lr: float,
    weight_decay: float,
    clip_threshold: float,
):
    """
    Launch Triton kernels for 2D parameter update.

    Uses Triton for reductions to avoid materializing grad**2 tensor.
    """
    n_rows, n_cols = grad.shape
    n_elements = n_rows * n_cols

    # Convert to float32 for computations
    grad_f32 = grad.float() if grad.dtype != torch.float32 else grad
    row_f32 = row.float() if row.dtype != torch.float32 else row
    col_f32 = col.float() if col.dtype != torch.float32 else col

    # Allocate temporary buffers for row/col sums (much smaller than grad_sq)
    row_sums = torch.empty(n_rows, device=grad.device, dtype=torch.float32)
    col_sums = torch.empty(n_cols, device=grad.device, dtype=torch.float32)

    # Compute row sums using Triton (no grad_sq materialization)
    BLOCK_SIZE_COL = 1024
    BLOCK_SIZE_ROW = 256
    factored_row_reduction_kernel[(n_rows,)](
        grad_f32,
        row_f32,
        row_sums,
        n_rows,
        n_cols,
        eps1,
        BLOCK_SIZE_COL,
    )

    # Compute column sums using Triton (no grad_sq materialization)
    factored_col_reduction_kernel[(n_cols,)](
        grad_f32,
        col_f32,
        col_sums,
        n_rows,
        n_cols,
        eps1,
        BLOCK_SIZE_ROW,
    )

    # Update states with EMA using PyTorch (these are small 1D ops)
    row_f32.lerp_(row_sums, 1.0 - beta2t)
    col_f32.lerp_(col_sums, 1.0 - beta2t)

    # Copy back to original dtype if needed
    if row.dtype != torch.float32:
        row.copy_(row_f32)
    if col.dtype != torch.float32:
        col.copy_(col_f32)

    # Compute row sum for normalization
    row_sum = row_f32.sum().item()

    # Compute RMS of updates using Triton (avoid materializing update tensor)
    update_sq_sum = torch.zeros(1, device=grad.device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (n_blocks,)

    compute_update_rms_kernel[grid](
        grad_f32,
        row_f32,
        col_f32,
        update_sq_sum,
        n_rows,
        n_cols,
        row_sum,
        BLOCK_SIZE,
    )

    # Compute clipping scale
    update_rms = torch.sqrt(update_sq_sum / n_elements)
    clip_scale = max(1.0, update_rms.item() / clip_threshold)

    # Apply updates with clipping using Triton (element-wise, no materialization)
    param_f32 = param.float() if param.dtype != torch.float32 else param

    apply_update_with_clipping_kernel[grid](
        param_f32,
        grad_f32,
        row_f32,
        col_f32,
        n_rows,
        n_cols,
        row_sum,
        lr,
        weight_decay,
        clip_scale,
        BLOCK_SIZE,
    )

    # Copy back to original dtype if needed
    if param.dtype != torch.float32:
        param.copy_(param_f32)


def adafactor_step_1d_triton(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: torch.Tensor,
    beta2t: float,
    eps1: float,
    lr: float,
    weight_decay: float,
    clip_threshold: float,
):
    """
    Launch Triton kernels for 1D parameter update.

    For simplicity, use PyTorch for state update and Triton for parameter update.
    """
    n_elements = grad.numel()

    # Convert to float32 for computations
    grad_f32 = grad.float() if grad.dtype != torch.float32 else grad
    state_f32 = state.float() if state.dtype != torch.float32 else state

    # Use PyTorch for moment update
    grad_sq = grad_f32**2 + eps1
    state_f32.lerp_(grad_sq, 1.0 - beta2t)

    # Copy back to original dtype if needed
    if state.dtype != torch.float32:
        state.copy_(state_f32)

    # Compute RMS of updates using Triton
    update_sq_sum = torch.zeros(1, device=grad.device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (n_blocks,)

    vector_compute_update_rms_kernel[grid](
        grad_f32,
        state_f32,
        update_sq_sum,
        n_elements,
        BLOCK_SIZE,
    )

    # Compute clipping scale
    update_rms = torch.sqrt(update_sq_sum / n_elements)
    clip_scale = max(1.0, update_rms.item() / clip_threshold)

    # Apply updates with clipping using Triton
    param_f32 = param.float() if param.dtype != torch.float32 else param

    vector_apply_update_kernel[grid](
        param_f32,
        grad_f32,
        state_f32,
        n_elements,
        lr,
        weight_decay,
        clip_scale,
        BLOCK_SIZE,
    )

    # Copy back to original dtype if needed
    if param.dtype != torch.float32:
        param.copy_(param_f32)
