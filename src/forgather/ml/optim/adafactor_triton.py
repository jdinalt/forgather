"""
Speed-optimized Triton kernels for Adafactor optimizer.

Key optimizations over the memory-optimized version:
1. Zero CPU-GPU synchronization (no .item() calls)
2. Persistent kernel fusing RMS computation + clip + parameter update
   into a single kernel launch (reads gradient once for both RMS and apply)
3. Precomputed rsqrt vectors to reduce per-element work in kernels
4. Avoids materializing intermediate tensors (grad^2, update)

Target configuration:
- relative_step=False (use fixed lr)
- bf16_stochastic_round: True or False
- weight_decay >= 0
- clip_threshold=1.0 (enabled)
"""

import torch
import triton
import triton.language as tl

# Cache SM count per device
_num_sms_cache = {}


def _get_num_sms(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _num_sms_cache:
        _num_sms_cache[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _num_sms_cache[idx]


# ============================================================
# Stochastic rounding helper
# ============================================================


@triton.jit
def _stochastic_round_bf16(val, seed, offs):
    """
    Stochastically round f32 values to bf16-representable f32.

    Extracts the lower 16 bits of the f32 representation (the bits lost
    when converting to bf16), generates a random 16-bit value, and rounds
    away from zero if the random value is less than the fractional bits.
    This matches the behavior of fp32_to_bf16_stochastic_round().
    """
    val_bits = val.to(tl.int32, bitcast=True)
    fraction = val_bits & 0xFFFF  # lower 16 bits (lost in bf16 truncation)
    val_rounded = val_bits - fraction  # round toward zero (== val_bits & 0xFFFF0000)
    rand_bits = tl.randint(seed, offs).to(tl.int32) & 0xFFFF
    # Round away from zero with probability fraction/2^16
    val_bits = tl.where(rand_bits < fraction, val_rounded + 0x10000, val_rounded)
    return val_bits.to(tl.float32, bitcast=True)


# ============================================================
# Row/Col reduction kernels (no intermediate tensor)
# ============================================================


@triton.jit
def _row_reduction_kernel(
    grad_ptr,
    row_sums_ptr,
    n_cols,
    eps1,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """Row sums of grad^2 + eps. One program per row, coalesced access."""
    row_idx = tl.program_id(0)
    row_sum = 0.0

    for col_start in range(0, n_cols, BLOCK_SIZE_COL):
        col_offs = col_start + tl.arange(0, BLOCK_SIZE_COL)
        mask = col_offs < n_cols
        grad_offs = row_idx * n_cols + col_offs
        grad_vals = tl.load(grad_ptr + grad_offs, mask=mask, other=0.0).to(tl.float32)
        grad_sq = grad_vals * grad_vals + eps1
        row_sum += tl.sum(tl.where(mask, grad_sq, 0.0))

    tl.store(row_sums_ptr + row_idx, row_sum)


@triton.jit
def _col_reduction_kernel(
    grad_ptr,
    col_sums_ptr,
    n_rows,
    n_cols,
    eps1,
    BLOCK_SIZE_ROW: tl.constexpr,
):
    """Column sums of grad^2 + eps. One program per column."""
    col_idx = tl.program_id(0)
    if col_idx < n_cols:
        col_sum = 0.0
        for row_start in range(0, n_rows, BLOCK_SIZE_ROW):
            row_offs = row_start + tl.arange(0, BLOCK_SIZE_ROW)
            mask = row_offs < n_rows
            grad_offs = row_offs * n_cols + col_idx
            grad_vals = tl.load(grad_ptr + grad_offs, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_sq = grad_vals * grad_vals + eps1
            col_sum += tl.sum(tl.where(mask, grad_sq, 0.0))
        tl.store(col_sums_ptr + col_idx, col_sum)


# ============================================================
# 2D persistent kernel: fused RMS computation + clip + apply
# ============================================================


@triton.jit
def _persistent_update_2d_kernel(
    param_ptr,
    grad_ptr,
    row_rsqrt_ptr,
    col_rsqrt_ptr,
    update_sq_sum_ptr,
    counter_ptr,
    n_elements,
    N_COLS: tl.constexpr,
    lr,
    weight_decay,
    clip_threshold,
    seed,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BF16_STOCHASTIC_ROUND: tl.constexpr,
):
    """
    Persistent kernel: compute update RMS via global reduction, then apply
    clipped update to parameters. Two phases with spin-wait barrier.

    Phase 1: Each block accumulates partial sum of update^2 over its chunks,
             then atomically adds to the global accumulator.
    Barrier: Spin-wait until all blocks have completed phase 1.
    Phase 2: Each block computes clip_scale from the global sum, then
             recomputes and applies the clipped update to parameters.

    The gradient is read twice (once per phase) but the second read is
    likely served from L1/L2 cache since the same block processes the
    same memory regions in both phases.
    """
    pid = tl.program_id(0)
    n_elements_f = n_elements.to(tl.float32)

    # Phase 1: accumulate update^2
    partial_sq_sum = 0.0
    chunk_id = pid
    while chunk_id * BLOCK_SIZE < n_elements:
        offs = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        row_idx = offs // N_COLS
        col_idx = offs % N_COLS

        grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
        r_rs = tl.load(row_rsqrt_ptr + row_idx, mask=mask, other=0.0)
        c_rs = tl.load(col_rsqrt_ptr + col_idx, mask=mask, other=0.0)

        update = grad * r_rs * c_rs
        partial_sq_sum += tl.sum(tl.where(mask, update * update, 0.0))

        chunk_id += NUM_SMS

    tl.atomic_add(update_sq_sum_ptr, partial_sq_sum)

    # Barrier
    tl.atomic_add(counter_ptr, 1)
    while tl.atomic_add(counter_ptr, 0) < NUM_SMS:
        pass

    # Compute clip scale (redundantly per block, cheap)
    total_sq = tl.atomic_add(update_sq_sum_ptr, 0.0)
    rms = tl.sqrt(total_sq / n_elements_f)
    clip_scale = tl.maximum(1.0, rms / clip_threshold)

    # Phase 2: apply clipped update
    chunk_id = pid
    while chunk_id * BLOCK_SIZE < n_elements:
        offs = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        row_idx = offs // N_COLS
        col_idx = offs % N_COLS

        grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
        r_rs = tl.load(row_rsqrt_ptr + row_idx, mask=mask, other=0.0)
        c_rs = tl.load(col_rsqrt_ptr + col_idx, mask=mask, other=0.0)

        update = grad * r_rs * c_rs / clip_scale

        param = tl.load(param_ptr + offs, mask=mask, other=0.0)
        param = param * (1.0 - lr * weight_decay) - lr * update
        if BF16_STOCHASTIC_ROUND:
            param = _stochastic_round_bf16(param, seed, offs)
        tl.store(param_ptr + offs, param, mask=mask)

        chunk_id += NUM_SMS


# ============================================================
# 2D non-persistent fallback: two separate kernels
# ============================================================


@triton.jit
def _compute_rms_2d_kernel(
    grad_ptr,
    row_rsqrt_ptr,
    col_rsqrt_ptr,
    update_sq_sum_ptr,
    n_elements,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute sum of update^2 for RMS calculation (2D)."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    row_idx = offs // N_COLS
    col_idx = offs % N_COLS

    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    r_rs = tl.load(row_rsqrt_ptr + row_idx, mask=mask, other=0.0)
    c_rs = tl.load(col_rsqrt_ptr + col_idx, mask=mask, other=0.0)

    update = grad * r_rs * c_rs
    sq_sum = tl.sum(tl.where(mask, update * update, 0.0))
    tl.atomic_add(update_sq_sum_ptr, sq_sum)


@triton.jit
def _apply_update_2d_kernel(
    param_ptr,
    grad_ptr,
    row_rsqrt_ptr,
    col_rsqrt_ptr,
    update_sq_sum_ptr,
    n_elements,
    N_COLS: tl.constexpr,
    lr,
    weight_decay,
    clip_threshold,
    seed,
    BLOCK_SIZE: tl.constexpr,
    BF16_STOCHASTIC_ROUND: tl.constexpr,
):
    """Apply clipped update to parameters (2D). Reads clip info from device memory."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    row_idx = offs // N_COLS
    col_idx = offs % N_COLS

    total_sq = tl.load(update_sq_sum_ptr)
    n_elements_f = n_elements.to(tl.float32)
    rms = tl.sqrt(total_sq / n_elements_f)
    clip_scale = tl.maximum(1.0, rms / clip_threshold)

    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    r_rs = tl.load(row_rsqrt_ptr + row_idx, mask=mask, other=0.0)
    c_rs = tl.load(col_rsqrt_ptr + col_idx, mask=mask, other=0.0)

    update = grad * r_rs * c_rs / clip_scale

    param = tl.load(param_ptr + offs, mask=mask, other=0.0)
    param = param * (1.0 - lr * weight_decay) - lr * update
    if BF16_STOCHASTIC_ROUND:
        param = _stochastic_round_bf16(param, seed, offs)
    tl.store(param_ptr + offs, param, mask=mask)


# ============================================================
# 1D kernels (biases, layernorm params)
# ============================================================


@triton.jit
def _compute_rms_1d_kernel(
    grad_ptr,
    state_ptr,
    update_sq_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute sum of update^2 for RMS calculation (1D)."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    state = tl.load(state_ptr + offs, mask=mask, other=1.0)

    update = grad * tl.rsqrt(state)
    sq_sum = tl.sum(tl.where(mask, update * update, 0.0))
    tl.atomic_add(update_sq_sum_ptr, sq_sum)


@triton.jit
def _apply_update_1d_kernel(
    param_ptr,
    grad_ptr,
    state_ptr,
    update_sq_sum_ptr,
    n_elements,
    lr,
    weight_decay,
    clip_threshold,
    seed,
    BLOCK_SIZE: tl.constexpr,
    BF16_STOCHASTIC_ROUND: tl.constexpr,
):
    """Apply clipped update to parameters (1D). Reads clip info from device memory."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    total_sq = tl.load(update_sq_sum_ptr)
    n_elements_f = n_elements.to(tl.float32)
    rms = tl.sqrt(total_sq / n_elements_f)
    clip_scale = tl.maximum(1.0, rms / clip_threshold)

    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    state = tl.load(state_ptr + offs, mask=mask, other=1.0)

    update = grad * tl.rsqrt(state) / clip_scale

    param = tl.load(param_ptr + offs, mask=mask, other=0.0)
    param = param * (1.0 - lr * weight_decay) - lr * update
    if BF16_STOCHASTIC_ROUND:
        param = _stochastic_round_bf16(param, seed, offs)
    tl.store(param_ptr + offs, param, mask=mask)


# ============================================================
# Public API
# ============================================================

_PERSISTENT_THRESHOLD = 65536


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
    bf16_stochastic_round: bool = False,
    sr_seed: int = 0,
):
    """
    Speed-optimized 2D Adafactor step.

    Uses Triton reduction kernels (same style as original, no grad_sq
    materialization), then a persistent Triton kernel for the fused
    RMS + clip + parameter update with zero CPU-GPU synchronization.
    """
    n_rows, n_cols = grad.shape
    n_elements = n_rows * n_cols

    # Convert gradient to f32
    if grad.dtype == torch.float32 and grad.is_contiguous():
        grad_f32 = grad
    else:
        grad_f32 = grad.float().contiguous()

    # Row/col reduction via Triton (no grad_sq materialization)
    row_sums = torch.empty(n_rows, device=grad.device, dtype=torch.float32)
    col_sums = torch.empty(n_cols, device=grad.device, dtype=torch.float32)

    BLOCK_COL = 1024
    BLOCK_ROW = 256
    _row_reduction_kernel[(n_rows,)](
        grad_f32, row_sums, n_cols, eps1, BLOCK_COL,
    )
    _col_reduction_kernel[(n_cols,)](
        grad_f32, col_sums, n_rows, n_cols, eps1, BLOCK_ROW,
    )

    # EMA state update (small vectors, fast PyTorch ops)
    row_f32 = row.float()
    col_f32 = col.float()
    row_f32.lerp_(row_sums, 1.0 - beta2t)
    col_f32.lerp_(col_sums, 1.0 - beta2t)
    if row.dtype != torch.float32:
        row.copy_(row_f32)
    if col.dtype != torch.float32:
        col.copy_(col_f32)

    # Precompute rsqrt vectors (small, cheap, stays on device)
    row_sum = row_f32.sum()
    row_rsqrt = torch.rsqrt(row_f32 / row_sum).contiguous()
    col_rsqrt = torch.rsqrt(col_f32).contiguous()

    # Flatten for 1D kernel indexing
    grad_flat = grad_f32.reshape(-1)
    if param.dtype == torch.float32:
        param_flat = param.reshape(-1)
    else:
        param_flat = param.float().reshape(-1)

    # Scratch tensors
    update_sq_sum = torch.zeros(1, device=grad.device, dtype=torch.float32)

    # Stochastic rounding only applies when converting f32 -> bf16
    do_stochastic_round = bf16_stochastic_round and param.dtype != torch.float32
    seed = sr_seed if do_stochastic_round else 0

    BLOCK_SIZE = 1024
    num_sms = _get_num_sms(grad.device)
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    if n_elements >= _PERSISTENT_THRESHOLD:
        counter = torch.zeros(1, device=grad.device, dtype=torch.int32)
        grid_size = min(num_sms, n_blocks)
        _persistent_update_2d_kernel[(grid_size,)](
            param_flat,
            grad_flat,
            row_rsqrt,
            col_rsqrt,
            update_sq_sum,
            counter,
            n_elements,
            n_cols,
            lr,
            weight_decay,
            clip_threshold,
            seed,
            grid_size,
            BLOCK_SIZE,
            do_stochastic_round,
        )
    else:
        _compute_rms_2d_kernel[(n_blocks,)](
            grad_flat,
            row_rsqrt,
            col_rsqrt,
            update_sq_sum,
            n_elements,
            n_cols,
            BLOCK_SIZE,
        )
        _apply_update_2d_kernel[(n_blocks,)](
            param_flat,
            grad_flat,
            row_rsqrt,
            col_rsqrt,
            update_sq_sum,
            n_elements,
            n_cols,
            lr,
            weight_decay,
            clip_threshold,
            seed,
            BLOCK_SIZE,
            do_stochastic_round,
        )

    if param.dtype != torch.float32:
        param.copy_(param_flat.view(param.shape))


def adafactor_step_1d_triton(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: torch.Tensor,
    beta2t: float,
    eps1: float,
    lr: float,
    weight_decay: float,
    clip_threshold: float,
    bf16_stochastic_round: bool = False,
    sr_seed: int = 0,
):
    """
    Speed-optimized 1D Adafactor step.

    Uses PyTorch for state update, two small Triton kernels for
    update computation with no CPU-GPU sync.
    """
    n_elements = grad.numel()

    # Convert to f32
    grad_f32 = grad.float().contiguous()
    state_f32 = state.float()

    # Update second moment state
    grad_sq = grad_f32 * grad_f32 + eps1
    state_f32.lerp_(grad_sq, 1.0 - beta2t)
    del grad_sq
    if state.dtype != torch.float32:
        state.copy_(state_f32)

    # Param in f32
    if param.dtype == torch.float32:
        param_f32 = param.contiguous()
    else:
        param_f32 = param.float().contiguous()

    # Compute RMS then apply (two kernels, no CPU sync)
    update_sq_sum = torch.zeros(1, device=grad.device, dtype=torch.float32)

    # Stochastic rounding only applies when converting f32 -> bf16
    do_stochastic_round = bf16_stochastic_round and param.dtype != torch.float32
    seed = sr_seed if do_stochastic_round else 0

    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    _compute_rms_1d_kernel[(n_blocks,)](
        grad_f32,
        state_f32,
        update_sq_sum,
        n_elements,
        BLOCK_SIZE,
    )

    _apply_update_1d_kernel[(n_blocks,)](
        param_f32,
        grad_f32,
        state_f32,
        update_sq_sum,
        n_elements,
        lr,
        weight_decay,
        clip_threshold,
        seed,
        BLOCK_SIZE,
        do_stochastic_round,
    )

    if param.dtype != torch.float32:
        param.copy_(param_f32)
