# Adafactor Triton Kernel Performance

Performance summary for the speed-optimized Triton implementation of the Adafactor optimizer (`src/forgather/ml/optim/adafactor_triton.py`).

## Background

The original Triton implementation was optimized for **memory consumption** (avoiding materialization of intermediate tensors like `grad^2`). However, it was slower than the baseline PyTorch implementation due to CPU-GPU synchronization overhead and excessive kernel launches.

The rewrite targets **speed** while retaining the memory-efficient reduction approach.

## Key Optimizations

1. **Zero CPU-GPU synchronization**: The original used `.item()` calls to transfer `row_sum` and `update_rms` to the CPU for scalar math. Each `.item()` forces a full GPU pipeline flush (~20-50us). The new version keeps all values on-device -- `row_sum` stays as a device tensor, and `clip_scale` is computed redundantly per-block inside the Triton kernel.

2. **Persistent kernel for fused RMS + clip + apply (2D)**: A single kernel launch handles both the update RMS computation (via inter-block atomic reduction with spin-wait barrier) and the clipped parameter update. This replaces two separate kernels (`compute_update_rms_kernel` + `apply_update_with_clipping_kernel`), eliminating one kernel launch and reducing gradient reads from 2 to ~1 (L1 cache reuse between phases within the same block).

3. **Precomputed rsqrt vectors**: `rsqrt(row / row_sum)` and `rsqrt(col)` are computed once in PyTorch on the small state vectors and passed to the update kernel, avoiding redundant per-element rsqrt computation.

4. **Same reduction approach as original**: The row and column reduction kernels avoid materializing the full `grad^2 + eps` tensor (O(n*m) allocation), computing sums on-the-fly from the gradient.

## Benchmark Results

Measured on NVIDIA RTX 4090 (128 SMs, 72MB L2 cache), bf16 parameters and gradients.

### 2D Parameters (Weight Matrices)

| Size | PyTorch (ms) | Old Triton (ms) | New Triton (ms) | vs PyTorch | vs Old Triton |
|------|-------------|-----------------|-----------------|------------|---------------|
| 256x256 | 0.218 | 0.239 | 0.214 | 1.02x | 1.12x |
| 512x512 | 0.222 | 0.235 | 0.215 | 1.03x | 1.09x |
| 1024x1024 | 0.226 | 0.237 | 0.215 | 1.05x | 1.10x |
| 1024x4096 | 0.227 | 0.255 | 0.213 | 1.07x | 1.20x |
| 2048x2048 | 0.225 | 0.245 | 0.213 | 1.06x | 1.15x |
| 4096x4096 | 1.842 | 0.923 | 0.918 | 2.01x | 1.01x |
| 4096x11008 | 5.421 | 2.888 | 2.863 | 1.89x | 1.01x |

### 1D Parameters (Biases, LayerNorm)

| Size | PyTorch (ms) | Old Triton (ms) | New Triton (ms) | vs PyTorch | vs Old Triton |
|------|-------------|-----------------|-----------------|------------|---------------|
| 512 | 0.145 | 0.167 | 0.123 | 1.17x | 1.36x |
| 1024 | 0.143 | 0.167 | 0.123 | 1.16x | 1.36x |
| 2048 | 0.144 | 0.167 | 0.124 | 1.16x | 1.35x |
| 4096 | 0.143 | 0.166 | 0.124 | 1.16x | 1.34x |
| 8192 | 0.143 | 0.166 | 0.124 | 1.16x | 1.34x |

### Summary

- **Faster than both PyTorch and old Triton across all tested sizes.**
- Small/medium 2D matrices: 5-7% faster than PyTorch, 10-20% faster than old Triton.
- Large 2D matrices: ~2x faster than PyTorch, on par with old Triton (reduction-dominated).
- 1D parameters: ~16% faster than PyTorch, ~35% faster than old Triton.

## Architecture

### 2D Step (`adafactor_step_2d_triton`)

```
Phase 1: Row/col reduction kernels
  _row_reduction_kernel  -- one block per row, coalesced column access
  _col_reduction_kernel  -- one block per column, strided row access
  (reads gradient twice total, no intermediate allocation)

Phase 2: EMA state update
  PyTorch lerp_ on small row/col vectors
  Precompute row_rsqrt, col_rsqrt vectors

Phase 3: Persistent update kernel
  _persistent_update_2d_kernel
    Phase A: Each block computes partial update^2 sum, atomic add
    Barrier: Spin-wait on completion counter
    Phase B: Compute clip_scale, apply clipped update to parameters
  (single kernel launch, gradient read ~1x via L1 cache reuse)
```

### 1D Step (`adafactor_step_1d_triton`)

```
Phase 1: PyTorch state update (small tensors)
Phase 2: _compute_rms_1d_kernel -> _apply_update_1d_kernel
  (two kernels, clip_scale passed via device memory, no CPU sync)
```

## Numerical Precision

The new implementation produces results within bf16 precision (~1e-3) of both the PyTorch baseline and the old Triton implementation. Small differences arise from:

- Different floating-point accumulation order in atomic reductions (RMS computation).
- Weight decay applied in f32 (Triton) vs bf16 intermediate (PyTorch baseline).

These differences are well within acceptable bounds for training.

## Supported Configuration

- `relative_step=False` (fixed learning rate)
- `bf16_stochastic_round=False` (standard conversion)
- `weight_decay >= 0`
- `clip_threshold > 0`
