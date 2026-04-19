"""
Benchmark comparing Adafactor implementations:
- PyTorch baseline (_adafactor)
- Original Triton (memory-optimized)
- New Triton (speed-optimized)

Tests both correctness (numerical agreement) and speed.
"""

import copy
import os
import sys
import time

import torch

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.forgather.ml.optim.adafactor import Adafactor, _adafactor
from src.forgather.ml.optim import adafactor_triton as triton_new

# Import original triton for comparison
from src.forgather.ml.optim import adafactor_triton_original as triton_old


def make_test_tensors(n_rows, n_cols, device, dtype=torch.bfloat16):
    """Create param, grad, row, col tensors for testing."""
    param = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    grad = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    row = torch.rand(n_rows, device=device, dtype=dtype) + 0.1
    col = torch.rand(n_cols, device=device, dtype=dtype) + 0.1
    return param, grad, row, col


def make_test_tensors_1d(n, device, dtype=torch.bfloat16):
    """Create param, grad, state tensors for 1D testing."""
    param = torch.randn(n, device=device, dtype=dtype)
    grad = torch.randn(n, device=device, dtype=dtype)
    state = torch.rand(n, device=device, dtype=dtype) + 0.1
    return param, grad, state


def reference_step_2d(param, grad, row, col, beta2t, eps1, lr, wd, clip_thr):
    """Run the PyTorch baseline step and return updated tensors."""
    p, g, r, c = param.clone(), grad.clone(), row.clone(), col.clone()
    # Weight decay
    if wd > 0:
        p.add_(p, alpha=(-lr * wd))
    g32 = g.float()
    update = g32 ** 2 + eps1
    r32 = r.float()
    c32 = c.float()
    r32.lerp_(update.sum(dim=-1), 1.0 - beta2t)
    c32.lerp_(update.sum(dim=-2), 1.0 - beta2t)
    update = g32 * torch.outer(torch.rsqrt(r32 / r32.sum()), torch.rsqrt(c32))
    rms_val = update.square().mean().sqrt()
    update /= (rms_val / clip_thr).clamp_(min=1.0)
    if p.dtype == update.dtype:
        p.add_(update, alpha=-lr)
    else:
        p.copy_(p.float() - lr * update)
    r.copy_(r32)
    c.copy_(c32)
    return p, r, c


def reference_step_1d(param, grad, state, beta2t, eps1, lr, wd, clip_thr):
    """Run the PyTorch baseline step for 1D and return updated tensors."""
    p, g, s = param.clone(), grad.clone(), state.clone()
    if wd > 0:
        p.add_(p, alpha=(-lr * wd))
    g32 = g.float()
    upd = g32 ** 2 + eps1
    s32 = s.float()
    s32.lerp_(upd, 1.0 - beta2t)
    upd = g32 / s32.sqrt()
    rms_val = upd.square().mean().sqrt()
    upd /= (rms_val / clip_thr).clamp_(min=1.0)
    if p.dtype == upd.dtype:
        p.add_(upd, alpha=-lr)
    else:
        p.copy_(p.float() - lr * upd)
    s.copy_(s32)
    return p, s


def test_correctness_2d(device, sizes=None):
    """Test numerical correctness of new Triton vs PyTorch baseline."""
    if sizes is None:
        sizes = [(512, 512), (1024, 2048), (4096, 4096)]

    print("\n=== 2D Correctness Tests ===")
    beta2t = 0.98
    eps1 = 1e-30
    lr = 1e-3
    wd = 0.001
    clip_thr = 1.0
    all_pass = True

    for n_rows, n_cols in sizes:
        torch.manual_seed(42)
        param, grad, row, col = make_test_tensors(n_rows, n_cols, device)

        # Reference
        ref_p, ref_r, ref_c = reference_step_2d(
            param, grad, row, col, beta2t, eps1, lr, wd, clip_thr
        )

        # New Triton
        p_new, g_new, r_new, c_new = (
            param.clone(), grad.clone(), row.clone(), col.clone(),
        )
        triton_new.adafactor_step_2d_triton(
            p_new, g_new, r_new, c_new, beta2t, eps1, lr, wd, clip_thr
        )

        p_diff = (ref_p.float() - p_new.float()).abs().max().item()
        r_diff = (ref_r.float() - r_new.float()).abs().max().item()
        c_diff = (ref_c.float() - c_new.float()).abs().max().item()

        # Compare new Triton vs old Triton (should match closely)
        p_old, g_old, r_old, c_old = (
            param.clone(), grad.clone(), row.clone(), col.clone(),
        )
        triton_old.adafactor_step_2d_triton(
            p_old, g_old, r_old, c_old, beta2t, eps1, lr, wd, clip_thr
        )
        new_vs_old = (p_new.float() - p_old.float()).abs().max().item()

        # Use 1e-2 for reference comparison (weight decay rounding order differs)
        # Use 1e-2 for new vs old Triton (bf16 precision differences)
        passed = p_diff < 1e-2 and new_vs_old < 1e-2
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {n_rows}x{n_cols}: "
              f"vs_ref={p_diff:.2e}, vs_old_triton={new_vs_old:.2e}, "
              f"row_diff={r_diff:.2e}, col_diff={c_diff:.2e}")

    return all_pass


def test_correctness_1d(device, sizes=None):
    """Test numerical correctness of new Triton 1D vs PyTorch baseline."""
    if sizes is None:
        sizes = [512, 1024, 4096]

    print("\n=== 1D Correctness Tests ===")
    beta2t = 0.98
    eps1 = 1e-30
    lr = 1e-3
    wd = 0.001
    clip_thr = 1.0
    all_pass = True

    for n in sizes:
        torch.manual_seed(42)
        param, grad, state = make_test_tensors_1d(n, device)

        # Reference
        ref_p, ref_s = reference_step_1d(
            param, grad, state, beta2t, eps1, lr, wd, clip_thr
        )

        # New Triton
        p_new, g_new, s_new = param.clone(), grad.clone(), state.clone()
        triton_new.adafactor_step_1d_triton(
            p_new, g_new, s_new, beta2t, eps1, lr, wd, clip_thr
        )

        p_diff = (ref_p.float() - p_new.float()).abs().max().item()
        s_diff = (ref_s.float() - s_new.float()).abs().max().item()

        passed = p_diff < 1e-3 and s_diff < 1e-3
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] n={n}: param_diff={p_diff:.2e}, state_diff={s_diff:.2e}")

    return all_pass


def benchmark_step_2d(device, n_rows, n_cols, n_warmup=10, n_iter=50):
    """Benchmark 2D step for all implementations."""
    beta2t = 0.98
    eps1 = 1e-30
    lr = 1e-3
    wd = 0.001
    clip_thr = 1.0

    results = {}

    # PyTorch baseline
    torch.manual_seed(42)
    param, grad, row, col = make_test_tensors(n_rows, n_cols, device)

    def run_pytorch():
        p, g, r, c = param.clone(), grad.clone(), row.clone(), col.clone()
        if wd > 0:
            p.add_(p, alpha=(-lr * wd))
        g32 = g.float()
        upd = g32 ** 2 + eps1
        r32 = r.float()
        c32 = c.float()
        r32.lerp_(upd.sum(dim=-1), 1.0 - beta2t)
        c32.lerp_(upd.sum(dim=-2), 1.0 - beta2t)
        upd = g32 * torch.outer(torch.rsqrt(r32 / r32.sum()), torch.rsqrt(c32))
        upd /= (upd.square().mean().sqrt() / clip_thr).clamp_(min=1.0)
        p.copy_(p.float() - lr * upd)
        r.copy_(r32)
        c.copy_(c32)

    for _ in range(n_warmup):
        run_pytorch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_pytorch()
    torch.cuda.synchronize()
    results["pytorch"] = (time.perf_counter() - t0) / n_iter * 1000

    # Old Triton
    def run_old_triton():
        p, g, r, c = param.clone(), grad.clone(), row.clone(), col.clone()
        triton_old.adafactor_step_2d_triton(p, g, r, c, beta2t, eps1, lr, wd, clip_thr)

    for _ in range(n_warmup):
        run_old_triton()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_old_triton()
    torch.cuda.synchronize()
    results["old_triton"] = (time.perf_counter() - t0) / n_iter * 1000

    # New Triton
    def run_new_triton():
        p, g, r, c = param.clone(), grad.clone(), row.clone(), col.clone()
        triton_new.adafactor_step_2d_triton(p, g, r, c, beta2t, eps1, lr, wd, clip_thr)

    for _ in range(n_warmup):
        run_new_triton()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_new_triton()
    torch.cuda.synchronize()
    results["new_triton"] = (time.perf_counter() - t0) / n_iter * 1000

    return results


def benchmark_step_1d(device, n, n_warmup=10, n_iter=50):
    """Benchmark 1D step for all implementations."""
    beta2t = 0.98
    eps1 = 1e-30
    lr = 1e-3
    wd = 0.001
    clip_thr = 1.0

    results = {}
    torch.manual_seed(42)
    param, grad, state = make_test_tensors_1d(n, device)

    # PyTorch baseline
    def run_pytorch():
        p, g, s = param.clone(), grad.clone(), state.clone()
        if wd > 0:
            p.add_(p, alpha=(-lr * wd))
        g32 = g.float()
        upd = g32 ** 2 + eps1
        s32 = s.float()
        s32.lerp_(upd, 1.0 - beta2t)
        upd = g32 / s32.sqrt()
        upd /= (upd.square().mean().sqrt() / clip_thr).clamp_(min=1.0)
        p.copy_(p.float() - lr * upd)
        s.copy_(s32)

    for _ in range(n_warmup):
        run_pytorch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_pytorch()
    torch.cuda.synchronize()
    results["pytorch"] = (time.perf_counter() - t0) / n_iter * 1000

    # Old Triton
    def run_old_triton():
        p, g, s = param.clone(), grad.clone(), state.clone()
        triton_old.adafactor_step_1d_triton(p, g, s, beta2t, eps1, lr, wd, clip_thr)

    for _ in range(n_warmup):
        run_old_triton()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_old_triton()
    torch.cuda.synchronize()
    results["old_triton"] = (time.perf_counter() - t0) / n_iter * 1000

    # New Triton
    def run_new_triton():
        p, g, s = param.clone(), grad.clone(), state.clone()
        triton_new.adafactor_step_1d_triton(p, g, s, beta2t, eps1, lr, wd, clip_thr)

    for _ in range(n_warmup):
        run_new_triton()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_new_triton()
    torch.cuda.synchronize()
    results["new_triton"] = (time.perf_counter() - t0) / n_iter * 1000

    return results


def main():
    # When CUDA_VISIBLE_DEVICES is set, the visible GPUs are remapped starting at 0
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"SMs: {torch.cuda.get_device_properties(device).multi_processor_count}")

    # Correctness tests
    passed_2d = test_correctness_2d(device)
    passed_1d = test_correctness_1d(device)

    if not (passed_2d and passed_1d):
        print("\nCorrectness tests FAILED - proceeding with benchmarks anyway.")
    else:
        print("\nAll correctness tests passed.")

    # 2D benchmarks
    print("\n=== 2D Benchmarks (ms per step) ===")
    print(f"{'Size':>14s} | {'PyTorch':>10s} | {'Old Triton':>10s} | {'New Triton':>10s} | {'Speedup':>8s}")
    print("-" * 70)

    sizes_2d = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (1024, 4096),
        (2048, 2048),
        (4096, 4096),
        (4096, 11008),
    ]

    for n_rows, n_cols in sizes_2d:
        results = benchmark_step_2d(device, n_rows, n_cols)
        speedup = results["pytorch"] / results["new_triton"]
        print(
            f"{n_rows}x{n_cols:>5d} | "
            f"{results['pytorch']:>10.3f} | "
            f"{results['old_triton']:>10.3f} | "
            f"{results['new_triton']:>10.3f} | "
            f"{speedup:>7.2f}x"
        )

    # 1D benchmarks
    print("\n=== 1D Benchmarks (ms per step) ===")
    print(f"{'Size':>14s} | {'PyTorch':>10s} | {'Old Triton':>10s} | {'New Triton':>10s} | {'Speedup':>8s}")
    print("-" * 70)

    sizes_1d = [512, 1024, 2048, 4096, 8192]

    for n in sizes_1d:
        results = benchmark_step_1d(device, n)
        speedup = results["pytorch"] / results["new_triton"]
        print(
            f"{n:>14d} | "
            f"{results['pytorch']:>10.3f} | "
            f"{results['old_triton']:>10.3f} | "
            f"{results['new_triton']:>10.3f} | "
            f"{speedup:>7.2f}x"
        )


if __name__ == "__main__":
    main()
