"""
Benchmark comparing Triton kernel implementations vs PyTorch baselines
for fused GLU activation and RoPE rotation.

Usage:
    python tests/benchmark_triton_kernels.py
"""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "modelsrc/transformer")

from glu_feedforward import _HAS_TRITON

if _HAS_TRITON:
    import triton
    from glu_feedforward import _FusedSiLUMul, _FusedGELUMul
    from rotary_embeddings import (
        _FusedRoPERotation,
        _prepare_cos_sin_for_triton,
        rotate_half,
    )
else:
    print("Triton not installed. Cannot run benchmarks.")
    sys.exit(1)


def bench_fn(fn, warmup=50, rep=200):
    """Benchmark a function using triton.testing.do_bench."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


# ============================================================
# SiLU-Mul Benchmarks
# ============================================================


def benchmark_silu_mul():
    """Benchmark fused SiLU*up kernel vs PyTorch."""
    device = torch.device("cuda")

    print("\n=== SiLU * Up Benchmark ===")
    print(
        f"{'Shape':>30s} | {'PyTorch (ms)':>12s} | {'Triton (ms)':>12s} | {'Speedup':>8s}"
    )
    print("-" * 70)

    # Shapes: (batch*seq, d_ff) typical for GLU feedforward
    configs = [
        (512, 1024, "Small"),
        (512, 4096, "Medium"),
        (512, 11008, "Llama-7B d_ff"),
        (512, 14336, "Llama-70B d_ff"),
        (2048, 4096, "Long seq, medium"),
        (2048, 11008, "Long seq, Llama"),
        (8192, 4096, "Very long seq"),
    ]

    for n_tokens, d_ff, label in configs:
        for dtype, dtype_name in [
            (torch.float32, "f32"),
            (torch.bfloat16, "bf16"),
        ]:
            gate = torch.randn(n_tokens, d_ff, device=device, dtype=dtype)
            up = torch.randn(n_tokens, d_ff, device=device, dtype=dtype)

            # PyTorch baseline
            def pytorch_fn():
                return up * F.silu(gate)

            # Triton fused
            gate_c = gate.contiguous()
            up_c = up.contiguous()

            def triton_fn():
                return _FusedSiLUMul.apply(gate_c, up_c)

            pytorch_ms = bench_fn(pytorch_fn)
            triton_ms = bench_fn(triton_fn)
            speedup = pytorch_ms / triton_ms

            desc = f"{label} ({dtype_name})"
            print(
                f"{desc:>30s} | {pytorch_ms:>12.3f} | {triton_ms:>12.3f} | {speedup:>7.2f}x"
            )

    # Backward benchmark
    print("\n=== SiLU * Up Backward Benchmark ===")
    print(
        f"{'Shape':>30s} | {'PyTorch (ms)':>12s} | {'Triton (ms)':>12s} | {'Speedup':>8s}"
    )
    print("-" * 70)

    for n_tokens, d_ff, label in [(512, 11008, "Llama-7B"), (2048, 11008, "Long Llama")]:
        gate = torch.randn(
            n_tokens, d_ff, device=device, dtype=torch.float32, requires_grad=True
        )
        up = torch.randn(
            n_tokens, d_ff, device=device, dtype=torch.float32, requires_grad=True
        )
        grad_out = torch.randn(n_tokens, d_ff, device=device, dtype=torch.float32)

        def pytorch_bwd():
            g = gate.detach().clone().requires_grad_(True)
            u = up.detach().clone().requires_grad_(True)
            out = u * F.silu(g)
            out.backward(grad_out)
            return g.grad, u.grad

        def triton_bwd():
            g = gate.detach().clone().requires_grad_(True)
            u = up.detach().clone().requires_grad_(True)
            out = _FusedSiLUMul.apply(g.contiguous(), u.contiguous())
            out.backward(grad_out)
            return g.grad, u.grad

        pytorch_ms = bench_fn(pytorch_bwd)
        triton_ms = bench_fn(triton_bwd)
        speedup = pytorch_ms / triton_ms

        print(
            f"{label:>30s} | {pytorch_ms:>12.3f} | {triton_ms:>12.3f} | {speedup:>7.2f}x"
        )


# ============================================================
# RoPE Rotation Benchmarks
# ============================================================


def benchmark_rope():
    """Benchmark fused RoPE rotation kernel vs PyTorch."""
    device = torch.device("cuda")

    print("\n=== RoPE Rotation Benchmark ===")
    print(
        f"{'Config':>40s} | {'PyTorch (ms)':>12s} | {'Triton (ms)':>12s} | {'Speedup':>8s}"
    )
    print("-" * 80)

    configs = [
        (2, 128, 8, 64, "Small (2x128, 8h, d64)"),
        (2, 512, 8, 64, "Medium (2x512, 8h, d64)"),
        (2, 2048, 8, 64, "Long (2x2048, 8h, d64)"),
        (2, 128, 32, 128, "Large heads (2x128, 32h, d128)"),
        (2, 2048, 32, 128, "Large + long (2x2048, 32h, d128)"),
        (4, 4096, 32, 128, "Very large (4x4096, 32h, d128)"),
        (1, 1, 32, 128, "Decode step (1x1, 32h, d128)"),
        (8, 1, 32, 128, "Decode batch (8x1, 32h, d128)"),
    ]

    for batch, seq_len, num_heads, d_head, label in configs:
        for dtype, dtype_name in [
            (torch.float32, "f32"),
            (torch.bfloat16, "bf16"),
        ]:
            q = torch.randn(
                batch, seq_len, num_heads, d_head, device=device, dtype=dtype
            )

            half_dim = d_head // 2
            freqs = torch.randn(seq_len, half_dim, device=device, dtype=dtype)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(2)
            sin = emb.sin().unsqueeze(0).unsqueeze(2)

            # PyTorch baseline
            def pytorch_fn():
                return (q * cos) + (rotate_half(q) * sin)

            # Triton fused
            cos_t, sin_t = _prepare_cos_sin_for_triton(
                cos, sin, batch, seq_len, d_head
            )
            q_c = q.contiguous()

            def triton_fn():
                return _FusedRoPERotation.apply(q_c, cos_t, sin_t)

            pytorch_ms = bench_fn(pytorch_fn)
            triton_ms = bench_fn(triton_fn)
            speedup = pytorch_ms / triton_ms

            desc = f"{label} ({dtype_name})"
            print(
                f"{desc:>40s} | {pytorch_ms:>12.3f} | {triton_ms:>12.3f} | {speedup:>7.2f}x"
            )

    # Backward benchmark
    print("\n=== RoPE Rotation Backward Benchmark ===")
    print(
        f"{'Config':>40s} | {'PyTorch (ms)':>12s} | {'Triton (ms)':>12s} | {'Speedup':>8s}"
    )
    print("-" * 80)

    for batch, seq_len, num_heads, d_head, label in [
        (2, 512, 32, 128, "Typical (2x512, 32h, d128)"),
        (2, 2048, 32, 128, "Long (2x2048, 32h, d128)"),
    ]:
        q = torch.randn(
            batch, seq_len, num_heads, d_head, device=device, requires_grad=True
        )

        half_dim = d_head // 2
        freqs = torch.randn(seq_len, half_dim, device=device)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)
        grad_out = torch.randn_like(q)

        cos_t, sin_t = _prepare_cos_sin_for_triton(cos, sin, batch, seq_len, d_head)

        def pytorch_bwd():
            q_ = q.detach().clone().requires_grad_(True)
            out = (q_ * cos) + (rotate_half(q_) * sin)
            out.backward(grad_out)
            return q_.grad

        def triton_bwd():
            q_ = q.detach().clone().requires_grad_(True)
            out = _FusedRoPERotation.apply(q_.contiguous(), cos_t, sin_t)
            out.backward(grad_out)
            return q_.grad

        pytorch_ms = bench_fn(pytorch_bwd)
        triton_ms = bench_fn(triton_bwd)
        speedup = pytorch_ms / triton_ms

        print(
            f"{label:>40s} | {pytorch_ms:>12.3f} | {triton_ms:>12.3f} | {speedup:>7.2f}x"
        )


# ============================================================
# Memory Bandwidth Analysis
# ============================================================


def memory_bandwidth_analysis():
    """Analyze memory bandwidth utilization of kernels."""
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    # Get theoretical peak bandwidth (approximate for common GPUs)
    props = torch.cuda.get_device_properties(device)

    print(f"\n=== Memory Bandwidth Analysis ===")
    print(f"GPU: {gpu_name}")
    print(f"SMs: {props.multi_processor_count}")

    # SiLU*up: reads 2 tensors, writes 1 (forward)
    n_tokens = 2048
    d_ff = 11008  # Llama-7B
    dtype = torch.bfloat16
    bytes_per_elem = 2

    total_bytes_fwd = n_tokens * d_ff * bytes_per_elem * 3  # 2 reads + 1 write
    total_bytes_bwd = n_tokens * d_ff * bytes_per_elem * 5  # 3 reads + 2 writes

    gate = torch.randn(n_tokens, d_ff, device=device, dtype=dtype)
    up = torch.randn(n_tokens, d_ff, device=device, dtype=dtype)

    triton_ms_fwd = bench_fn(
        lambda: _FusedSiLUMul.apply(gate.contiguous(), up.contiguous())
    )

    bw_fwd = total_bytes_fwd / (triton_ms_fwd / 1000) / 1e9  # GB/s

    print(f"\nSiLU*up ({n_tokens}x{d_ff}, bf16):")
    print(f"  Forward:  {triton_ms_fwd:.3f} ms, {bw_fwd:.1f} GB/s effective bandwidth")

    # RoPE: reads 4 values (x1, x2, cos, sin), writes 2 (out1, out2) per pair
    batch, seq_len, num_heads, d_head = 2, 2048, 32, 128
    half_dim = d_head // 2
    n_pairs = batch * seq_len * num_heads * half_dim
    total_bytes_rope = n_pairs * bytes_per_elem * 6  # 4 reads + 2 writes

    q = torch.randn(batch, seq_len, num_heads, d_head, device=device, dtype=dtype)
    freqs = torch.randn(seq_len, half_dim, device=device, dtype=dtype)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(2)
    sin = emb.sin().unsqueeze(0).unsqueeze(2)
    cos_t, sin_t = _prepare_cos_sin_for_triton(cos, sin, batch, seq_len, d_head)

    triton_ms_rope = bench_fn(
        lambda: _FusedRoPERotation.apply(q.contiguous(), cos_t, sin_t)
    )
    bw_rope = total_bytes_rope / (triton_ms_rope / 1000) / 1e9

    print(f"\nRoPE ({batch}x{seq_len}x{num_heads}x{d_head}, bf16):")
    print(f"  Forward:  {triton_ms_rope:.3f} ms, {bw_rope:.1f} GB/s effective bandwidth")


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"SMs: {torch.cuda.get_device_properties(device).multi_processor_count}")

    benchmark_silu_mul()
    benchmark_rope()
    memory_bandwidth_analysis()


if __name__ == "__main__":
    main()
