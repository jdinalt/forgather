"""
Benchmark trainable-ALiBi + flex_attention vs RoPE + flex_attention.

Measures forward + backward latency and peak memory for both attention
mechanisms across the backends supported by the Forgather attention
interface (eager / sdpa / flex_attention), at geometries matching the
deepone medium config.

Usage:
    python tests/benchmark_alibi_vs_rope.py                  # defaults
    python tests/benchmark_alibi_vs_rope.py --seq-lens 1024,2048,4096
    python tests/benchmark_alibi_vs_rope.py --no-kernel-options

Notes:
- Requires CUDA and a Triton-capable GPU for the flex_attention path.
- Uses deepone's `flex_attn_kernel_options` by default, which matches the
  production training configuration. Pass `--no-kernel-options` to see
  what autotune picks on its own (can OOM on RTX 30xx / 40xx for some
  choices; see templatelib/examples/flex_kernel_options/default.yaml).
- The RoPE row uses a minimal inline rotary encoder -- this is not the
  production forgather RoPE implementation, but reproduces the flex
  attention cost with the same Q/K rotation overhead.
"""

import argparse
import math
import sys
import time

import torch
import torch._dynamo as dynamo
import torch.nn as nn

REPO_ROOT = "/home/dinalt/code/forgather"

# modelsrc/transformer is not a package; add to path for direct import.
sys.path.insert(0, f"{REPO_ROOT}/modelsrc/transformer")

from attention_interface import (  # noqa: E402
    eager_attention_forward,
    flex_attention_forward,
    sdpa_attention_forward,
)
from causal_alibi_attn import CausalAlibiAttn  # noqa: E402
from causal_multihead_attn import CausalMultiheadAttn  # noqa: E402

# Geometry matching deepone/medium.yaml: hidden=768, 8 heads, d_head=96.
HIDDEN = 768
HEADS = 8
BATCH = 4
DTYPE = torch.bfloat16
DEVICE = "cuda"

# deepone's default.yaml flex_attn_kernel_options (pre-PT 2.9 workaround
# for RTX 30xx/40xx autotune OOM).
DEEPONE_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_M1": 16,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 16,
}


def build_block_mask(seq):
    from torch.nn.attention.flex_attention import create_block_mask

    def causal(b, h, q, kv):
        return q >= kv

    return create_block_mask(
        causal, B=None, H=None, Q_LEN=seq, KV_LEN=seq, device=DEVICE
    )


def build_causal_4d(seq, dtype):
    mask = torch.full(
        (1, 1, seq, seq), float("-inf"), device=DEVICE, dtype=torch.float32
    )
    mask = torch.triu(mask, diagonal=1).to(dtype)
    return mask


def make_inputs(seq):
    """Return a single `hidden_states` tensor wired up to the module's
    Q/K/V/O projections. Matches how training would call the attention
    module."""
    return torch.randn(BATCH, seq, HIDDEN, device=DEVICE, dtype=DTYPE)


def rope_cos_sin(seq, d_head, base=10000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, d_head, 2, device=DEVICE).float() / d_head)
    )
    t = torch.arange(seq, device=DEVICE, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0).to(DTYPE), emb.sin().unsqueeze(0).to(DTYPE)


def make_alibi_module(impl, kernel_options=None):
    attn_functions = {
        "eager": eager_attention_forward,
        "sdpa": sdpa_attention_forward,
        "flex_attention": flex_attention_forward,
    }
    mod = CausalAlibiAttn(
        d_model=HIDDEN,
        num_heads=HEADS,
        attn_implementation=impl,
        attn_functions=attn_functions,
        bias=False,
        dropout=0.0,
        trainable_alibi=True,
        alt_alibi_init=True,
        layer_idx=0,
    )
    return mod.to(device=DEVICE, dtype=DTYPE)


def make_rope_module(impl, seq, d_head):
    attn_functions = {
        "eager": eager_attention_forward,
        "sdpa": sdpa_attention_forward,
        "flex_attention": flex_attention_forward,
    }
    mod = CausalMultiheadAttn(
        d_model=HIDDEN,
        num_heads=HEADS,
        attn_implementation=impl,
        attn_functions=attn_functions,
        bias=False,
        dropout=0.0,
        layer_idx=0,
    )
    mod = mod.to(device=DEVICE, dtype=DTYPE)

    cos, sin = rope_cos_sin(seq, d_head)

    def rotate_half(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def pos_encoder(q, k, **kwargs):
        c = cos.unsqueeze(2)  # (1, S, 1, D)
        s = sin.unsqueeze(2)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        return q, k

    mod.pos_encoder = pos_encoder
    return mod


def bench_step(mod, hidden, mask, extra_kwargs, warmup=3, iters=10):
    """Measure (cold first-call, steady-state, peak memory) for fwd+bwd."""
    extra_kwargs = extra_kwargs or {}

    def run():
        hs = hidden.detach().clone().requires_grad_(True)
        out = mod(hs, attention_mask=mask, **extra_kwargs)
        out.float().pow(2).mean().backward()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first = time.perf_counter() - t0

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    steady = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    return first, steady, peak


def run_suite(seq, kernel_options, warmup, iters):
    print(
        f"\n=== seq={seq} batch={BATCH} hidden={HIDDEN} heads={HEADS} "
        f"dtype={DTYPE} kernel_options={'deepone' if kernel_options else 'default'} ==="
    )

    d_head = HIDDEN // HEADS
    hidden = make_inputs(seq)
    block_mask = build_block_mask(seq)
    mask_4d = build_causal_4d(seq, DTYPE)

    rows = []

    for label, is_alibi, impl in [
        ("ALiBi eager", True, "eager"),
        ("ALiBi sdpa", True, "sdpa"),
        ("ALiBi flex_attention", True, "flex_attention"),
        ("RoPE  eager", False, "eager"),
        ("RoPE  sdpa", False, "sdpa"),
        ("RoPE  flex_attention", False, "flex_attention"),
    ]:
        dynamo.reset()
        mask = block_mask if impl == "flex_attention" else mask_4d
        extra = {}
        if impl == "flex_attention" and kernel_options is not None:
            extra["kernel_options"] = kernel_options

        if is_alibi:
            mod = make_alibi_module(impl)
        else:
            mod = make_rope_module(impl, seq, d_head)

        try:
            first, steady, peak = bench_step(
                mod, hidden, mask, extra, warmup=warmup, iters=iters
            )
            rows.append((label, first, steady, peak))
        except Exception as e:
            msg = str(e).split("\n")[0][:80]
            rows.append((label, float("nan"), float("nan"), float("nan")))
            print(f"  {label:<22}  FAILED: {msg}")
            continue

        del mod
        torch.cuda.empty_cache()

    # Print summary table
    print(
        f"{'backend':<22}  {'first (ms)':>10}  {'steady (ms)':>12}  {'peak (MiB)':>10}"
    )
    for label, first, steady, peak in rows:
        if math.isnan(steady):
            continue
        print(
            f"  {label:<22}  {first*1000:>10.1f}  {steady*1000:>12.2f}  {peak:>10.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq-lens",
        default="1024,2048,4096",
        help="Comma-separated sequence lengths to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--no-kernel-options",
        action="store_true",
        help="Omit deepone's FlexKernelOptions override (lets autotune choose).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for flex_attention bench; skipping.")
        return

    print(f"PyTorch: {torch.__version__}  GPU: {torch.cuda.get_device_name(0)}")

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    kernel_options = None if args.no_kernel_options else DEEPONE_KERNEL_OPTIONS

    for seq in seq_lens:
        run_suite(seq, kernel_options, args.warmup, args.iters)


if __name__ == "__main__":
    main()
