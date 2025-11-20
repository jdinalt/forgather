#!/usr/bin/env python3
"""
Memory profiling script for large vocabulary output layers.

This reproduces the memory profile of Qwen3's last pipeline stage to understand
where the 20GB peak comes from.

Configuration matching Qwen3 1.7B:
- Hidden dim: 2048
- Vocabulary size: 151936
- Sequence length: 4096
- Batch size: 1 (per microbatch in pipeline parallel)
- dtype: bfloat16
"""

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
import sys
import os

# Add forgather to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from forgather.ml.loss import CausalLoss, ChunkedCausalLoss, FusedLinearCrossEntropy

# Try to import Apple's Cut Cross-Entropy
try:
    from cut_cross_entropy import linear_cross_entropy
    HAS_CCE = True
except ImportError:
    HAS_CCE = False
    linear_cross_entropy = None
    print("Note: cut-cross-entropy not installed. Install with:")
    print('  pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"')
    print()


def print_memory(label: str):
    """Print current and peak CUDA memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{label:40s} | Allocated: {allocated:6.3f} GB | Reserved: {reserved:6.3f} GB | Peak: {max_allocated:6.3f} GB")


def reset_memory():
    """Reset CUDA memory stats."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def profile_standard_approach(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    vocab_size: int,
    n_microbatches: int = 8,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Profile memory usage of standard approach:
    Linear layer → logits → loss function

    Simulates pipeline parallel with overlapped forward/backward.
    """
    print("\n" + "="*80)
    print("STANDARD APPROACH: Linear → Logits → Loss")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model components
    output_layer = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=dtype, device=device)
    loss_fn = CausalLoss()

    reset_memory()
    print_memory("After model creation")

    # Simulate pipeline parallel with microbatching
    # Key question: how many microbatches worth of activations are in flight?

    print(f"\nSimulating {n_microbatches} microbatches (batch_size={batch_size} each):")
    print("-" * 80)

    # Track activations that would be alive during overlapped execution
    alive_activations = []

    for mb_idx in range(n_microbatches):
        print(f"\nMicrobatch {mb_idx + 1}/{n_microbatches}:")

        # Forward pass
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        print_memory(f"  After creating inputs")

        # Output layer
        logits = output_layer(hidden_states)
        print_memory(f"  After output layer (logits created)")

        # Compute loss
        loss = loss_fn(logits, labels)
        print_memory(f"  After loss computation")

        # Store for backward (simulates pipeline keeping activations alive)
        # In real pipeline, activations are kept until backward completes
        alive_activations.append((hidden_states, logits, loss))
        print_memory(f"  After storing (alive={len(alive_activations)} mbs)")

        # Simulate overlapped backward: start backward when we have 3+ microbatches queued
        if len(alive_activations) >= 3:
            print(f"\n  Starting backward for microbatch {mb_idx - 2}...")
            old_hidden, old_logits, old_loss = alive_activations.pop(0)

            old_loss.backward(retain_graph=False)
            print_memory(f"    After backward")

            # Note: skipping optimizer.step() to avoid in-place modifications
            # In real training, gradients accumulate and optimizer runs once per step

            # Free references
            del old_hidden, old_logits, old_loss
            torch.cuda.empty_cache()
            print_memory(f"    After freeing refs (alive={len(alive_activations)} mbs)")

    # Process remaining microbatches
    print("\n\nProcessing remaining microbatches:")
    print("-" * 80)
    while alive_activations:
        print(f"\nBackward for remaining microbatch...")
        old_hidden, old_logits, old_loss = alive_activations.pop(0)

        old_loss.backward(retain_graph=False)
        print_memory(f"  After backward")

        del old_hidden, old_logits, old_loss
        torch.cuda.empty_cache()
        print_memory(f"  After freeing (alive={len(alive_activations)} mbs)")

    print("\n" + "="*80)
    print_memory("FINAL PEAK")
    print("="*80 + "\n")

    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0


def profile_chunked_loss_approach(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    vocab_size: int,
    n_microbatches: int = 8,
    chunk_size: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Profile memory with chunked loss (but still materializing full logits).
    """
    print("\n" + "="*80)
    print(f"CHUNKED LOSS APPROACH: Linear → Logits → ChunkedLoss (chunk_size={chunk_size})")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_layer = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=dtype, device=device)
    loss_fn = ChunkedCausalLoss(chunk_size=chunk_size)

    reset_memory()
    print_memory("After model creation")

    alive_activations = []

    for mb_idx in range(n_microbatches):
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        logits = output_layer(hidden_states)
        loss = loss_fn(logits, labels)

        alive_activations.append((hidden_states, logits, loss))

        if len(alive_activations) >= 3:
            old_hidden, old_logits, old_loss = alive_activations.pop(0)
            old_loss.backward(retain_graph=False)
            del old_hidden, old_logits, old_loss
            torch.cuda.empty_cache()

    while alive_activations:
        old_hidden, old_logits, old_loss = alive_activations.pop(0)
        old_loss.backward(retain_graph=False)
        del old_hidden, old_logits, old_loss
        torch.cuda.empty_cache()

    print_memory("FINAL PEAK")

    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0


def profile_fused_approach(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    vocab_size: int,
    n_microbatches: int = 8,
    chunk_size: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Profile memory with fused linear + cross-entropy (no logits materialization).

    This simulates what would happen if we could avoid materializing logits entirely.
    """
    print("\n" + "="*80)
    print(f"FUSED APPROACH: FusedLinearCrossEntropy (chunk_size={chunk_size}, NO LOGITS!)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fused_layer = FusedLinearCrossEntropy(
        hidden_dim, vocab_size, chunk_size=chunk_size, bias=False
    ).to(dtype=dtype, device=device)

    reset_memory()
    print_memory("After model creation")

    alive_activations = []

    for mb_idx in range(n_microbatches):
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Fused layer computes loss directly without materializing logits!
        loss = fused_layer(hidden_states, labels)

        alive_activations.append((hidden_states, loss))

        if len(alive_activations) >= 3:
            old_hidden, old_loss = alive_activations.pop(0)
            old_loss.backward(retain_graph=False)
            del old_hidden, old_loss
            torch.cuda.empty_cache()

    while alive_activations:
        old_hidden, old_loss = alive_activations.pop(0)
        old_loss.backward(retain_graph=False)
        del old_hidden, old_loss
        torch.cuda.empty_cache()

    print_memory("FINAL PEAK")

    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0


def profile_apple_cce_approach(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    vocab_size: int,
    n_microbatches: int = 8,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Profile memory with Apple's Cut Cross-Entropy (optimized Triton kernels).
    """
    if not HAS_CCE:
        print("\n" + "="*80)
        print("APPLE CCE APPROACH: Skipped (not installed)")
        print("="*80)
        return 0

    print("\n" + "="*80)
    print("APPLE CCE APPROACH: linear_cross_entropy (Triton kernels)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output layer (CCE takes weight matrix separately)
    output_weight = nn.Parameter(
        torch.randn(vocab_size, hidden_dim, dtype=dtype, device=device)
    )

    reset_memory()
    print_memory("After creating weight matrix")

    alive_activations = []

    for mb_idx in range(n_microbatches):
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_dim, dtype=dtype, device=device, requires_grad=True
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Apple's CCE with automatic shifting
        # e = embeddings (hidden_states), c = classifier weight, shift=1 for causal LM
        loss = linear_cross_entropy(
            hidden_states,  # e
            output_weight,  # c
            labels,         # targets
            shift=1,        # Automatic causal shifting
            reduction="mean"
        )

        alive_activations.append((hidden_states, loss))

        if len(alive_activations) >= 3:
            old_hidden, old_loss = alive_activations.pop(0)
            old_loss.backward(retain_graph=False)
            del old_hidden, old_loss
            torch.cuda.empty_cache()

    while alive_activations:
        old_hidden, old_loss = alive_activations.pop(0)
        old_loss.backward(retain_graph=False)
        del old_hidden, old_loss
        torch.cuda.empty_cache()

    print_memory("FINAL PEAK")

    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0


def main():
    """Run memory profiling experiments."""

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires a GPU.")
        return

    # Qwen3 1.7B configuration
    config = {
        "batch_size": 1,  # Per microbatch
        "seq_len": 4096,
        "hidden_dim": 2048,
        "vocab_size": 151936,
        "n_microbatches": 8,
        "dtype": torch.bfloat16,
    }

    print("="*80)
    print("MEMORY PROFILING: Large Vocabulary Output Layer")
    print("="*80)
    print("\nConfiguration (matching Qwen3 1.7B last pipeline stage):")
    print(f"  Batch size per microbatch: {config['batch_size']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Hidden dimension: {config['hidden_dim']}")
    print(f"  Vocabulary size: {config['vocab_size']}")
    print(f"  Number of microbatches: {config['n_microbatches']}")
    print(f"  dtype: {config['dtype']}")
    print()

    # Theoretical calculation
    logits_size = config['batch_size'] * config['seq_len'] * config['vocab_size'] * 2 / 1024**3
    hidden_size = config['batch_size'] * config['seq_len'] * config['hidden_dim'] * 2 / 1024**3
    weight_size = config['hidden_dim'] * config['vocab_size'] * 2 / 1024**3

    print("Theoretical memory per microbatch:")
    print(f"  Hidden states: {hidden_size:.3f} GB")
    print(f"  Logits: {logits_size:.3f} GB")
    print(f"  Output layer weight: {weight_size:.3f} GB")
    print(f"  Output layer gradient: {weight_size:.3f} GB (accumulated)")
    print()

    # Run experiments
    results = {}

    print("\n" + "#"*80)
    print("# EXPERIMENT 1: Standard Approach")
    print("#"*80)
    results['standard'] = profile_standard_approach(**config)

    print("\n" + "#"*80)
    print("# EXPERIMENT 2: Chunked Loss (still materializes logits)")
    print("#"*80)
    results['chunked_loss'] = profile_chunked_loss_approach(**config, chunk_size=4096)

    print("\n" + "#"*80)
    print("# EXPERIMENT 3: Fused Approach (no logits materialization)")
    print("#"*80)
    results['fused'] = profile_fused_approach(**config, chunk_size=4096)

    print("\n" + "#"*80)
    print("# EXPERIMENT 4: Apple CCE (Triton-optimized kernels)")
    print("#"*80)
    results['apple_cce'] = profile_apple_cce_approach(**config)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nPeak memory usage:")
    print(f"  Standard approach:        {results['standard']:6.3f} GB")
    print(f"  Chunked loss:             {results['chunked_loss']:6.3f} GB")
    print(f"  Fused approach:           {results['fused']:6.3f} GB")
    if HAS_CCE and results['apple_cce'] > 0:
        print(f"  Apple CCE:                {results['apple_cce']:6.3f} GB")
    print()
    print(f"Memory savings:")
    print(f"  Chunked loss vs standard: {results['standard'] - results['chunked_loss']:6.3f} GB ({(results['standard'] - results['chunked_loss'])/results['standard']*100:.1f}%)")
    print(f"  Fused vs standard:        {results['standard'] - results['fused']:6.3f} GB ({(results['standard'] - results['fused'])/results['standard']*100:.1f}%)")
    if HAS_CCE and results['apple_cce'] > 0:
        print(f"  Apple CCE vs standard:    {results['standard'] - results['apple_cce']:6.3f} GB ({(results['standard'] - results['apple_cce'])/results['standard']*100:.1f}%)")
    print()

    print("Interpretation:")
    if results['chunked_loss'] < results['standard'] * 0.95:
        print("  ✓ Chunked loss provides meaningful memory savings")
    else:
        print("  ✗ Chunked loss does NOT provide meaningful savings (logits still materialized)")

    if results['fused'] < results['standard'] * 0.7:
        print("  ✓ Fused approach provides significant memory savings")
    else:
        print("  ~ Fused approach provides moderate savings")

    if HAS_CCE and results['apple_cce'] > 0:
        if results['apple_cce'] < results['standard'] * 0.7:
            print("  ✓ Apple CCE provides significant memory savings (production-ready!)")
        else:
            print("  ~ Apple CCE provides moderate savings")

    print()


if __name__ == "__main__":
    main()
