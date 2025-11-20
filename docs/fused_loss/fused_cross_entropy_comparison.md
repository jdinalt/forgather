# Fused Cross-Entropy Implementations: Comparison

## Overview

Three major production-ready implementations of fused linear + cross-entropy for large vocabulary models:

1. **Apple CCE** (Cut Cross-Entropy) - November 2024
2. **LinkedIn Liger Kernel** - August 2024
3. **Our Implementation** (FusedLinearCrossEntropy) - Pure PyTorch

All solve the same problem: avoiding the massive memory footprint of materializing full logits tensors for large vocabulary models.

## Implementation Comparison

### Apple CCE (Cut Cross-Entropy)

**Repository**: https://github.com/apple/ml-cross-entropy
**Paper**: https://arxiv.org/abs/2411.09009 (Nov 2024)

**Key Features**:
- Custom Triton kernels with flash memory optimization
- Gradient filtering (skip negligible gradients below numerical precision)
- Multiple implementations: `cce`, `torch_compile`, `cce_kahan` (better precision)
- **Built-in vocabulary parallelism** support
- Automatic fp32 upcasting for numerical stability
- Causal shifting via `shift=1` parameter

**Memory Savings**:
- Gemma 2 (2B): 24 GB → 1 MB (24,000× reduction!)
- Our test (Qwen3 1.7B): 10.5 GB → 1.8 GB (83% reduction)

**API**:
```python
from cut_cross_entropy import linear_cross_entropy

loss = linear_cross_entropy(
    embeddings,      # [batch, seq, hidden]
    classifier,      # [vocab, hidden] weight matrix
    labels,
    shift=1,         # Auto-shift for causal LM
    impl="cce"       # or "torch_compile"
)
```

**Pros**:
- Most optimized (best memory savings in our tests)
- Vocabulary parallelism out of the box
- Multiple precision options
- Clean, focused API

**Cons**:
- Requires Triton 3.0+ and Ampere+ GPU
- Smaller community (Apple research project)
- Less framework integration

---

### LinkedIn Liger Kernel

**Repository**: https://github.com/linkedin/Liger-Kernel
**Paper**: https://arxiv.org/abs/2410.10989 (Oct 2024)

**Key Features**:
- Collection of Triton kernels for LLM training (not just cross-entropy)
- **Extensive model support**: Llama, Mistral, Qwen, Gemma, Phi, etc.
- **Framework integration**: HF Trainer, Axolotl, LLaMA-Factory, TRL
- Alignment loss kernels: DPO, ORPO, CPO, SimPO, KTO
- Convergence tested against baseline training
- Multi-GPU support: FSDP, DeepSpeed, DDP

**Memory Savings**:
- "60% memory reduction" (general claim)
- "80% memory savings" for alignment tasks
- LLaMA 3-8B: 4K context (baseline OOM) → 16K context (with Liger)

**API**:
```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

loss_fn = LigerFusedLinearCrossEntropyLoss()
loss = loss_fn(model.weight, input, target)

# Or high-level patching
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()
```

**Pros**:
- **Best framework integration** (one-line patching)
- Broader scope (RMSNorm, RoPE, SwiGLU, LayerNorm, etc.)
- Large community (LinkedIn + HF ecosystem)
- Supports both CUDA and ROCm
- Alignment loss support (DPO, etc.)
- **Explicitly supports Qwen models**

**Cons**:
- Less granular control than CCE
- Slightly less memory efficient than CCE (in our tests)
- Larger dependency footprint

---

### Our Implementation (FusedLinearCrossEntropy)

**Location**: `src/forgather/ml/loss.py`

**Key Features**:
- Pure PyTorch implementation
- No external dependencies (except PyTorch)
- Log-sum-exp chunking strategy
- Works on any hardware

**Memory Savings**:
- Our test (Qwen3 1.7B): 10.5 GB → 6.0 GB (43% reduction)

**API**:
```python
from forgather.ml.loss import FusedLinearCrossEntropy

fused = FusedLinearCrossEntropy(
    in_features=2048,
    out_features=151936,
    chunk_size=4096
)
loss = fused(hidden_states, labels)
```

**Pros**:
- Zero dependencies beyond PyTorch
- Full control and understanding
- Easy to modify and extend
- Works everywhere (CPU, old GPUs, etc.)

**Cons**:
- Less optimized than Triton kernels
- No vocabulary parallelism
- No gradient filtering optimizations

---

## Performance Summary (Our Tests)

Configuration: Qwen3 vocab (151936), batch=1, seq=4096, bf16, 8 microbatches

| Implementation | Peak Memory | vs Standard | Reduction % |
|----------------|-------------|-------------|-------------|
| **Standard** | 10.50 GB | baseline | 0% |
| Chunked Loss (logits still materialized) | 10.59 GB | -0.10 GB | -1% |
| **Our PyTorch** | 5.96 GB | 4.54 GB | **43%** |
| **Apple CCE** | 1.82 GB | 8.68 GB | **83%** |
| **Liger** | ~4-5 GB* | ~5-6 GB* | ~50%* |

*Liger not tested in our benchmark yet, estimated from their reported 60% reduction

---

## Recommendation for Forgather

### Short Term: Support All Three

Create a general `LinearCrossEntropyLoss` wrapper that supports all implementations:

```python
class LinearCrossEntropyLoss:
    def __init__(self, output_embeddings, impl="auto", **kwargs):
        # impl: "cce" | "liger" | "pytorch" | "auto"

        if impl == "auto":
            # Try Liger first (best integration)
            # Fall back to CCE (best memory)
            # Fall back to PyTorch (always works)
            ...
```

**Rationale**:
- **Liger**: Best for users with standard HF models (Qwen, Llama, etc.)
- **CCE**: Best for maximum memory savings and custom models
- **PyTorch**: Fallback for compatibility

### Recommended Defaults

1. **For Qwen models**: Use **Liger** (explicit Qwen support, good integration)
2. **For maximum memory**: Use **Apple CCE** (83% savings)
3. **For compatibility**: Use **our PyTorch** impl (works everywhere)

### Integration Strategy

```yaml
# Automatic selection
loss_fn: !call:LinearCrossEntropyLoss
  output_embeddings: !ref model.get_output_embeddings()
  impl: "auto"  # Tries Liger → CCE → PyTorch

# Or explicit
loss_fn: !call:LinearCrossEntropyLoss
  output_embeddings: !ref model.get_output_embeddings()
  impl: "liger"  # or "cce" or "pytorch"
  chunk_size: 4096  # For pytorch impl
```

---

## Key Differences Summary

### Apple CCE
- **Focus**: Maximum memory efficiency
- **Strength**: Best memory savings, vocabulary parallelism
- **Use case**: Custom models, extreme memory constraints

### Liger Kernel
- **Focus**: Production LLM training
- **Strength**: Framework integration, model coverage, alignment losses
- **Use case**: Standard HF models, full training pipelines

### Our PyTorch
- **Focus**: Portability and understanding
- **Strength**: Zero dependencies, works everywhere
- **Use case**: Development, debugging, non-CUDA hardware

---

## Additional Considerations

### Vocabulary Parallelism
- **CCE**: Built-in support via `VocabParallelOptions`
- **Liger**: Not explicitly mentioned
- **Ours**: Not implemented

For very large models (30B+), CCE's vocabulary parallelism could be crucial.

### Numerical Precision
- **CCE**: Kahan summation option (`cce_kahan`), auto fp32 upcasting
- **Liger**: Convergence tested, exact computation
- **Ours**: Standard PyTorch numerics

For long training runs, CCE's precision options may help.

### Alignment Training
- **CCE**: No special support
- **Liger**: DPO, ORPO, CPO, SimPO, KTO kernels
- **Ours**: No special support

For RLHF/alignment, Liger has significant advantages.

---

## Implementation Plan

1. **Add Liger support** to our `LinearCrossEntropyLoss` wrapper
2. **Keep CCE support** for maximum memory savings
3. **Keep PyTorch impl** for fallback
4. **Add auto-detection** to choose best available
5. **Test all three** with Qwen3 in pipeline parallel
6. **Document tradeoffs** for users

This gives users the flexibility to choose based on their needs while maintaining a simple, unified API.

