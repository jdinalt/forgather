# Peak Memory

Experiments in peak GPU memory utilization

## Overview

This project systematically compares various memory optimization techniques for training large language models. Using a 1.6B parameter transformer model (Deep One) with TinyStories dataset, we evaluated 9 different memory optimization configurations to understand their individual and combined effects on peak GPU memory usage.

## Techniques Tested

### 1. **BFloat16 Precision** (`bfloat16.yaml`)
Converts model weights and activations from 32-bit floating point to 16-bit Brain Floating Point format. BFloat16 maintains the same dynamic range as FP32 but with reduced precision, offering better numerical stability than FP16 while halving memory requirements for model parameters.

### 2. **Activation Checkpointing** (`checkpoint.yaml`)
Trades computation for memory by discarding intermediate activations during forward pass and recomputing them during backpropagation. This technique can reduce memory complexity from O(n) to O(√n) for n-layer networks, with typically 10-20% increase in training time.

### 3. **PyTorch Compilation** (`compile.yaml`)
Uses `torch.compile` with the default inductor backend to optimize the model through kernel fusion, memory access pattern optimization, and other graph-level optimizations. The compilation process specializes kernels for specific tensor shapes and operations.

### 4. **Optimizer Fusion** (`fused.yaml`)
Enables `fuse_optim_with_backward=True` to perform optimizer steps during the backward pass rather than as separate operations. This reduces peak memory by eliminating the need to store gradients separately before applying optimizer updates.

### 5. **Memory Budget Control** (`memory_budget_05.yaml, memory_budget_60.yaml`)
This makes use of a new, experimental, PyTorch "compile" feature, which uses a knapsack solver to find the minimal recomputation necessary to stay below the activation memory budget.
1.0 represents the activation memmory from the default "compile" strategy, while 0.0 corresponds to appling activation checkpointing to everything within the full compiled region.

### 6. **Combined Techniques**
- **Fused Checkpoint** (`fused_checkpoint.yaml`): Combines activation checkpointing with optimizer fusion
- **BF16 Fused Checkpoint** (`bf16_fused_chkpt.yaml`): Combines all three: BFloat16, activation checkpointing, and optimizer fusion
- **Fused Memory Budget** (`fused_memory_budget_05.yaml`): Combines torch compile memory budget with optimizer fusion

## Results

| Configuration | Peak Memory (GiB) | Memory Reduction | Speed (samples/sec) | Relative Time | Eval Perplexity |
|---------------|------------------|------------------|------------------|------------------|------------------|
| **Control** (FP32 baseline) | 21.165 | - | 2.518 | 1m 40s | 152.1 |
| **BFloat16** | 12.418 | **41.3%** | 6.79 | 41s | 180.0 |
| **Activation Checkpoint** | 12.529 | **40.8%** | 2.03 | 2m 15s | 152.1 |
| **Torch Compile** | 19.712 | **6.9%** | 2.307 | 1m 36s | 152.1 |
| **Optimizer Fusion** | 21.164 | **~0%** | 2.535 | 1m 40s | 152.1 |
| **Fused Checkpoint** | 7.425 | **64.9%** | 2.049 | 2m 14s | 152.1 |
| **BF16 + Fused + Checkpoint** | 3.975 | **81.2%** | 5.834 ↑ | 49s | 182.5 |
| **Memory Budget 0.05** | 13.561 | **35.9%** | 1.574 | 2m 5s | 152.1 |
| **Memory Budget 0.60** | 15.636 | **26.1%** | 1.760 | 1m 38s | 152.1 |
| **Fused Memory Budget 0.05** | 13.562 | **35.9%** | 1.938 | 1m 40s | 152.1 |
| **BF16 + Budget + 0.60** | 6.709 | **68.3%** | 2.76 | 45s | 182.4 |

## Analysis and Interpretation

### Most Effective Individual Techniques
1. **BFloat16** achieved 41.3% memory reduction while **doubling training speed** - the best single optimization
2. **Activation Checkpointing** provided 40.8% memory reduction but slowed training by ~20%
3. **Torch Compile** provided modest 6.9% memory savings with slight speed reduction
4. **Optimizer Fusion** alone showed negligible memory impact but small speed improvement

### Synergistic Effects
The most impressive results came from combining techniques:
- **Fused Checkpoint** (checkpointing + optimizer fusion) achieved 64.9% memory reduction
- **BF16 + Fused + Checkpoint** delivered the ultimate 81.2% memory reduction while maintaining good training speed

### Key Insights

1. **BFloat16 is the clear winner** for single-technique optimization, providing massive memory and speed benefits with minimal downsides
2. **Activation checkpointing scales well** when combined with other techniques, despite its individual speed penalty
3. **Torch compile's memory budget feature** can provide signifcant savings and compares favorably with traditional checkpointing in terms of both memory and performance
4. **Optimizer fusion's value** emerges primarily when combined which normal activation checkpointing; this may hint at a potential bug in PyTorch?
5. **Combined approaches unlock exponential gains** - the best configuration used only 18.8% of baseline memory

### Practical Recommendations

- **For immediate gains**: Start with BFloat16 conversion - easiest to implement with best ROI
- **For memory-constrained training**: Use the "BF16 + Fused + Checkpoint" combination for maximum memory savings
- **For production training**: BFloat16 alone often provides the best speed/memory trade-off
- **Avoid**: Relying solely on torch compile or optimizer fusion for memory optimization

### Technical Notes

All experiments used identical model architecture (32-layer transformer, 2048 hidden size, 1.6B parameters) and training configuration (batch size 4, 100 steps) to ensure fair comparison. Memory measurements represent peak allocated CUDA memory during training, not just model parameter memory.

The training speed metrics includes the additional time required for torch compile, where applicable. Given that we only sampled over 100 steps, this distorts the actual performance. To help mitigate this issue, the reported Tensor Board relative completion times have been reported, which is the time between step 10 and step 100.

Given the lack of results for optimizer fusion, without activation checkpoiting, it seems plausible that something in the backward pass is holding references to the gradients until complete, whereas with activation checkpointing, something about the discontiguous gradeint computations is likely mitigating the problem. TBD.

**Hardware**

- CPU: AMD Ryzen Threadripper PRO 5955WX 16-Cores
- GPU: 1 x RTX 4090
- Memory: 258GiB

**Software**

- PyTorch v2.8.0
- CUDA Version: 12.4
- NVIDIA-SMI 550.54.15
- torch.set_float32_matmul_precision = "highest"

**Comand Line**

The basic command structure for running the experiments:

```bash
# List experiments
forgather ls

# Run experiment
forgather -t control.yaml train
```

The memory budget API can be configured to generate a compute vs memory plot when executed like this:
```bash
PARTITIONER_MEMORY_BUDGET_PARETO=1 PARTITIONER_MEMORY_BUDGET_PARETO_DIR="./" forgather -t memoy_budget_05.yaml train
```
This takes a considerable amount of time, as the model must be recompiled for each data-point.

**Memory Budget Pareto**

- [memory_budget_pareto_model__0_joint_0.svg](./memory_budget_pareto_model__0_joint_0.svg)
- [memory_budget_pareto_model__1_joint_3.svg](./memory_budget_pareto_model__1_joint_3.svg)

### References

- [Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [How to save memory by fusing the optimizer step into the backward pass](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)
- [activation_memory_budget source code](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/config.py#L140)
- [torch.utils.checkpoint](https://docs.pytorch.org/docs/2.8/checkpoint.html)
