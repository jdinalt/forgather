# Tiny Experiments

A collection of small-scale experiments and integration tests built around the [TinyStories](https://arxiv.org/abs/2305.07759) dataset. Because a ~4M parameter model trained for one epoch on TinyStories takes only a few minutes on a single GPU, this workspace is well-suited for quickly validating trainer features, comparing architectures, and benchmarking training techniques without committing to a long run. Projects in this collection serve dual roles: as working examples of Forgather features and as regression tests for the library itself.

The workspace shares a common template library and base project configuration (`forgather_workspace/`) so that individual projects only need to declare what differs from the common defaults.

## Model Architectures

- **[tiny_models](./tiny_models/README.md)** - Train and compare small (~4M parameter) causal LM architectures — Vanilla Transformer, Llama, DeepOne, Mistral, Qwen3, and Llama+Canon — under identical conditions with a shared BPE tokenizer.
- **[canon](./canon/README.md)** - Experiments with Canon layers (depthwise causal 1D convolutions inserted into transformer blocks) from *Physics of Language Models: Part 4.1*. Investigates how Canon interacts with and partially substitutes for positional encoding.

## Trainers

- **[compare_trainers](./compare_trainers/README.md)** - Side-by-side comparison of Forgather's built-in trainer, the Accelerate-based trainer, and the HuggingFace Transformers trainer on the same workload.
- **[ddp_trainer](./ddp_trainer/README.md)** - Integration tests and usage examples for the DDP trainer: distributed checkpointing, dataset distribution strategies, and gradient accumulation across data-parallel ranks.
- **[fsdp2_trainer](./fsdp2_trainer/README.md)** - Integration test for the FSDP2 (`fully_shard`) trainer. Exercises layer-wise sharding, sharded DTensor checkpoint save/resume, and CPU offload.
- **[pipeline_parallel](./pipeline_parallel/README.md)** - Test harness for the Pipeline Parallel trainer covering all supported PyTorch pipeline schedules, checkpoint save/resume, and activation checkpointing.
- **[diloco](./diloco/README.md)** - Demonstrates DiLoCo (Distributed Low-Communication) training via `DiLoCoCallback`: multiple independent workers synchronize outer gradients at configurable intervals over standard network interfaces.
- **[diloco_lowprec](./diloco_lowprec/README.md)** - Low-precision wire transport for DiLoCo: bf16 pseudo-gradient / parameter legs with optional stochastic rounding, and true-bf16 weight training — measuring the convergence cost of halving sync bandwidth.
- **[diloco_features](./diloco_features/README.md)** - Measures what DiLoCo's communication-reduction knobs (streaming, async, DN-buffer, DyLU) cost in convergence, on a controlled ~1B-token run against the synchronous baseline.

## Training Techniques

- **[checkpointing](./checkpointing/README.md)** - Demonstrates optimizer and learning-rate scheduler checkpoint save/restore, automatic checkpoint discovery, and robust resume logic across multiple training sessions.
- **[grad_accumulation](./grad_accumulation/README.md)** - Validates that gradient accumulation produces equivalent training curves to proportionally larger batch sizes across the standard, pipeline parallel, and Accelerate DDP trainers.
- **[peak_memory](./peak_memory/README.md)** - Systematic comparison of memory optimization techniques on a 1.6B parameter model: bfloat16, activation checkpointing, `torch.compile`, fused optimizer kernels, and combinations thereof.
- **[optimizers](./optimizers/README.md)** - Benchmarks ten optimizer configurations (AdamW, SGD variants, AMP, pure bfloat16, gradient accumulation) on a 30M parameter Llama model, examining whether adaptive optimizers are necessary for small-batch LLM pre-training.
- **[sinkgd](./sinkgd/README.md)** - Hyperparameter sweep for the Sinkhorn GD optimizer, a rounding-aware optimizer that applies doubly-stochastic normalization to weight matrices.

## Scaffold

- **[tiny_experiment](./tiny_experiment/README.md)** - Minimal project that exercises the default `projects/tiny.yaml` base template. Used to validate that the base project templates work correctly and as a starting point for new ablations.
