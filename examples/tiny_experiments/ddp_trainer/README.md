# DDP Trainer Example and Integration Tests

This project serves multiple purposes:
- **Integration testing** for DDP trainer features (distributed training, checkpointing, dataset strategies, gradient accumulation)
- **Usage examples** for configuring DDP training with different dataset patterns
- **Performance benchmarking** for data-parallel configurations

## Quick Start

```bash
# 2-GPU training (default configuration)
forgather -t 2gpu.yaml train

# 4-GPU training with sharded dataset
forgather -t 4gpu.yaml train

# Single process (baseline comparison)
forgather -t single_processs.yaml train

# View training logs
forgather logs list
forgather logs summary --all --format one-line
forgather logs plot --loss-curves
```

## Overview

The DDP (Distributed Data Parallel) trainer implements efficient multi-GPU training using PyTorch's `DistributedDataParallel`. It provides:

- **Automatic gradient synchronization** across GPUs
- **Flexible dataset loading** strategies (dispatch vs. sharding)
- **Distributed checkpointing** with full state restoration
- **Gradient accumulation** support
- **Transparent scaling** from single GPU to multi-node

**Trainer Implementation**: `src/forgather/ml/trainer/ddp/ddp_trainer.py`

## Configuration Files

All configurations extend `project.yaml` which sets up a small transformer model on Tiny Stories dataset:

### Basic DDP Configurations

| Config | Purpose | GPUs | Batch Size | Dataset Strategy |
|--------|---------|------|------------|------------------|
| **2gpu.yaml** | Default 2-GPU setup | 2 | 64 (32×2) | Dispatched batches |
| **4gpu.yaml** | 4-GPU training | 4 | 128 (32×4) | Sharded dataset |
| **single_processs.yaml** | Single-process baseline | 1 | 32 | N/A |
| **4cpu.yaml** | CPU-based DDP | 4 CPUs | 64 (32×2) | Dispatched batches |

### Dataset Strategy Configurations

| Config | Dataset Type | Sharding | Purpose |
|--------|--------------|----------|---------|
| **sharded_dataset.yaml** | Standard (Arrow) | Per-rank sharding | Test dataset sharding on standard HF datasets |
| **iterable_dataset.yaml** | HF IterableDataset | None (dispatched) | Baseline iterable dataset performance |
| **sharded_iterable_dataset.yaml** | HF IterableDataset | Per-rank sharding | Sharded iterable dataset |
| **sharded_fast_dataset.yaml** | Fast Iterable | Per-rank sharding | Optimized fast iterable dataset |

### Advanced Configurations

| Config | Feature | Description |
|--------|---------|-------------|
| **grad_accum.yaml** | Gradient Accumulation | Tests DDP with `gradient_accumulation_steps: 4`, smaller batch size |
| **checkpoint_train.yaml** | Checkpoint Save | Trains 500 steps with full checkpointing (model, optimizer, scheduler, dataset, RNG) |
| **checkpoint_resume.yaml** | Checkpoint Resume | Resumes from checkpoint and continues to 1000 steps |

## Performance Analysis

Performance data from actual training runs on Tiny Stories dataset (10% subset, ~212k samples):

### Scaling Efficiency

```bash
# View performance comparison
forgather logs summary --all --format one-line
```

| Configuration | GPUs | Steps | Time | Throughput | Speedup | Efficiency |
|--------------|------|-------|------|------------|---------|------------|
| Single Process | 1 | 6600 | 1:57 | 1,804 samples/s | 1.0× | 100% |
| 2-GPU (dispatched) | 2 | 3300 | 1:35 | 2,224 samples/s | 1.23× | 62% |
| 2-GPU (sharded) | 2 | 3300 | 1:15 | 2,818 samples/s | 1.56× | 78% |
| 4-GPU (sharded) | 4 | 1600 | 0:40 | 5,262 samples/s | 2.92× | 73% |

**Key Findings**:
- **Sharded datasets outperform dispatched batches**: ~27% faster (2,818 vs 2,224 samples/s on 2 GPUs)
- **Near-linear scaling to 4 GPUs**: 2.92× speedup with 73% efficiency
- **Communication overhead**: ~20-40% efficiency loss due to gradient synchronization

### Dataset Loading Strategies

| Strategy | Throughput | Pros | Cons |
|----------|------------|------|------|
| **Dispatched** (default) | 2,224 samples/s | Simple, centralized state | Rank-0 bottleneck, higher latency |
| **Sharded** | 2,818 samples/s | Parallel loading, higher throughput | Requires explicit sharding, complex state |
| **Iterable (HF)** | 1,498 samples/s | Streaming, memory efficient | Slower due to HF overhead |
| **Iterable (Fast)** | 1,971 samples/s | Optimized streaming | Still slower than Arrow-based |
| **Sharded Iterable** | 1,977 samples/s | Streaming + parallel | Complex coordination |

**Recommendation**: Use **sharded datasets** for maximum throughput when dataset supports it. Fall back to **dispatched batches** for simplicity when performance is not critical.

## DDP Features

### 1. Batch Dispatching (`dispatch_batches`)

**How it works**: Rank 0 loads and preprocesses all batches, then dispatches them to other ranks via `torch.distributed`.

```yaml
[trainer_args]
    dispatch_batches: True  # Default
```

**When to use**:
- ✅ Dataset doesn't support sharding
- ✅ Simple checkpoint management (single global state)
- ✅ Smaller datasets where rank-0 isn't a bottleneck

**Trade-offs**:
- ❌ Rank-0 becomes bottleneck for data loading
- ❌ Higher communication overhead
- ✅ Easier to debug and reason about

### 2. Dataset Sharding

**How it works**: Each rank loads its own shard of the dataset independently.

```yaml
[dataset_project]
    shard_dataset: True

[trainer_args]
    dispatch_batches: False  # Must disable dispatching
```

**When to use**:
- ✅ Large datasets where loading is a bottleneck
- ✅ Dataset supports clean sharding (e.g., Arrow-based HF datasets)
- ✅ Maximum throughput is critical

**Trade-offs**:
- ✅ 20-30% faster data loading
- ✅ Parallel preprocessing on all ranks
- ❌ Must ensure each rank gets different examples
- ❌ More complex checkpoint state management

### 3. Gradient Accumulation

**How it works**: Accumulate gradients over multiple micro-batches before synchronizing.

```yaml
[trainer_args]
    gradient_accumulation_steps: 4
    per_device_train_batch_size: 8  # Smaller micro-batch
```

**When to use**:
- ✅ Large models that don't fit with full batch size
- ✅ Want effective batch size > per-device memory limit
- ✅ Reduce communication frequency (sync every N steps)

**Trade-offs**:
- ✅ Enables training larger models
- ✅ Reduces gradient sync overhead
- ❌ Slightly slower due to more forward/backward passes
- ❌ Must carefully tune accumulation steps

### 4. Distributed Checkpointing

Full checkpoint support with automatic state management:

```yaml
[trainer_args]
    save_strategy: "steps"
    save_steps: 200
    save_optimizer_state: True
    save_scheduler_state: True
    save_dataset_state: True    # Critical for reproducibility
    save_rng_state: True         # Ensures identical randomness
```

**Checkpoint contents**:
- **Model weights**: REPLICATED (synchronized across all ranks via DDP)
- **Optimizer state**: REPLICATED (same due to gradient sync)
- **Dataset state**: GLOBAL (dispatched) or PER_RANK (sharded)
- **RNG state**: PER_RANK (each rank needs different random numbers)

**Resume behavior**:
```bash
# Train with checkpointing
forgather -t checkpoint_train.yaml train    # Saves at step 200, 400

# Resume and continue
forgather -t checkpoint_resume.yaml train   # Resumes from step 400, continues to 1000
```

See `docs/checkpointing/user_guide.md` for detailed documentation.

## Usage Examples

### Basic Multi-GPU Training

```bash
# 2-GPU training with batch dispatching
forgather -t 2gpu.yaml train

# 4-GPU training with dataset sharding (faster)
forgather -t 4gpu.yaml train -d 0,1,2,3

# View results
forgather logs summary
forgather logs plot --loss-curves -e
```

### Compare Dataset Strategies

```bash
# Train with different strategies
forgather -t sharded_dataset.yaml train
forgather -t iterable_dataset.yaml train
forgather -t sharded_fast_dataset.yaml train

# Compare performance
forgather logs summary --all --format one-line

# Visualize comparison
forgather logs plot --compare \
    output_models/default_model/runs/sharded_*/trainer_logs.json \
    output_models/default_model/runs/iterable_*/trainer_logs.json \
    --loss-curves -e
```

### Test Checkpoint Functionality

```bash
# Initial training with checkpointing
forgather -t checkpoint_train.yaml train

# Verify checkpoints created
ls -lh output_models/default_model/checkpoint-*

# Resume from checkpoint
forgather -t checkpoint_resume.yaml train

# Verify training continued from correct step
forgather logs summary output_models/default_model/runs/checkpoint_*/trainer_logs.json
```

### Gradient Accumulation

```bash
# Train with gradient accumulation
forgather -t grad_accum.yaml train

# Compare memory usage and throughput
# (Note: requires additional memory profiling)
```

## Architecture Details

### DDP Trainer Class Hierarchy

```
Trainer (base)
    └── DDPTrainer
         ├── Wraps model with DDP
         ├── Handles gradient synchronization
         ├── Manages distributed checkpoints
         └── Optional: DataloaderDispatcher for batch broadcasting
```

### Key Methods

**`_init_distributed()`**: Initialize device mesh and DDP process group

**`_wrap()`**: Wrap model in DDP, optionally wrap dataloaders with dispatcher

**`unwrapped_model()`**: Access original model (stored in `model.module`)

**`_distributed_loss()`**: All-reduce loss across ranks for logging

**`_forward_backward_step()`**: Skip gradient sync during accumulation steps using `no_sync()`

**`get_state_components()`**: Define checkpoint sharing patterns:
- Model/Optimizer: `REPLICATED` (DDP synchronized)
- Dataset: `GLOBAL` (dispatched) or `PER_RANK` (sharded)
- RNG: `PER_RANK` (different per rank)

### Checkpoint State Patterns

```python
# From get_state_components()
components = [
    StateComponent(
        key="model",
        sharing_pattern=SharingPattern.REPLICATED,  # DDP synced
        validate_replication=True,  # Catch sync bugs
    ),
    StateComponent(
        key="dataset",
        sharing_pattern=(
            SharingPattern.GLOBAL if dispatch_batches
            else SharingPattern.PER_RANK
        ),
    ),
    StateComponent(
        key="rng",
        sharing_pattern=SharingPattern.PER_RANK,  # Different per rank
    ),
]
```

## Integration Tests

This project validates:

- ✅ **DDP initialization and model wrapping**
- ✅ **Gradient synchronization across ranks**
- ✅ **Batch dispatching via DataloaderDispatcher**
- ✅ **Dataset sharding with proper rank assignment**
- ✅ **Checkpoint save with REPLICATED/GLOBAL/PER_RANK patterns**
- ✅ **Checkpoint resume with full state restoration**
- ✅ **Gradient accumulation with selective sync**
- ✅ **Single-process fallback (world_size == 1)**
- ✅ **CPU-based distributed training**
- ✅ **Iterable dataset integration**
- ✅ **Fast dataset loader performance**

## Troubleshooting

### Common Issues

**1. Slow training with dispatched batches**
```bash
# Solution: Use sharded dataset
[dataset_project]
    shard_dataset: True
[trainer_args]
    dispatch_batches: False
```

**2. Different data on each rank**
```bash
# Problem: Sharded dataset but dispatch_batches=True
# Solution: Ensure consistency
dispatch_batches: False  # When using sharding
```

**3. Checkpoint resume fails**
```bash
# Ensure all state is saved
save_dataset_state: True  # Critical!
save_rng_state: True
```

**4. Out of memory on GPUs**
```bash
# Use gradient accumulation
gradient_accumulation_steps: 4
per_device_train_batch_size: 8  # Reduce from 32
```

### Debugging

```bash
# Check distributed environment
forgather -t 2gpu.yaml train --dry-run

# View detailed logs
tail -f output_models/default_model/runs/*/trainer_logs.json

# Analyze performance
forgather logs summary --all --format one-line
forgather logs plot --metrics "loss,grad_norm,learning_rate" -e
```

## Further Reading

- **DDP Implementation**: `src/forgather/ml/trainer/ddp/ddp_trainer.py`
- **Checkpointing Guide**: `docs/checkpointing/user_guide.md`
- **Fast Dataset Loader**: `docs/datasets/fast-hf-loader.md`
- **Dataset Sharding**: Template examples in `templates/configs/`
- **PyTorch DDP**: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

## Performance Tips

1. **Use sharded datasets** for >2 GPUs: 20-30% faster than dispatched
2. **Enable gradient checkpointing** if memory bound (in model config)
3. **Tune batch size**: Larger batches = better GPU utilization
4. **Monitor gradient norms**: Track with `forgather logs plot --metrics "grad_norm"`
5. **Profile first**: Use single-process baseline to identify bottlenecks
6. **Use fast iterables** when streaming: 30% faster than standard HF iterables

## Contributing

When adding new DDP features:
1. Add configuration in `templates/configs/`
2. Run training and save logs
3. Update this README with performance data
4. Add integration test validation

Use the log analysis tools to gather metrics:
```bash
forgather logs summary --all --format one-line > performance.txt
forgather logs plot --compare runs/*.json --output comparison.png
```
