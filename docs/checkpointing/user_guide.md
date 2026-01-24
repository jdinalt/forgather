# Distributed Checkpointing User Guide

**Last Updated**: 2026-01-24

## Overview

Forgather's checkpoint system automatically handles distributed training across multiple GPUs and nodes. Checkpoints include complete training state (model, optimizer, scheduler, dataset position, RNG state) and can be resumed seamlessly.

## Basic Usage

### Enabling Checkpointing

Configure checkpointing in your training arguments:

```python
from forgather.ml.trainer import TrainingArguments

args = TrainingArguments(
    output_dir="output_models/my_model",

    # Checkpoint strategy
    save_strategy="steps",           # Save every N steps
    save_steps=500,                   # Save every 500 steps
    save_total_limit=3,               # Keep only last 3 checkpoints

    # What to save
    save_optimizer_state=True,        # Save optimizer state (recommended)
    save_scheduler_state=True,        # Save LR scheduler state
    save_dataset_state=True,          # Save dataset position (important!)
    save_rng_state=True,              # Save RNG state for reproducibility

    # Optional
    save_safetensors=True,            # Use safetensors format (default)
)

trainer = Trainer(model=model, args=args, ...)
trainer.train()
```

**Output:**
```
output_models/my_model/
├── checkpoint-500/
│   ├── model.safetensors           # Model weights
│   ├── optimizer_state.pt          # Optimizer state
│   ├── scheduler_state.pt          # Scheduler state
│   ├── dataset_state.pt            # Dataset position
│   ├── rng_state.pt                # RNG state
│   ├── trainer_state.pt            # Training progress
│   └── checkpoint_manifest.json    # Checkpoint metadata
├── checkpoint-1000/
│   └── ...
└── checkpoint-1500/
    └── ...
```

### Resuming from Checkpoint

```python
# Resume from latest checkpoint
args = TrainingArguments(
    output_dir="output_models/my_model",
    resume_from_checkpoint=True,      # Auto-finds latest checkpoint
    max_steps=2000,                   # Continue training
)

trainer = Trainer(model=model, args=args, ...)
trainer.train()  # Continues from step 1500

# Or specify explicit checkpoint
args = TrainingArguments(
    resume_from_checkpoint="output_models/my_model/checkpoint-1000",
    ...
)
```

**Result**: Training continues from the exact step/epoch where it left off, with identical optimizer state, learning rate schedule, dataset position, and RNG state.

## DDP Training (Data Parallel)

### Centralized Data Loading (Recommended)

Rank 0 loads and dispatches data to all ranks:

```python
from forgather.ml.trainer.ddp import DDPTrainer, DDPTrainingArguments

args = DDPTrainingArguments(
    output_dir="output_models/my_ddp_model",
    dispatch_batches=True,            # Rank 0 loads, others receive
    save_strategy="steps",
    save_steps=1000,
    save_optimizer_state=True,
    save_scheduler_state=True,
    save_dataset_state=True,
)

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py

trainer = DDPTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    ...
)

trainer.train()
```

**Checkpoint behavior:**
- **Model weights**: Saved once (rank 0) - identical across all ranks due to DDP synchronization
- **Optimizer state**: Saved once (rank 0) - DDP synchronizes gradients
- **Dataset state**: Saved once (rank 0) - centralized loading
- **RNG state**: Saved per rank - each rank needs different random numbers

**Benefits:**
- No data duplication across ranks
- Simplified dataset management
- Single dataset checkpoint (smaller, faster)

### Independent Data Loading (Advanced)

Each rank loads its own data shard:

```python
args = DDPTrainingArguments(
    output_dir="output_models/my_ddp_model",
    dispatch_batches=False,           # Each rank loads independently
    save_strategy="steps",
    save_steps=1000,
    save_dataset_state=True,          # Important! Each rank saves its position
)

# Ensure each rank gets different data
# Use DistributedSampler or shard dataset manually
from torch.utils.data import DistributedSampler

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.per_device_train_batch_size,
    sampler=train_sampler,
)

trainer = DDPTrainer(
    model=model,
    args=args,
    train_dataloader=train_dataloader,  # Custom dataloader with sharding
    ...
)

trainer.train()
```

**Checkpoint behavior:**
- **Model weights**: Saved once (rank 0)
- **Optimizer state**: Saved once (rank 0)
- **Dataset state**: Saved per rank - each rank has independent position
- **RNG state**: Saved per rank

**Use cases:**
- Very large datasets (avoid rank-0 bottleneck)
- Reading from different storage locations per rank
- Custom data sharding strategies

**Important:** When resuming, ensure each rank uses the same sharding strategy (same `num_replicas` and `rank`).

## Pipeline Parallel Training

Pipeline parallelism splits the model across ranks:

```python
from forgather.ml.trainer.pipeline import PipelineTrainer, PipelineTrainingArguments

args = PipelineTrainingArguments(
    output_dir="output_models/my_pipeline_model",
    save_strategy="steps",
    save_steps=500,
)

# Define how to split model into pipeline stages
def split_model(model):
    return [
        model.layers[0:4],   # Stage 0 (rank 0)
        model.layers[4:8],   # Stage 1 (rank 1)
        model.layers[8:12],  # Stage 2 (rank 2)
        model.layers[12:16], # Stage 3 (rank 3)
    ]

trainer = PipelineTrainer(
    model_splitter=split_model,
    args=args,
    train_dataset=train_dataset,
    ...
)

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py

trainer.train()
```

**Checkpoint behavior:**
- **Model weights**: Saved per rank - each rank has different pipeline stage
- **Optimizer state**: Saved per rank - each rank optimizes different parameters
- **Dataset state**: Saved once (rank 0) - centralized loading and broadcast
- **Scheduler**: Saved once (rank 0) - same LR schedule across all ranks
- **RNG state**: Saved per rank

**Checkpoint files:**
```
checkpoint-500/
├── model_state_rank_0.pt           # Stage 0 weights
├── model_state_rank_1.pt           # Stage 1 weights
├── model_state_rank_2.pt           # Stage 2 weights
├── model_state_rank_3.pt           # Stage 3 weights
├── optimizer_state_rank_0.pt       # Stage 0 optimizer
├── optimizer_state_rank_1.pt       # Stage 1 optimizer
├── optimizer_state_rank_2.pt       # Stage 2 optimizer
├── optimizer_state_rank_3.pt       # Stage 3 optimizer
├── scheduler_state.pt              # Shared scheduler
├── dataset_state.pt                # Shared dataset
├── rng_state_rank_0.pt             # Per-rank RNG
├── rng_state_rank_1.pt
├── rng_state_rank_2.pt
├── rng_state_rank_3.pt
└── checkpoint_manifest.json        # Metadata
```

**Resume requirement**: Must use same number of ranks and same model splitting strategy.

## Hybrid Parallelism (Future)

### Data Parallel + Tensor Parallel (DP x TP)

**Use case**: Train very large models (too big for single GPU) with data parallelism for efficiency.

**Configuration (example for 8 GPUs):**
- 2 DP groups (data parallelism)
- 4 TP ranks per group (tensor parallelism)

```
Rank topology:
DP Group 0: [0, 1, 2, 3]  (TP ranks sharing same data, different model shards)
DP Group 1: [4, 5, 6, 7]  (TP ranks sharing same data, different model shards)
```

**Checkpoint behavior:**
- **Model shards**: Saved per DP group (ranks [0, 4] save shard 0, [1, 5] save shard 1, etc.)
- **Dataset**: Saved per DP group (one dataloader per group)
- **Optimizer**: Saved per DP group (matches model sharding)

**Not yet tested** - Implementation ready, needs testing.

### Data Parallel + Pipeline Parallel (DP x PP)

**Use case**: Maximum model size with data parallel throughput.

**Configuration (example for 8 GPUs):**
- 2 DP groups
- 4 PP stages per group

```
Rank topology:
DP Group 0: [0, 1, 2, 3]  (PP stages 0-3)
DP Group 1: [4, 5, 6, 7]  (PP stages 0-3)
```

**Checkpoint behavior:**
- **Model stages**: Saved per DP group (each PP stage saved once per group)
- **Dataset**: Saved per DP group
- **Optimizer**: Saved per DP group

**Not yet tested** - Implementation ready, needs testing.

## Multi-Node Training

Checkpointing works across multiple nodes with proper distributed initialization:

```python
# Initialize distributed training
import torch.distributed as dist

dist.init_process_group(
    backend='nccl',
    init_method='env://',
)

# Configure checkpoint location accessible to all nodes
args = DDPTrainingArguments(
    output_dir="/shared/storage/my_model",  # Must be accessible to all nodes
    save_strategy="steps",
    save_steps=1000,
)

trainer = DDPTrainer(model=model, args=args, ...)
trainer.train()
```

**Important:**
- Checkpoint directory must be on shared storage (NFS, Lustre, etc.)
- All nodes must have access to same checkpoint path
- Rank 0 (global) saves model weights and manifest
- Other components saved based on sharing pattern

## Checkpoint Manifest

Every checkpoint includes a manifest with metadata:

```json
{
  "checkpoint_path": "/path/to/checkpoint-500",
  "world_size": 4,
  "timestamp": "2026-01-24T10:30:45",
  "pytorch_version": "2.9.1+cu130",
  "components": {
    "model": {
      "key": "model",
      "sharing_pattern": "replicated",
      "ranks": [0],
      "size_bytes": 445678123
    },
    "optimizer": {
      "key": "optimizer",
      "sharing_pattern": "replicated",
      "ranks": [0],
      "size_bytes": 33478195
    },
    "dataset": {
      "key": "dataset",
      "sharing_pattern": "global",
      "ranks": [0],
      "size_bytes": 1553
    },
    "rng": {
      "key": "rng",
      "sharing_pattern": "per_rank",
      "ranks": [0, 1, 2, 3],
      "size_bytes": 14042
    }
  }
}
```

**Use manifest to:**
- Debug checkpoint issues
- Verify all components saved
- Check world size matches
- Identify missing files

## Validation

For DDP trainers, checkpointing includes optional validation to verify model/optimizer state is synchronized:

```python
# Validation is automatically enabled for DDPTrainer
# Model: tensor-level validation (per-tensor checksums)
# Optimizer: quick validation (hash-based)

trainer = DDPTrainer(...)
trainer.train()

# If validation fails, you'll see an error like:
# "Replication validation failed for component 'model'"
# This indicates a DDP synchronization bug
```

**Validation catches:**
- DDP synchronization bugs
- Accidental divergence across ranks
- Incorrect sharing pattern configuration

## Best Practices

### 1. Always Save Dataset State

```python
save_dataset_state=True  # Critical for resuming mid-epoch
```

Without dataset state, resuming will restart from beginning of epoch, wasting compute.

### 2. Save RNG State for Reproducibility

```python
save_rng_state=True  # Ensures exact reproducibility
```

Without RNG state, data augmentation and dropout will differ after resume.

### 3. Use Reasonable Checkpoint Frequency

```python
save_steps=1000  # Balance between safety and storage
```

Too frequent: Wastes time on I/O and storage
Too rare: Risk losing more progress if crash occurs

**Rule of thumb**: Save every 30-60 minutes of training

### 4. Limit Checkpoint Retention

```python
save_total_limit=3  # Keep only last 3 checkpoints
```

Saves storage space while maintaining recovery options.

### 5. Use Shared Storage for Multi-Node

```python
output_dir="/shared/nfs/checkpoints/my_model"  # Accessible to all nodes
```

Local storage won't work - other nodes can't access it.

### 6. Test Resume Early

After starting training, immediately test resume:

```bash
# Start training
torchrun --nproc_per_node=4 train.py

# Stop after first checkpoint (Ctrl+C)

# Resume from checkpoint
torchrun --nproc_per_node=4 train.py --resume
```

Verify resuming works before investing days in training.

## Troubleshooting

### Training hangs during checkpoint save

**Symptom**: GPU utilization drops, processes hang indefinitely

**Cause**: Distributed barrier deadlock

**Solution**: Ensure all ranks participate in checkpoint save. This is automatic in built-in trainers. For custom trainers, see `CheckpointManager` implementation.

### "Replication validation failed"

**Symptom**: Checkpoint save fails with validation error

**Possible causes:**
1. DDP not properly synchronizing (bug in training code)
2. Using wrong sharing pattern (e.g., GLOBAL instead of REPLICATED)
3. AccelTrainer with optimizer validation (known issue - validation disabled by default)

**Solution**: Check DDP synchronization is working correctly. For AccelTrainer, optimizer validation is disabled.

### Missing checkpoint files on resume

**Symptom**: "File not found" error during checkpoint load

**Possible causes:**
1. Checkpoint incomplete (training crashed during save)
2. Wrong checkpoint path
3. Required component wasn't saved

**Solution:**
- Use previous checkpoint if available
- Set `strict=False` when loading to skip missing optional components
- Check `checkpoint_manifest.json` to see what was saved

### Different eval loss after resume

**Symptom**: Resume produces different results than training straight through

**Possible causes:**
1. Missing RNG state (`save_rng_state=False`)
2. Missing dataset state (`save_dataset_state=False`)
3. Dataset not deterministic

**Expected**: With full state saved, results should match within floating-point precision (~0.01% difference acceptable).

## Advanced: Custom Checkpoint Components

For custom trainers, you can add additional components:

```python
from forgather.ml.trainer.checkpoint_types import StateComponent, SharingPattern

class MyCustomTrainer(BaseTrainer):
    def get_state_components(self) -> List[StateComponent]:
        components = super().get_state_components()

        # Add custom component
        if hasattr(self, 'my_custom_state'):
            components.append(
                StateComponent(
                    key="custom",
                    stateful=self.my_custom_state,  # Must have state_dict/load_state_dict
                    sharing_pattern=SharingPattern.GLOBAL,  # Or appropriate pattern
                    required=False,  # Optional component
                )
            )

        return components
```

See `docs/checkpointing/migration_guide.md` for full details on implementing custom trainers.

## Related Documentation

- **Technical Details**: `docs/checkpointing/distributed_checkpoint_abstraction.md`
- **Migration Guide**: `docs/checkpointing/migration_guide.md`
- **Dataset Checkpointing**: `docs/datasets/fast-hf-loader-checkpoints.md`
