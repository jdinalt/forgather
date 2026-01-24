# Migration Guide: Distributed Checkpoint Abstraction

**Audience**: Developers implementing custom trainers or extending existing trainers
**Status**: Phase 3 Complete - All built-in trainers migrated

## Overview

This guide explains how to implement the new state-centric checkpoint API for custom trainers. All built-in Forgather trainers now use this system. The new API provides better support for hybrid parallelism and makes checkpoint semantics explicit.

## Why Use the New API?

The new checkpoint system offers several advantages:

1. **Explicit semantics**: No guessing which ranks should save what
2. **Hybrid parallelism support**: Easily express complex DP/MP/PP combinations
3. **Dynamic patterns**: Runtime determination of sharing (e.g., dataset state)
4. **Validation**: Built-in replication validation and manifest checking
5. **Debugging**: Complete checkpoint inventory via manifest
6. **Automatic coordination**: No manual rank checks needed
7. **Production-ready**: All built-in trainers tested and working

## When to Implement

You need to implement the new checkpoint API when:

1. **Creating a custom trainer** - Inherit from `BaseTrainer` and override `get_state_components()`
2. **Adding a new parallelism strategy** - Use appropriate `SharingPattern` for your components
3. **Extending existing trainers** - Follow the pattern from similar built-in trainers

## Quick Start

### Step 1: Import New Types

```python
from forgather.ml.trainer.checkpoint_types import (
    StateComponent,
    SharingPattern,
)
from forgather.ml.trainer.checkpoint_manager import RNGState
```

### Step 2: Implement `get_state_components()`

Replace the old `get_statefuls_for_save()` method with `get_state_components()`:

**Before (Legacy API):**
```python
class MyTrainer:
    def get_statefuls_for_save(self) -> Dict[str, Stateful]:
        statefuls = {}
        if self.args.save_optimizer_state:
            statefuls["optimizer"] = self.optimizer
        if self.args.save_scheduler_state:
            statefuls["scheduler"] = self.lr_scheduler
        # ... etc
        return statefuls

    def get_statefuls_for_load(self) -> Dict[str, Stateful]:
        # Similar to above
        ...
```

**After (New API):**
```python
class MyTrainer:
    def get_state_components(self) -> List[StateComponent]:
        return [
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.GLOBAL,
                required=self.args.save_optimizer_state,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.GLOBAL,
                required=self.args.save_scheduler_state,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=self._get_dataset_sharing_pattern(),
                required=self.args.save_dataset_state,
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
                required=self.args.save_rng_state,
            ),
        ]
```

### Step 3: Handle Dynamic Patterns (Optional)

For components with runtime-determined sharing (like datasets):

```python
def _get_dataset_sharing_pattern(self) -> SharingPattern:
    """Determine dataset state sharing pattern based on dataloader type."""
    if isinstance(self.train_dataloader, DataloaderDispatcher):
        # Dispatcher coordinates data loading
        if self.train_dataloader._dp_size == 1:
            # Pure MP mode: all ranks get same batch, rank 0 loads
            return SharingPattern.GLOBAL
        elif self.train_dataloader._mp_size == 1:
            # Pure DP mode: rank 0 loads and dispatches
            return SharingPattern.GLOBAL
        else:
            # Hybrid: each DP group has one loader
            return SharingPattern.PER_GROUP
    else:
        # Each rank has independent dataloader
        return SharingPattern.PER_RANK
```

## Pattern Selection Guide

Choose the right `SharingPattern` for each component:

### GLOBAL
**Use when**: Only one copy exists across all ranks
**Examples**:
- Training progress when using centralized data dispatch
- Global metrics/counters

**Save behavior**: Rank 0 saves
**Load behavior**: All ranks load same file

```python
StateComponent(
    key="trainer_state",
    stateful=self.state,
    sharing_pattern=SharingPattern.GLOBAL,
)
```

### PER_RANK
**Use when**: Each rank has unique state
**Examples**:
- RNG state (each rank needs different random numbers)
- Pipeline stage parameters (different stage per rank)
- Rank-specific optimizer state (when optimizing different parameters)

**Save behavior**: Every rank saves its own file
**Load behavior**: Each rank loads its specific file

```python
StateComponent(
    key="rng",
    stateful=RNGState(),
    sharing_pattern=SharingPattern.PER_RANK,
)
```

### REPLICATED
**Use when**: State is identical across all ranks
**Examples**:
- DDP model weights (synchronized by DDP)
- DDP optimizer state (synchronized by DDP)
- LR scheduler state (same schedule across all ranks)

**Save behavior**: Rank 0 saves (avoids redundancy)
**Load behavior**: All ranks load same file
**Optional**: Validate that all ranks actually have identical state

```python
StateComponent(
    key="model",
    stateful=self.unwrapped_model(),
    sharing_pattern=SharingPattern.REPLICATED,
    validate_replication=True,  # Verify DDP synchronization
)
```

### PER_GROUP
**Use when**: State is shared within process groups, different across groups
**Examples**:
- Model shard shared within DP group but different across PP stages
- Dataset state shared within DP group
- Optimizer state for grouped parameters

**Save behavior**: One rank per group saves
**Load behavior**: Ranks load based on group membership

```python
StateComponent(
    key="model_shard",
    stateful=self.model_shard,
    sharing_pattern=SharingPattern.PER_GROUP,
    process_group_name="dp_group",
)
```

**Note**: Also implement `get_process_groups()`:
```python
def get_process_groups(self) -> Dict[str, ProcessGroup]:
    return {
        "dp_group": self.dp_process_group,
        "pp_group": self.pp_process_group,
    }
```

### PER_NODE
**Use when**: State is local to each node
**Examples**:
- Node-local caches
- Node-specific resources

**Save behavior**: Local rank 0 on each node saves
**Load behavior**: Ranks load based on node membership

```python
StateComponent(
    key="node_cache",
    stateful=self.cache,
    sharing_pattern=SharingPattern.PER_NODE,
)
```

## Complete Migration Examples

### Example 1: Single-GPU Trainer

```python
class SimpleTrainer(BaseTrainer):
    def get_state_components(self) -> List[StateComponent]:
        """All state is GLOBAL in single-GPU setting."""
        return [
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.GLOBAL,
                required=self.args.save_optimizer_state,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.GLOBAL,
                required=self.args.save_scheduler_state,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.GLOBAL,
                required=self.args.save_dataset_state,
            ),
            StateComponent(
                key="trainer",
                stateful=self,  # TrainerState
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
                required=self.args.save_rng_state,
            ),
        ]
```

### Example 2: DDP Trainer

```python
class DDPTrainer(BaseTrainer):
    def get_state_components(self) -> List[StateComponent]:
        """DDP synchronizes weights, so use REPLICATED pattern."""
        return [
            StateComponent(
                key="model",
                stateful=self.unwrapped_model(),
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=True,  # Catch DDP sync bugs
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.REPLICATED,
                required=self.args.save_optimizer_state,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
                required=self.args.save_scheduler_state,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=self._get_dataset_sharing_pattern(),
                required=self.args.save_dataset_state,
            ),
            StateComponent(
                key="trainer",
                stateful=self,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
                required=self.args.save_rng_state,
            ),
        ]
```

### Example 3: Pipeline Parallel Trainer

```python
class PipelineTrainer(BaseTrainer):
    def get_state_components(self) -> List[StateComponent]:
        """Each rank has different pipeline stage."""
        return [
            StateComponent(
                key="model",
                stateful=self.pipeline_modules,  # Different per rank
                sharing_pattern=SharingPattern.PER_RANK,
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,  # Different parameters per rank
                sharing_pattern=SharingPattern.PER_RANK,
                required=self.args.save_optimizer_state,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,  # Same schedule across ranks
                sharing_pattern=SharingPattern.REPLICATED,
                required=self.args.save_scheduler_state,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,  # Centralized loading
                sharing_pattern=SharingPattern.GLOBAL,
                required=self.args.save_dataset_state,
            ),
            StateComponent(
                key="trainer",
                stateful=self,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
                required=self.args.save_rng_state,
            ),
        ]
```

### Example 4: Hybrid DDP x Pipeline

```python
class HybridDDPPipelineTrainer(BaseTrainer):
    def get_state_components(self) -> List[StateComponent]:
        """Hybrid parallelism: DP groups with PP stages."""
        return [
            StateComponent(
                key="model",
                stateful=self.pipeline_modules,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="pp_group",  # Same within PP, different across DP
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="pp_group",
                required=self.args.save_optimizer_state,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
                required=self.args.save_scheduler_state,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="dp_group",  # One per DP group
                required=self.args.save_dataset_state,
            ),
            StateComponent(
                key="trainer",
                stateful=self,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
                required=self.args.save_rng_state,
            ),
        ]

    def get_process_groups(self) -> Dict[str, ProcessGroup]:
        return {
            "dp_group": self.dp_process_group,
            "pp_group": self.pp_process_group,
        }
```

## Common Pitfalls

### ❌ Wrong: Using GLOBAL for DDP weights
```python
# This saves redundantly on every rank!
StateComponent(
    key="model",
    stateful=self.model,
    sharing_pattern=SharingPattern.GLOBAL,  # Wrong!
)
```

**✅ Correct:**
```python
StateComponent(
    key="model",
    stateful=self.unwrapped_model(),
    sharing_pattern=SharingPattern.REPLICATED,  # DDP synchronizes
)
```

### ❌ Wrong: Using REPLICATED for RNG state
```python
# This makes all ranks use the same random numbers!
StateComponent(
    key="rng",
    stateful=RNGState(),
    sharing_pattern=SharingPattern.REPLICATED,  # Wrong!
)
```

**✅ Correct:**
```python
StateComponent(
    key="rng",
    stateful=RNGState(),
    sharing_pattern=SharingPattern.PER_RANK,  # Each rank needs unique RNG
)
```

### ❌ Wrong: Forgetting to implement `get_process_groups()`
```python
StateComponent(
    key="model",
    stateful=self.model,
    sharing_pattern=SharingPattern.PER_GROUP,
    process_group_name="dp_group",  # But get_process_groups() not implemented!
)
```

**✅ Correct:**
```python
def get_process_groups(self) -> Dict[str, ProcessGroup]:
    return {"dp_group": self.dp_process_group}
```

## Testing Your Migration

### 1. Unit Tests

Verify your `get_state_components()` implementation:

```python
def test_state_components():
    trainer = MyTrainer(...)
    components = trainer.get_state_components()

    # Check all expected components present
    keys = {c.key for c in components}
    assert "model" in keys
    assert "optimizer" in keys

    # Check sharing patterns are correct
    model_component = next(c for c in components if c.key == "model")
    assert model_component.sharing_pattern == SharingPattern.REPLICATED  # For DDP
```

### 2. Integration Tests

Test actual save/load cycles:

```python
def test_checkpoint_save_load():
    trainer = MyTrainer(...)

    # Train for a few steps
    trainer.train()

    # Save checkpoint
    checkpoint_path = trainer.save_checkpoint()

    # Verify manifest exists
    assert os.path.exists(os.path.join(checkpoint_path, "checkpoint_manifest.json"))

    # Create new trainer
    trainer2 = MyTrainer(...)
    trainer2.load_checkpoint(checkpoint_path)

    # Verify state was restored
    # ... assertions
```

### 3. Distributed Tests

Test with multiple ranks:

```bash
# Test DDP save/load
torchrun --nproc_per_node=4 test_ddp_checkpoint.py

# Test pipeline parallel save/load
torchrun --nproc_per_node=4 test_pipeline_checkpoint.py
```

## Built-in Trainer Reference

All built-in trainers provide reference implementations:

### BaseTrainer (Single-Device)
- **Model**: GLOBAL
- **Optimizer**: GLOBAL
- **Scheduler**: GLOBAL
- **Dataset**: GLOBAL
- **RNG**: PER_RANK
- **Location**: `src/forgather/ml/trainer/base_trainer.py:get_state_components()`

### DDPTrainer (Data Parallel)
- **Model**: REPLICATED (with validation)
- **Optimizer**: REPLICATED (with validation)
- **Scheduler**: REPLICATED
- **Dataset**: GLOBAL or PER_RANK (dynamic - depends on `dispatch_batches`)
- **RNG**: PER_RANK
- **Location**: `src/forgather/ml/trainer/ddp/ddp_trainer.py:get_state_components()`

### AccelTrainer (Accelerate DDP)
- **Model**: REPLICATED (with validation)
- **Optimizer**: REPLICATED (validation disabled - AcceleratedOptimizer wrapper)
- **Scheduler**: REPLICATED
- **Dataset**: PER_RANK
- **RNG**: PER_RANK
- **Location**: `src/forgather/ml/trainer/accelerate/accel_trainer.py:get_state_components()`

### PipelineTrainer (Pipeline Parallel)
- **Model**: PER_RANK (different stages)
- **Optimizer**: PER_RANK (different parameters)
- **Scheduler**: REPLICATED
- **Dataset**: GLOBAL (rank 0 loads and broadcasts)
- **RNG**: PER_RANK
- **Location**: `src/forgather/ml/trainer/pipeline/pipeline_trainer.py:get_state_components()`

## Backward Compatibility

CheckpointManager automatically detects which API your trainer implements:

```python
# CheckpointManager initialization (automatic):
state_components = stateful_provider.get_state_components()
if state_components is not None:
    # NEW API: Use CheckpointCoordinator
    self.coordinator = CheckpointCoordinator(...)
else:
    # OLD API: Use legacy get_statefuls_for_save/load
    self.coordinator = None
```

**For custom trainers:**
- Implement `get_state_components()` for new API (recommended)
- Keep old `get_statefuls_for_save/load()` for backward compatibility (optional)
- CheckpointManager will use new API if available, fall back to old API otherwise

## Migration Checklist

- [ ] Import new types (`StateComponent`, `SharingPattern`)
- [ ] Implement `get_state_components()` method
- [ ] Choose correct `SharingPattern` for each component
- [ ] Implement `get_process_groups()` if using PER_GROUP
- [ ] Add dynamic pattern resolution for dataset state (if applicable)
- [ ] Add validation flags (e.g., `validate_replication=True` for DDP)
- [ ] Test with unit tests
- [ ] Test with integration tests (save/load cycles)
- [ ] Test with distributed training (if applicable)
- [ ] Update documentation
- [ ] Remove legacy `get_statefuls_for_save/load()` (after testing)

## Getting Help

- **Main Documentation**: `docs/checkpointing/distributed_checkpoint_abstraction.md`
- **User Guide**: `docs/checkpointing/user_guide.md` - Troubleshooting and best practices
- **Built-in Trainers**: Check source code for reference implementations
- **Issues**: Report issues at https://github.com/anthropics/forgather/issues

## Current Status

- **Phase 3 Complete**: All built-in trainers migrated and tested
- **New API**: Default for all built-in trainers
- **Old API**: Still supported for custom trainers (backward compatibility)
- **Production Ready**: 5/5 trainer types tested successfully
