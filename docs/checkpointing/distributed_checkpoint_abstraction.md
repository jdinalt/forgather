# Distributed Checkpoint Abstraction for Hybrid Parallelism

**Status**: Phase 3 Complete (All Trainers Migrated and Tested)
**Last Updated**: 2026-01-24

## Overview

This document describes the new state-centric checkpoint abstraction system for Forgather trainers. The system enables automatic distributed checkpoint coordination based on explicit state sharing patterns, replacing ad-hoc rank checks with declarative semantics.

**All Forgather trainers now use this system**. Checkpoints are backward compatible and include detailed manifests for debugging.

## Motivation

Forgather's previous checkpoint system used boolean flags (`save_on_all_ranks`, `save_on_each_node`) to control which ranks save state. This doesn't scale to hybrid parallelism where different state components have different sharing patterns:

- **Some state is globally shared** (one copy total): e.g., training progress when using centralized data dispatch
- **Some state is per-rank** (different on every rank): e.g., pipeline stage parameters, RNG state
- **Some state is replicated** (identical across ranks): e.g., DDP model weights, scheduler state
- **Some state is per-group** (shared within parallelism groups): e.g., model shards shared within DP group but different across PP stages
- **Some state has dynamic sharing patterns**: e.g., dataset state can be GLOBAL (when using DataloaderDispatcher) or PER_RANK (when each rank has independent iteration)

The new system addresses these challenges through explicit state classification and automatic coordination.

## Architecture

### Core Types

#### 1. `SharingPattern` Enum

Defines how a state component is distributed across ranks:

```python
class SharingPattern(Enum):
    GLOBAL = "global"           # One copy total (e.g., trainer state)
    PER_RANK = "per_rank"       # Each rank has unique copy (e.g., RNG state)
    REPLICATED = "replicated"   # Identical across ranks (e.g., DDP weights)
    PER_GROUP = "per_group"     # Shared within group (e.g., DP group shard)
    PER_NODE = "per_node"       # Local to each node
```

#### 2. `StateComponent` Dataclass

Describes a checkpointable component with its sharing semantics:

```python
@dataclass
class StateComponent:
    key: str                                    # Component identifier
    stateful: Stateful                          # Object with state_dict/load_state_dict
    sharing_pattern: SharingPattern             # How state is distributed
    process_group_name: Optional[str] = None    # For PER_GROUP pattern
    required: bool = True                       # Whether required for training
    validate_replication: bool = False          # Verify REPLICATED state matches
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 3. `ComponentManifest` and `CheckpointManifest`

Track what was saved for validation and debugging:

```python
@dataclass
class ComponentManifest:
    key: str
    sharing_pattern: str
    ranks: List[int]                       # Which ranks saved
    replicated_across: Optional[List[int]] # For REPLICATED/PER_GROUP
    group_name: Optional[str]              # For PER_GROUP
    size_bytes: int = 0
    checksum: Optional[str] = None

@dataclass
class CheckpointManifest:
    checkpoint_path: str
    world_size: int
    timestamp: datetime
    components: Dict[str, ComponentManifest]
    training_args_hash: Optional[str] = None
    forgather_version: Optional[str] = None
    pytorch_version: Optional[str] = None
```

### CheckpointCoordinator

Orchestrates distributed save/load based on sharing patterns:

```python
class CheckpointCoordinator:
    def __init__(
        self,
        state_components: List[StateComponent],
        process_groups: Dict[str, ProcessGroup],
        dist: DistributedEnvInterface,
        output_dir: str,
        ...
    ):
        ...

    def save_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        validate: bool = False,
    ) -> str:
        """Save distributed checkpoint with automatic coordination."""
        ...

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        """Load checkpoint with automatic coordination."""
        ...
```

## Usage Examples

### Single Trainer (No Parallelism)

```python
from forgather.ml.trainer.checkpoint_types import StateComponent, SharingPattern
from forgather.ml.trainer.checkpoint_manager import RNGState

class SimpleTrainer:
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
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=self._get_dataset_sharing_pattern(),
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]

    def _get_dataset_sharing_pattern(self) -> SharingPattern:
        """Determine dataset state sharing pattern at runtime."""
        if isinstance(self.train_dataloader, DataloaderDispatcher):
            return SharingPattern.GLOBAL  # Centralized dispatch
        else:
            return SharingPattern.PER_RANK  # Independent iteration
```

### DDP Trainer (Data Parallel)

```python
class DDPTrainer:
    def get_state_components(self) -> List[StateComponent]:
        return [
            StateComponent(
                key="model",
                stateful=self.unwrapped_model(),
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=True,  # Verify DDP synchronization
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=self._get_dataset_sharing_pattern(),
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]
```

### Pipeline Parallel Trainer

```python
class PipelineTrainer:
    def get_state_components(self) -> List[StateComponent]:
        return [
            StateComponent(
                key="model",
                stateful=self.pipeline_modules,  # Different stage per rank
                sharing_pattern=SharingPattern.PER_RANK,
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.PER_RANK,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.GLOBAL,  # Rank 0 loads, broadcasts
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]
```

### Hybrid DDP x Pipeline Trainer

```python
class HybridTrainer:
    def get_state_components(self) -> List[StateComponent]:
        return [
            StateComponent(
                key="model",
                stateful=self.pipeline_modules,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="pp_group",  # Shared within PP, different across DP
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="pp_group",
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="dp_group",  # One per DP group
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]

    def get_process_groups(self) -> Dict[str, ProcessGroup]:
        return {
            "pp_group": self.pp_group,
            "dp_group": self.dp_group,
        }
```

## Implementation Status

### ✅ Phase 1: Core Abstractions (Complete)

**Files Created:**
- `src/forgather/ml/trainer/checkpoint_types.py` - Core type definitions
- `src/forgather/ml/trainer/checkpoint_coordinator.py` - Coordination logic
- `tests/unit/ml/test_checkpoint_types.py` - Comprehensive unit tests (24 tests, all passing)

**Files Modified:**
- `src/forgather/ml/trainer/trainer_types.py` - Updated `StatefulProvider` protocol
- `src/forgather/ml/trainer/__init__.py` - Export new types

**What Works:**
- ✅ `SharingPattern` enum with 5 patterns
- ✅ `StateComponent` with validation
- ✅ `ComponentManifest` and `CheckpointManifest` with serialization
- ✅ State hashing for replication validation (4 levels: NONE, QUICK, TENSOR, FULL)
- ✅ `CheckpointCoordinator` with pattern-specific handlers:
  - ✅ GLOBAL pattern (rank 0 saves/all load)
  - ✅ PER_RANK pattern (each rank saves/loads own state)
  - ✅ REPLICATED pattern (rank 0 saves/all load, optional validation)
  - ✅ PER_GROUP pattern (basic implementation)
  - ✅ PER_NODE pattern (basic implementation)
- ✅ Manifest generation and validation
- ✅ Legacy checkpoint loading (backward compatibility)
- ✅ Optional vs required component handling
- ✅ Comprehensive unit tests

### ✅ Phase 2: Pattern Implementations (Complete)

**Enhancements:**
- ✅ REPLICATED pattern with 4 validation levels (NONE, QUICK, TENSOR, FULL)
- ✅ Tensor-level validation with per-tensor checksums
- ✅ Detailed error reporting for validation failures
- ✅ State hashing improvements for large state dicts

### ✅ Phase 3: Trainer Migration (Complete)

**All trainers migrated to new checkpoint API:**

1. ✅ **BaseTrainer** - Base implementation with GLOBAL patterns
2. ✅ **Trainer (SimpleTrainer)** - Inherits from BaseTrainer
3. ✅ **AccelTrainer** - REPLICATED patterns for DDP
4. ✅ **PipelineTrainer** - PER_RANK for model/optimizer, GLOBAL for dataset
5. ✅ **DDPTrainer** - REPLICATED patterns with dynamic dataset pattern

**Integration:**
- ✅ CheckpointManager now uses CheckpointCoordinator
- ✅ Model weights continue using sharded checkpoint (efficient for large models)
- ✅ Training state uses CheckpointCoordinator (non-model components)
- ✅ Backward compatibility maintained (old API still works)

**Testing Results (5/5 trainer types):**
- ✅ SimpleTrainer - Basic checkpoint (eval loss ~2.34 vs baseline ~2.32)
- ✅ SimpleTrainer (iterable) - Dataset state checkpointing (eval loss ~2.32)
- ✅ PipelineTrainer - PER_RANK pattern with 2 GPUs (resumed step 237→500)
- ✅ DDPTrainer - REPLICATED pattern with 2 GPUs (eval loss ~2.17, validation enabled)
- ✅ AccelTrainer - REPLICATED pattern with 2 GPUs (eval loss ~2.08, model validation enabled)

**Key Issues Resolved:**
1. **Distributed Barrier Deadlock**: Fixed CheckpointManager to ensure ALL ranks call CheckpointCoordinator (has barriers)
2. **AccelTrainer Optimizer Validation**: Disabled for AcceleratedOptimizer wrapper (has rank-specific state)

### 🚧 Phase 4: Advanced Testing (Future)

**Remaining work:**
1. Test PER_GROUP pattern with hybrid parallelism (DP x PP, DP x TP)
2. Test PER_NODE pattern with multi-node distributed training
3. Test DDP with independent dataloaders (dispatch_batches=False)
4. Performance benchmarks (checkpoint save/load times)

### 🚧 Phase 5: Advanced Features (Future)

**Future enhancements:**
1. Transactional checkpointing (atomic saves)
2. Async checkpoint saves (save in background)
3. Checkpoint migration tools
4. Advanced validation and diagnostics

## Untested Scenarios and Future Work

The following scenarios are designed into the system but not yet tested in production:

### DDP with Independent Dataloaders (dispatch_batches=False)

**Scenario**: Each DDP rank loads its own data shard independently instead of using centralized dispatch.

**Configuration:**
```python
trainer = DDPTrainer(
    args=DDPTrainingArguments(
        dispatch_batches=False,  # Each rank loads independently
        ...
    ),
    train_dataset=train_dataset,
    ...
)
```

**Checkpoint Behavior:**
- **Model**: REPLICATED (DDP synchronizes)
- **Optimizer**: REPLICATED (DDP synchronizes gradients)
- **Dataset**: PER_RANK (each rank has independent state)
- **RNG**: PER_RANK (different random numbers per rank)

**Why useful**: Avoids rank-0 bottleneck for large datasets, each rank can read from different storage.

**Testing needed**: Verify that each rank correctly saves/loads its own dataset position.

### Hybrid Data Parallel + Tensor Parallel (DP x TP)

**Scenario**: Combine DDP for data parallelism with tensor parallelism for large models.

**Example Configuration:**
```python
# 8 GPUs: 2 DP groups x 4 TP ranks per group
# DP Group 0: ranks [0, 1, 2, 3]
# DP Group 1: ranks [4, 5, 6, 7]

class DPTPTrainer(BaseTrainer):
    def get_state_components(self) -> List[StateComponent]:
        return [
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="tp_group",  # Shared within TP, different across DP
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="tp_group",
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="dp_group",  # One dataset per DP group
            ),
            # ... other components
        ]

    def get_process_groups(self) -> Dict[str, ProcessGroup]:
        return {
            "tp_group": self.tp_process_group,
            "dp_group": self.dp_process_group,
        }
```

**Checkpoint Behavior:**
- **Model**: PER_GROUP (tp_group) - Ranks [0,4] save (TP shard 0 from each DP group)
- **Dataset**: PER_GROUP (dp_group) - Ranks [0,1,2,3] share one dataloader, [4,5,6,7] share another
- **Optimizer**: PER_GROUP (tp_group) - Matches model sharding

**Why useful**: Enables training models too large for single GPU while maintaining data parallel efficiency.

**Testing needed**: Verify correct group rank detection and checkpoint coordination.

### Hybrid Data Parallel + Pipeline Parallel (DP x PP)

**Scenario**: Combine DDP with pipeline parallelism for even larger models.

**Example Configuration:**
```python
# 8 GPUs: 2 DP groups x 4 PP stages per group
# DP Group 0: ranks [0, 1, 2, 3] (PP stages 0-3)
# DP Group 1: ranks [4, 5, 6, 7] (PP stages 0-3)

class DPPPTrainer(BaseTrainer):
    def get_state_components(self) -> List[StateComponent]:
        return [
            StateComponent(
                key="model",
                stateful=self.pipeline_modules,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="pp_group",  # Shared within PP, different across DP
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="dp_group",  # One per DP group
            ),
            # ... other components
        ]
```

**Checkpoint Behavior:**
- **Model**: PER_GROUP (pp_group) - Each PP stage saved once per DP group
- **Dataset**: PER_GROUP (dp_group) - One dataloader state per DP group
- **Scheduler**: REPLICATED - Same schedule across all ranks

**Why useful**: Maximum model size (pipeline parallel) with data parallel for throughput.

**Testing needed**: Verify PER_GROUP pattern with complex group hierarchies.

### Multi-Node Training with PER_NODE Pattern

**Scenario**: Training across multiple nodes with node-local resources.

**Example Configuration:**
```python
class MultiNodeTrainer(BaseTrainer):
    def get_state_components(self) -> List[StateComponent]:
        return [
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.REPLICATED,  # DDP across all nodes
            ),
            StateComponent(
                key="node_cache",
                stateful=self.local_cache,
                sharing_pattern=SharingPattern.PER_NODE,  # Different per node
            ),
            # ... other components
        ]
```

**Checkpoint Behavior:**
- **Model**: REPLICATED - Rank 0 saves globally
- **Node cache**: PER_NODE - Local rank 0 on each node saves

**Why useful**: Handle node-specific resources (local SSD caches, node-local preprocessing).

**Testing needed**: Verify node leader detection and per-node file creation.

## Benefits

1. **Explicit Semantics**: Code clearly expresses intent (REPLICATED vs PER_RANK vs GLOBAL)
2. **Dynamic Patterns**: Handles runtime decisions like DataloaderDispatcher vs independent iteration
3. **Automatic Coordination**: No manual rank checks in trainer code
4. **Validation**: Detect configuration mismatches, verify replication, detect corruption
5. **Composability**: Easy to express hybrid parallelism (DDP x Pipeline, DP x TP, etc.)
6. **Debuggability**: Manifest provides complete checkpoint inventory
7. **Extensibility**: Easy to add new sharing patterns as needed
8. **Eliminates Redundancy**: REPLICATED pattern saves once instead of on every DDP rank
9. **Type Safety**: Explicit StateComponent dataclass vs ad-hoc dictionaries
10. **Production-Ready**: All current trainers tested and working

## Practical Usage

All Forgather trainers now use the new checkpoint system automatically. No code changes required for existing training scripts.

### Basic Training with Checkpointing

```python
from forgather.ml.trainer import Trainer
from forgather.ml.trainer import TrainingArguments

args = TrainingArguments(
    output_dir="output_models/my_model",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    save_optimizer_state=True,
    save_scheduler_state=True,
    save_dataset_state=True,
    save_rng_state=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Training automatically saves checkpoints every 500 steps
trainer.train()

# Checkpoints include manifest for debugging:
# output_models/my_model/checkpoint-500/checkpoint_manifest.json
```

### Resuming from Checkpoint

```python
# Resume from latest checkpoint
args = TrainingArguments(
    output_dir="output_models/my_model",
    resume_from_checkpoint=True,  # Finds latest checkpoint
    max_steps=1000,
)

trainer = Trainer(model=model, args=args, ...)
trainer.train()  # Continues from where it left off

# Or specify explicit checkpoint path
args = TrainingArguments(
    resume_from_checkpoint="output_models/my_model/checkpoint-500",
    ...
)
```

### DDP Training with Validation

```python
from forgather.ml.trainer.ddp import DDPTrainer, DDPTrainingArguments

args = DDPTrainingArguments(
    output_dir="output_models/my_ddp_model",
    dispatch_batches=True,  # Centralized data loading (GLOBAL dataset pattern)
    save_strategy="steps",
    save_steps=1000,
)

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py

trainer = DDPTrainer(model=model, args=args, ...)
trainer.train()

# Model and optimizer automatically validated for replication
# Dataset state saved once (GLOBAL pattern with dispatch_batches=True)
# RNG state saved per rank (PER_RANK pattern)
```

### Pipeline Parallel Training

```python
from forgather.ml.trainer.pipeline import PipelineTrainer, PipelineTrainingArguments

args = PipelineTrainingArguments(
    output_dir="output_models/my_pipeline_model",
    save_strategy="steps",
    save_steps=500,
)

trainer = PipelineTrainer(
    model_splitter=my_splitter,
    args=args,
    ...
)

trainer.train()

# Model/optimizer saved per rank (PER_RANK - different pipeline stages)
# Dataset saved once (GLOBAL - rank 0 loads and broadcasts)
# Scheduler saved once (REPLICATED - same schedule across ranks)
```

### Inspecting Checkpoint Manifests

```python
import json
from pathlib import Path

# Load manifest
manifest_path = Path("output_models/my_model/checkpoint-500/checkpoint_manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)

print(f"World size: {manifest['world_size']}")
print(f"Timestamp: {manifest['timestamp']}")
print(f"Components:")
for key, comp in manifest['components'].items():
    print(f"  {key}:")
    print(f"    Pattern: {comp['sharing_pattern']}")
    print(f"    Ranks: {comp['ranks']}")
    print(f"    Size: {comp['size_bytes']} bytes")
```

## Backward Compatibility

The new system maintains backward compatibility:

1. **Dual API Support**: CheckpointManager checks for `get_state_components()` and falls back to old API if not implemented
2. **Legacy Checkpoint Loading**: Can load old checkpoints without manifest
3. **No Breaking Changes**: All existing training scripts continue to work
4. **All Trainers Migrated**: BaseTrainer, SimpleTrainer, AccelTrainer, PipelineTrainer, DDPTrainer all use new API

**Current Status**: All built-in trainers use the new API. Custom trainers can continue using old API temporarily.

## Troubleshooting

### Issue: Training hangs during checkpoint save (distributed mode)

**Symptoms:**
- GPU power consumption drops to ~80W and stays flat
- Processes never complete checkpoint save
- No error messages, just hangs

**Cause:** Distributed barrier deadlock. CheckpointCoordinator has barriers that ALL ranks must reach, but only some ranks called it.

**Solution:** Ensure ALL ranks call CheckpointCoordinator, not just rank 0. This is already fixed in CheckpointManager:

```python
# Correct (in CheckpointManager):
if self.coordinator is not None:
    # NEW API: all ranks participate (has barriers)
    self._save_training_state(checkpoint_path)
else:
    # OLD API: only saving ranks
    if self._should_save_common():
        self._save_training_state(checkpoint_path)
```

### Issue: AccelTrainer fails with "Replication validation failed for optimizer"

**Symptoms:**
- Checkpoint save fails during validation
- Error: "Replication validation failed for required component 'optimizer'"
- Accelerate-based training only

**Cause:** AcceleratedOptimizer wrapper includes rank-specific internal state that differs across ranks, even though underlying optimizer state is synchronized.

**Solution:** Disable validation for AccelTrainer optimizer (already implemented):

```python
StateComponent(
    key="optimizer",
    stateful=self.optimizer,
    sharing_pattern=SharingPattern.REPLICATED,
    validate_replication=False,  # Disabled: AcceleratedOptimizer wrapper
    validation_level="quick",
)
```

**Note:** Model validation remains enabled and works correctly.

### Issue: Checkpoint file not found during load

**Symptoms:**
- Load fails with "file not found" for optimizer_state.pt or similar
- Checkpoint directory exists but some files missing

**Cause:** Component was marked as `required=True` but wasn't saved (e.g., `save_optimizer_state=False` during save).

**Solution:** Either:
1. Set `required=False` for optional components
2. Ensure save flags match load expectations
3. Use `strict=False` when loading to skip missing components

```python
# Load with strict=False to skip missing components
trainer.load_checkpoint(checkpoint_path, strict=False)
```

### Issue: Wrong sharing pattern produces duplicate or missing files

**Symptoms:**
- Expected rank-specific files but only got one
- Got redundant files from all ranks when expected one

**Cause:** Wrong `SharingPattern` for the component.

**Solution:** Review sharing pattern selection:
- **GLOBAL**: One copy total, rank 0 saves
- **PER_RANK**: Each rank has unique copy, all ranks save
- **REPLICATED**: Identical across ranks, rank 0 saves (avoids redundancy)
- **PER_GROUP**: One copy per process group
- **PER_NODE**: One copy per node

Common mistakes:
- Using GLOBAL for DDP weights (should be REPLICATED)
- Using REPLICATED for RNG state (should be PER_RANK)
- Using PER_RANK for DDP optimizer (should be REPLICATED)

## Testing

All core functionality is tested:

### Unit Tests
```bash
# Run checkpoint system tests
python -m pytest tests/unit/ml/test_checkpoint_types.py -v

# Results: 24 tests, all passing
# - SharingPattern enum tests
# - StateComponent validation tests
# - Manifest serialization tests
# - State hashing tests
# - CheckpointCoordinator save/load tests
```

### Integration Tests
```bash
# Test basic checkpoint (SimpleTrainer)
cd examples/tiny_experiments/checkpointing
forgather -t train.yaml train         # Train to step 500
forgather -t resume.yaml train        # Resume to step 1000

# Test DDP checkpoint (DDPTrainer)
cd examples/tiny_experiments/ddp_trainer
forgather -t checkpoint_train.yaml train   # Save checkpoint
forgather -t checkpoint_resume.yaml train  # Resume

# Test pipeline parallel checkpoint (PipelineTrainer)
cd examples/tiny_experiments/pipeline_parallel
forgather -t checkpoint_test_train.yaml train
forgather -t checkpoint_test_resume.yaml train

# Test Accelerate DDP checkpoint (AccelTrainer)
cd examples/tiny_experiments/checkpointing
forgather -t accel_ddp.yaml train
forgather -t accel_ddp_resume.yaml train
```

**All tests passing** across 5 trainer types with different sharing patterns.

## Next Steps

Core implementation complete. Future enhancements:

1. **Test untested scenarios**: DDP with independent dataloaders, hybrid DP x TP, hybrid DP x PP
2. **Improve PER_GROUP pattern**: Better group rank tracking and coordination
3. **Performance optimization**: Async checkpoint saves, parallel component saves
4. **Advanced features**: Transactional semantics, checkpoint migration tools

## Related Documentation

- **Migration Guide**: `docs/checkpointing/migration_guide.md` - How to implement custom trainers
- **User Guide**: `docs/checkpointing/user_guide.md` - Practical usage guide
- **Dataset Checkpoints**: `docs/datasets/fast-hf-loader-checkpoints.md` - Stateful dataset support

## References

### Source Code
- **Core Types**: `src/forgather/ml/trainer/checkpoint_types.py`
- **Coordinator**: `src/forgather/ml/trainer/checkpoint_coordinator.py`
- **Manager**: `src/forgather/ml/trainer/checkpoint_manager.py`
- **Protocol**: `src/forgather/ml/trainer/trainer_types.py` (`StatefulProvider`)

### Trainer Implementations
- **Base**: `src/forgather/ml/trainer/base_trainer.py`
- **Simple**: `src/forgather/ml/trainer/trainer.py`
- **Accelerate**: `src/forgather/ml/trainer/accelerate/accel_trainer.py`
- **DDP**: `src/forgather/ml/trainer/ddp/ddp_trainer.py`
- **Pipeline**: `src/forgather/ml/trainer/pipeline/pipeline_trainer.py`

### Tests
- **Unit Tests**: `tests/unit/ml/test_checkpoint_types.py`
- **Integration Tests**: `examples/tiny_experiments/checkpointing/`, `examples/tiny_experiments/ddp_trainer/`, `examples/tiny_experiments/pipeline_parallel/`
