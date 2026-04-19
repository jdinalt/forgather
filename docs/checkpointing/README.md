# Distributed Checkpointing Documentation

Forgather's distributed checkpoint system provides automatic coordination for multi-GPU and multi-node training. The system uses explicit state sharing patterns to handle complex parallelism strategies.

## Documentation Structure

### For Users

**[User Guide](user_guide.md)** - Start here for practical usage
- Basic checkpointing setup
- Resuming from checkpoints
- DDP training (centralized and independent data loading)
- Pipeline parallel training
- Hybrid parallelism (DP x TP, DP x PP)
- Multi-node training
- Best practices and troubleshooting

**[Divergence Detection & Checkpoint Preservation](divergence_detection.md)** - Advanced features
- Prevent loss of best checkpoints
- Early divergence detection with stateful callbacks
- Decoupled eval/save scheduling
- Multiple detection strategies

### For Developers

**[Migration Guide](migration_guide.md)** - Implementing custom trainers
- Step-by-step API implementation
- Choosing correct sharing patterns
- Dynamic pattern resolution
- Complete examples for all parallelism types
- Common pitfalls and solutions

**[Technical Documentation](distributed_checkpoint_abstraction.md)** - System architecture
- Core abstractions and types
- Checkpoint coordinator design
- Pattern implementations (GLOBAL, PER_RANK, REPLICATED, PER_GROUP, PER_NODE)
- Validation system (4 levels)
- Manifest format
- Implementation status

### Implementation Details

All implementation details are documented in the main documentation:
- **Test Results**: All 5 trainers tested successfully (see Technical Documentation)
- **Migration Details**: Complete trainer migration guide available in Migration Guide
- **Known Issues**: Documented in User Guide troubleshooting section

## Quick Reference

### Sharing Patterns

| Pattern | Use Case | Save Behavior | Example |
|---------|----------|---------------|---------|
| **GLOBAL** | One copy total | Rank 0 saves | Training state with centralized dispatch |
| **PER_RANK** | Unique per rank | All ranks save | RNG state, pipeline stages |
| **REPLICATED** | Identical across ranks | Rank 0 saves | DDP model weights, DDP optimizer |
| **PER_GROUP** | Shared within groups | One per group | DP x PP hybrid models |
| **PER_NODE** | Local to each node | Local rank 0 saves | Node-local caches |

### Trainer Support

| Trainer | Status | Model Pattern | Optimizer Pattern | Dataset Pattern |
|---------|--------|---------------|-------------------|-----------------|
| **SimpleTrainer** | ✅ Tested | GLOBAL | GLOBAL | GLOBAL |
| **DDPTrainer** | ✅ Tested | REPLICATED | REPLICATED | GLOBAL or PER_RANK |
| **AccelTrainer** | ✅ Tested | REPLICATED | REPLICATED | PER_RANK |
| **PipelineTrainer** | ✅ Tested | PER_RANK | PER_RANK | GLOBAL |
| **Hybrid DP x TP** | 🚧 Designed | PER_GROUP | PER_GROUP | PER_GROUP |
| **Hybrid DP x PP** | 🚧 Designed | PER_GROUP | PER_GROUP | PER_GROUP |

### Validation Levels

| Level | Speed | What It Checks | Use Case |
|-------|-------|----------------|----------|
| **NONE** | Fastest | Nothing | Production (trusted code) |
| **QUICK** | Fast | Full state hash | Default for optimizer |
| **TENSOR** | Moderate | Per-tensor checksums | Default for model (DDP) |
| **FULL** | Slow | Full tensor comparison | Debugging only |

## Key Features

- ✅ **Automatic Coordination**: No manual rank checks needed
- ✅ **Explicit Semantics**: Clear declaration of state sharing
- ✅ **Dynamic Patterns**: Runtime determination (e.g., dataset state)
- ✅ **Validation**: Optional replication verification
- ✅ **Manifests**: Complete checkpoint inventory for debugging
- ✅ **Backward Compatible**: Old checkpoints still load
- ✅ **Production Ready**: All trainers tested successfully
- ✅ **Checkpoint Preservation**: Keep best N checkpoints safe from cleanup
- ✅ **Divergence Detection**: Catch training issues early with stateful callbacks
- ✅ **Stateful Callbacks**: Callback state saved/restored with checkpoints
- ✅ **Decoupled Eval/Save**: Flexible eval/save scheduling without alignment

## Getting Started

1. **Read the User Guide**: `docs/checkpointing/user_guide.md`
2. **Enable checkpointing** in your training arguments:
   ```python
   args = TrainingArguments(
       save_strategy="steps",
       save_steps=1000,
   )
   # All state is saved automatically:
   # - Model, optimizer, scheduler, dataset, RNG, training progress
   ```
3. **Train and checkpoint** - automatic!
4. **Resume from checkpoint**:
   ```python
   args = TrainingArguments(
       resume_from_checkpoint=True,  # Auto-finds latest
       ...
   )
   ```

## Common Use Cases

### Single-GPU Training
```python
trainer = Trainer(model, args, ...)
trainer.train()
```
All state saved as GLOBAL pattern.

### DDP Training (Recommended Setup)
```python
args = DDPTrainingArguments(
    dispatch_batches=True,  # Centralized data loading
    ...
)
trainer = DDPTrainer(model, args, ...)
trainer.train()
```
Model/optimizer saved as REPLICATED (validation enabled), dataset as GLOBAL.

### Pipeline Parallel Training
```python
trainer = PipelineTrainer(
    model_splitter=split_function,
    args=args,
    ...
)
trainer.train()
```
Model/optimizer saved as PER_RANK (different stages), dataset as GLOBAL.

## Troubleshooting

**Training hangs during save?**
- Distributed barrier deadlock (already fixed in built-in trainers)

**Validation failure?**
- Check DDP synchronization
- AccelTrainer optimizer validation is automatically disabled

**Different results after resume?**
- Enable `save_rng_state=True` and `save_dataset_state=True`

**Missing checkpoint files?**
- Check `checkpoint_manifest.json` to see what was saved
- Use `strict=False` when loading to skip missing optional components

See [User Guide - Troubleshooting](user_guide.md#troubleshooting) for details.

## Contributing

When adding new parallelism strategies:

1. Choose appropriate `SharingPattern` for each component
2. Implement `get_state_components()` method
3. Add integration tests (save/resume cycle)
4. Update documentation

See [Migration Guide](migration_guide.md) for implementation details.

## Related Documentation

- **Dataset Checkpointing**: `docs/datasets/fast-hf-loader-checkpoints.md`
- **Trainer Overview**: (coming soon)
- **Configuration System**: `docs/configuration/README.md`

## Implementation Status

- **Phase 1**: ✅ Core abstractions complete
- **Phase 2**: ✅ Pattern implementations complete
- **Phase 3**: ✅ All trainers migrated and tested
- **Phase 4**: 🚧 Hybrid parallelism testing (in progress)
- **Phase 5**: 🚧 Advanced features (future)

**Current status**: Production-ready for all single parallelism strategies (DDP, Pipeline, Accelerate). Hybrid parallelism designed but needs testing.
