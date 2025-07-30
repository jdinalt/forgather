# Checkpointing Improvement Plan

## Current Issues Identified
1. **Pipeline trainer only saves rank 0 optimizer state** - Other ranks lose optimizer states
2. **Missing final checkpoint** - Training steps lost if max_steps not multiple of save_steps  
3. **Limited distributed support** - Manual coordination required for complex scenarios

## PyTorch DCP Considerations
- **Version compatibility risk**: DCP checkpoints may not work across PyTorch versions
- **No backward compatibility needed**: User has no existing checkpoints to preserve
- **Benefits**: Native distributed support, resharding, async checkpointing, FSDP integration

## Implementation Plan

### Phase 1: Critical Fixes (Immediate)
1. **Fix final checkpoint save**: Add `_save_checkpoint()` call at end of training loop in trainer.py
2. **Fix pipeline optimizer states**: Modify pipeline_trainer.py to save/load per-rank optimizer states properly
3. **Add configuration validation**: Ensure checkpoint settings are consistent

### Phase 2: DCP Integration (Medium-term)
1. **Add DCP backend option**: Implement as alternative to current system with `use_distributed_checkpoint` flag
2. **No backward compatibility needed**: Can replace existing system entirely
3. **Add conversion utilities**: For migrating between formats if needed

### Phase 3: Advanced Features (Long-term)
1. **Async checkpointing**: Non-blocking saves for better performance
2. **Cross-topology loading**: Support different world sizes
3. **Intelligent scheduling**: Optimize checkpoint timing

## Technical Details
- Pipeline trainer file: `/home/dinalt/ai_assets/forgather/src/forgather/ml/pipeline_trainer.py`
- Base trainer file: `/home/dinalt/ai_assets/forgather/src/forgather/ml/base_trainer.py`  
- Main training loop: `/home/dinalt/ai_assets/forgather/src/forgather/ml/trainer.py`
- Training args: `/home/dinalt/ai_assets/forgather/src/forgather/ml/trainer_types.py`

## Key Code Locations
- Pipeline optimizer state save: `pipeline_trainer.py:791` - Only rank 0 saves
- Training loop end: `trainer.py:313-316` - No final checkpoint save
- Periodic checkpoint: `trainer.py:296-297` - Only periodic saves

## Priority
Start with Phase 1 fixes immediately, then evaluate DCP integration timeline.