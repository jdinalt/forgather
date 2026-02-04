# Checkpoint Management Refactor - Summary

**Date**: 2026-02-02
**Branch**: feature/divergence-detection-checkpoint-preservation
**Commits**: b3ca1fc (bug fixes), fd74218 (refactor)

## What Was Fixed

### 1. Race Condition (Critical Bug)

**Problem**: Best checkpoints were being deleted because the preserved list was updated AFTER checkpoint deletion.

**The Flow Was:**
1. `save_checkpoint()` → deletes old checkpoints using preserved list
2. `_update_best_model()` → adds new checkpoint to preserved list (too late!)

**Now:**
1. `update_best_checkpoints()` → updates list FIRST
2. `save_checkpoint()` → deletes using CORRECT preserved list

### 2. Duplicate Logging (All Ranks)

**Problem**:
- "Updating best model to..." logged 5 times (once per GPU)
- "New best checkpoint..." logged 5 times
- "Best checkpoints: [...]" logged 5 times

**Fix**: All logging now checks `is_world_process_zero` or happens inside checkpoint_manager methods that receive the flag.

### 3. Wrong Checkpoint Count

**Problem**: `preserve_n_best=2` but list showed 3 checkpoints

**Root Cause**: The old code appended to the list before trimming, and trimming happened in the wrong place.

**Fix**: Moved all logic to `checkpoint_manager.update_best_checkpoints()` which properly sorts and trims in one place.

### 4. Deleted Best Checkpoint

**Problem**: checkpoint-23750 deleted even though it was in the "best" list

**Root Cause**: Race condition #1 - list updated after deletion

**Fix**: update_best_checkpoints() called before save

## API Changes

### New CheckpointManager Methods

```python
class CheckpointManager:
    def update_best_checkpoints(
        self,
        checkpoint_path: str,
        metrics: dict[str, float],
        metric_key: str,
        greater_is_better: bool | None,
        preserve_n_best: int,
        is_world_process_zero: bool = True,
    ) -> bool:
        """
        Update best checkpoints list if this checkpoint qualifies.

        Returns True if checkpoint is one of the best.

        Call this BEFORE save_checkpoint() so preserved list is correct.
        """

    def get_best_checkpoints_summary(self, metric_key: str = "loss") -> str:
        """Get formatted summary with metrics for end-of-training logging."""
```

### Trainer Changes

**Old (broken):**
```python
checkpoint_path = checkpoint_manager.save_checkpoint(...)
if eval_metrics:
    _update_best_model(checkpoint_path, eval_metrics)  # After save!
```

**New (fixed):**
```python
checkpoint_path = next_checkpoint_path(output_dir, step)
if eval_metrics:
    checkpoint_manager.update_best_checkpoints(...)  # Before save!
checkpoint_manager.save_checkpoint(...)
```

### Removed APIs

**Completely removed** (as you requested):
- `load_best_model_at_end` support
- `metric_for_best_model` references
- `greater_is_better` backward compatibility
- `_update_best_model()` method
- All migration and validation code

## What You'll See Now

### During Training

**Single-rank logging only:**
```
INFO:forgather.ml.trainer.checkpoint_manager:Saving checkpoint at .../checkpoint-25000
INFO:forgather.ml.trainer.trainer:New best checkpoint: .../checkpoint-25000 (loss=2.8557)
INFO:forgather.ml.trainer.checkpoint_manager:Best checkpoints:
INFO:forgather.ml.trainer.checkpoint_manager:  .../checkpoint-25000 (loss=2.8557)
INFO:forgather.ml.trainer.checkpoint_manager:  .../checkpoint-23750 (loss=2.8901)
```

**Correct preservation:**
- `preserve_n_best=2` → exactly 2 best checkpoints kept
- `save_total_limit=3` → up to 3 recent checkpoints + 2 best = 5 total max
- Best checkpoints never deleted

### At Training End

```
============================================================
Training complete!
Best checkpoints (N=2):
  /path/to/checkpoint-25000: loss=2.8557
  /path/to/checkpoint-23750: loss=2.8901
============================================================
```

### After Divergence

The divergence detector will stop training, and the best checkpoints will be preserved. You'll see which checkpoints are best in the final summary.

## Migration Guide

If you were using `load_best_model_at_end`:

**Before:**
```python
args = TrainingArguments(
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)
```

**After:**
```python
args = TrainingArguments(
    preserve_best_model=True,
    best_model_metric="loss",
    eval_on_save=True,  # Force eval at save steps for accurate tracking
)
```

**Note**: The old API no longer works - you must update your configs.

## Testing

Run your training again with these changes. You should see:

1. ✅ Single-rank logging (no more 5x duplicates)
2. ✅ Correct checkpoint count (preserve_n_best=2 → exactly 2)
3. ✅ Best checkpoints preserved (never deleted)
4. ✅ Metrics shown in checkpoint list
5. ✅ Summary at end with all best checkpoints
6. ✅ Divergence detector stops training and preserves good checkpoints

## Files Changed

- `src/forgather/ml/trainer/checkpoint_manager.py` (+91 lines)
  - Added update_best_checkpoints() method
  - Added get_best_checkpoints_summary() method
  - Removed old set_best_checkpoint()

- `src/forgather/ml/trainer/trainer.py` (-103 lines)
  - Refactored checkpoint save flow
  - Removed _update_best_model() method
  - Added end-of-training summary logging
  - Imports next_checkpoint_path

- `src/forgather/ml/trainer/base_trainer.py` (-54 lines)
  - Removed load_best_model_at_end backward compatibility
  - Removed validation code

- `BUG_FIXES_SUMMARY.md` (NEW)
  - Detailed analysis of both critical bugs

- `DIVERGENCE_DETECTION_IMPLEMENTATION.md` (updated)
  - Migration notes updated

## Next Steps

1. Re-run your training with these fixes
2. Verify:
   - No duplicate logging
   - Correct checkpoint count
   - Best checkpoints preserved
   - Divergence detection works
3. Update any configs using `load_best_model_at_end` to use `preserve_best_model`

## Questions or Issues?

The design is now much simpler:
- All best checkpoint logic in checkpoint_manager
- No backward compatibility cruft
- Clear separation of concerns
- Single-rank logging throughout

If you see any issues, the code is much easier to debug now!
