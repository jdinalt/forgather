# Critical Bug Fixes - Divergence Detection and Checkpoint Preservation

**Date**: 2026-02-02
**Branch**: feature/divergence-detection-checkpoint-preservation
**Commit**: b3ca1fc

## Summary

Fixed two critical bugs discovered during real training run where:
- Loss diverged from 3.15 → 8.27 but training didn't stop
- Good checkpoints (checkpoint-6250) were deleted despite being marked as best
- Training continued until segfault

## Bug #1: Best Checkpoint List Management

### Root Cause
In `checkpoint_manager.py`, `set_best_checkpoint()` called `pop()` BEFORE the caller sorted the list:

```python
# BUGGY CODE
def set_best_checkpoint(self, checkpoint_path, metric_value):
    self.best_checkpoints.append((checkpoint_path, metric_value))
    if len(self.best_checkpoints) > self.preserve_n_best:
        self.best_checkpoints.pop()  # ❌ Removes LAST item before sorting!
```

### The Problem
1. Append new checkpoint (might be the BEST one!)
2. Pop last item (removes the one we just added if list was full!)
3. THEN the caller sorts the list (too late!)

### Example
- `best_checkpoints = [(cp-2500, 3.5), (cp-1250, 3.6)]` (preserve_n_best=2)
- New checkpoint: `(cp-6250, 3.15)` ← BEST loss so far!
- After append: `[(cp-2500, 3.5), (cp-1250, 3.6), (cp-6250, 3.15)]`
- After pop: `[(cp-2500, 3.5), (cp-1250, 3.6)]` ← cp-6250 removed!
- Caller sorts (too late): `[(cp-6250, 3.15), ...]` but cp-6250 is gone!

### Solution
Remove the pop() call since the caller already handles sorting and trimming:

```python
# FIXED CODE
def set_best_checkpoint(self, checkpoint_path, metric_value):
    self.best_checkpoints.append((checkpoint_path, metric_value))
    # Sorting and trimming handled by caller based on greater_is_better
    logger.info(f"Best checkpoints: {[cp[0] for cp in self.best_checkpoints]}")
```

The caller in `trainer.py` already does the correct sequence:
1. Call `set_best_checkpoint()` to append
2. Sort the list by metric value
3. Trim to N best using slice: `best_checkpoints[:preserve_n_best]`

## Bug #2: Divergence Detector Not Receiving Eval Metrics

### Root Cause
Eval metrics are dispatched via `on_evaluate` event, but divergence detectors only implemented `on_log`.

In `trainer.py`:
```python
def _eval_loop(self):
    metrics = {"eval_loss": (total_loss / (step + 1)).item()}
    self._dispatch_event("on_evaluate", metrics=metrics)  # ← on_evaluate, not on_log!
    return metrics
```

Divergence detectors only had:
```python
class DualTimeScaleDivergenceDetector(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # This only gets train loss, NEVER eval_loss!
        ...
```

### The Problem
With `use_eval_loss=True` (default), the detector was configured to monitor eval_loss but never received those values. The detector sat idle while loss diverged from 3.15 → 8.27.

### Evidence from Real Training Run
```
Step 6250: eval_loss = 3.154 (best)
Step 6875: eval_loss = 7.860 (divergence starts!)
Step 7500: eval_loss = 8.272
Step 8125: eval_loss = 8.389 (should have triggered here!)
Step 8750: eval_loss = 8.407
```

With the configured parameters:
- short_alpha=0.1, long_alpha=0.01, threshold=1.0

The detector SHOULD have triggered at step 8125:
- short_ema ≈ 4.55, long_ema ≈ 3.35
- Divergence: 4.55 - 3.35 = 1.20 >= 1.0 ✓

But it never saw the eval_loss values!

### Solution
Refactor to a shared `_check_divergence()` method called from both `on_log` and `on_evaluate`:

```python
class DualTimeScaleDivergenceDetector(TrainerCallback):
    def _check_divergence(self, args, state, control, logs=None, metrics=None):
        """Check for divergence given metrics dict."""
        data = logs or metrics  # Accept either logs or metrics
        if not data:
            return control
        # ... rest of divergence detection logic ...

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update EMA filters when training metrics are logged."""
        return self._check_divergence(args, state, control, logs=logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Update EMA filters when evaluation metrics are available."""
        return self._check_divergence(args, state, control, metrics=metrics)
```

Applied to both `DualTimeScaleDivergenceDetector` and `DualWindowDivergenceDetector`.

## Impact

With these fixes:
1. **Best checkpoints are correctly preserved** - checkpoint-6250 would have been kept safe
2. **Divergence detection works** - training would have stopped at step 8125 (before deleting checkpoint-6250)
3. **No checkpoint loss** - best model preserved even after divergence

## Testing

All 17 tests pass:
```bash
python -m pytest tests/test_divergence_detection.py -v
============================= 17 passed in 0.02s ==============================
```

## Files Changed

- `src/forgather/ml/trainer/checkpoint_manager.py`: Removed premature pop()
- `src/forgather/ml/trainer/callbacks/divergence_detector.py`: Added on_evaluate support

## Recommended Next Steps

1. Re-run the training with these fixes
2. Verify divergence detection triggers at expected step
3. Confirm best checkpoint is preserved
4. Consider lowering threshold to 0.75 for faster detection (would trigger at step 7500 instead of 8125)
