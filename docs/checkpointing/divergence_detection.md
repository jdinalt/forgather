# Divergence Detection and Checkpoint Preservation

**Last Updated**: 2026-03-17

## Overview

Forgather provides advanced checkpoint management features to prevent loss of good checkpoints and detect training divergence early:

1. **Checkpoint Preservation**: Keep best checkpoints safe from cleanup
2. **Stateful Callbacks**: Save and restore callback state with checkpoints
3. **Decoupled Eval/Save**: Force evaluation before saving to ensure metrics available
4. **Divergence Detection**: Catch training divergence within a few log entries of a loss spike

## Quick Start

### Prevent Best Checkpoint Deletion

```python
from forgather.ml.trainer import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="output_models/my_model",
    save_steps=1000,
    save_total_limit=3,           # Keep only 3 recent checkpoints

    # Checkpoint preservation (new!)
    preserve_best_model=True,     # Don't delete best checkpoint
    best_model_metric="loss",
    preserve_n_best=2,            # Keep top 2 checkpoints

    # Decoupled eval/save (new!)
    eval_on_save=True,            # Force eval when saving
)

trainer = Trainer(model, args=args, ...)
trainer.train()
```

### Detect Training Divergence

```python
from forgather.ml.trainer.callbacks import DivergenceDetector

detector = DivergenceDetector(
    smoothing=0.3,           # EMA alpha for loss smoothing
    threshold=1.0,           # Stop if smoothed - best >= 1.0
    patience=3,              # Require 3 consecutive observations
    action="stop",
)

trainer = Trainer(model, args=args, callbacks=[detector], ...)
trainer.train()
```

## Checkpoint Preservation

### Basic Preservation

Prevent best checkpoints from being deleted by `save_total_limit`:

```python
args = TrainingArguments(
    save_total_limit=3,           # Keep 3 recent checkpoints
    preserve_best_model=True,     # Plus preserve best
    best_model_metric="loss",
)
```

**Result**: If checkpoint-2000 is best, it won't be deleted even after checkpoint-3000, 4000, 5000 are created.

### Track Multiple Best Checkpoints

Keep top N checkpoints for ensembling or comparison:

```python
args = TrainingArguments(
    save_total_limit=5,
    preserve_best_model=True,
    preserve_n_best=3,           # Keep top 3 checkpoints
    best_model_metric="eval_accuracy",
    best_model_greater_is_better=True,
)
```

**Result**:
- Top 3 checkpoints by accuracy preserved
- Last 5 checkpoints preserved
- Total: up to 8 checkpoints (if best not in recent 5)

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `preserve_best_model` | bool | False | Enable best checkpoint preservation |
| `best_model_metric` | str | "loss" | Metric to compare checkpoints |
| `best_model_greater_is_better` | bool\|None | None | Higher is better? (auto-detected if None) |
| `preserve_n_best` | int | 1 | Keep top N checkpoints |

## Decoupled Eval/Save

Force evaluation before saving to ensure metrics are available for best model tracking:

```python
args = TrainingArguments(
    save_steps=1000,       # Save every 1000 steps
    eval_steps=250,        # Eval every 250 steps
    eval_on_save=True,     # Force eval when saving (new!)
)
```

**Without `eval_on_save`**: Required `save_steps % eval_steps == 0` (e.g., save at 1000, eval at 250, 500, 750, 1000)

**With `eval_on_save`**: No alignment required. Eval forced at save steps (e.g., eval at 250, 500, 750, 1000, 1250, ...)

**Benefits**:
- Flexible scheduling (different eval/save frequencies)
- No manual alignment calculation needed
- Guaranteed metrics available for best model tracking

## Divergence Detection

### DivergenceDetector

Detects training divergence by comparing smoothed loss against its best observed value:

```python
from forgather.ml.trainer.callbacks import DivergenceDetector

detector = DivergenceDetector(
    smoothing=0.3,                # EMA alpha for loss smoothing (0-1)
    threshold=1.0,                # Absolute: smoothed - best >= threshold
    relative_threshold=None,      # Relative: smoothed >= best * factor (e.g. 1.5)
    patience=3,                   # Consecutive observations above threshold
    warmup=10,                    # Skip first N observations
    action="stop",                # "stop" (graceful) or "abort" (immediate)
    use_eval_loss=False,          # Monitor train loss (more frequent)
    metric_key=None,              # Optional: custom metric to monitor
)
```

**How it works**:
1. Smooths the raw loss with an EMA to reduce noise
2. Tracks the running minimum of the smoothed loss (best observed baseline)
3. Triggers when smoothed loss exceeds the baseline by the threshold
4. Requires `patience` consecutive observations above threshold to avoid false positives
5. Detects NaN/Inf values immediately (no patience required)

**Why this approach?** The previous dual-EMA approach (comparing fast vs slow EMA) fails
when loss decreases monotonically during normal training. The slow EMA lags above the fast
EMA, creating a negative divergence buffer that masks real spikes. The "smoothed vs best"
approach correctly handles this common training pattern.

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smoothing` | float | 0.3 | EMA alpha (0-1). Higher = more responsive. Window ~ 1/alpha |
| `threshold` | float\|None | 1.0 | Absolute threshold: smoothed - best >= threshold |
| `relative_threshold` | float\|None | None | Relative threshold: smoothed >= best * factor |
| `patience` | int | 3 | Consecutive observations above threshold before triggering |
| `warmup` | int | 10 | Skip first N observations (avoids early high-loss phase) |
| `action` | str | "stop" | "stop" (save checkpoint then stop) or "abort" (stop immediately) |
| `use_eval_loss` | bool | False | True: monitor eval_loss, False: monitor train loss |
| `metric_key` | str\|None | None | Custom metric key (overrides use_eval_loss) |

At least one of `threshold` or `relative_threshold` must be set. Both can be set
simultaneously; either condition firing counts toward patience.

### Choosing Threshold Type

**Absolute threshold** (`threshold=1.0`): triggers when smoothed loss rises 1.0 above
the best. Works well when you know the expected loss scale and want a fixed tolerance.

**Relative threshold** (`relative_threshold=1.5`): triggers when smoothed loss is 1.5x
the best (a 50% increase). Works well across different loss scales since it adapts
proportionally to the baseline.

**Both**: set both to catch either condition. For example, `threshold=1.0` catches
divergence from a low baseline (e.g., best=0.5, spike to 1.5) while
`relative_threshold=2.0` catches proportional spikes at any scale.

### Detection Speed

With defaults (`smoothing=0.3, threshold=1.0, patience=3`) and training logs every 32
steps:

- A spike from loss 3.8 to 9.7 is detected within **3 log entries** (~96 training steps)
- No false positives on healthy runs with normal loss fluctuations

Higher `smoothing` (e.g., 0.5) makes detection faster but more sensitive to noise.
Lower `patience` (e.g., 1) triggers immediately but risks false positives from transient
spikes.

### Monitoring Custom Metrics

Monitor gradient norms, accuracy drops, or other metrics:

```python
# Monitor gradient norm
detector = DivergenceDetector(
    threshold=5.0,
    metric_key="grad_norm",
)

# Monitor accuracy drops (use relative threshold)
detector = DivergenceDetector(
    threshold=None,
    relative_threshold=1.3,    # 30% degradation from best
    metric_key="eval_accuracy",
)
```

## Stateful Callbacks

Callbacks implementing the `Stateful` protocol have their state automatically saved/restored with checkpoints:

```python
from forgather.ml.trainer.callbacks import TrainerCallback
from torch.distributed.checkpoint.stateful import Stateful

class MyDetector(TrainerCallback, Stateful):
    def __init__(self):
        self.my_state = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Your detection logic
        return control

    def state_dict(self):
        """Save callback state to checkpoint."""
        return {'my_state': self.my_state}

    def load_state_dict(self, state_dict):
        """Restore callback state from checkpoint."""
        self.my_state = state_dict['my_state']
```

**Saved to**: `checkpoint_path/callback_states.pt`

**Automatic handling**: CheckpointManager detects Stateful callbacks and saves/loads their state

**Resume correctness**: DivergenceDetector resumes with correct smoothed loss, best baseline, observation count, and consecutive trigger count after checkpoint load

## Advanced Patterns

### Multiple Safeguards

Combine absolute and relative thresholds:

```python
from forgather.ml.trainer.callbacks import DivergenceDetector

# Catches both absolute spikes and proportional divergence
detector = DivergenceDetector(
    smoothing=0.3,
    threshold=1.0,              # Absolute: any 1.0 increase from best
    relative_threshold=1.5,     # Relative: 50% increase from best
    patience=3,
    action="abort",
)

args = TrainingArguments(
    preserve_best_model=True,
    preserve_n_best=3,
    eval_on_save=True,
)

trainer = Trainer(..., args=args, callbacks=[detector])
```

### Production Configuration

Balanced settings for production training:

```python
detector = DivergenceDetector(
    smoothing=0.3,
    threshold=1.0,
    patience=5,          # Higher patience for fewer false positives
    warmup=20,           # Longer warmup for large models with noisy early loss
    action="stop",       # Graceful stop (saves checkpoint)
)

args = TrainingArguments(
    preserve_best_model=True,
    preserve_n_best=2,
    save_total_limit=5,
    eval_on_save=True,
    eval_steps=500,
    save_steps=1000,
)
```

### LR Sweep Configuration

For learning rate sweeps where you expect some runs to diverge:

```python
detector = DivergenceDetector(
    smoothing=0.3,
    threshold=1.0,
    patience=3,
    warmup=10,
    action="abort",      # Abort without saving (diverged checkpoints are useless)
)
```

### Experimentation Configuration

Frequent checkpoints for rapid iteration:

```python
args = TrainingArguments(
    preserve_best_model=True,
    preserve_n_best=5,      # Keep top 5 for comparison
    save_total_limit=10,    # Keep 10 recent
    eval_on_save=True,
    eval_steps=100,         # Frequent feedback
    save_steps=500,
)
```

## Backward Compatibility

### Divergence Detector

The old `DualTimeScaleDivergenceDetector` and `DualWindowDivergenceDetector` names are
kept as aliases for `DivergenceDetector`. Existing imports will continue to work, but the
parameter interface has changed. Update your code to use `DivergenceDetector` with the
new parameters.

### Checkpoint Preservation

Old `load_best_model_at_end` API still works with deprecation warning:

```python
# Old API (deprecated)
args = TrainingArguments(
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_steps=1000,
    eval_steps=1000,  # Required alignment
)
```

**Auto-migrates to**:
```python
# New API
args = TrainingArguments(
    preserve_best_model=True,
    best_model_metric="loss",
    eval_on_save=True,
    save_steps=1000,
    eval_steps=250,   # No alignment needed
)
```

## Troubleshooting

### "No metrics available for best model tracking"

**Cause**: No evaluation metrics when saving

**Fix**: Set `eval_on_save=True` or align eval/save schedules

### "Divergence detector not triggering"

**Possible causes**:
- Threshold too high for the loss scale of your model
- Warmup too long (detector is still skipping observations)
- Using `use_eval_loss=True` with infrequent evaluation

**Fix**: Lower threshold, reduce warmup, increase smoothing for faster response,
or switch to `use_eval_loss=False` to use the more frequent train loss

### "Callback state not restored after checkpoint load"

**Cause**: Callback not implementing Stateful protocol

**Fix**: Inherit from both TrainerCallback and Stateful, implement state_dict/load_state_dict

### "Too many checkpoints preserved"

**Cause**: save_total_limit + preserve_n_best

**Fix**: Total checkpoints = save_total_limit + preserve_n_best (when best not in recent)

## Examples

See `examples/snippets/checkpoint_management/` for complete examples:
- `preserve_best_checkpoints.py` - Checkpoint preservation examples
- `divergence_detection.py` - Divergence detection usage
- `advanced_usage.py` - Advanced patterns

## References

- User Guide: `docs/checkpointing/user_guide.md`
- Technical Details: `docs/checkpointing/distributed_checkpoint_abstraction.md`
- Migration Guide: `docs/checkpointing/migration_guide.md`
- Tests: `tests/test_divergence_detection.py`
