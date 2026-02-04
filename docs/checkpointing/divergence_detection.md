# Divergence Detection and Checkpoint Preservation

**Last Updated**: 2026-02-02

## Overview

Forgather provides advanced checkpoint management features to prevent loss of good checkpoints and detect training divergence early:

1. **Checkpoint Preservation**: Keep best checkpoints safe from cleanup
2. **Stateful Callbacks**: Save and restore callback state with checkpoints
3. **Decoupled Eval/Save**: Force evaluation before saving to ensure metrics available
4. **Divergence Detection**: Catch training issues in ~10-20 eval steps vs 50,000 train steps

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
from forgather.ml.trainer.callbacks import DualTimeScaleDivergenceDetector

detector = DualTimeScaleDivergenceDetector(
    short_alpha=0.1,         # Fast EMA (~10 step window)
    long_alpha=0.01,         # Slow EMA (~100 step window)
    threshold=2.0,           # Stop if short - long >= 2.0
    action="stop",
    use_eval_loss=True,      # Monitor eval_loss (more stable)
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

### DualTimeScaleDivergenceDetector

Detects sustained loss increases using dual-timescale exponential moving averages:

```python
from forgather.ml.trainer.callbacks import DualTimeScaleDivergenceDetector

detector = DualTimeScaleDivergenceDetector(
    short_alpha=0.1,         # Fast filter (0-1, higher = faster response)
    long_alpha=0.01,         # Slow filter (0-1, lower = slower response)
    threshold=2.0,           # Trigger when short - long >= threshold
    action="stop",           # "stop" (graceful) or "abort" (immediate)
    use_eval_loss=True,      # Monitor eval_loss (vs train loss)
    metric_key=None,         # Optional: custom metric to monitor
)
```

**How it works**:
- Maintains fast EMA (tracks current state) and slow EMA (tracks baseline)
- Fast filter responds quickly to changes, slow filter stays stable
- Triggers when fast - slow >= threshold (sustained divergence)
- Classic signal processing approach for detecting sustained changes vs transient noise

**Effective window**: ~1/alpha steps
- `short_alpha=0.1` → ~10 step window
- `long_alpha=0.01` → ~100 step window

### DualWindowDivergenceDetector

Detects divergence using finite moving average windows:

```python
from forgather.ml.trainer.callbacks import DualWindowDivergenceDetector

detector = DualWindowDivergenceDetector(
    short_window=10,         # Recent average (last N steps)
    long_window=100,         # Long-term average (last N steps)
    threshold=2.0,
    action="stop",
    use_eval_loss=True,
)
```

**How it works**:
- Maintains finite buffers of recent observations
- Computes simple moving averages over windows
- Triggers when short_avg - long_avg >= threshold

**When to use**:
- More intuitive than EMA ("last 10 steps" vs "alpha=0.1")
- Exact cutoff at window boundary (vs exponential decay)
- Better for detecting gradual degradation over fixed horizons

### Tuning for Quick Spike Detection

For catching sudden spikes (e.g., loss 2.8 → 8.0):

```python
detector = DualTimeScaleDivergenceDetector(
    short_alpha=0.2,         # Faster response (~5 steps)
    long_alpha=0.01,         # Stable baseline
    threshold=1.5,           # More sensitive
    use_eval_loss=True,
)

args = TrainingArguments(
    eval_steps=500,          # Eval every 500 steps
    eval_on_save=True,
)
```

**Detection speed**: ~10-20 eval steps (5,000-10,000 train steps) vs 50,000 train steps!

### Monitoring Custom Metrics

Monitor gradient norms, accuracy drops, or other metrics:

```python
# Monitor gradient norm
detector = DualTimeScaleDivergenceDetector(
    short_alpha=0.1,
    long_alpha=0.01,
    threshold=5.0,           # Higher threshold for grad norm
    metric_key="grad_norm",  # Custom metric
)

# Monitor accuracy drops
detector = DualTimeScaleDivergenceDetector(
    short_alpha=0.1,
    long_alpha=0.01,
    threshold=0.1,           # 10% accuracy drop
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

**Resume correctness**: Divergence detectors resume with correct EMA/buffer state after checkpoint load

## Advanced Patterns

### Multiple Safeguards

Combine multiple detection strategies:

```python
from forgather.ml.trainer.callbacks import (
    DualTimeScaleDivergenceDetector,
    DualWindowDivergenceDetector,
)

callbacks = [
    # Fast spike detection (EMA)
    DualTimeScaleDivergenceDetector(
        short_alpha=0.2, long_alpha=0.01, threshold=1.5
    ),
    # Gradual degradation (window)
    DualWindowDivergenceDetector(
        short_window=10, long_window=100, threshold=0.5
    ),
]

args = TrainingArguments(
    preserve_best_model=True,
    preserve_n_best=3,
    eval_on_save=True,
)

trainer = Trainer(..., args=args, callbacks=callbacks)
```

### Production Configuration

Balanced settings for production training:

```python
detector = DualTimeScaleDivergenceDetector(
    short_alpha=0.1,
    long_alpha=0.01,
    threshold=2.0,
    use_eval_loss=True,
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

**Cause**: Threshold too high or alpha too low

**Fix**: Lower threshold or increase short_alpha for faster response

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
