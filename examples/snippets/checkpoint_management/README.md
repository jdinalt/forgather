# Checkpoint Management Examples

This directory contains examples for Forgather's checkpoint preservation and divergence detection features.

## Files

- `preserve_best_checkpoints.py` - Basic checkpoint preservation examples
- `divergence_detection.py` - Divergence detection callback usage
- `advanced_usage.py` - Advanced patterns combining multiple features

## Quick Start

### Prevent Best Checkpoint Deletion

```python
from forgather.ml.trainer import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="output_models/my_model",
    save_steps=1000,
    save_total_limit=3,           # Keep only 3 recent checkpoints
    preserve_best_model=True,     # Don't delete best checkpoint
    best_model_metric="loss",
    preserve_n_best=2,            # Keep top 2 checkpoints
    eval_on_save=True,            # Force eval at save steps
)
```

### Detect Training Divergence

```python
from forgather.ml.trainer.callbacks import DualTimeScaleDivergenceDetector

detector = DualTimeScaleDivergenceDetector(
    short_alpha=0.1,         # Fast EMA (~10 step window)
    long_alpha=0.01,         # Slow EMA (~100 step window)
    threshold=2.0,           # Stop if short - long >= 2.0
    action="stop",
    use_eval_loss=True,
)

trainer = Trainer(model, args=args, callbacks=[detector], ...)
```

## Key Features

1. **Checkpoint Preservation**: Keep best checkpoints safe from cleanup
2. **Decoupled Eval/Save**: Different eval/save frequencies without constraints
3. **Divergence Detection**: Catch training issues in ~10-20 eval steps
4. **Stateful Callbacks**: Callback state saved/restored with checkpoints

## Documentation

See `docs/checkpointing/` for complete documentation:
- `user_guide.md` - User-facing documentation
- `divergence_detection.md` - Divergence detection guide
- `example_usage.py` - Comprehensive examples

## Testing

Unit tests available in `tests/test_divergence_detection.py`
