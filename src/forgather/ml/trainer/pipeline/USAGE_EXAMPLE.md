# Pipeline Parallel Splitter Usage Examples

## Overview

The PipelineTrainer now uses dependency injection for model splitting, allowing you to choose between automatic (torch.export-based) and manual (layer deletion) splitting strategies.

## Manual Splitting for CasualLM Models

Manual splitting is recommended for CasualLM-based models as it:
- Supports external attention mask creation (avoiding pipeline transport issues)
- Works with models that use BlockMask for flex_attention
- Provides predictable layer distribution across stages

```python
from functools import partial
from forgather.ml.trainer.pipeline import (
    PipelineTrainer,
    PipelineTrainingArguments,
    create_manual_causal_lm_splitter,
)

# Create splitter - auto-detects num_layers from model.config
splitter = create_manual_causal_lm_splitter(
    num_layers=None,      # Auto-detect from model.config.num_hidden_layers
    input_weight=1,       # Computational weight for input_encoder
    output_weight=1,      # Computational weight for output modules
)

# Or use partial to hide configuration
splitter = partial(
    create_manual_causal_lm_splitter,
    input_weight=2,       # Heavier weight if input processing is expensive
    output_weight=2,
)()

# Training arguments - only basic types
args = PipelineTrainingArguments(
    output_dir="./output",
    pipeline_chunks=4,
    stages_per_rank=1,
    per_device_train_batch_size=8,
    # ... other training args
)

# Create trainer with injected splitter
trainer = PipelineTrainer(
    args=args,
    model_splitter=splitter,     # Injected here
    model_init=model_init_fn,
    train_dataset=train_dataset,
    # ... other trainer params
)

trainer.train()
```

## Custom Splitters

You can implement custom splitters by creating a function matching the ModelSplitter signature:

```python
def my_custom_splitter(
    model, example_args, example_kwargs, stage_indices, train, device, rank, pp_group
):
    # Your custom splitting logic here
    # Must return: (all_modules, rank_modules, rank_stages, mask_creator)
    ...
    return all_pipeline_modules, pipeline_modules, pipeline_stages, attention_mask_creator

trainer = PipelineTrainer(
    args=args,
    model_splitter=my_custom_splitter,
    # ...
)
```

## Benefits

1. **Clean separation**: Splitting logic isolated from trainer
2. **Flexible**: Easy to add new splitting strategies
3. **Testable**: Splitters can be tested independently
4. **Type-safe**: Only basic types in args, complex objects injected
5. **Attention mask support**: Manual splitter supports external mask creation
