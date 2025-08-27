# Forgather ML Trainer System Overview

The Forgather ML framework provides a sophisticated trainer system designed to be compatible with HuggingFace Transformers while offering enhanced flexibility, extensibility, and support for distributed training scenarios.

## Design Philosophy

The trainer system follows these core principles:

- **HuggingFace Compatibility**: Maintains API compatibility with `transformers.Trainer` for easy migration
- **Extensibility**: Built on abstract base classes with clear extension points
- **Distributed Training**: Multiple implementations for different parallelization strategies
- **Configuration-Driven**: Deep integration with Forgather's template-based configuration system
- **Checkpoint Robustness**: Advanced checkpoint management with state preservation

## Architecture Overview

### Class Hierarchy

```
AbstractBaseTrainer (ABC)
└── ExtensibleTrainer (ABC)
    └── BaseTrainer (ABC)
        ├── Trainer
        ├── AccelTrainer 
        └── PipelineTrainer
```

### Core Abstractions

1. **AbstractBaseTrainer**: Minimal core interface (`train`, `evaluate`, `save_model`)
2. **ExtensibleTrainer**: Adds callback system support
3. **BaseTrainer**: Common functionality implementation (still abstract)
4. **Concrete Trainers**: Specific implementations for different training scenarios

## Available Trainer Implementations

### 1. Trainer
- **Use Case**: Single GPU/CPU training, prototyping, small models
- **Features**: Lightweight, minimal dependencies, full HF compatibility
- **Best For**: Development, debugging, models that fit on single GPU

### 2. AccelTrainer
- **Use Case**: Multi-GPU training with automatic distributed coordination
- **Features**: HuggingFace Accelerate integration, automatic device placement
- **Best For**: Data parallel training, models that need multiple GPUs

### 3. PipelineTrainer
- **Use Case**: Very large models requiring pipeline parallelism
- **Features**: Model partitioning, multiple scheduling strategies
- **Best For**: Models too large for single GPU, pipeline parallel workloads

## Quick Start

### Basic Training
```python
from forgather.ml import Trainer, TrainingArguments

# Create trainer
trainer = Trainer(
    model_init=model_factory,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_steps=100,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
result = trainer.train()
```

### Multi-GPU Training
```python
from forgather.ml import AccelTrainer, AccelTrainingArguments

trainer = AccelTrainer(
    model_init=model_factory,
    args=AccelTrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,  # Per device
        # Accelerate handles multi-GPU automatically
    ),
    train_dataset=train_dataset,
)

result = trainer.train()
```

### Pipeline Parallel Training
```python
from forgather.ml import PipelineTrainer, PipelineTrainingArguments
from forgather.ml.distributed import DistributedEnvironment

trainer = PipelineTrainer(
    distributed_env=DistributedEnvironment(),
    loss_fn=loss_function,
    args=PipelineTrainingArguments(
        output_dir="./output",
        split_spec={"layer.4": "end", "layer.8": "end"},  # Pipeline stages
        pipeline_chunks=4,
    ),
    train_dataset=train_dataset,
)

result = trainer.train()
```

## Key Features

### Checkpoint Management
- **Automatic Resume**: `resume_from_checkpoint=True` finds latest checkpoint
- **State Preservation**: Saves optimizer, scheduler, and global step state
- **Robust Discovery**: Uses modification time for reliable checkpoint selection

### Callback System
- **HuggingFace Compatible**: Same event system as `transformers.Trainer`
- **Built-in Callbacks**: Progress bars, logging, TensorBoard, text generation
- **Easy Extension**: Add custom callbacks for monitoring and control

### Configuration Integration
- **Template-Based**: Define trainers in YAML templates with inheritance
- **Factory Pattern**: Flexible component construction and dependency injection
- **Reproducible**: Complete configuration serialization and versioning

## Migration from HuggingFace

Forgather trainers are designed as drop-in replacements for `transformers.Trainer`:

```python
# HuggingFace Trainer
from transformers import Trainer, TrainingArguments

# Forgather Trainer (minimal changes)
from forgather.ml import Trainer, TrainingArguments

# Same API, enhanced functionality
trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # Additional Forgather features:
    optimizer_factory=optimizer_factory,
    lr_scheduler_factory=scheduler_factory,
)
```

## Next Steps

- [Trainer API Reference](api-reference.md) - Detailed API documentation
- [Configuration Guide](configuration.md) - Template-based configuration
- [Distributed Training](distributed-training.md) - Multi-GPU and pipeline parallelism
- [Callbacks](callbacks.md) - Callback system and custom callbacks
- [Checkpointing](checkpointing.md) - Advanced checkpoint management
- [Trainer Control](trainer-control.md) - External control of running training jobs