# Trainer Configuration Guide

This guide covers how to configure Forgather trainers using the template-based configuration system.

## Configuration Overview

Forgather uses a sophisticated template system that combines:
- **YAML configuration files** with Jinja2 preprocessing
- **Template inheritance** for sharing common configurations
- **Factory pattern** for flexible component construction
- **Dependency injection** through configuration references

## Basic Configuration Structure

### Minimal Trainer Configuration

```yaml
# templates/configs/basic_training.yaml
-- extends 'trainers/trainer.yaml'

-- block trainer_args
    output_dir: "./output"
    per_device_train_batch_size: 16
    num_train_epochs: 3
    logging_steps: 100
-- endblock trainer_args
```

### Complete Training Configuration

```yaml
-- extends 'projects/base.yaml'

-- block trainer_definition
    -- include 'my_trainer_config'
-- endblock trainer_definition

#-------------------- my_trainer_config --------------------
-- extends 'trainers/trainer.yaml'

-- block trainer_args
    == super()  # Include parent arguments
    
    # Basic training settings
    output_dir: "./output_models/my_model"
    logging_dir: "./output_models/my_model/runs"
    
    # Training schedule
    num_train_epochs: 5
    max_steps: -1  # Use epochs instead
    per_device_train_batch_size: 32
    per_device_eval_batch_size: 64
    
    # Logging and evaluation
    logging_steps: 50
    logging_strategy: "steps"
    eval_steps: 200
    eval_strategy: "steps"
    
    # Saving and checkpoints
    save_steps: 500
    save_strategy: "steps"
    save_total_limit: 3
    
    # Checkpoint management
    save_optimizer_state: true
    save_scheduler_state: true
    resume_from_checkpoint: false
    
    # Performance optimization
    torch_compile: false
    dataloader_num_workers: 4
    dataloader_pin_memory: true
-- endblock trainer_args
```

## Trainer Type Selection

### Simple Trainer (Single GPU)

```yaml
-- extends 'trainers/trainer.yaml'

-- block trainer_args
    output_dir: "./output"
    per_device_train_batch_size: 16
    # Additional simple trainer args...
-- endblock trainer_args
```

### Accelerate Trainer (Multi-GPU)

```yaml
-- extends 'trainers/accel_trainer.yaml'

-- block trainer_args
    == super()
    output_dir: "./output"
    per_device_train_batch_size: 8  # Per device
    
    # Accelerate-specific settings
    accelerator_args:
        device_placement: true
        split_batches: false
        dispatch_batches: true
-- endblock trainer_args
```

### Pipeline Trainer (Large Models)

```yaml
-- extends 'trainers/pipeline_trainer.yaml'

-- block trainer_args
    == super()
    output_dir: "./output"
    per_device_train_batch_size: 4
    
    # Pipeline-specific settings
    split_spec:
        "layer.4": "end"
        "layer.8": "end"
        "layer.12": "end"
    pipeline_chunks: 4
    stages_per_rank: 1
    
    # Pipeline scheduling
    schedule_class: "ScheduleGPipe"
    checkpoint_sequential: true
-- endblock trainer_args

-- block distributed_env
    !singleton:forgather.ml.distributed:DistributedEnvironment@distributed_env
-- endblock distributed_env
```

## Component Configuration

### Optimizer Configuration

```yaml
-- block optimizer
    !partial:torch.optim:AdamW
        lr: 1.0e-3
        weight_decay: 0.01
        betas: [0.9, 0.999]
        eps: 1.0e-8
-- endblock optimizer
```

### Learning Rate Scheduler

```yaml
-- block lr_scheduler
    !partial:transformers:get_cosine_schedule_with_warmup
        num_warmup_steps: 1000
        num_training_steps: !var "max_steps"
-- endblock lr_scheduler
```

### Custom Scheduler Example

```yaml
-- block lr_scheduler
    !partial:forgather.ml.optim.infinite_lr_scheduler:InfiniteLRScheduler
        warmup_steps: 5000
        cooldown_steps: 50000
        constant_lr: 1.0e-4
-- endblock lr_scheduler
```

### Data Collator

```yaml
-- block data_collator
    !singleton:transformers:DataCollatorForLanguageModeling
        tokenizer: *tokenizer
        mlm: false
        return_tensors: "pt"
-- endblock data_collator
```

### Callbacks Configuration

```yaml
-- block trainer_callbacks
    !list:@trainer_callbacks
        # Progress and info callbacks (included by default)
        - !singleton:forgather.ml.tb_logger:TBLogger
            args: [*summary_writer]
            kwargs:
                experiment_info: *experiment_info
        
        - !singleton:forgather.ml.textgen_callback:TextgenCallback
            summary_writer: *summary_writer
            prompts: *test_prompts
            generation_config: *generation_config
            max_new_tokens: 50
            generation_steps: 1000
        
        - !singleton:forgather.ml.json_logger:JsonLogger
            <<: *experiment_info
-- endblock trainer_callbacks
```

## Advanced Configuration Patterns

### Environment-Specific Configurations

```yaml
# Base configuration
-- set training_env = "development"

-- if training_env == "development"
    -- set batch_size = 8
    -- set max_steps = 100
-- elif training_env == "production"
    -- set batch_size = 32
    -- set max_steps = 10000
-- endif

-- block trainer_args
    per_device_train_batch_size: !var "batch_size"
    max_steps: !var "max_steps"
-- endblock trainer_args
```

### Conditional Checkpoint Settings

```yaml
-- set enable_checkpointing = true
-- set checkpoint_frequency = 500

-- block trainer_args
    == super()
    
    -- if enable_checkpointing
    save_strategy: "steps"
    save_steps: !var "checkpoint_frequency"
    save_optimizer_state: true
    save_scheduler_state: true
    save_total_limit: 3
    -- else
    save_strategy: "no"
    -- endif
-- endblock trainer_args
```

### Template Inheritance Chains

```yaml
# base_trainer.yaml
-- block trainer_args
    output_dir: "./output"
    logging_steps: 500
    save_strategy: "no"
-- endblock trainer_args

# checkpointing_trainer.yaml
-- extends 'base_trainer.yaml'
-- block trainer_args
    == super()
    save_strategy: "steps"
    save_steps: 100
    save_optimizer_state: true
    save_scheduler_state: true
-- endblock trainer_args

# my_project.yaml
-- extends 'checkpointing_trainer.yaml'
-- block trainer_args
    == super()
    output_dir: "./my_model_output"
    per_device_train_batch_size: 16
-- endblock trainer_args
```

## Configuration Variables and References

### Using Variables

```yaml
-- set model_name = "my_transformer"
-- set base_lr = 1e-3
-- set warmup_ratio = 0.1

-- block trainer_args
    output_dir: "./output_models/{{ model_name }}"
    learning_rate: !var "base_lr"
-- endblock trainer_args

-- block lr_scheduler
    !partial:transformers:get_linear_schedule_with_warmup
        num_warmup_steps: !calc "int(max_steps * warmup_ratio)"
        num_training_steps: !var "max_steps"
-- endblock lr_scheduler
```

### Shared References

```yaml
# Define shared components
experiment_info: &experiment_info !dict:@experiment_info
    name: "My Training Experiment"
    description: "Testing new architecture"
    date: "2024-01-01"

summary_writer: &summary_writer !singleton:torch.utils.tensorboard:SummaryWriter
    - "./runs/experiment_001"

# Use in trainer callbacks
-- block trainer_callbacks
    !list:@trainer_callbacks
        - !singleton:forgather.ml.tb_logger:TBLogger
            args: [*summary_writer]
            kwargs:
                <<: *experiment_info
-- endblock trainer_callbacks
```

## Configuration Validation and Debugging

### Preprocessing Output

Use the CLI tool to examine preprocessed configuration:

```bash
# View preprocessed configuration
fgcli.py -t my_config.yaml pp

# View specific component
fgcli.py -t my_config.yaml pp trainer_args
```

### Common Configuration Issues

#### Template Inheritance Problems

```yaml
# Problem: trainer_args not appearing in final config
-- extends 'projects/base.yaml'
-- block trainer_args  # Gets overridden by parent
    save_steps: 100
-- endblock trainer_args

# Solution: Use specific trainer configuration
-- block trainer_definition
    -- include 'my_specific_trainer'
-- endblock trainer_definition

#-------------------- my_specific_trainer --------------------
-- extends 'trainers/trainer.yaml'
-- block trainer_args
    == super()
    save_steps: 100
-- endblock trainer_args
```

#### Missing Dependencies

```yaml
# Problem: Undefined variable reference
-- block lr_scheduler
    num_training_steps: !var "max_steps"  # max_steps not defined
-- endblock lr_scheduler

# Solution: Define or calculate variable
-- set max_steps = 1000
# OR
-- block trainer_args
    max_steps: 1000
-- endblock trainer_args
```

## Best Practices

### 1. Use Template Inheritance

Create reusable base configurations and extend them:

```yaml
# templates/base/training_base.yaml
-- block trainer_args
    logging_steps: 100
    eval_strategy: "steps"
    eval_steps: 500
-- endblock trainer_args

# Your specific config
-- extends 'base/training_base.yaml'
-- block trainer_args
    == super()
    output_dir: "./my_output"
-- endblock trainer_args
```

### 2. Separate Concerns

Keep trainer configuration separate from model/data configuration:

```yaml
-- extends 'projects/my_model.yaml'

-- block trainer_definition
    -- include 'training_config'
-- endblock trainer_definition

-- block data_definition
    -- include 'data_config'
-- endblock data_definition
```

### 3. Use Environment Variables

```yaml
-- set output_dir = os.environ.get("TRAINING_OUTPUT_DIR", "./default_output")
-- set batch_size = int(os.environ.get("BATCH_SIZE", "16"))

-- block trainer_args
    output_dir: !var "output_dir"
    per_device_train_batch_size: !var "batch_size"
-- endblock trainer_args
```

### 4. Document Configuration

```yaml
# Configuration: Multi-GPU Training with Checkpointing
# Purpose: Train large model with automatic checkpointing
# Usage: fgcli.py train.yaml

-- extends 'trainers/accel_trainer.yaml'

-- block trainer_args
    == super()
    
    # Training schedule - 5 epochs with early evaluation
    num_train_epochs: 5
    eval_steps: 1000
    
    # Aggressive checkpointing for long training runs
    save_steps: 500
    save_optimizer_state: true
    save_scheduler_state: true
    save_total_limit: 5
-- endblock trainer_args
```

## Configuration Examples

See the `examples/` directory for complete configuration examples:

- `examples/tiny_experiments/checkpointing/` - Checkpoint functionality demo
- `examples/trainers/simple/` - Basic single-GPU training
- `examples/trainers/distributed/` - Multi-GPU training with Accelerate
- `examples/trainers/pipeline/` - Pipeline parallel training