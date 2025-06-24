# Checkpoint Management

Forgather provides comprehensive checkpoint management that goes beyond simple model weight saving to include complete training state preservation and robust resumption capabilities.

## Overview

### Key Features

- **Complete State Preservation**: Model weights, optimizer state, LR scheduler state, and training progress
- **Robust Discovery**: Modification time-based checkpoint selection for reliability
- **Flexible Resumption**: Support for auto-discovery and explicit checkpoint paths
- **Format Compatibility**: Works with PyTorch, SafeTensors, and sharded checkpoint formats
- **Configuration Control**: Fine-grained control over what gets saved and restored

### Checkpoint Structure

Each checkpoint directory contains:
```
checkpoint-{step}/
├── config.json              # Model configuration
├── generation_config.json   # Generation parameters (if applicable)
├── model.safetensors         # Model weights (SafeTensors format)
├── pytorch_model.bin         # Model weights (PyTorch format, alternative)
├── tokenizer_config.json    # Tokenizer configuration
├── special_tokens_map.json  # Special token mappings
└── training_state.pt        # Training state (optimizer, scheduler, global step)
```

## Configuration Options

### Basic Checkpoint Settings

```yaml
-- block trainer_args
    # Checkpoint saving
    save_strategy: "steps"           # "steps" | "epoch" | "no"
    save_steps: 500                  # Save every N steps
    save_total_limit: 3              # Keep only N most recent checkpoints
    
    # Training state persistence
    save_optimizer_state: true       # Save optimizer state
    save_scheduler_state: true       # Save LR scheduler state
    
    # Checkpoint resumption
    resume_from_checkpoint: false    # false | true | "/path/to/checkpoint"
    restore_optimizer_state: true    # Restore optimizer state when resuming
    restore_scheduler_state: true    # Restore scheduler state when resuming
-- endblock trainer_args
```

### Advanced Checkpoint Configuration

```yaml
-- block trainer_args
    == super()
    
    # Saving configuration
    save_strategy: "steps"
    save_steps: 100                  # Frequent saves for long training
    save_total_limit: 5              # Keep more checkpoints
    save_safetensors: true           # Use SafeTensors format
    overwrite_output_dir: true       # Allow overwriting existing models
    
    # State management
    save_optimizer_state: true
    save_scheduler_state: true
    restore_optimizer_state: true
    restore_scheduler_state: true
    
    # Resume configuration
    resume_from_checkpoint: true     # Auto-find latest checkpoint
-- endblock trainer_args
```

## Checkpoint Discovery and Validation

### Automatic Discovery

When `resume_from_checkpoint: true`, the trainer automatically:

1. **Searches checkpoint directory**: `{output_dir}/checkpoints/checkpoint-*`
2. **Validates checkpoints**: Ensures required files are present
3. **Selects by modification time**: Uses most recently modified checkpoint
4. **Logs selection**: Reports which checkpoint was selected and why

```python
# Automatic discovery
args = TrainingArguments(
    output_dir="./output_models/my_model",
    resume_from_checkpoint=True,  # Auto-discover latest
)
```

### Explicit Checkpoint Path

```python
# Explicit checkpoint path
args = TrainingArguments(
    resume_from_checkpoint="./output_models/my_model/checkpoints/checkpoint-1000"
)
```

### Checkpoint Validation

The system validates checkpoints by checking for:
- **Model files**: At least one of `model.safetensors`, `pytorch_model.bin`, or sharded variants
- **Directory structure**: Proper checkpoint directory format
- **File accessibility**: All required files are readable

Invalid checkpoints are automatically skipped during discovery.

## Training State Persistence

### What Gets Saved

**training_state.pt** contains:
```python
{
    'optimizer': optimizer.state_dict(),      # Optimizer state (momentum, etc.)
    'lr_scheduler': scheduler.state_dict(),   # Scheduler state (step count, etc.)
    'global_step': trainer.state.global_step # Current training step
}
```

### Optimizer State

Preserves optimizer-specific state including:
- **AdamW**: Exponential moving averages (beta1, beta2), step counts
- **SGD**: Momentum buffers
- **Custom optimizers**: All optimizer-specific state variables

### LR Scheduler State

Preserves scheduler state including:
- **Step counts**: Current scheduler step
- **Internal state**: Scheduler-specific variables
- **Learning rate history**: For schedulers that track history

### Global Step Restoration

Ensures training continues from the exact step where it was interrupted:
- **Step counter**: Restored from checkpoint
- **Progress tracking**: Proper epoch and step calculations
- **Logging alignment**: Metrics continue from correct step

## Complete Checkpoint Workflow

### Initial Training with Checkpointing

```yaml
# templates/configs/train.yaml
-- extends 'project.yaml'

-- block trainer_definition
    -- include 'checkpoint_trainer'
-- endblock trainer_definition

#-------------------- checkpoint_trainer --------------------
-- extends 'trainers/trainer.yaml'

-- block trainer_args
    == super()
    
    # Enable checkpointing
    save_strategy: "steps"
    save_steps: 500
    save_total_limit: 3
    
    # Save complete training state
    save_optimizer_state: true
    save_scheduler_state: true
    
    # Training configuration
    max_steps: 2000
    output_dir: "./output_models/my_model"
-- endblock trainer_args
```

### Resume Training Configuration

```yaml
# templates/configs/resume.yaml
-- extends 'project.yaml'

-- block trainer_definition
    -- include 'resume_trainer'
-- endblock trainer_definition

#-------------------- resume_trainer --------------------
-- extends 'trainers/trainer.yaml'

-- block trainer_args
    == super()
    
    # Resume from checkpoint
    resume_from_checkpoint: true
    
    # Restore complete training state
    restore_optimizer_state: true
    restore_scheduler_state: true
    
    # Continue training to higher step count
    max_steps: 4000
    output_dir: "./output_models/my_model"
-- endblock trainer_args
```

### Execution

```bash
# Initial training
RANK=0 python scripts/train_script.py train.yaml -p .

# Resume training (automatic checkpoint discovery)
RANK=0 python scripts/train_script.py resume.yaml -p .

# Resume from specific checkpoint (configure path in resume.yaml)
RANK=0 python scripts/train_script.py resume.yaml -p .
```

## Checkpoint Management in Code

### Manual Checkpoint Saving

```python
# During training loop
if step % args.save_steps == 0:
    checkpoint_path = trainer._save_checkpoint()
    print(f"Checkpoint saved at: {checkpoint_path}")
```

### Manual Checkpoint Loading

```python
# Load specific checkpoint
checkpoint_path = "./output_models/my_model/checkpoints/checkpoint-1000"
trainer._load_model_from_checkpoint(checkpoint_path)
trainer._load_training_state(checkpoint_path)
```

### Checkpoint Discovery

```python
# Find latest checkpoint
latest_checkpoint = trainer._find_latest_checkpoint()
if latest_checkpoint:
    print(f"Latest checkpoint: {latest_checkpoint}")
else:
    print("No checkpoints found")

# Validate checkpoint
is_valid = trainer._validate_checkpoint(checkpoint_path)
print(f"Checkpoint valid: {is_valid}")
```

## Multi-Process and Distributed Training

### Accelerate Trainer Checkpoints

```python
from forgather.ml import AccelTrainer, AccelTrainingArguments

trainer = AccelTrainer(
    args=AccelTrainingArguments(
        save_strategy="steps",
        save_steps=1000,
        save_optimizer_state=True,
        save_scheduler_state=True,
        resume_from_checkpoint=True,
    ),
    # ... other args
)
```

**Features:**
- **Coordinated saving**: Only main process saves checkpoints
- **State synchronization**: All processes load same checkpoint state
- **Distributed validation**: Checkpoint validation across all processes

### Pipeline Trainer Checkpoints

```python
from forgather.ml import PipelineTrainer, PipelineTrainingArguments

trainer = PipelineTrainer(
    args=PipelineTrainingArguments(
        save_strategy="steps", 
        save_steps=500,
        # Pipeline-specific checkpointing
        save_optimizer_state=True,
        save_scheduler_state=True,
    ),
    # ... other args
)
```

**Features:**
- **Sharded checkpoints**: Model weights split across pipeline stages
- **Coordinated restoration**: All pipeline stages restore consistently
- **Stage-specific state**: Each stage's optimizer/scheduler state preserved

## Troubleshooting

### Common Issues

#### 1. Checkpoint Not Found

```
WARNING: No valid checkpoints found in checkpoint directory
```

**Solutions:**
- Verify checkpoint directory exists: `{output_dir}/checkpoints/`
- Check file permissions and accessibility
- Ensure checkpoint directories follow naming pattern: `checkpoint-{step}`

#### 2. Invalid Checkpoint Error

```
WARNING: Checkpoint appears to be incomplete (no model files found)
```

**Solutions:**
- Check if training was interrupted during checkpoint saving
- Verify disk space wasn't exhausted during saving
- Look for any of: `model.safetensors`, `pytorch_model.bin`, or sharded files

#### 3. Device Mapping Error

```
ERROR: 'int' object is not callable
```

**Solution:**
This was a historical issue that has been fixed. Update to latest version where `torch.device()` is used properly.

#### 4. Training State Loading Failed

```
ERROR: Failed to load training state from training_state.pt
```

**Solutions:**
- Check if `save_optimizer_state` and `save_scheduler_state` were enabled during saving
- Verify optimizer and scheduler are initialized before loading
- Ensure checkpoint was saved completely (not interrupted)

### Debugging Checkpoint Issues

#### Enable Debug Logging

```python
import logging
logging.getLogger('forgather.ml.base_trainer').setLevel(logging.DEBUG)
```

#### Inspect Checkpoint Contents

```python
import torch
import os

checkpoint_path = "./output_models/my_model/checkpoints/checkpoint-1000"

# List checkpoint files
print("Checkpoint files:", os.listdir(checkpoint_path))

# Examine training state
if os.path.exists(os.path.join(checkpoint_path, "training_state.pt")):
    training_state = torch.load(
        os.path.join(checkpoint_path, "training_state.pt"), 
        map_location='cpu'
    )
    print("Training state keys:", list(training_state.keys()))
    if 'global_step' in training_state:
        print("Global step:", training_state['global_step'])
```

#### Validate Checkpoint Discovery

```python
# Check checkpoint discovery logic
checkpoints_dir = "./output_models/my_model/checkpoints"
checkpoints = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
print("Found checkpoints:", checkpoints)

for cp in checkpoints:
    mtime = os.path.getmtime(cp)
    print(f"{cp}: {time.ctime(mtime)}")
```

## Best Practices

### 1. Checkpoint Frequency

Balance between:
- **Recovery granularity**: More frequent saves = less work lost
- **Storage overhead**: Each checkpoint uses significant disk space
- **Training performance**: Frequent saves can slow training

```yaml
# For long training runs (days/weeks)
save_steps: 1000
save_total_limit: 5

# For experimental runs (hours)
save_steps: 500
save_total_limit: 3

# For debugging (minutes)
save_steps: 100
save_total_limit: 2
```

### 2. Storage Management

```yaml
# Limit checkpoint storage
save_total_limit: 3        # Keep only 3 most recent
save_safetensors: true     # More efficient format

# For critical training runs
save_total_limit: 10       # Keep more checkpoints
# Consider external backup of key checkpoints
```

### 3. Resume Strategy

```yaml
# Automatic resume (recommended)
resume_from_checkpoint: true

# Explicit resume (for specific cases)
resume_from_checkpoint: "/path/to/specific/checkpoint"

# No resume (fresh start)
resume_from_checkpoint: false
```

### 4. State Management

```yaml
# Complete state preservation (recommended)
save_optimizer_state: true
save_scheduler_state: true
restore_optimizer_state: true
restore_scheduler_state: true

# Model-only checkpointing (for inference or different optimizer)
save_optimizer_state: false
save_scheduler_state: false
```

### 5. Validation

Always validate checkpoint functionality:

```python
# Test checkpoint save/load cycle
trainer.train()  # Train briefly
checkpoint_path = trainer._save_checkpoint()

# Create new trainer and restore
new_trainer = Trainer(...)
new_trainer._load_model_from_checkpoint(checkpoint_path)
new_trainer._load_training_state(checkpoint_path)
```

## Integration with External Tools

### TensorBoard Monitoring

Checkpoint events are automatically logged to TensorBoard when using `TBLogger`:

```yaml
trainer_callbacks: &trainer_callbacks !list:@trainer_callbacks
    - !singleton:forgather.ml.tb_logger:TBLogger
        args: [*summary_writer]
```

### Model Versioning

Combine with Git for complete reproducibility:

```bash
# Tag model versions
git tag -a model-v1.0 -m "Model after 10k steps"

# Save commit hash in checkpoint metadata
echo $GIT_COMMIT > ./output_models/my_model/git_commit.txt
```

### Cloud Storage Integration

```python
# Example: Upload checkpoints to cloud storage
class CloudCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = f"{args.output_dir}/checkpoints/checkpoint-{state.global_step}"
        upload_to_cloud(checkpoint_dir)
```