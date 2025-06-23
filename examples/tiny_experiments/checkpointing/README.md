# Checkpoint Testing Project

This project demonstrates the new optimizer and learning rate scheduler checkpoint functionality in Forgather.

## Features

- **Optimizer State Checkpointing**: Save and restore optimizer state including momentum and other parameter-specific state
- **LR Scheduler State Checkpointing**: Maintain learning rate schedules across training interruptions  
- **Automatic Checkpoint Discovery**: Find the most recent checkpoint by modification time
- **Robust Resume Logic**: Handle multiple training sessions with out-of-order checkpoint names

## Project Structure

```
checkpointing/
├── README.md
├── meta.yaml                    # Project metadata
├── project_index.ipynb          # Interactive demo and documentation
└── templates/
    ├── project.yaml            # Base project configuration
    └── configs/
        ├── train.yaml          # Initial training with checkpointing
        └── resume.yaml         # Resume from latest checkpoint
```

## Quick Start

### 1. Initial Training
Train a model with checkpointing enabled:

```bash
cd examples/tiny_experiments/checkpointing
python ../../../bin/fgcli.py -t train.yaml train -d 0
```

This should:
- Train for 500 steps (currently using full epoch due to configuration inheritance)
- Save checkpoints every 100 steps
- Save optimizer and scheduler state with each checkpoint

### 2. Resume Training  
Continue from the latest checkpoint:

```bash
python ../../../bin/fgcli.py -t resume.yaml train -d 0
```

This should:
- Automatically find the latest checkpoint
- Restore model, optimizer, and scheduler state
- Continue training from step 500 to 800

## Current Status

The checkpoint functionality has been successfully implemented and tested with unit tests, but the project configuration needs refinement to properly override the inherited training settings. The core checkpoint features work as demonstrated by the comprehensive test suite.

## Configuration Options

The checkpoint functionality is controlled by these training arguments:

| Option | Description | Default |
|--------|-------------|---------|
| `save_optimizer_state` | Save optimizer state in checkpoints | `true` |
| `save_scheduler_state` | Save LR scheduler state in checkpoints | `true` |
| `restore_optimizer_state` | Restore optimizer state when resuming | `true` |
| `restore_scheduler_state` | Restore scheduler state when resuming | `true` |
| `resume_from_checkpoint` | `true` (auto-find) or path string | `false` |
| `save_total_limit` | Maximum checkpoints to keep | `3` |

## Implementation Details

### Checkpoint Discovery
- Uses file modification time rather than step numbers
- Robust across multiple training sessions
- Validates checkpoints before selection

### State Management  
- Optimizer state includes momentum, variance estimates, etc.
- Scheduler state preserves step counts and learning rate schedules
- Graceful handling of missing state files

### Multi-Trainer Support
- Works with `Trainer`, `AccelTrainer`, and `PipelineTrainer`
- Proper synchronization in distributed training
- Trainer-specific optimizations for state handling

## Testing

The project includes comprehensive unit tests in `tests/unit/ml/test_checkpoints.py` covering:
- Checkpoint validation and discovery
- State saving and loading
- Configuration option handling
- Integration with trainer workflow

Run tests with:
```bash
pytest tests/unit/ml/test_checkpoints.py -v
```