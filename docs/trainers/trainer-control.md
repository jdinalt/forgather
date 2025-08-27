# Trainer Control System

The Trainer Control system enables external control of running training jobs through HTTP commands, allowing you to interact with training processes from other terminals or scripts without interrupting the training loop.

## Overview

The system provides:
- **External control** of running distributed training jobs
- **HTTP-based communication** for multi-node support  
- **Service discovery** via filesystem endpoint files
- **Non-blocking command processing** during training
- **Graceful shutdown** with proper cleanup
- **On-demand checkpointing** and evaluation

## Quick Start

### 1. Enable Control in Your Training Script

```python
from forgather.ml.trainer.callbacks import TrainerControlCallback

# Add to your trainer callbacks
callbacks = [
    TrainerControlCallback(
        job_id="my_training_job",  # Optional: auto-generated if not provided
    ),
    # ... your other callbacks
]

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    callbacks=callbacks
)

# Start training - control interface will be automatically available
trainer.train()
```

### 2. Control from Another Terminal

```bash
# List all discoverable training jobs
forgather control list

# Get job status
forgather control status JOB_ID

# Save checkpoint (with evaluation if configured)  
forgather control save JOB_ID

# Gracefully stop training
forgather control stop JOB_ID

# Save checkpoint then gracefully stop
forgather control save-stop JOB_ID

# Abort training without saving (for hyperparameter experiments)
forgather control abort JOB_ID
```

## Architecture

### Communication Protocol

The system uses a two-tier communication protocol:

1. **HTTP Server (Rank 0 only)**: Receives external commands
2. **Distributed Broadcast**: Coordinates commands across all ranks using `torch.distributed`

```
External Client → HTTP (Rank 0) → torch.distributed.broadcast → All Ranks
                     ↓
                Service Discovery Files (~/.forgather/jobs/JOB_ID/)
```

### Service Discovery

Training jobs register themselves by creating endpoint files:

```
~/.forgather/jobs/JOB_ID/
├── endpoint.json      # Host, port, PID, start time
└── status.json        # Current training status (optional)
```

This enables automatic job discovery without requiring manual configuration.

### Distributed Coordination

- **Rank 0**: Runs HTTP server, receives commands, broadcasts to other ranks
- **All Ranks**: Listen for broadcast commands during logging events  
- **Device-Aware**: Automatically handles CPU/GPU tensor placement for broadcasts
- **Non-Blocking**: Commands are processed during existing `on_log` callback events

## API Reference

### TrainerControlCallback

```python
class TrainerControlCallback(TrainerCallback):
    def __init__(
        self, 
        job_id: str = None,           # Auto-generated if not provided
        port: int = None,             # Auto-selected if not provided  
        host: str = "localhost",      # Server bind address
        enable_http: bool = True,     # Enable HTTP control interface
        cleanup_on_exit: bool = True  # Clean up endpoint files on exit
    ):
```

**Parameters:**
- `job_id`: Unique identifier for the training job. Auto-generated from hostname and timestamp if not provided.
- `port`: HTTP server port. Auto-selected from available ports if not provided.
- `host`: Server bind address for HTTP interface.
- `enable_http`: Whether to enable HTTP control interface (vs file-based only).
- `cleanup_on_exit`: Whether to automatically clean up endpoint files when training completes.

### Control Commands

| Command | Description | Behavior |
|---------|-------------|----------|
| `graceful_stop` | Stop training gracefully | Sets `should_training_stop=True`, allows final checkpoint |
| `save_checkpoint` | Save model checkpoint | Triggers immediate checkpoint save with evaluation |
| `save_and_stop` | Save checkpoint then stop | Combines save + graceful stop |  
| `abort` | Stop without saving | Sets `should_abort_without_save=True`, skips final checkpoint |

### CLI Interface

```bash
# List and manage jobs
forgather control list [--remote HOST:PORT]  # List jobs, optionally include remote
forgather control cleanup [--force]          # Remove dead job files

# Job control  
forgather control status JOB_ID              # Get detailed status
forgather control save JOB_ID                # Save checkpoint  
forgather control stop JOB_ID                # Graceful stop
forgather control save-stop JOB_ID           # Save then stop
forgather control abort JOB_ID [--force]     # Abort without saving
```

### HTTP API

Direct HTTP access (primarily for advanced integration):

```bash
# Get job status
curl http://localhost:PORT/jobs/JOB_ID/status

# Send control command
curl -X POST http://localhost:PORT/jobs/JOB_ID/control \
     -H "Content-Type: application/json" \
     -d '{"command": "graceful_stop"}'
```

## Advanced Usage

### Custom Job IDs

Provide meaningful job identifiers for easier management:

```python
TrainerControlCallback(
    job_id=f"experiment_{model_name}_{dataset_name}_{timestamp}"
)
```

### Multi-Node Training

The system automatically handles multi-node distributed training. Only rank 0 runs the HTTP server, but all ranks participate in command execution through distributed broadcasts.

### Integration with Best Model Tracking

When using `load_best_model_at_end=True`, the save command will trigger evaluation and potentially update the best model checkpoint:

```python
training_args = TrainingArguments(
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    # ... other args
)
```

### Error Handling and Recovery

The system is designed to be robust:
- HTTP server failures don't crash training
- Malformed commands are logged but ignored
- Distributed communication errors are handled gracefully
- Endpoint cleanup occurs automatically on normal exit

## Best Practices

### Job Management

1. **Use descriptive job IDs** that include experiment details
2. **Run cleanup regularly** to remove dead job files: `forgather control cleanup`
3. **Monitor job status** before sending commands to ensure jobs are responsive

### Command Usage

1. **Use `save-stop` for normal completion** to ensure final checkpoint is saved
2. **Use `abort` for failed hyperparameter experiments** to avoid saving bad checkpoints
3. **Check job status** after commands to verify execution
4. **Use `--force` flags carefully** to avoid accidental data loss

### Distributed Training

1. **Only interact with rank 0** - other ranks automatically coordinate
2. **Ensure network connectivity** between nodes for distributed broadcasts
3. **Monitor all ranks** for successful command execution in logs

## Troubleshooting

### Common Issues

**Job not appearing in list:**
- Check if training process is still running: `ps aux | grep python`
- Verify endpoint file exists: `ls ~/.forgather/jobs/`
- Ensure TrainerControlCallback was added to trainer

**Commands not responding:**
- Check training logs for error messages
- Verify HTTP server started successfully (rank 0 logs)
- Test direct HTTP access: `curl http://localhost:PORT/jobs/JOB_ID/status`

**Distributed training issues:**
- Ensure `torch.distributed` is properly initialized
- Check that all ranks have the same device configuration
- Verify network connectivity between nodes

### Debug Logging

Enable detailed logging to diagnose issues:

```python
import logging
logging.getLogger('forgather.ml.trainer.callbacks').setLevel(logging.DEBUG)
```

## Examples

See `examples/trainer_control/` for a complete working example demonstrating all features of the trainer control system.

## Implementation Details

### Security Considerations

- HTTP server binds to localhost by default for security
- No authentication is implemented - rely on filesystem permissions
- Command validation prevents arbitrary code execution
- All communication uses structured JSON payloads

### Performance Impact

- Minimal overhead: ~1μs per logging step for command checking
- Non-blocking I/O during training loop
- HTTP server runs in separate thread on rank 0 only
- Memory usage: <1MB for endpoint management

### Compatibility

- **HuggingFace Trainer API**: Fully compatible callback interface
- **PyTorch Distributed**: Supports all DDP backends (nccl, gloo, mpi)
- **Multiple Trainers**: Tested with SimpleTrainer, AccelTrainer, PipelineTrainer
- **Checkpointing**: Integrates with existing checkpoint/restoration system