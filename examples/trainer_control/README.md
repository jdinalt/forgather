# Trainer Control Examples

This directory contains examples of using the TrainerControlCallback system for external control of running training jobs.

## Files

- `trainer_control_demo.py` - Complete demo showing how to add TrainerControlCallback to your training job and control it from another terminal

## Usage

### Running the Demo

```bash
# Terminal 1: Start training with control enabled
cd examples/trainer_control
python trainer_control_demo.py

# Terminal 2: Control the training job
forgather control list                    # Find your job
forgather control status JOB_ID          # Check status  
forgather control save JOB_ID            # Save checkpoint
forgather control stop JOB_ID            # Gracefully stop
forgather control abort JOB_ID           # Abort without saving
```

The demo creates a simple transformer model and trains it on synthetic data, demonstrating all aspects of the trainer control system.

## Key Features Demonstrated

1. **Adding TrainerControlCallback** to enable external control
2. **HTTP-based communication** for distributed training support
3. **Service discovery** via filesystem endpoint files
4. **Command execution** during training without interruption
5. **Graceful shutdown** with proper cleanup
6. **Checkpoint management** on demand

## Integration with Your Code

To add trainer control to your own training script:

```python
from forgather.ml.trainer.callbacks import TrainerControlCallback

# Add to your trainer callbacks
callbacks = [
    TrainerControlCallback(
        job_id="my_training_job",  # Optional: auto-generated if not provided
        # port=None,               # Optional: auto-select port
        # enable_http=True         # Optional: enable HTTP control (default)
    ),
    # ... your other callbacks
]

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    callbacks=callbacks
)
```

Then control from another terminal using the `forgather control` commands.