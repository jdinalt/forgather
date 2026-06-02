# Trainer Control Examples

This directory contains examples of using the TrainerControlCallback system for external control of running training jobs.

## Files

- `trainer_control_demo.py` - Demo showing how to add `TrainerControlCallback` to a training job so it exposes an external control endpoint.

## Usage

### Running the demo

```bash
cd examples/trainer_control
python trainer_control_demo.py
```

The demo trains a small transformer on synthetic data with
`TrainerControlCallback` enabled, which exposes a control endpoint
(`~/.config/forgather/jobs/<job_id>/endpoint.json`).

### Controlling a training job

The callback is what makes a job controllable; how you reach it depends on how
the job was launched:

- **Server-managed jobs (recommended).** Run training through the forgather
  server's scheduler — `forgather submit`, `forgather train --schedule`, or the
  webui — and control it with `forgather job`:

  ```bash
  forgather job list                 # find the queue_id
  forgather job status <queue_id>    # check status
  forgather job save <queue_id>      # checkpoint
  forgather job stop <queue_id>      # graceful stop
  forgather job abort <queue_id>     # abort without saving
  ```

- **Locally-launched jobs** (like this standalone demo) aren't registered with
  the forgather server, so `forgather job` won't see them. Drive their control
  endpoint programmatically with the `forgather.trainer_control` client.

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

Then control it from another terminal: run the job through the scheduler
(`forgather submit`) and use `forgather job`, or drive the local control
endpoint directly via the `forgather.trainer_control` client.