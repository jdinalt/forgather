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

The callback is what makes a job controllable. There are two ways to reach it:

- **Via `forgather job` (CLI).** As long as the forgather server is running
  (typically on the same host), it discovers the trainer's control endpoint
  (`~/.config/forgather/jobs/<id>/endpoint.json`) and relays commands to it.
  This works for any run with the callback enabled -- a plain foreground
  `forgather train`, a `forgather submit` / `forgather train --schedule` job,
  or a run launched from the webui:

  ```bash
  forgather job list                 # find the job id
  forgather job status <job_id>      # check status
  forgather job save <job_id>        # checkpoint
  forgather job stop <job_id>        # graceful stop
  forgather job abort <job_id>       # abort without saving
  ```

  `forgather job` requires a running server; `--schedule` (or
  `forgather submit`) is only needed when you also want the scheduler to queue
  and manage the run.

- **Programmatically.** Drive the control endpoint directly with the
  `forgather.trainer_control` client (no server required). This is how the
  standalone demo in this directory is intended to be controlled.

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

Then control it from another terminal with `forgather job` (when the
forgather server is running, it discovers the trainer's control endpoint),
or drive the control endpoint directly via the `forgather.trainer_control`
client.