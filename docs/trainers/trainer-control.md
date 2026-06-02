# Trainer Control

Forgather supports external control of running training jobs. From a separate
terminal (or script), you can query job status, trigger checkpoint saves, and
gracefully stop training -- without touching the process that is training.

This is useful for:

- Saving a checkpoint at an interesting point during training (e.g., a loss plateau)
- Gracefully stopping training early when results look good (or bad)
- Aborting failed hyperparameter experiments without saving
- Scripting control decisions based on training metrics

## Quick start

**1. Enable control in your training job** by adding `TrainerControlCallback`:

```python
from forgather.ml.trainer.callbacks import TrainerControlCallback

callbacks = [
    TrainerControlCallback(
        job_id="my_experiment",   # Optional: auto-generated if not provided
    ),
]
```

Or in a configuration template:

```yaml
[callback_list]
    == super()
    trainer_control: !singleton:forgather.ml.trainer.callbacks:TrainerControlCallback
```

**2. Start training** with the control callback enabled:

```bash
forgather -t config.yaml train
```

A plain foreground `forgather train` is controllable via `forgather job`
as long as the forgather server is running (typically on the same host):
the server discovers the trainer's control endpoint and relays commands to
it. `forgather job` needs a running server -- that is its one requirement.
Use `--schedule` (or `forgather submit`) only when you additionally want
the scheduler to queue and manage the run; it is not required for control.

**3. Control from another terminal** via `forgather job` (which proxies
through the forgather server):

```bash
forgather job list                    # Find running jobs
forgather job status JOB_ID          # Check training progress
forgather job save JOB_ID            # Save a checkpoint now
forgather job stop JOB_ID            # Gracefully stop after current step
forgather job save-stop JOB_ID       # Save checkpoint, then stop
forgather job abort JOB_ID           # Stop immediately without saving
```

## CLI reference

| Command | Description |
|---------|-------------|
| `forgather job list` | List queued and active jobs (shows status, type, priority, GPUs, project/config) |
| `forgather job status JOB_ID` | Show current step, epoch, max_steps, and latest logged metrics |
| `forgather job save JOB_ID` | Trigger a checkpoint save (runs evaluation first if configured) |
| `forgather job stop JOB_ID` | Graceful stop -- training finishes the current step, then exits |
| `forgather job save-stop JOB_ID` | Save a checkpoint, then gracefully stop |
| `forgather job abort JOB_ID` | Abort immediately without saving (prompts for confirmation) |
| `forgather job cleanup [JOB_ID]` | Remove terminal job records (all, or a specific job) |

## How it works

### Architecture

The control system has two sides:

- **Server side** (`TrainerControlCallback`): An HTTP server running in a background
  thread on rank 0 of the training process. It accepts commands and queues them for
  the training loop to process.
- **Client side** (`forgather job` CLI): Sends requests to the forgather server,
  which proxies them to the trainer's control endpoint.

### Job discovery

When `TrainerControlCallback` starts, rank 0 writes an endpoint file to
`~/.config/forgather/jobs/<job_id>/endpoint.json` containing the host, port, and PID.
The forgather server scans this directory to find running jobs and checks whether
each process is still alive; `forgather job list` surfaces them.

A trainer the server did not itself launch (e.g. a foreground `forgather train`)
is **promoted** to a first-class job record when discovered, so it shows up in
`forgather job` with normal status/lifecycle (marked as externally launched; it
reserves no scheduler GPUs). It is reaped by PID liveness like a re-attached job.

When training ends (or is stopped), the endpoint file is automatically cleaned up.
If a job crashes without cleanup, the endpoint directory is left behind; the server
treats it as dead via a PID-liveness check (so it stops appearing in
`forgather job list`) and a periodic GC sweep removes the stale directory once it
is older than the TTL (`FORGATHER_ORPHAN_JOB_DIR_TTL_SECONDS`, default 1h).

### Distributed coordination

Only rank 0 runs the HTTP server. When a command arrives, it is broadcast to all
ranks via `torch.distributed.broadcast` at the next log step. All ranks then apply
the command to the `TrainerControl` state simultaneously.

Commands are checked on each `on_log` callback event (controlled by the
`logging_steps` training argument). There is a latency of up to `logging_steps`
training steps between sending a command and it taking effect.

### Command behavior

| Command | Effect |
|---------|--------|
| `graceful_stop` | Training finishes the current step and exits. If `save_strategy` is not `"no"`, a final checkpoint is saved automatically on exit. |
| `save_checkpoint` | Triggers a checkpoint save regardless of `save_strategy`. If `load_best_model_at_end` is configured, also triggers evaluation. |
| `save_and_stop` | Forces a checkpoint save and then stops. See note below. |
| `abort` | Stops training immediately without saving. |

**`stop` vs `save-stop`:** When `save_strategy="steps"` or `"epoch"`, `stop` already
saves a final checkpoint on exit (this is the trainer's normal exit behavior), so
`stop` and `save-stop` produce the same result. The difference matters when
`save_strategy="no"`: `stop` exits without saving, while `save-stop` forces a save
before exiting. This is useful when you have disabled periodic checkpointing but
want to keep the option of saving on demand.

## TrainerControlCallback parameters

```python
TrainerControlCallback(
    job_id: str = None,         # Auto-generated if not provided
    port: int = None,           # Auto-selected starting from 8900
    enable_http: bool = None,   # Auto-detected based on aiohttp availability
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `job_id` | Auto-generated | Unique identifier for this job. Format: `job_{timestamp}_{hostname}_{pid}` |
| `port` | Auto-selected | HTTP server port. Scans from 8900 upward for an available port |
| `enable_http` | Auto-detected | Enabled if `aiohttp` is installed; falls back to file-based control otherwise |

## Programmatic API

The control system can also be used from Python:

```python
from forgather.trainer_control import list_jobs, get_job_status, graceful_stop, save_checkpoint

# List running jobs
jobs = list_jobs()
for job in jobs:
    print(f"{job.job_id} on {job.host}:{job.port}")

# Check status
status = get_job_status("my_experiment")
print(f"Step {status['global_step']} / {status['max_steps']}")

# Send commands
save_checkpoint("my_experiment")
graceful_stop("my_experiment")
```

## Example

See [`examples/trainer_control/trainer_control_demo.py`](../examples/trainer_control/trainer_control_demo.py)
for a complete working example showing how to set up and use the control system.
