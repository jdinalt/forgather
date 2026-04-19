# DiLoCo Distributed Training Example

This project demonstrates DiLoCo (Distributed Local-SGD) integration with the
Forgather trainer via `DiLoCoCallback`. It uses a tiny 4M parameter transformer
model on TinyStories for quick iteration.

All commands below assume you are in the project directory:

```bash
cd examples/tiny_experiments/diloco
```

## Quick Start

For a fully automated run, use the launch script:

```bash
./run_diloco.sh                    # 2 workers, default settings
./run_diloco.sh -n 4 -s 100       # 4 workers, sync every 100 steps
```

The script handles model construction, server startup, worker launch, and
cleanup on Ctrl-C. See below for the manual step-by-step process.

### 1. Construct the Model (First Time Only)

The DiLoCo server needs a model with saved weights. Build and save weights using
the model project (not this training project):

```bash
# To start fresh, delete any existing models
rm -rf output_models

# Create a freshly initialized model instance
forgather -p ../../models/causal_lm -t 4M.yaml \
    model --device cpu --save-checkpoint --safetensors \
    --output-dir output_models/default_model \
    construct
```

### 2. Start the Server

On any reachable machine (GPU not required):

```bash
forgather diloco server \
    -o output_models/default_model \
    -n 2 \
    --port 8512
```

The server binds to localhost by default. For remote workers, use SSH port
forwarding or `--host 0.0.0.0` (see `docs/trainers/diloco.md` for details).

### 3. Start Workers

Each worker needs a unique dataset shard. With 2 workers, use `--num-shards 2`
and assign each worker a different `--shard-index`.

**Option A: Using `forgather diloco worker` CLI** (recommended):

Note that the `-d N` arguments are equivalent to `CUDA_VISIBLE_DEVICES=N`, which controls
which GPUs are avaialble to each worker.

```bash
# Worker A (shard 0)
forgather diloco worker \
    --server localhost:8512 \
    --sync-every 500 \
    -t default.yaml \
    train --num-shards 2 --shard-index 0 -d 0

# Worker B (shard 1)
forgather diloco worker \
    --server localhost:8512 \
    --sync-every 500 \
    -t default.yaml \
    train --num-shards 2 --shard-index 1 -d 1
```

Note: If you only have a single GPU, you can run both workers on the same GPU by setting `-d 0` on both. It will not train any faster than with a single GPU, but it at least allows for testing.

**Option B: Using dynamic args** (configuration-level control):

```bash
DILOCO_SERVER=localhost:8512 DILOCO_SYNC_EVERY=500 \
forgather -t default.yaml \
    train --num-shards 2 --shard-index 0
```

**Option C: Standalone** (no server, callback is a no-op):

```bash
forgather -t default.yaml train
```

### 4. Monitor

```bash
watch -n 1 forgather diloco status --server localhost:8512
```

Or visit `http://localhost:8512/dashboard` in a browser.

### 5. Stopping

Workers stop automatically when training completes (reaching `max_steps`). They
deregister from the server on exit, so the server updates its worker count.

The server runs until explicitly stopped. There are three ways to stop it:

- **Ctrl-C** in the server terminal.
- **Dashboard**: Click the Shutdown button at `http://localhost:8512/dashboard`.
- **HTTP API**: `curl -X POST http://localhost:8512/control/shutdown`

To save server state on demand (without stopping):

```bash
curl -X POST http://localhost:8512/control/save_state
```

If you used the launch script (`run_diloco.sh`), Ctrl-C stops all processes
(server and workers) together.

## Test the Model

For a quick inference test:

```bash
# Link the model base model weights to the weights in the latest checkpoint
# Note that `-f` will force-overwrite the original initialization weights.
forgather checkpoint link -f

# Test model inference with tiny-stories prompts.
../../snippets/prompt_test.py output_models/default_model ../../../prompts/tiny_stories.yaml
```

## Configuration Files

| Config | Description |
|--------|-------------|
| `default.yaml` | Basic DiLoCo training with standard full-model sync |
| `streaming.yaml` | DiLoCo with 4-fragment streaming for overlapped communication |

## Dynamic Arguments

| Argument | Description |
|----------|-------------|
| `--num-shards N` | Number of dataset shards (set to number of workers) |
| `--shard-index I` | Dataset shard index for this worker (0-based) |
| `--diloco-server HOST:PORT` | DiLoCo server address |
| `--diloco-sync-every N` | Local optimizer steps between syncs |
| `--diloco-worker-id ID` | Unique worker ID |
| `--diloco-no-bf16` | Disable bfloat16 pseudo-gradient compression |
| `--diloco-dylu` | Enable Dynamic Local Updates |
| `--diloco-heartbeat SECS` | Seconds between heartbeats |
| `--diloco-fragments N` | Number of streaming fragments |

## How It Works

The `DiLoCoCallback` bridges the DiLoCo worker system with Forgather's trainer:

1. **on_train_begin**: Creates and starts a `DiLoCoWorker` that hooks into the
   optimizer. Every `sync_every` steps, pseudo-gradients are sent to the server.
2. **on_log**: Injects DiLoCo metrics (sync_count, sync_time, bandwidth) into
   the training logs.
3. **on_train_end**: Stops the worker and deregisters from the server.
4. **Checkpointing**: The callback implements `Stateful`, so sync progress is
   automatically saved and restored by the checkpoint manager.

When no server address is configured (no `--diloco-server`, no `DILOCO_SERVER`
env var), the callback does nothing, allowing the same configuration for
standalone training.

## Dataset Sharding

Training data is sharded across workers so each sees a unique subset. Evaluation
data is **not** sharded -- each worker evaluates on the full validation set,
producing comparable eval loss values across workers.

This is configured in `project.yaml` via the `shard_eval: False` pp_kwarg
passed to the dataset sub-project.

## Streaming Mode

The `streaming.yaml` config splits the model into 4 fragments. Each fragment
syncs at staggered intervals in a background thread while training continues,
hiding communication latency behind computation:

```
sync_every=500, num_fragments=4 -> fragment interval = 125 steps

Step 125:  Submit fragment 0 in background
Step 250:  Apply fragment 0, submit fragment 1
Step 375:  Apply fragment 1, submit fragment 2
Step 500:  Apply fragment 2, submit fragment 3, reset
```

## Output Directories

When `--shard-index` is provided, the model name is automatically suffixed
with the shard index (e.g., `default_model_shard0`, `default_model_shard1`).
This gives each worker its own output directory under `output_models/`,
preventing checkpoint race conditions.

| Scenario | Output directory |
|----------|----------------|
| Standalone | `output_models/default_model/` |
| Worker shard 0 | `output_models/default_model_shard0/` |
| Worker shard 1 | `output_models/default_model_shard1/` |

## Notes

- When using `forgather diloco worker`, DiLoCo parameters (sync_every, bf16, etc.)
  are passed via environment variables. All DiLoCo callback parameters default to
  `null` in the config, so env var values take effect automatically.
- The server must use the same model architecture as the training project. If the
  worker uses a different model, the server will return a clear error identifying
  the parameter name mismatch.
