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

Each worker needs a unique `--worker-id`. The project template appends that
id to the output directory (`ns.output_dir = <main>_<worker_id>`) so each
worker lands in its own output directory — no race on checkpoints / logs /
generation samples.

Manual dataset sharding (`--num-shards` / `--shard-index`) has been removed.
For a single-node smoke run like this one, all workers iterate the full
dataset and average pseudo-gradients via DiLoCo; for a multi-node run with
distinct dataset slices per worker, point all workers at a shared
`forgather dataset_server` and the DiLoCo server's work-unit dispatch will
hand out non-overlapping row ranges automatically.

**Option A: Using `forgather diloco worker` CLI** (recommended):

`-d N` maps to `CUDA_VISIBLE_DEVICES=N` and controls which GPU the worker sees.

```bash
# Worker A
forgather diloco worker \
    --server localhost:8512 \
    --sync-every 500 \
    --worker-id w0 \
    -t default.yaml \
    train -d 0

# Worker B
forgather diloco worker \
    --server localhost:8512 \
    --sync-every 500 \
    --worker-id w1 \
    -t default.yaml \
    train -d 1
```

If you only have a single GPU, you can run both workers on the same GPU by
setting `-d 0` on both. It won't train any faster than a single-GPU run, but
it lets you smoke-test the DiLoCo flow.

**Option B: Using env vars** (configuration-level control):

```bash
DILOCO_SERVER=localhost:8512 DILOCO_SYNC_EVERY=500 DILOCO_WORKER_ID=w0 \
forgather -t default.yaml train
```

**Option C: Standalone** (no server, callback isn't constructed):

```bash
forgather -t default.yaml train
```

### 4. Monitor

```bash
watch -n 1 forgather diloco status --server localhost:8512
```

Or open the forgather webui's **DiLoCo** view, which lists known servers
on the left, polls `/status` for the selected one, and renders the
workers table + outer-optimizer metrics + work-unit dispatch heatmap
inline.

### 5. Stopping

Workers stop automatically when training completes (reaching `max_steps`). They
deregister from the server on exit, so the server updates its worker count.

The server runs until explicitly stopped. There are three ways to stop it:

- **Ctrl-C** in the server terminal.
- **Webui**: forgather webui → **DiLoCo** view → select the server →
  **Control** card → **Shutdown server**.
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
| `baseline.yaml` | Non-DiLoCo baseline (same tinyv2 hyperparameters, for comparison) |
| `default.yaml` | Basic DiLoCo training with standard full-model sync |
| `streaming.yaml` | DiLoCo with 4-fragment streaming for overlapped communication |

## Dynamic Arguments

| Argument | Description |
|----------|-------------|
| `--diloco-server HOST:PORT` | DiLoCo server address |
| `--diloco-sync-every N` | Local optimizer steps between syncs |
| `--diloco-worker-id ID` | Unique worker identity (also drives the output-dir suffix) |
| `--diloco-bf16-comm` | Cast pseudo-gradients to bf16 before sending (default on) |
| `--diloco-dylu` | Enable Dynamic Local Updates |
| `--diloco-heartbeat-interval SECS` | Seconds between heartbeats |
| `--diloco-fragments N` | Number of streaming fragments |

## How It Works

The `DiLoCoCallback` bridges the DiLoCo worker system with Forgather's trainer:

1. **on_train_begin**: Verifies the configured DiLoCo server is reachable
   (`/status` round-trip), then creates and starts a `DiLoCoWorker` that hooks
   into the optimizer. Every `sync_every` steps, pseudo-gradients are sent to
   the server. If the server is unset or unreachable, training aborts here
   instead of silently running as a no-op.
2. **on_log**: Injects DiLoCo metrics (sync_count, sync_time, bandwidth) into
   the training logs.
3. **on_train_end**: Stops the worker and deregisters from the server.
4. **Checkpointing**: The callback implements `Stateful`, so sync progress is
   automatically saved and restored by the checkpoint manager.

When no server is configured (`DILOCO_SERVER` unset and `--diloco-server`
not passed), the template gates the callback include off entirely — the same
config behaves as a vanilla tinyv2 single-node training run.

The DiLoCo-specific YAML lives in `templatelib/examples/mixins/diloco.yaml`
and is composed into this project's `templates/project.yaml` via `{% from %}`
macro imports. See that mixin file for the full list of injected fragments
(callback singleton, dynamic args, eval-bypass kwargs).

## Dataset partitioning

Two layers do different things here, and the difference matters in
this smoke-run setup:

- **Eval / test loads** run the full dataset on every worker so eval
  loss values are comparable across the cohort. The DiLoCo mixin
  injects `diloco_work_dispatch: False` into the eval dataset
  project's `load_dataset_args` (via `[eval_dataset_project_pp_args]`
  in `templates/project.yaml`).
- **Train load partitioning** is server-driven via the DiLoCo
  server's work-unit dispatch when the worker routes through a
  `forgather dataset_server` (`FORGATHER_DATASET_SERVER` env). The
  default `run_diloco.sh` setup in this example does NOT spin up a
  dataset_server, so each worker iterates the full train stream and
  trains on identical rows; pseudo-gradient averaging still works
  but there's no data-parallel speedup. To exercise actual row
  partitioning, spawn a `forgather dataset_server` for the same
  dataset and set `FORGATHER_DATASET_SERVER` in each worker's env;
  the wrap then registers a `(dataset_id, shuffle_seed)` queue on
  the DiLoCo server and workers pull non-overlapping unit ranges.

The legacy `--num-shards` / `--shard-index` flow is gone — the
project template never declared those dynamic args, and the
worker CLI ignores them.

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

Each worker passes a unique `--worker-id` (e.g. `w0`, `w1`) — the
project template's `[globals]` block appends it to `ns.output_dir` so
each worker lands in its own output directory under `output_models/`,
preventing checkpoint and log races.

| Scenario | Output directory |
|----------|----------------|
| Standalone (no DiLoCo) | `output_models/tinyv2/` |
| Worker `--worker-id w0` | `output_models/tinyv2_w0/` |
| Worker `--worker-id w1` | `output_models/tinyv2_w1/` |

When launched via the webui, the scheduler sets `DILOCO_WORKER_ID` from
the job's `queue_id` automatically, so the suffix is unique by
construction even without an explicit `--diloco-worker-id`.

## Notes

- When using `forgather diloco worker`, DiLoCo parameters (sync_every, bf16, etc.)
  are passed via environment variables. All DiLoCo callback parameters default to
  `null` in the config, so env var values take effect automatically.
- The server validates each worker's model against its own at
  `/register` time — both the param-name set and the per-param shapes
  must match. A mismatched worker is rejected with a 422 + a
  diagnostic naming the divergent param, so an operator pointing a
  worker at the wrong `--model-id-or-path` finds out before any
  training rounds happen.
