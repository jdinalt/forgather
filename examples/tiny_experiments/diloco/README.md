# DiLoCo Distributed Training Example

This project demonstrates DiLoCo (Distributed Local-SGD) integration with the
Forgather trainer via `DiLoCoCallback`. It uses a tiny transformer model on
Tiny Stories for quick iteration.

## Quick Start

### 1. Start the Server

On any reachable machine (GPU not required):

```bash
forgather diloco server \
    -m output_models/diloco/  \
    -n 2 \
    --port 8512
```

Or use a pretrained model:

```bash
forgather diloco server \
    -m path/to/model \
    -n 2 \
    --port 8512
```

### 2. Start Workers

**Option A: Using `forgather diloco worker` CLI** (sets env vars automatically):

```bash
# Machine A
forgather diloco worker \
    --server 192.168.1.100:8512 \
    --sync-every 500 \
    -p examples/tiny_experiments/diloco \
    -t default.yaml \
    train

# Machine B
forgather diloco worker \
    --server 192.168.1.100:8512 \
    --sync-every 500 \
    -p examples/tiny_experiments/diloco \
    -t default.yaml \
    train
```

**Option B: Using dynamic args** (configuration-level control):

```bash
forgather -p examples/tiny_experiments/diloco -t default.yaml \
    train --diloco-server 192.168.1.100:8512 --diloco-sync-every 500
```

**Option C: Standalone** (no server, callback is a no-op):

```bash
forgather -p examples/tiny_experiments/diloco -t default.yaml train
```

### 3. Monitor

```bash
forgather diloco status --server 192.168.1.100:8512
```

## Configuration Files

| Config | Description |
|--------|-------------|
| `default.yaml` | Basic DiLoCo training with standard full-model sync |
| `streaming.yaml` | DiLoCo with 4-fragment streaming for overlapped communication |

## How It Works

The `DiLoCoCallback` bridges the DiLoCo worker system with Forgather's trainer:

1. **on_train_begin**: Creates and starts a `DiLoCoWorker` that hooks into the
   optimizer. Every `sync_every` steps, pseudo-gradients are sent to the server.
2. **on_log**: Injects DiLoCo metrics (sync_count, sync_time, bandwidth) into
   the training logs.
3. **on_train_end**: Stops the worker and deregisters from the server.
4. **Checkpointing**: The callback implements `Stateful`, so sync progress is
   automatically saved and restored by the checkpoint manager.

When `server_addr` is empty (no `--diloco-server` and no `DILOCO_SERVER` env var),
the callback does nothing, allowing the same configuration for standalone training.

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
