# DiLoCo: Distributed Local-SGD Training

DiLoCo (Distributed Local SGD with Communication) enables distributed training
across multiple heterogeneous machines on a standard LAN. Unlike DDP, which
requires high-bandwidth interconnects (NVLink, InfiniBand), DiLoCo reduces
communication by ~500x, making 1 Gig Ethernet practical for multi-machine
training.

The system supports two operating modes:
- **Synchronous**: All workers must submit before the server applies the outer
  optimizer. Simple and deterministic.
- **Asynchronous**: Workers submit independently without waiting. Supports
  heterogeneous hardware (different GPU types, different numbers of GPUs per
  machine) with Delayed Nesterov (DN) momentum and Dynamic Local Updates (DyLU).

## How It Works

Each machine runs any existing Forgather trainer (single GPU, DDP, or pipeline)
as an independent "worker." Workers train locally for H steps using their inner
optimizer (e.g., AdamW), then synchronize with a central parameter server. The
server averages the workers' updates and applies an outer optimizer (SGD with
Nesterov momentum) to produce new global parameters that all workers adopt.

```
                    +-------------------+
                    |   DiLoCo Server   |
                    | (standalone proc) |
                    |                   |
                    | - Global params   |
                    | - Outer optimizer |
                    | - Worker registry |
                    +--------+----------+
                             |
                 HTTP over 1G Ethernet
                             |
         +-------------------+-------------------+
         |                   |                   |
   +-----+-----+      +-----+-----+      +-----+-----+
   |  Worker 0  |      |  Worker 1  |      |  Worker 2  |
   | (Machine A)|      | (Machine B)|      | (Machine C)|
   |            |      |            |      |            |
   | Pipeline   |      | Single GPU |      | DDP        |
   | Trainer    |      | Trainer    |      | Trainer    |
   | (4x 3090)  |      | (1x 4090)  |      | (2x A6000) |
   +------------+      +------------+      +------------+
```

### Synchronous Protocol

In the default synchronous mode, each round follows these steps:

1. Workers train locally for `sync_every` optimizer steps (the "inner loop")
2. Each worker computes pseudo-gradients: `global_params - local_params`
3. Workers submit pseudo-gradients to the server over HTTP
4. Server waits until all workers have submitted (synchronous barrier)
5. Server averages the pseudo-gradients across all workers
6. Server applies the outer optimizer step (SGD with Nesterov momentum)
7. Updated global parameters are returned to all workers
8. Workers load the new parameters and begin the next inner loop

### Asynchronous Protocol

In async mode (`--async`), the barrier is removed. Each worker submits
pseudo-gradients and receives updated global params immediately without waiting
for other workers. This is essential for heterogeneous clusters where machines
have different training speeds.

The server applies each worker's pseudo-gradients as they arrive. To mitigate
the momentum amplification problem caused by stale gradients, the server
supports **Delayed Nesterov (DN)** momentum and **Dynamic Local Updates (DyLU)**.

See [Async Mode](#async-mode) for configuration details.

### Bandwidth Efficiency

Pseudo-gradients are optionally cast to bfloat16 before transmission, halving
bandwidth with minimal quality impact. With `sync_every=500`, a 1B parameter
model transfers ~2 GB every 500 training steps, achieving >97% compute
utilization on 1 Gig Ethernet.

| Model Size | BF16 Size | Transfer Time (1 Gbps) | H=500 steps @ 1s/step | Utilization |
|------------|-----------|------------------------|----------------------|-------------|
| 150M       | 300 MB    | 2.4s                   | 500s compute         | 99.5%       |
| 1B         | 2 GB      | 16s                    | 500s compute         | 97%         |
| 7B         | 14 GB     | 112s                   | 500s compute         | 82%         |

## Quick Start

### 1. Start the Server

The server is a standalone process that holds global model parameters. Start it
on any reachable machine (it does not need a GPU):

```bash
# Synchronous mode (default)
forgather diloco server \
    -o path/to/model \
    -n 2 \
    --port 8512

# Asynchronous mode (for heterogeneous hardware)
forgather diloco server \
    -o path/to/model \
    -n 3 \
    --async \
    --dn-buffer-size 3 \
    --dylu \
    --dylu-base-sync-every 500
```

Server arguments:
- `-o`: Path to a model/output directory
- `-n`: Number of expected workers
- `--port`: Server port (default: 8512)
- `--async`: Enable asynchronous mode
- `--dn-buffer-size N`: Delayed Nesterov buffer size (async only, default: 0 = disabled)
- `--dylu`: Enable Dynamic Local Updates (async only)
- `--dylu-base-sync-every N`: Base sync interval for the fastest worker (default: 500)
- `--from-checkpoint FROM_CHECKPOINT`: Load model from specified checkpoint path. Overrides loading from newest.

```bash
# Load a specific checkpoint and save checkpoints to specified directory.
forgather diloco server -o path/to/output --from-checkpoint output_models/my_model/checkpoint-1000 -n 2
```

### 2. Start Workers

On each machine, launch a worker that wraps the normal training command.
Each worker needs a unique `--worker-id` so its output directory doesn't
collide with the others (the project template appends the worker id to
`ns.model_name`):

```bash
# sync mode
forgather diloco worker \
    --server 192.168.1.100:8512 \
    --sync-every 500 \
    --worker-id w0 \
    -p my_project -t train.yaml \
    train -d 0

# with DyLU - server adjusts sync frequency dynamically
forgather diloco worker \
    --server 192.168.1.100:8512 \
    --sync-every 500 \
    --worker-id w1 \
    --dylu \
    --heartbeat-interval 30 \
    -p my_project -t train.yaml \
    train -d 1
```

Worker arguments:
- `--server`: Server address as `host:port`
- `--sync-every`: Local steps between syncs (default: 500)
- `--worker-id`: Unique worker identity. Drives the per-worker output-dir
  suffix the project template appends to `ns.model_name`, and the
  uniqueness key the server enforces on `/register`. Auto-generated when
  omitted but operators typically set it explicitly so logs / output dirs
  are predictable.
- `--no-bf16`: Send full-precision pseudo-gradients instead of bfloat16
- `--dylu`: Enable dynamic sync frequency adjustment from server
- `--heartbeat-interval`: Seconds between heartbeats for speed reporting (default: 30)
- `-d`: CUDA visible devices

Dataset partitioning across workers is handled by the server's **work-unit
dispatch**: each worker registers its train dataset with the DiLoCo server
on first iteration and pulls per-unit row ranges on demand, so no row is
trained on twice within an epoch. There's no operator-facing toggle —
dispatch is active whenever `DILOCO_SERVER` is set on the worker process.
See the *Work-unit dispatch* section below.

### 3. Monitor

```bash
watch -n 1 forgather diloco status --server localhost:8512
```

Shows sync round, registered workers, their hostnames, training speeds, and
pending sync submissions. In async mode, also shows total submissions, DN buffer
status, and DyLU configuration.

## Programmatic API

The DiLoCo system can also be used directly in Python, independent of the CLI.

### DiLoCoWorker

The worker is a composable wrapper that hooks into any optimizer via
`register_step_post_hook`. It works as a context manager:

```python
import torch
from forgather.ml.diloco import DiLoCoWorker

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

with DiLoCoWorker(
    model=model,
    optimizer=optimizer,
    server_addr="192.168.1.100:8512",
    sync_every=500,
    bf16_comm=True,
) as diloco:
    # Train normally - DiLoCo syncs happen automatically every 500 optimizer steps
    for batch in dataloader:
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Access sync metrics
    print(diloco.sync_metrics)
```

Key parameters:
- `model`: The model being trained
- `optimizer`: The inner optimizer (any `torch.optim.Optimizer`)
- `server_addr`: Server address as `"host:port"`
- `sync_every`: Steps between syncs (H in the DiLoCo paper)
- `bf16_comm`: Cast pseudo-gradients to bfloat16 (default: True)
- `worker_id`: Unique ID (auto-generated if None)
- `dylu`: Enable dynamic sync frequency adjustment (default: False)
- `heartbeat_interval`: Seconds between heartbeats for DyLU (default: 30)

### DiLoCoServer

```python
from forgather.ml.diloco import DiLoCoServer

# Synchronous server (default)
server = DiLoCoServer(
    "path/to/model",
    num_workers=3,
    port=8512,
    outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.7, momentum=0.9, nesterov=True),
)
server.run()

# Asynchronous server with DN momentum and DyLU
server = DiLoCoServer(
    "path/to/model",
    num_workers=3,
    port=8512,
    async_mode=True,
    dn_buffer_size=3,
    dylu_enabled=True,
    dylu_base_sync_every=500,
)
server.run()

# Or start in background
server.start()
# ... do other things ...
server.stop()
```

### DiLoCoClient

Low-level client for direct server communication:

```python
from forgather.ml.diloco import DiLoCoClient

client = DiLoCoClient("192.168.1.100:8512")

# Register and get initial params
params = client.register("my_worker", {"hostname": "machine-a"})

# Submit pseudo-gradients (blocks until all workers sync)
new_params = client.submit_pseudogradients("my_worker", pseudograds)

# Other operations
status = client.get_status()
client.heartbeat("my_worker", steps_per_second=3.5)
client.deregister("my_worker")
```

## Server Configuration

### Outer Optimizer

The default outer optimizer is SGD with Nesterov momentum (lr=0.7, momentum=0.9),
following the DiLoCo paper. You can customize it via CLI flags or the factory
function:

```bash
# CLI
forgather diloco server -o path/to/model -n 2 --outer-lr 0.5 --outer-momentum 0.95

# Or disable Nesterov
forgather diloco server -o path/to/model -n 2 --no-nesterov
```

Any `torch.optim.Optimizer` can be used as the outer optimizer via the
programmatic API. The server wraps global parameters as `nn.Parameter` objects,
so standard optimizers work directly.

### Server State Persistence

The server can periodically save its state (global params + outer optimizer
state) for crash recovery:

```bash
forgather diloco server -o path/to/model -n 2 --save-every 10
```

To resume a server from saved state:

```bash
forgather diloco server -o path/to/model -n 2
```

To resume from a specific checkpoint:

```bash
forgather diloco server -o ./model -n 2 --from-checkpoint ./model/checkpoints/checkpoint-25
```

## Async Mode

Asynchronous mode removes the synchronization barrier, allowing workers to
submit pseudo-gradients and receive updated parameters independently. This is
the recommended mode for heterogeneous clusters where machines have different
training speeds.

### Delayed Nesterov (DN) Momentum

In standard (synchronous) DiLoCo, the outer optimizer uses SGD with Nesterov
momentum. In async mode, applying momentum on every single worker submission can
amplify stale gradients, leading to training instability.

**Delayed Nesterov** addresses this by buffering pseudo-gradient submissions.
Between buffered steps, the server applies simple gradient descent (no momentum):

```
param -= lr * grad
```

When the buffer fills (every `dn_buffer_size` submissions), the server averages
the buffered gradients and applies a full outer optimizer step with momentum.

This prevents momentum from tracking the direction of stale individual worker
updates while still benefiting from momentum's acceleration over longer windows.

```bash
# Buffer 3 submissions, then apply momentum
forgather diloco server -o ./model -n 3 --async --dn-buffer-size 3
```

When `dn_buffer_size=0` (the default), the outer optimizer with momentum is
applied on every submission, which is appropriate when staleness is low.

### Dynamic Local Updates (DyLU)

When workers have different hardware (e.g., 4x RTX 3090 vs 1x RTX 4090), they
train at different speeds. Without adjustment, the faster worker submits far more
updates, potentially biasing the global model.

**DyLU** adapts each worker's sync frequency proportional to its relative speed:

```
H_w = floor((v_w / v_max) * H_base)
```

Where `v_w` is the worker's training speed (steps/second), `v_max` is the
fastest worker's speed, and `H_base` is the base sync interval. This ensures
faster workers do more local steps between syncs, so all workers contribute
updates at approximately the same wall-clock rate.

DyLU requires:
1. **Server**: `--dylu` flag and `--dylu-base-sync-every` (default: 500)
2. **Workers**: `--dylu` flag and `--heartbeat-interval` (default: 30s)

Workers periodically report their training speed via heartbeats. The server
computes the recommended sync interval and returns it in the heartbeat response.
Workers adjust their `sync_every` dynamically.

```bash
# Server with DyLU
forgather diloco server -o ./model -n 3 --async --dylu --dylu-base-sync-every 500

# Worker with DyLU enabled
forgather diloco worker --server host:8512 --sync-every 500 --worker-id w0 --dylu -- train
```

### Staleness Tracking

In async mode, the server tracks **staleness** for each worker submission: the
number of server-side updates that have occurred since the worker last synced.
High staleness means the worker's pseudo-gradients are computed against an
outdated reference, which can reduce training efficiency. Staleness is logged
on each submission and visible in the status endpoint for monitoring.

## Streaming DiLoCo (Fragment Sync)

Streaming DiLoCo splits the model into N **fragments** and staggers their
synchronization. Instead of one large transfer every H steps, each fragment
syncs every H/N steps, with communication happening in a background thread
while training continues on the remaining fragments.

### How It Works

```
sync_every=600, num_fragments=3 -> fragment interval = 200 steps

Step 1-200:   Training
Step 200:     Submit fragment 0 in background thread
Step 201-400: Training continues (fragment 0 transfer in background)
Step 400:     Apply fragment 0 result, submit fragment 1
Step 401-600: Training continues (fragment 1 transfer in background)
Step 600:     Apply fragment 1 result, submit fragment 2, reset counter
Step 1-200:   Training continues (fragment 2 transfer in background)
...
```

The total data transferred per `sync_every` steps is the same as standard mode
(full model), but latency is hidden behind computation. With enough fragments,
communication becomes fully overlapped.

### Bandwidth Analysis (Streaming)

| Model Size | Fragments | Fragment Size | Transfer Time | Compute Window | Hidden? |
|------------|-----------|---------------|---------------|----------------|---------|
| 150M       | 3         | 100 MB        | 0.8s          | 167s           | Yes     |
| 1B         | 7         | 286 MB        | 2.3s          | 71s            | Yes     |
| 7B         | 7         | 2 GB          | 16s           | 71s            | Yes     |

### CLI Usage

```bash
# Worker with 4 streaming fragments
forgather diloco worker \
    --server 192.168.1.100:8512 \
    --sync-every 500 \
    --worker-id w0 \
    --num-fragments 4 \
    -p my_project -t train.yaml \
    train
```

### Programmatic Usage

```python
from forgather.ml.diloco import DiLoCoWorker

with DiLoCoWorker(
    model=model,
    optimizer=optimizer,
    server_addr="192.168.1.100:8512",
    sync_every=500,
    num_fragments=4,       # Split model into 4 fragments
) as diloco:
    trainer.train()        # Fragment syncs happen in background
```

### FragmentManager

The `FragmentManager` handles parameter-to-fragment assignment:

```python
from forgather.ml.diloco import FragmentManager

fm = FragmentManager(model, num_fragments=4)

# Query fragment contents
print(fm.fragments[0])           # List of param names in fragment 0
print(fm.param_to_fragment)      # Dict: param_name -> fragment_id

# Check sync schedule
frag_id = fm.get_fragment_schedule(local_step=200, sync_every=800)
```

Parameters are split into contiguous groups by default, which naturally aligns
with pipeline stages where adjacent layers are on the same rank.

### Design Notes

- When `num_fragments=1` (default), the standard non-streaming path is used.
  No background threads, no fragment overhead.
- At most one fragment is in-flight at a time. Before submitting the next
  fragment, the previous one's result is applied.
- `force_sync()` always does a full-model sync regardless of fragment mode.
- The server's outer optimizer handles partial pseudo-gradient submissions by
  only setting `.grad` on the fragment's parameters. PyTorch optimizers skip
  parameters with `None` grad, so momentum buffers for other fragments remain
  untouched.

## Fault Tolerance

The DiLoCo system includes fault tolerance features to handle worker failures,
dynamic membership changes, and server restarts.

### Health Monitoring

The server runs a background **HealthMonitor** thread that periodically checks
worker heartbeat timestamps. Workers that haven't sent a heartbeat within the
`heartbeat_timeout` window are considered dead and automatically evicted.

```bash
# Server with health monitoring (default: 120s timeout)
forgather diloco server -o ./model -n 3 --heartbeat-timeout 120

# Disable health monitoring
forgather diloco server -o ./model -n 3 --heartbeat-timeout 0

# Require at least 2 workers to proceed
forgather diloco server -o ./model -n 3 --min-workers 2
```

Workers send heartbeats automatically (default: every 30 seconds). This is
independent of DyLU -- heartbeats are always active unless explicitly disabled
with `--heartbeat-interval 0`.

### Worker Death and Barrier Release

When a worker dies (heartbeat timeout or explicit deregistration):

1. The worker is removed from the registry
2. `num_workers` is decremented (but never below `min_workers`)
3. Any pending pseudo-gradient submissions from the dead worker are removed
4. The sync barrier is re-evaluated -- if the remaining workers have all
   submitted, the barrier releases and training continues

This prevents a dead worker from blocking all other workers indefinitely in
synchronous mode.

### Dynamic Worker Joining

New workers can join an active training run at any time:

1. The new worker registers with the server and receives the current global
   parameters
2. It begins training from the latest global state
3. `num_workers` is automatically increased if more workers than initially
   expected are registered
4. The new worker is not expected to submit for the current sync round --
   it participates starting from the next round

This enables elastic scaling: start with a few workers and add more as machines
become available.

### Worker Reconnection

Workers automatically retry sync operations on connection failure:

```python
# Worker with retry configuration
with DiLoCoWorker(
    model=model,
    optimizer=optimizer,
    server_addr="host:8512",
    sync_every=500,
    max_sync_retries=3,     # Retry sync up to 3 times on failure
) as diloco:
    trainer.train()
```

On connection failure, the worker:
1. Waits with exponential backoff
2. Re-registers with the server (getting fresh global params)
3. Recomputes pseudo-gradients against the new global state
4. Retries the sync submission

If all retries fail, the sync is skipped and training continues. This handles
transient network failures and server restarts gracefully.

### Server Restart Recovery

The server's `save_state` / `load_state` mechanism (see
[Server State Persistence](#server-state-persistence)) enables recovery from
server crashes. After restart:

1. The server loads the latest saved state from `output_dir` (or from `--from-checkpoint` if specified)
2. Workers detect the connection failure and enter their retry loop
3. Workers re-register and receive the saved global parameters
4. Training continues from the last saved checkpoint

### Monitoring Fault Tolerance

The status endpoint includes fault tolerance fields:

```bash
forgather diloco status --server host:8512
```

Shows `heartbeat_timeout`, `min_workers`, and `total_worker_deaths` (if any
workers have been evicted).

## How Pseudo-Gradients Work

The pseudo-gradient computation follows the TorchFt approach:

1. When a worker registers or completes a sync round, it saves a CPU snapshot
   of the model parameters (`_save_global_params_snapshot`)
2. The worker trains normally on GPU for `sync_every` steps
3. At sync time, the worker computes: `pseudo_grad = snapshot_cpu - model_params.cpu()`
4. The pseudo-gradient is optionally cast to bfloat16 and sent to the server
5. The server averages pseudo-gradients from all workers and applies the outer
   optimizer: `global_params -= lr * avg_pseudo_grad` (with momentum)

This design keeps the CPU snapshot in host memory without interfering with GPU
training, and the delta computation is done on CPU to avoid disrupting the
training computation graph.

## HTTP API Reference

The server exposes these HTTP endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/register` | Worker registration; returns global params |
| POST | `/submit_pseudograd` | Submit full-model pseudo-gradients; returns updated params |
| POST | `/submit_fragment_pseudograd` | Submit fragment pseudo-gradients; returns updated fragment params |
| GET | `/global_params` | Fetch current global parameters |
| POST | `/heartbeat` | Worker heartbeat with training speed; returns DyLU recommendation if enabled |
| POST | `/deregister` | Worker departure |
| GET | `/status` | Server status (mode, workers, sync round, fragment/async fields) |
| GET | `/info` | Static facts a client needs to negotiate settings (output_dir, num_parameters, expected_client_settings) |
| POST | `/control/{action}` | Control endpoints: `save_state`, `kick_worker`, `update_optimizer`, `update_num_workers`, `shutdown` |

Tensor data is serialized using `torch.save` to `BytesIO` and sent as
`application/octet-stream`. The pseudo-gradient submission uses a
length-prefixed JSON header followed by the tensor payload.

The `/status` endpoint returns additional fields in async mode:
- `mode`: `"sync"` or `"async"`
- `total_submissions`: Total pseudo-gradient submissions received
- `dn_buffer_size`: Configured DN buffer size
- `dn_buffered`: Current number of buffered submissions
- `dylu_enabled`: Whether DyLU is active
- `dylu_base_sync_every`: Base sync interval for DyLU

## Forgather Integration

The `DiLoCoCallback` integrates DiLoCo with the Forgather trainer ecosystem.
It manages the `DiLoCoWorker` lifecycle automatically and integrates with the
checkpoint system via the `Stateful` protocol.

### Callback Usage

Add `DiLoCoCallback` to your trainer's callback list. When `server_addr` is
empty (and `DILOCO_SERVER` is unset), the callback is a no-op, so the same
configuration works for both DiLoCo and standalone training.

```python
from forgather.ml.trainer.callbacks import DiLoCoCallback

# Explicit configuration
callback = DiLoCoCallback(
    server_addr="192.168.1.100:8512",
    sync_every=500,
    bf16_comm=True,
    num_fragments=1,
)

# Or rely on environment variables (set by `forgather diloco worker`)
callback = DiLoCoCallback()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    callbacks=[callback],
)
trainer.train()
```

All constructor parameters fall back to `DILOCO_*` environment variables:

| Parameter | Env Var | Default |
|-----------|---------|---------|
| `server_addr` | `DILOCO_SERVER` | `""` (no-op) |
| `sync_every` | `DILOCO_SYNC_EVERY` | `500` |
| `worker_id` | `DILOCO_WORKER_ID` | auto-generated |
| `bf16_comm` | `DILOCO_BF16_COMM` | `True` |
| `dylu` | `DILOCO_DYLU` | `False` |
| `heartbeat_interval` | `DILOCO_HEARTBEAT_INTERVAL` | `30.0` |
| `num_fragments` | `DILOCO_NUM_FRAGMENTS` | `1` |

### Configuration Template

Include the DiLoCo callback template to add DiLoCo support to any project:

```yaml
-- extends 'callbacks/diloco.yaml'
```

Or add the callback directly in your project template:

```yaml
[callback_list]
    == super()
    diloco_callback: !singleton:forgather.ml.trainer.callbacks:DiLoCoCallback
        server_addr: {{ diloco_server | default(None) }}
        sync_every: {{ diloco_sync_every | default(None) }}
        num_fragments: {{ diloco_num_fragments | default(None) }}
```

See `examples/tiny_experiments/diloco/` for a complete working example.

### Checkpoint Behavior

The `DiLoCoCallback` implements the `Stateful` protocol, so the checkpoint
manager automatically saves and restores its state:

- **Saved**: sync_count, local_step, sync_every, worker_id, total_sync_time,
  retry/reconnection counters, DyLU adjustments, fragment sync count
- **Not saved**: global_params snapshot (the server provides fresh params when
  the worker re-registers on resume)

On checkpoint resume, the callback's `load_state_dict` is called during
`_prepare()` (before the worker exists). The state is deferred and applied
in `on_train_begin` after the worker is created and registered with the server.

### Model fingerprint check

The callback also runs a `/status` round-trip in `on_train_begin` before
constructing the worker, and the worker's `/register` call ships a
`{name: shape}` map of every named parameter (the `param_shapes` field in
the register body). The server compares against its own `_param_list`
shapes and rejects mismatched models with HTTP 422 + a diagnostic naming
the divergent param. This catches the operator-misconfiguration case —
pointing a worker at the wrong `--model-id-or-path` — at register time
rather than letting it surface hundreds of steps later in the first
sync's optimizer step.

The check only fires when the worker actually ships `param_shapes`
(post-#51 builds; on by default in this codebase). Workers from an
older build that omit the field still register cleanly and would
crash later in sync if the model is wrong — there's no server-side
way to detect a missing fingerprint as suspicious without breaking
those callers.

The webui's Submit-training-job modal pre-fills `--model-id-or-path`
from the selected DiLoCo server's `/info.output_dir` to keep the easy
path easy.

## Work-unit dispatch

Workers in a DiLoCo run partition the training dataset through a
server-driven dispatch loop, not via manual `--num-shards` / `--shard-index`
flags. The server holds a per-`(dataset_id, shuffle_seed)` queue of `K`
work units (default `K=1024`); each worker requests the next available
unit, streams that unit's row range from its dataset backend, and asks
for another. Issuance is one-way — once a unit is issued it's consumed
from the queue regardless of worker fate, so within an epoch no row is
ever trained on twice.

### Activation

Work-unit dispatch is **unconditional** when DiLoCo is enabled. No
operator-facing toggle. The wrap fires when:

- `DILOCO_SERVER` is set in the worker's environment, AND
- `DILOCO_WORKER_ID` is set (the scheduler defaults this to the
  queue_id when DiLoCo is enabled), AND
- The dataset is loaded via the iterable-backend path —
  `forgather.ml.datasets:fast_load_iterable_dataset` routing through a
  `forgather dataset_server` (`FORGATHER_DATASET_SERVER` env var) or
  the cluster auto-routing (`cluster-auto://`) variant. The
  in-process local loader (no `FORGATHER_DATASET_SERVER`) doesn't
  participate; if you point a DiLoCo worker at a local-loader
  config, the wrap aborts with `DiLoCoWorkDispatchUnavailable` at
  startup so you see the misconfiguration before training begins.

Any failure to wire up the wrap when `DILOCO_SERVER` is set —
unreachable `/datasets/register`, missing `DILOCO_WORKER_ID`, a
backend without `__len__`, or an invalid load arg — is fatal at
startup, surfaced as `DiLoCoWorkDispatchUnavailable` in the worker's
TTY. Silently falling back to a bare backend would mean every
worker iterates the full row stream on identical rows, which is the
broken-data-parallelism class of failure the system explicitly
guards against.

**Where dispatch lives in the dataset pipeline.** The wrap is
applied *inside* ``ComposableIterableDataset`` (state set by
``enable_work_dispatch(client, worker_id)``; the dispatch loop lives
in ``_iter_window``). ``preprocess_dataset`` calls it after slice and
shard are settled — the dispatch operates on the post-slice view
bounds, and `shard()` and `enable_work_dispatch()` are mutually
exclusive. The implementation isn't a backend wrapper; it's a method
on the composable itself.

**How operators select dispatch.** The ``shard_dataset.method``
field on the dataset preprocess block controls partitioning:

```yaml
# Conventional DDP sharding (default when DiLoCo is off):
shard_dataset: True                            # WORLD_SIZE / RANK
shard_dataset: {num_shards: 4, index: 0}       # explicit
shard_dataset: {method: "conventional"}        # alias

# DiLoCo work-unit dispatch (default when DiLoCo is on):
shard_dataset: {method: "work_units"}

# No partitioning:
shard_dataset: False                           # full dataset per process
```

The validity matrix is enforced at preprocess time, and the rules
depend on ``partition_purpose`` (stamped per-singleton in
``load_dataset.yaml`` — train splits get ``"train"``, eval / test
splits get ``"eval"``):

**``partition_purpose='train'``** (strict — train requires cross-host
coordination under DiLoCo):

| Config                        | `DILOCO_SERVER` unset | `DILOCO_SERVER` set |
|-------------------------------|----------------------|---------------------|
| `False`                       | OK                   | OK                  |
| `True` / `conventional`       | OK                   | **error**            |
| `work_units`                  | **error**            | OK                  |

Conventional + DiLoCo is rejected for the train dataset because
asymmetric DDP topologies (e.g. DDPx4 on one host, DDPx8 on another)
produce overlapping per-rank shards — workers train on the same rows.
Work-unit dispatch replaces conventional sharding entirely for train:
all DDP ranks across all DiLoCo hosts compete for units in one shared
queue.

**``partition_purpose='eval'``** (replicated across hosts — every host
runs the full eval, metrics averaged across hosts):

| Config                        | `DILOCO_SERVER` unset | `DILOCO_SERVER` set |
|-------------------------------|----------------------|---------------------|
| `False`                       | OK                   | OK                  |
| `True` / `conventional`       | OK                   | OK                  |
| `work_units`                  | **error**            | **error**            |

Under DiLoCo, eval is intentionally replicated across hosts — the
cross-host duplication is harmless because metrics get averaged. But
within a single host we still want DDP sharding to split eval work
across the ranks of that host, otherwise every DDP rank runs the full
eval locally for an identical result (W× wasted compute). So
``shard_dataset: True`` (= conventional within-host shard) is the
right value for eval under DiLoCo, and the validity check allows it.
``work_units`` is refused for eval — that would route eval through
the cross-host work queue, which makes no sense given eval is
replicated by design.

``lm_training_project.yaml`` wires this automatically: train picks
``method='work_units'`` under DiLoCo (else conventional), and eval
picks ``True`` unconditionally (DDP within host, replicated across
hosts).

### dataset_id

The `dataset_id` is a stable 16-hex hash of the normalized load args
(`path`, `name`, `split`, `data_files`, `revision`) plus the
composable's resolved slice bounds (`slice_start`, `slice_end`). Two
workers loading "the same dataset with the same slice" agree on the
dataset_id by construction; two workers loading **different slices**
of the same source dataset get **different** dataset_ids (so they
share work only within the same slice). Shard info is deliberately
**not** part of the hash: under DiLoCo, all DDP ranks across all
hosts share one queue, so per-rank sharding must not key separate
queues. The worker also ships the human-readable fields alongside the
hash so the webui can label queues with
`roneneldan/TinyStories@train` instead of just the hex.

### Interleaved / multi-source datasets

Each call to `fast_load_iterable_dataset` registers its own queue.
A training run that interleaves two HF paths produces two queues; the
webui renders each as its own heatmap card. Workers in a multi-source
run hold one issued unit per source at a time.

### DataLoader `num_workers > 1`

Allowed and correct under work-unit dispatch — each forked DataLoader
worker has its own `DiLoCoClient` and competes for units in the same
queue. Server atomicity prevents double-issuance, so the rows are
still partitioned correctly.

Not recommended, though:

- **Connection multiplication.** Each fork opens its own HTTP keep-
  alive to the DiLoCo server *and* its own to the dataset_server. For
  W=8 DDP × N=4 DataLoader workers that's 32 connections per host to
  each service.
- **Shuffle quality degrades.** The reservoir buffer is per-process;
  N forks → N independent 1000-row buffers spread across the queue,
  effectively worse shuffle than one big buffer.
- **Throughput rarely improves.** With `ResilientRemoteBackend`
  already streaming, the bottleneck under DiLoCo dispatch is usually
  the dispatch round-trip, not the iter loop.

Default to `dataloader_num_workers=0` (or `1`) for iterable datasets
under DiLoCo unless you have a measured reason otherwise.

### Crash recovery

If a worker dies holding an issued unit, that unit is lost (the
server's one-way issuance design — at most `N_workers` units lost per
epoch). The DiLoCo server's `_work_queues` is **not** persisted across
server restarts (pre-#46 it was, but cross-experiment state-bleed from
a stale checkpoint outweighed crash-recovery utility). On server
restart, workers re-register their datasets on first contact and the
queue map is reconstructed fresh.

Design details: `docs/design/diloco-work-unit-dispatch.md`.

## Monitoring & Control

The DiLoCo server itself only exposes JSON endpoints — the operator-facing
view lives in the **forgather webui's DiLoCo panel**
(`tools/forgather_server/webui/src/components/DiLoCoPanel.tsx`). The
panel lists known servers on the left and renders per-server detail
on the right:

1. **Header**: server mode (sync/async badge), sync round, uptime,
   parameter count + model size.
2. **Workers table**: per-worker health dot (green/yellow/red by
   heartbeat age), ID (hover for full id), hostname, sync round,
   steps/s, relative heartbeat age, and a per-row **Kick** button.
3. **Server metrics**: outer LR / momentum, worker-death count,
   heartbeat timeout. Sync mode adds a pending-submissions progress
   bar; async mode adds total-submissions, DN buffer status, and DyLU
   state.
4. **Control card**: **Save checkpoint**, **Shutdown** (confirm
   overlay), live **Optimizer** tuning (LR + momentum + Apply),
   **Workers** expected-count adjustment.
5. **Work-unit dispatch**: per-queue heatmap (K cells, three states:
   available / issued / completed), with per-worker counters.

An earlier version of the server shipped its own Alpine.js dashboard
at `/dashboard`; that page was removed when the webui panel took
over. The control endpoints below are unchanged and are what the
webui's Control card talks to under the hood.

### Control Endpoints

| Endpoint | Body | Action |
|----------|------|--------|
| `POST /control/save_state` | `{}` | Save server state to disk |
| `POST /control/kick_worker` | `{"worker_id": "..."}` | Evict a worker |
| `POST /control/update_optimizer` | `{"lr": 0.5, "momentum": 0.8}` | Update optimizer hyperparameters |
| `POST /control/update_num_workers` | `{"num_workers": 4}` | Change expected worker count |
| `POST /control/shutdown` | `{}` | Save state (if configured) and stop |

All endpoints return `{"status": "ok", ...}` on success or `{"error": "..."}` on
failure.

### Security Note

The DiLoCo server's HTTP endpoints have no authentication. They
provide full control over the training run, including shutdown and
optimizer mutation. Only expose the server on trusted networks. Do
not expose the server port to the public internet without additional
access controls (e.g., a reverse proxy with authentication, or the
forgather webui in front of it).

## Network Configuration

By default, the server binds to `127.0.0.1` (localhost only). This is the safest
configuration when workers run on the same machine.

### Remote Workers via SSH Port Forwarding

For remote workers, the recommended approach is SSH port forwarding. This avoids
exposing the server on all interfaces and provides encrypted communication:

```bash
# On each remote worker machine, forward the server port:
ssh -L 8512:localhost:8512 server-machine

# Then start the worker pointing to localhost:
forgather diloco worker --server localhost:8512 ...
```

The `-L 8512:localhost:8512` flag forwards the worker's local port 8512 to port
8512 on the server machine. The worker connects to `localhost:8512` as if the
server were local.

For persistent tunnels (e.g., in tmux), add `-N` to keep the SSH connection
open without a shell:

```bash
ssh -N -L 8512:localhost:8512 server-machine &
```

### Binding to All Interfaces

If SSH tunneling is impractical (e.g., trusted LAN with many workers), you can
bind to all interfaces:

```bash
forgather diloco server -o ./model -n 4 --host 0.0.0.0
```

**Warning**: This exposes the server's HTTP control endpoints (including
`/control/shutdown`, `/control/update_optimizer`, etc., which the webui
DiLoCo view's Control card calls into) to any machine on the network.
Only use this on trusted networks with appropriate firewall rules.

## References

- Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models" (2024)
- Douillard et al., "DiPaCo: Distributed Path Composition" (2024)
- Douillard et al., "Asynchronous Local-SGD Training for Language Modeling" (2024) - Async DiLoCo, Delayed Nesterov, DyLU
- Douillard et al., "Streaming DiLoCo with Overlapping Communication" (2024) - Fragment-based staggered sync
- TorchFt (Meta) - fault-tolerant distributed training library
