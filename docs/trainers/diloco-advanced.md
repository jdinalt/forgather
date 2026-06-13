# DiLoCo Programmatic API & Advanced Modes

> Part of the [DiLoCo documentation](diloco.md). This chapter is the deep
> reference: the Python API, server configuration, unified statistics, async
> mode, streaming/fragment sync, fault tolerance, the HTTP API, Forgather
> integration, and work-unit dispatch. See the [hub](diloco.md) for concepts
> and quick start, and the [CLI reference](diloco-cli.md) for command usage.

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
    # Wire precision is server-authoritative; workers normally adopt
    # upload/download dtype + SR from the server's /info.
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
- `upload_dtype` / `upload_sr` / `download_dtype` / `download_sr`: wire precision
  for the pseudo-gradient (upload) and averaged-params (download) legs, with
  optional stochastic rounding. Normally adopted from the server's `/info`;
  `bf16_comm=True` is kept as a deprecated alias for `upload_dtype="bf16"`. See
  [Wire precision](diloco.md#wire-precision).
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

## Unified statistics

The DiLoCo server has no training loop of its own, so the run-level picture of
"how is training going" lives in the workers. The server aggregates it into a
single view: each worker reports its training metrics on the heartbeat
(sourced from the **DiLoCo callback**, in parallel to the control callback —
server stats do not depend on it), and the server folds them into an aggregate
exposed on `/status` (and so in `forgather diloco status` and the webui DiLoCo
view).

Collected metrics:

| Metric | Aggregation | Persisted in checkpoints |
|---|---|---|
| `total_tokens` | sum of per-worker increments (keyed by worker_id) | yes |
| `total_flos` | sum of per-worker increments | yes |
| `total_steps` | sum of per-worker increments | yes |
| `tok_per_sec` | sum over currently-reporting workers | no (live) |
| `mfu` | FLOPs-weighted mean (falls back to tokens) | no (live) |
| `peak_memory` | sum over currently-reporting workers | no (live) |
| `grad_norm` | token-weighted mean | no (live) |
| `train_loss` | token-weighted EMA | yes (EMA state) |
| `eval_loss` | token-weighted weak EMA (with the step it was computed at) | yes (EMA state) |

Lifetime counters and the loss EMAs persist in `server_state.pt`, so they
survive a restart; because the per-worker token/step/FLOP increments are keyed
by `worker_id`, relaunching a worker under the same id (`submit
--resume-workers`) continues the totals rather than double-counting the resumed
history. Live gauges are recomputed from the workers currently reporting and a
worker the server evicts drops out of them.

### Run directories and logs

When the server has an `output_dir`, each freshly-started run logs to its own
directory under `<output_dir>/runs/<timestamp>_<run-name>` (the same `runs/`
convention the trainers use, so common tooling and TensorBoard can read them
together). `--run-name` sets the label (defaults to the hostname); a resume
from checkpoint continues the prior run's directory rather than fragmenting
across a new one. Keeping each run in its own directory makes it easy to retain
and compare runs with different parameters.

Each run directory holds two things:

- `diloco_server_stats.jsonl` — the aggregate stream, one JSON record per line
  (append-only). The webui DiLoCo server view plots train/eval loss from it
  (scroll to zoom, drag to pan, double-click to reset), served via
  `GET /stats_history` (proxied as `GET /api/diloco/stats-history`). This
  in-panel plot is for quick diagnostics.
- TensorBoard event files — the same aggregates mirrored via the torch
  TensorBoard logger (resuming where it left off on restart, like the trainer).
  Point TensorBoard at `<output_dir>/runs` to overlay and compare runs, or to
  overlay the server aggregate against a worker's own run (the scalar tags
  `train-loss` / `eval-loss` / `grad-norm` match the trainer's).

Per-worker eval curves don't track precisely between syncs — for that detail,
point TensorBoard at an individual worker's `output_dir`. The server's
`eval_loss` is intentionally a lightly-smoothed cross-worker summary.

## Async Mode

Asynchronous mode removes the synchronization barrier, allowing workers to
submit pseudo-gradients and receive updated parameters independently. This is
the recommended mode for heterogeneous clusters where machines have different
training speeds.

### Delayed Nesterov (DN) Momentum

In standard (synchronous) DiLoCo, the outer optimizer uses SGD with Nesterov
momentum. In async mode, applying momentum on every single worker submission can
amplify stale gradients, leading to training instability.

**Delayed Nesterov** addresses this by *delaying* the momentum (Liu et al. 2024,
arXiv:2401.09135, Algorithm 3, with the default activation `c=0`). With buffer
size `N` (`--dn-buffer-size N`), every submission takes an immediate, momentum-free
descent step scaled by `1/N`, while the Nesterov momentum is refreshed and applied
only once every `N` submissions, on the averaged buffer:

```
every submission:   param -= lr * grad / N ;   Delta += grad
every N-th:          m <- beta * m + Delta / N
                     param -= lr * beta * m ;   Delta <- 0
```

This prevents momentum from tracking the direction of stale individual worker
updates while still benefiting from momentum's acceleration over the `N`-submission
window. The server keeps only the running sum `Delta` and the momentum `m`, so the
memory cost is `O(model)`, independent of `N`. For `N=1` this reduces exactly to
the plain Nesterov outer step.

```bash
# Delay momentum over 3 submissions
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
2. **Workers**: `--heartbeat-interval` (default: 30s). DyLU enablement and the
   base `sync_every` are taken from the server's `/info` — there is no worker
   `--dylu` flag.

Workers periodically report their training speed via heartbeats. The server
computes the recommended sync interval and returns it in the heartbeat response.
Workers adjust their `sync_every` dynamically.

```bash
# Server with DyLU (this is where dylu / sync_every are configured)
forgather diloco server -o ./model -n 3 --async --dylu --dylu-base-sync-every 500

# Worker — picks up dylu + sync_every from the server's /info
forgather submit --diloco --diloco-server host:8512 --worker-id w0 -- train
```

### Grace Period

DN and DyLU reduce the *impact* of staleness; the **grace period** reduces the
staleness itself by letting workers that finish close together resync against the
**same** model (Liu et al. 2024, Section 3, Algorithms 2/5).

With `--grace-period S` (seconds, async only), a submission's response is held and
any other workers that submit within the window — anchored to the **first**
arrival, so it always closes — are aggregated into **one** outer step; all of them
are then released with the same post-step params. The window flushes early if all
live workers have submitted.

```bash
forgather diloco server -o ./model -n 4 --async \
    --dn-buffer-size 4 --dylu --grace-period 2.0
```

The three async knobs **layer**: the grace period aggregates near-simultaneous
submissions *within* a round; **DN** delays the outer momentum *across* rounds (a
grace batch is one DN tick, regardless of how many workers it aggregated); and
**DyLU** makes workers co-terminate, so more of them land in each window and the
batches grow. `--grace-period 0` (default) disables it — each submission applies
immediately. The grace period is server-side only (workers don't adopt it); the
batch-size distribution is reported on `/status` (`mean_grace_batch_size`,
`grace_batch_histogram`) so you can see it working.

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
# Streaming fragments are configured on the server, not the worker —
# the worker reads num_fragments (and sync_every) from /info.
forgather submit --diloco \
    --diloco-server 192.168.1.100:8512 \
    --worker-id w0 \
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

fm = FragmentManager(model, num_fragments=4, assignment="strided")

# Query fragment contents
print(fm.fragments[0])           # List of param names in fragment 0
print(fm.param_to_fragment)      # Dict: param_name -> fragment_id
print(fm.block_faithful)         # True if split on transformer-block boundaries

# Check sync schedule
frag_id = fm.get_fragment_schedule(local_step=200, sync_every=800)
```

Following Streaming DiLoCo (arXiv:2501.18512), the model is split on
**transformer-block boundaries** — each fragment is a set of whole blocks, never
a partial block. Blocks are discovered from the model's `_no_split_modules` (the
HF convention for atomic transformer-block classes, the same signal vLLM uses;
see [vLLM integration](../inference/vllm_integration.md)). Two assignment modes,
set server-side with `--fragment-assignment`:

- **`strided`** (default, the paper's mild preference): block `i` goes to
  fragment `i % N`, so each fragment spans the depth of the model.
- **`sequential`**: contiguous runs of blocks per fragment.

The non-block parameters are attached deterministically — embeddings to the
first fragment, the final norm and LM head to the last — so every parameter sits
in exactly one fragment. A model that exposes no block plan (no
`_no_split_modules`, e.g. a non-transformer) falls back to an equal-param-count
contiguous split with a warning (`block_faithful == False`).

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
forgather diloco status --diloco-server host:8512
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
| GET | `/info` | Static facts a client needs to negotiate settings (output_dir, num_parameters, expected_client_settings, `model_hash`) |
| GET | `/model_def` | Model-definition bundle: a tar of config + custom code + tokenizer (no weights), used by workers to construct the model. Bearer-required; control-plane only. |
| POST | `/control/{action}` | Control endpoints: `save_state`, `kick_worker`, `update_optimizer`, `update_num_workers`, `shutdown` |

The `/model_def` response is an uncompressed tar (deterministic member
order) carrying only the non-weight files from the server's checkpoint
directory — `config.json`, every custom modeling/configuration `.py`
(the full `trust_remote_code` closure, including split two-file
definitions), and the tokenizer. Weights, shard indices, `server_state.pt`,
and the audit log are excluded. The `X-Forgather-Model-Hash` header carries
the bundle identity, which equals the `model_hash` advertised by `/info`
(parameter topology folded together with the definition-file contents) so a
worker never pairs a bundle with a mismatched parameter set.

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

# Explicit configuration (client-local knobs only)
callback = DiLoCoCallback(
    server_addr="192.168.1.100:8512",
    worker_id="w0",
    heartbeat_interval=30.0,
)

# Or rely on environment variables (set by `forgather submit --diloco`)
callback = DiLoCoCallback()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    callbacks=[callback],
)
trainer.train()
```

The callback's client-local constructor parameters fall back to `DILOCO_*`
environment variables:

| Parameter | Env Var | Default |
|-----------|---------|---------|
| `server_addr` | `DILOCO_SERVER` | `""` (no-op) |
| `worker_id` | `DILOCO_WORKER_ID` | auto-generated |
| `heartbeat_interval` | `DILOCO_HEARTBEAT_INTERVAL` | `30.0` |

`sync_every`, the wire-precision knobs (`upload_dtype`, `upload_sr`,
`download_dtype`, `download_sr`), `dylu`, and `num_fragments` are **not** callback
parameters or env vars — they must match across the group, so the worker
reads them from the server's `/info` at startup (set them on the server).
The `DiLoCoWorker` class still accepts them directly for the low-level
programmatic API; it's only the callback / CLI surface that defers to the
server.

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
        worker_id: {{ diloco_worker_id | default(None) }}
        heartbeat_interval: {{ diloco_heartbeat_interval | default(None) }}
```

See [`examples/tiny_experiments/diloco/`](../../examples/tiny_experiments/diloco/README.md)
— the canonical end-to-end CLI walkthrough — for a complete working example.

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

### Model definition comes from the server

A DiLoCo worker takes **no model path**. On startup the trainer stages the
model-definition bundle from the server's `GET /model_def` endpoint into
`<output_dir>/diloco_model_def/` (config + the custom modeling/configuration
`.py` closure + tokenizer; never weights), builds the model **empty** (from
that config, with no weights), and fills it from the parameter sync at
register. Every worker therefore builds the *same* model the server holds —
there is no `--model-id-or-path` to get wrong and no shared filesystem
requirement. See [Model-definition staging](#model-definition-staging) below.
The empty skeleton is built on the meta device (allocation-free), and the
worker checkpoints its training state (optimizer / scheduler / progress /
RNG) but **never model weights** — those are the server's authority. See
[Checkpoint state selection + empty-meta construction](diloco-architecture.md)
in the architecture doc.

### Model fingerprint check

As defense-in-depth, the worker's `/register` call still ships a
`{name: shape}` map of every named parameter (the `param_shapes` field in
the register body). The server compares against its own `_param_list`
shapes and rejects mismatched models with HTTP 422 + a diagnostic naming
the divergent param. With the bundle the worker builds from the server's
*own* config, so this should never fire — it stays as a backstop against
a stale staged definition. The coarse `/info` `model_hash` (parameter
topology folded with the definition-file contents) is the complementary
pre-construction gate the staging cache validates against.

The check only fires when the worker actually ships `param_shapes`
(post-#51 builds; on by default in this codebase). Workers from an
older build that omit the field still register cleanly and would
crash later in sync if the model is wrong — there's no server-side
way to detect a missing fingerprint as suspicious without breaking
those callers.

### Model-definition staging

Staging is wired in the training config as a cached `!singleton`
(`forgather.ml.diloco.model_stage:stage_model_def`) that closes over the
worker's computed `output_dir` and the server address. It is referenced by
the tokenizer, the model config, and the model factory, so the **first**
consumer to materialize — typically the tokenizer at dataset preprocessing,
which resolves into a real object well before the model is built — triggers
a single fetch; the rest reuse the cache. A `.forgather_model_hash` stamp
records the bundle identity: a later run (or DDP rank) with a matching
stamp short-circuits the network fetch, while a mismatch — the server was
restarted on a different model — forces a clean re-fetch. `file_lock_build`
serializes concurrent ranks/workers on one host. There is no offline
fallback: a worker that cannot reach the server fails loud rather than
build a divergent model.

> **Operator notes.** The worker loads the staged code with
> `trust_remote_code=True`, so the DiLoCo server is a *trusted code-
> distribution authority* for its workers — only register workers against
> servers you control (the same trust the worker already extends by pulling
> weights and authoritative settings from it). The bundle ships **every
> `.py`** found in the server's checkpoint directory (the full
> `trust_remote_code` closure, including split two-file definitions), so
> that directory should contain only model-definition files — don't stash
> unrelated scripts or secrets next to the model. Weights, optimizer state
> (`server_state.pt`), shard indices, and the audit log are always excluded.

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
picks ``{{ ns.dispatch_batches == False }}`` unconditionally — i.e.
``True`` (DDP within host, replicated across hosts) when
``dispatch_batches`` is off, ``False`` (full eval per process) when
``dispatch_batches`` is on, regardless of DiLoCo state.

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
epoch). The DiLoCo server's `_work_queues` **is** persisted with its
checkpoint and restored on restart, so a server bounce does not re-issue
already-consumed rows within the epoch: a re-registering worker resumes
at the next un-issued unit. A changed dataset hashes to a new
`dataset_id`, so stale queues from a prior dataset are never matched
(no cross-experiment bleed); the queue is flushed on graceful shutdown
as well as the periodic save cadence.

Design details: `docs/design/diloco-work-unit-dispatch.md`.

