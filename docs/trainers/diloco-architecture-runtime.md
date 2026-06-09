# DiLoCo Architecture: Runtime Behavior

> Part of the DiLoCo [Architecture & Maintainer Guide](diloco-architecture.md).
> This page covers what happens at runtime: the lifecycle and data flow of a
> sync round, fault tolerance, and unified statistics. See the
> [design page](diloco-architecture.md) for the data structures, wire protocol,
> and threading model, and the
> [maintainer reference](diloco-architecture-reference.md) for persistence,
> CLI, testing, troubleshooting, and extension points.

## Lifecycle and Data Flow

### Full sync round (synchronous, no fragments)

```
Worker                                  Server
  |                                       |
  |-- register(worker_id, info) --------->|  POST /register
  |<--- global_params (torch.save) -------|
  |                                       |
  | [save CPU snapshot of global params]  |
  | [train for sync_every steps]          |
  |                                       |
  | [compute pseudograds on CPU]          |
  | [cast to bf16 if enabled]             |
  |                                       |
  |-- submit_pseudograd(wid, pgs) ------->|  POST /submit_pseudograd
  |                                       |  [store in _pending_pseudograds]
  |                                       |  [wait at barrier...]
  |                                       |  [all workers submitted]
  |                                       |  [average pseudograds]
  |                                       |  [set as .grad on _param_list]
  |                                       |  [outer_optimizer.step()]
  |<--- updated global_params ------------|
  |                                       |
  | [copy global params to model (GPU)]   |
  | [save new CPU snapshot]               |
  | [reset _local_step = 0]              |
  | [continue training...]               |
```

### Streaming sync round (3 fragments)

```
Worker                                  Server
  |                                       |
  | [train 200 steps]                    |
  |                                       |
  | [compute frag 0 pseudograds, CPU]    |
  | [launch background thread]           |
  |    |-- submit_fragment(wid, 0, pgs)->|  POST /submit_fragment_pseudograd
  |    |                                  |  [barrier for frag 0]
  |    |                                  |  [apply outer opt to frag 0 params]
  | [train 200 steps]                    |
  |    |<-- frag 0 updated params -------|
  |                                       |
  | [join bg thread, apply frag 0 result]|
  | [compute frag 1 pseudograds]         |
  | [launch background thread]           |
  |    |-- submit_fragment(wid, 1, pgs)->|
  | [train 200 steps]                    |
  |    |<-- frag 1 updated params -------|
  |                                       |
  | [join bg thread, apply frag 1 result]|
  | [compute frag 2 pseudograds]         |
  | [launch background thread]           |
  |    |-- submit_fragment(wid, 2, pgs)->|
  | [train 200 steps (next round)]      |
  |    |<-- frag 2 updated params -------|
  | ...                                   |
```

### Model-definition staging (issue #53)

Under DiLoCo a worker takes **no model path**. It obtains the model
*definition* — `config.json`, the custom modeling/configuration `.py`
closure, and the tokenizer (never weights) — from the server, builds the
model **empty** (from that config, with no weights), and fills it from the
parameter sync. This removes the shared-filesystem requirement and makes it
impossible for a worker to build a different model than the rest of the
group. The empty skeleton is built on the meta device (allocation-free) and
the worker checkpoints its non-model training state only — see
[Checkpoint state selection + empty-meta construction](diloco-architecture-reference.md#checkpoint-state-selection--empty-meta-construction).

This happens **before** `DiLoCoWorker.start()` — it is part of trainer
*construction* (materializing the model/tokenizer), not the worker
lifecycle. The two stages are independent: staging supplies the *skeleton
and tokenizer*; `start()`'s register/`_apply_global_params` supplies the
*weights*.

#### Components

| Piece | Role |
|-------|------|
| `model_def.py` | The include/exclude policy (every `.py` + config + tokenizer; never weights/`server_state.pt`/indices/audit), deterministic `pack_model_def`, `compute_bundle_hash`, and traversal-safe `extract_model_def`. Shared by server and client. |
| Server `GET /model_def` | Streams the packed tar with an `X-Forgather-Model-Hash` header. Bearer-required, control-plane only. |
| `DiLoCoClient.fetch_model_def(dest)` | GET + validate header against `/info` + traversal-safe extract. |
| `model_stage.stage_model_def(addr, output_dir)` | Worker-side fetch-and-cache into `<output_dir>/diloco_model_def/`; returns the local dir. |
| Template `models/from_diloco_server.yaml` | Wires `stage_model_def` as a cached `!singleton` consumed by the tokenizer, the model config, and the model factory. |

#### Sequence — what happens when the config loads the model

The training config materializes the model assets in the worker's `train`
process (where `DILOCO_SERVER` and the worker-suffixed `output_dir` are
both known). The staging singleton closes over `output_dir` (a render-time
Jinja literal) and the server address, and is referenced by three nodes:

```
[render time]  output_dir, server_addr baked into the !singleton node
                     │
[train process: materialize trainer → model assets]
                     │
   FIRST consumer to resolve  (usually the TOKENIZER, at dataset
   preprocessing — it resolves into a real object before the model)
                     │
                     ▼
   stage_model_def(server_addr, output_dir):
     1. client.get_info()  ── want_hash = /info model_hash
     2. fast path: <output_dir>/diloco_model_def/.forgather_model_hash
        == want_hash?  ── yes → return dir (no lock, no tar fetch)
     3. file_lock_build(dir, force_lock=True):        # serialize ranks
          re-check stamp under lock → maybe return
          tmp = mkdtemp(dir=output_dir)
          client.fetch_model_def(tmp)                 # GET /model_def
            → validate X-Forgather-Model-Hash vs /info → extract tar
          write tmp/.forgather_model_hash (stamp LAST)
          rmtree(dir); os.replace(tmp, dir)           # atomic swap
        return dir
                     │
   AutoTokenizer.from_pretrained(dir, trust_remote_code=True)
                     │
   SECOND/THIRD consumers (model config, model factory) resolve the SAME
   cached singleton → no second fetch; AutoConfig.from_pretrained(dir),
   then the model !partial builds the empty structure (no weights)
                     │
[DiLoCoWorker.start()]  register → _apply_global_params overwrites the
   freshly-initialized persistent weights with the server's global params
```

The shared-singleton design is load-bearing: the tokenizer is needed at
dataset preprocessing, **earlier** than model construction, so staging
cannot live inside the model factory alone. The first of {tokenizer,
config, model} to materialize triggers exactly one fetch; the rest reuse
the cache.

#### Caching and invalidation

* **Worker-side stamp.** `<output_dir>/diloco_model_def/.forgather_model_hash`
  records the bundle identity. The fast path compares it to a fresh `/info`
  `model_hash`: a match short-circuits the network tar fetch entirely
  (cheap one `/info` round-trip), a mismatch — the server was restarted on
  a *different* model — forces a clean re-fetch. There is no offline
  fallback; an unreachable server fails loud.
* **Atomic, crash-safe writes.** The bundle is fetched into a sibling temp
  dir and `os.replace`'d into place, with the stamp written *last*, so an
  interrupted fetch never leaves a match-looking directory behind (the next
  run re-fetches).
* **DDP / multi-worker.** `file_lock_build(..., force_lock=True)` serializes
  ranks/workers sharing one host: one fetches under the lock, the rest
  re-check the stamp on acquiring it and reuse.
* **Server-side bundle cache.** `_model_def_dir` is content-stable for the
  server's lifetime, so the server packs the tar once (lazily, under
  `_model_def_lock`) and caches `self._model_def_bundle`. Concurrent worker
  fetches don't each re-walk the dir or hold separate in-memory copies.
* **Hash semantics.** `_model_hash` is the parameter `(name, shape)`
  topology folded with the definition-file *contents* (`compute_bundle_hash`,
  applied in `load_state`). So a config tweak or an edited modeling `.py`
  — not just a shape change — changes the advertised `model_hash` and
  invalidates worker stamps. The same value is returned by `/info` and the
  `/model_def` header, and is complementary to the fine-grained
  per-parameter `param_shapes` check at `/register`.

### Worker startup (`start()` / `__enter__`)

1. Send registration request to server
2. Receive global parameters from server
3. Copy global params into model (`_apply_global_params`)
4. Save CPU snapshot (`_save_global_params_snapshot`)
5. Register optimizer post-step hook
6. Start heartbeat thread if `heartbeat_interval > 0` (default: 30s)

### Worker shutdown (`stop()` / `__exit__`)

1. Wait for any in-flight fragment to complete
2. Stop heartbeat thread
3. Remove optimizer hooks
4. Send deregistration request to server

---

## Fault Tolerance

The system handles four fault scenarios: worker death, dynamic joining, worker
reconnection after transient failures, and server restart recovery.

### Worker death detection

The `HealthMonitor` (in `health.py`) runs a background daemon thread on the
server. Every `check_interval` seconds (default: `heartbeat_timeout / 3`) it
scans all registered workers:

```
for each worker in _workers:
    if now - worker.last_heartbeat > heartbeat_timeout:
        server._handle_worker_death(worker_id)
```

Workers update `last_heartbeat` via the `/heartbeat` endpoint. The heartbeat
thread runs unconditionally on workers when `heartbeat_interval > 0` (default
30s), regardless of DyLU setting.

### Worker death handling (`_handle_worker_death`)

When a worker is declared dead (by HealthMonitor or explicit deregistration):

```
1. Acquire _sync_cond -> _workers_lock (lock ordering preserved)
2. Remove worker from _workers registry
3. Increment _total_worker_deaths
4. Update num_workers = max(min_workers, remaining)
5. Remove worker's pending pseudo-gradients (if any)
6. Remove worker from _round_expected_workers set

7. Re-evaluate full-model sync barrier:
   - expected = len(_round_expected_workers)
   - if submitted >= expected: apply outer optimizer, complete round

8. Re-evaluate per-fragment barriers (for each active fragment):
   - Remove dead worker's fragment submission
   - If remaining submissions satisfy expected count: apply and complete

9. notify_all() to wake waiting threads
```

This ensures that a worker dying mid-sync doesn't deadlock the remaining
workers. The barrier dynamically adjusts to the reduced worker count.

**`min_workers` floor:** The `num_workers` field never drops below
`min_workers` (default 1). This prevents a scenario where all workers die and
the barrier releases with zero submissions.

### Dynamic worker joining

New workers can register at any time via `/register`. The registration handler:

1. If `_round_expected_workers` already exists (mid-round), the new worker is
   **not** added to it. The new worker participates starting the next round.
2. If more workers register than the current `num_workers`, `num_workers` is
   increased to accommodate them.
3. The new worker receives the current global parameters and begins local
   training immediately.

This design prevents a new worker from blocking the current round's barrier
(which would deadlock because existing workers already have the expected count
computed).

### Worker reconnection

Workers handle transient connection failures via retry with reconnection:

```python
# In _sync() - retry loop
for attempt in range(max_sync_retries + 1):
    try:
        new_global = client.submit_pseudogradients(worker_id, pseudograds)
        break
    except ConnectionError:
        if attempt < max_sync_retries:
            sleep(retry_delay)  # exponential backoff: 2s, 4s, 8s, ...
            retry_delay *= 2
            self._reconnect()   # re-register, get fresh global params
            pseudograds = self._compute_pseudogradients()  # recompute
        else:
            # Skip this sync round, continue training
```

The `_reconnect()` method re-registers the worker with the server, receives
the current global parameters, and updates the local snapshot. This handles:

- **Server restart:** Server comes back with saved state, worker re-registers
  and gets the latest global params.
- **Network partition:** Temporary disconnection resolves, worker re-registers.
- **Worker eviction:** If the server's HealthMonitor evicted this worker,
  re-registration adds it back.

After reconnection, pseudo-gradients are recomputed against the new global
params snapshot to avoid stale deltas.

### Client tensor retry

The `DiLoCoClient._request_tensor()` method accepts an optional `retries`
parameter. When set (used by internal reconnection logic), failed tensor
requests are retried with exponential backoff before raising `ConnectionError`.
By default (retries=0), tensor requests fail immediately (they are large,
stateful payloads where blind retry is not always appropriate).

### Interaction with async mode

In async mode, there is no barrier to deadlock, so worker death is less
critical. The `_handle_worker_death()` method still removes the worker from the
registry and adjusts `num_workers`. The HealthMonitor runs identically in both
modes.

### Status monitoring

The `/status` endpoint includes fault tolerance fields:

- `heartbeat_timeout`: configured timeout value
- `min_workers`: configured minimum workers
- `total_worker_deaths`: cumulative death count

Worker `sync_metrics` include `sync_retries` and `reconnections` counters.

---

## Unified statistics

`StatsAggregator` (`diloco/stats.py`) gives the server a run-level training
view it otherwise lacks (it has no training loop). Each worker's
`DiLoCoCallback` snapshots the trainer's metrics in `on_log` / `on_evaluate`
onto a normalized schema, stashes it on the `DiLoCoWorker`, and the worker
ships it as the optional `stats` field on its next heartbeat (consume-once, so
the loss EMA isn't re-fed the same sample). `_handle_heartbeat` stores the
snapshot on `WorkerInfo.stats` and folds it into the aggregator; `/status`
returns the result under `aggregate_stats`.

Aggregation rules:

- **Lifetime counters** (`total_tokens`, `total_flos`, `total_steps`) accumulate
  per-worker *deltas* keyed by `worker_id` — a worker reports its own cumulative
  value, the server adds the increment since that worker's last report. Reusing
  a `worker_id` on resume continues the count; a counter reset clamps to a
  non-negative delta. These persist in the checkpoint (`stats` key), as does the
  per-worker last-seen baseline needed to keep deltas correct across a restart.
- **Live gauges** are computed on demand from the latest snapshot of each
  currently-reporting worker: `tok_per_sec` and `peak_memory` sum (extensive);
  `mfu` and `grad_norm` are weighted means (intensive — summing MFU would
  exceed 100%), MFU weighted by each worker's per-report FLOPs increment
  (falling back to tokens), grad_norm by tokens. Not persisted, and
  `drop_worker` removes an evicted worker from them (its delta baseline is kept).
- **Loss** is a token-weighted EMA (`S = decay·S + w·loss`, `Z = decay·Z + w`,
  `loss = S/Z`); `S`/`Z` persist so smoothing survives a resume. `train_loss`
  uses a stronger decay than the weak-EMA `eval_loss`.
- **Per-worker training-state gauges** (`global_step`, `epoch`, `learning_rate`)
  mirror the trainer-control endpoint's `/status` payload. They aren't
  aggregated across workers (different workers can be at different points in
  their schedule) — the per-worker `WorkerInfo.stats` carries them verbatim
  through the heartbeat → server pipeline and the webui's per-worker stats
  row reads them from `/status`. This is what makes per-worker stats render
  in the cross-node case where the trainer's loopback-bound control endpoint
  isn't reachable from the webui's node.

When `output_dir` is set the server logs each aggregate snapshot (throttled to
one record per advance in total steps) into a per-run directory
`<output_dir>/runs/<time_ns>_<run_name>` — the trainer's `runs/` convention, so
the two are tooling-compatible. A fresh start makes a new dir (unique by
`time_ns`); the chosen subdir is persisted in the checkpoint (`stats_run_subdir`)
and a resume reuses it for continuity. `_run_name` comes from `--run-name`
(default hostname) and is sanitized to a safe path component. Each run dir holds:

- `diloco_server_stats.jsonl` — one JSON object per line, append-only.
  Deliberately not the trainer's JSON-array `JsonLogWriter`: append-only JSONL
  into a unique-per-run dir is robust to restarts (no exclusive-create race, no
  truncate-to-empty on resume) where the array format's bracket/close
  management was not. Served by `/stats_history` for the webui plot.
- TensorBoard event files — the same scalars via a `torch.utils.tensorboard`
  `SummaryWriter`, tags `train-loss`/`eval-loss`/`grad-norm` (matching the
  trainer, for overlay) plus `tokens-per-sec`/`mfu`/`total-tokens`/etc. On
  resume the writer is opened with `purge_step` = the restored total step.

Writes are guarded by `_stats_log_lock` (`_handle_heartbeat` runs on concurrent
threads). The logs are not otherwise checkpoint-coupled — only the run subdir
is persisted, for directory continuity.

