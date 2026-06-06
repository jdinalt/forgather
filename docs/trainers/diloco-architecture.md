# DiLoCo Architecture & Maintainer Guide

This document describes the internal architecture of Forgather's DiLoCo system.
It is intended for developers who need to understand how the system works,
troubleshoot issues, or implement new features.

For user-facing documentation (CLI usage, quick start, API examples), see
[diloco.md](diloco.md). For a runnable, end-to-end CLI walkthrough, see the
canonical example at
[`examples/tiny_experiments/diloco/`](../../examples/tiny_experiments/diloco/README.md).

## Contents

- [System Overview](#system-overview)
- [Source Layout](#source-layout)
- [Data Structures and State](#data-structures-and-state)
- [Wire Protocol](#wire-protocol)
- [Threading Model](#threading-model)
- [Synchronization Modes](#synchronization-modes)
- [Streaming DiLoCo (Fragments)](#streaming-diloco-fragments)
- [Outer Optimizer Integration](#outer-optimizer-integration)
- [Lifecycle and Data Flow](#lifecycle-and-data-flow)
- [Fault Tolerance](#fault-tolerance)
- [Server State Persistence](#server-state-persistence)
- [CLI Layer](#cli-layer)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Extension Points](#extension-points)
- [Checkpoint state selection + empty-meta construction](#checkpoint-state-selection--empty-meta-construction)
- [Known Limitations](#known-limitations)

---

## System Overview

DiLoCo is a client-server distributed training system. A central **server**
holds the global model parameters and outer optimizer state. Multiple **workers**
(each running any Forgather trainer) train locally and periodically submit
pseudo-gradients to the server over HTTP. The server applies an outer optimizer
step and returns updated global parameters.

```
                    DiLoCoServer (CPU-only process)
                    +---------------------------------+
                    | _param_list: ParameterList       |
                    | outer_optimizer: torch.optim.SGD  |
                    | _workers: Dict[str, WorkerInfo]   |
                    | ThreadingHTTPServer               |
                    +---------------------------------+
                                  |
                     HTTP (torch.save payloads)
                                  |
              +-------------------+-------------------+
              |                                       |
     DiLoCoWorker (GPU machine)             DiLoCoWorker (GPU machine)
     +---------------------------+          +---------------------------+
     | model: nn.Module           |         | model: nn.Module           |
     | optimizer: inner optimizer  |         | optimizer: inner optimizer  |
     | _global_params: CPU snapshot|         | _global_params: CPU snapshot|
     | DiLoCoClient               |         | DiLoCoClient               |
     | optimizer post-step hook    |         | optimizer post-step hook    |
     +---------------------------+          +---------------------------+
```

The system has three operating dimensions that can be combined:

| Dimension | Options | Key mechanism |
|-----------|---------|---------------|
| Sync mode | Synchronous / Asynchronous | Server barrier vs immediate apply |
| Momentum | Standard / Delayed Nesterov (DN) | Buffer submissions before momentum |
| Communication | Full-model / Streaming (fragments) | Background thread per fragment |

---

## Source Layout

```
src/forgather/ml/diloco/
  __init__.py        Exports: DiLoCoServer, DiLoCoClient, DiLoCoWorker, FragmentManager,
                     HealthMonitor, OuterSyncBackend, HttpStarBackend, SyncResult,
                     CoordinatorClient
  server.py          HTTP server, outer optimizer, sync barrier, fragment handling, fault tolerance
  client.py          HTTP client, tensor serialization, request construction, retry logic
  sync_backend.py    OuterSyncBackend seam + HttpStarBackend (the bulk tensor-leg transport)
  coordinator.py     CoordinatorClient (the coordination surface: heartbeat / info / model-def)
  worker.py          Optimizer hook, pseudo-gradient computation, streaming, reconnection
  fragments.py       FragmentManager: parameter splitting, scheduling
  health.py          HealthMonitor: background worker liveness detection
  model_def.py       Model-definition bundle: include/exclude policy, deterministic
                     tar packing, content hashing, traversal-safe extraction (issue #53)
  model_stage.py     stage_model_def(): worker-side fetch+cache of the bundle into
                     <output_dir>/diloco_model_def, with stamp reuse + file-lock

src/forgather/cli/
  diloco.py          CLI command handlers (_server_cmd, _status_cmd, _worker_cmd)
  diloco_args.py     Argument parser (create_diloco_parser)

tests/unit/ml/diloco/
  test_server.py          Server: outer optimizer correctness, serialization
  test_sync_backend.py    OuterSyncBackend delegation; worker backend-agnosticism
  test_coordinator.py     CoordinatorClient delegation; worker coordinator wiring
  test_server_client.py   HTTP round-trip: register, submit, status
  test_worker.py          Worker: pseudo-gradients, optimizer hooks, full sync cycle
  test_async.py           Async mode, DN momentum, DyLU
  test_streaming.py       FragmentManager, fragment server/client, streaming worker
  test_fault_tolerance.py Health monitor, worker death, barrier release, reconnection

docs/trainers/
  diloco.md               User-facing documentation
  diloco-architecture.md  This file
```

---

## Data Structures and State

### Server (`DiLoCoServer`)

**Global parameters:**

```python
_param_names: List[str]              # Ordered parameter names (matches model.state_dict() order)
_param_list: nn.ParameterList        # Global params as nn.Parameter (float32, CPU, requires_grad=False)
_param_name_to_idx: Dict[str, int]   # Reverse lookup: name -> index in _param_list
```

Parameters are stored as `nn.Parameter` objects inside a `ParameterList` so that
any standard `torch.optim.Optimizer` can be constructed against them. All
parameters are float32 on CPU regardless of what workers send (incoming bf16
pseudo-gradients are cast to float32 before accumulation).

**Worker registry:**

```python
_workers: Dict[str, WorkerInfo]      # worker_id -> metadata
_workers_lock: threading.Lock        # Protects _workers dict
```

`WorkerInfo` is a dataclass with: `worker_id`, `hostname`, `registered_at`,
`last_heartbeat`, `sync_round` (worker's count), `last_sync_server_round`
(server round at last sync), `steps_per_second`, `output_dir`, `extra`, and
`stats` (the worker's latest unified-stats snapshot — see
[Unified statistics](#unified-statistics)).

`output_dir` is the worker's local output directory, reported at
registration and surfaced per-worker in `/status`. It is used only by the
webui to correlate a worker back to its forgather job: the primary key is
`queue_id == worker_id`, but a run that reuses a stable custom worker-id
(e.g. to resume from its own checkpoint) registers under an id that no
longer equals the job's `queue_id`, so the panel falls back to matching the
worker's `output_dir` against the job's resolved `output_dir`. The
per-worker output-dir suffix and the registered worker-id share a single
resolved source (the `--diloco-worker-id` Jinja arg, else `DILOCO_WORKER_ID`
env), so the two stay in lockstep.

**Synchronous state:**

```python
_sync_round: int                                      # Global monotonic counter
_pending_pseudograds: Dict[str, Dict[str, Tensor]]    # worker_id -> pseudograds (waiting for barrier)
_sync_cond: threading.Condition                        # Barrier notification
_completed_rounds: Dict[int, Dict[str, Tensor]]       # round_number -> result (cached for late wakers)
```

**Async state:**

```python
_async_lock: threading.Lock                            # Serializes async submissions
_total_submissions: int                                # Total submissions received
_dn_grad_buffer: List[Dict[str, Tensor]]               # Delayed Nesterov buffer
```

**Fragment state (sync + async):**

```python
_fragment_pending: Dict[int, Dict[str, Dict[str, Tensor]]]   # frag_id -> worker_id -> pseudograds
_fragment_rounds: Dict[int, int]                               # frag_id -> current round
_completed_fragment_rounds: Dict[Tuple[int,int], Dict[str, Tensor]]  # (frag_id, round) -> result
_fragment_submissions: int                                     # Total fragment submissions
```

**Fault tolerance state:**

```python
_round_expected_workers: Optional[set]          # Worker IDs expected for current sync round (None before first submission)
_health_monitor: Optional[HealthMonitor]        # Background health checker (None if heartbeat_timeout=0)
_total_worker_deaths: int                       # Cumulative dead worker count
heartbeat_timeout: float                        # Seconds before a worker is considered dead (0 = disabled)
min_workers: int                                # Floor for num_workers during death handling
```

`_round_expected_workers` is the key data structure for fault-tolerant barriers.
It is snapshotted from `_workers.keys()` when a sync round completes (or lazily
on the first submission of a round). Workers that join mid-round are not added
to the current snapshot -- they participate starting next round. When a worker
dies, it is removed from this set, which may cause the barrier to release early
if the remaining submissions satisfy the reduced expected count.

### Worker (`DiLoCoWorker`)

```python
model: nn.Module                      # Live model (on GPU)
optimizer: torch.optim.Optimizer      # Inner optimizer (AdamW, etc.)
client: DiLoCoClient                  # HTTP client for server communication
_global_params: Dict[str, Tensor]     # CPU snapshot taken after each sync
_local_step: int                      # Steps since last sync (reset to 0 after sync)
_sync_count: int                      # Completed sync rounds
_hooks: List                          # Optimizer post-step hook handles
_fragment_manager: Optional[FragmentManager]  # None when num_fragments <= 1
_inflight_thread: Optional[Thread]    # Background thread for current fragment
_inflight_result: Optional[Tuple[int, Optional[Dict[str, Tensor]]]]  # (frag_id, result)
max_sync_retries: int                 # Max retry attempts per sync (default: 3)
_sync_retries: int                    # Cumulative sync retry count
_reconnections: int                   # Cumulative reconnection count
```

### FragmentManager

```python
fragments: List[List[str]]            # fragment_id -> list of param names
param_to_fragment: Dict[str, int]     # param_name -> fragment_id
num_fragments: int
```

Parameters are split into contiguous groups of roughly equal count (not equal
tensor size). The first `total % N` fragments get one extra parameter.

---

## Wire Protocol

All communication uses HTTP/1.1 over TCP. The server runs a
`ThreadingHTTPServer` (one thread per request).

### Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/register` | JSON: `{worker_id, hostname, ...}` | Tensor: global params |
| POST | `/submit_pseudograd` | Binary: header + tensors | Tensor: updated global params |
| POST | `/submit_fragment_pseudograd` | Binary: header + tensors | Tensor: updated fragment params |
| GET | `/global_params` | (none) | Tensor: global params |
| POST | `/heartbeat` | JSON: `{worker_id, steps_per_second, stats?}` | JSON: `{status, sync_round, recommended_sync_every?, command?}` |
| POST | `/deregister` | JSON: `{worker_id}` | JSON: `{status: "ok"}` |
| GET | `/status` | (none) | JSON: server state |
| GET | `/known_workers` | (none) | JSON: `{workers: [{worker_id, output_dir, last_registered, running}]}` |
| GET | `/info` | (none) | JSON: negotiation facts + `model_hash` |
| GET | `/model_def` | (none) | tar: model definition (config + code + tokenizer, no weights); `X-Forgather-Model-Hash` header |

**Wire precision.** Four server-authoritative knobs — `upload_dtype`,
`upload_sr`, `download_dtype`, `download_sr` — are advertised in `/info` and
adopted by every worker. The **upload** cast happens worker-side in
`compute_pseudograds` (via `_cast_for_upload`) before `/submit_pseudograd`; the
**download** cast happens server-side via `_cast_for_download` on the params
returned by `/register`, `/submit_pseudograd`, and `/submit_fragment_pseudograd`.
Each cast optionally uses stochastic rounding (`fp32_to_bf16_stochastic_round`).
The server's master parameters and outer-optimizer state always stay fp32;
incoming tensors are accumulated in fp32 regardless of wire dtype. The legacy
`bf16_comm` boolean is retained as a deprecated alias for `upload_dtype`.

`/known_workers` returns every `worker_id` the server has ever seen — the
roster `self._known_workers`, persisted with the server's checkpoints (see
"Server state persistence" below). Each entry carries the worker's
last-reported `output_dir`, its last registration time, and a `running`
flag (true iff currently registered). The webui's submit modal surfaces the
not-running entries as toggleable chips in its worker pool so an operator can
relaunch a worker under its old id and thereby resume from that worker's own
checkpoint — the checkpoint path is the worker-id-suffixed `output_dir`, so
reusing the id is the only way to find it. Routed on the control port only and
bearer-authenticated, like `/status`.

`/model_def` is served from `self._model_def_dir`, resolved in `load_state`
by `_resolve_model_def_dir`: the loaded checkpoint when it carries the
definition (a self-contained `--from-checkpoint` model dir), else
`output_dir` — the model's home. This fallback matters because a *rotated*
server checkpoint (`checkpoints/checkpoint-N/`) holds only weights +
`server_state.pt`; the definition (`config.json` + custom modeling/config
`.py` + tokenizer) lives at the `output_dir` top level. Without the
fallback, a server restarted off a rotated checkpoint would serve an empty
bundle and every worker's config load would fail with "Unrecognized model"
(issue #103). When neither dir carries a definition, `_model_def_dir` is
`None` and `/model_def` returns 503 (loud failure, no empty bundle); the
worker's `stage_model_def` independently refuses to stamp a bundle that has
no `config.json`, so a definition-less fetch can never poison the staging
cache. The folded `model_hash` is computed over `_model_def_dir`, so in the
common case — the definition lives at `output_dir` — it is stable across a
restart even though the loaded checkpoint dir changes. It shifts only if the
server was first started from a *separate* self-contained `--from-checkpoint`
dir that also carried the definition (first run resolves to that dir, a
restart-from-rotated-checkpoint falls back to `output_dir`); the worst case
there is a one-time bundle re-fetch by each worker, not an incorrect bundle.

The include/exclude policy, deterministic packing, and traversal-safe
extraction live in `forgather.ml.diloco.model_def`; the worker-side staging
into `<output_dir>/diloco_model_def/` lives in
`forgather.ml.diloco.model_stage`. Both `/info` and `/model_def` are
control-plane endpoints (bearer-required, never served on the bulk
listener).

### Binary tensor format (submit endpoints)

Pseudo-gradient submissions use a length-prefixed header format:

```
[4 bytes: header length (big-endian uint32)]
[header_length bytes: JSON header (UTF-8)]
[remaining bytes: torch.save payload]
```

The JSON header contains `worker_id` and optionally `fragment_id`. The tensor
payload is a serialized `Dict[str, torch.Tensor]` produced by
`torch.save(state_dict, BytesIO)`.

### Tensor serialization

Both client and server use `torch.save` / `torch.load` with `map_location="cpu"`
and `weights_only=True`. This is the same format used by PyTorch checkpoints.
Response payloads use `Content-Type: application/octet-stream`.

### Client retry behavior

- JSON requests (`register`, `heartbeat`, `deregister`): retried up to
  `max_retries` times with exponential backoff (default: 3 retries, 1s base)
- Tensor requests (`submit_pseudograd`, `submit_fragment_pseudograd`,
  `get_global_params`): configurable retries via the `retries` parameter on
  `_request_tensor()`. Default is 0 (no retries). The worker's `_sync()` method
  handles retry at a higher level via `_reconnect()` + resubmit.
- Default timeout: 600 seconds (sync submissions may block for a long time at
  the server barrier)

### Sync backend seam

The bulk tensor legs of a sync round — join (register + initial params),
full-model and per-fragment pseudo-gradient submission, and leave (deregister) —
are routed through an `OuterSyncBackend` (`sync_backend.py`) rather than called
on the client directly. The seam also exposes a contribute-free
`current_global_params` fetch for late-join/recovery. The worker owns everything
local (pseudo-gradient computation via `ParamView`, applying the returned
params, the DDP-rank broadcast, and scheduling); the backend owns *how* the
worker reaches agreement on the next global params.

`HttpStarBackend` is the only implementation: a thin adapter over `DiLoCoClient`
providing the HTTP central-parameter-server transport. The worker holds an
`OuterSyncBackend` — an `HttpStarBackend` wrapping its own client by default.
The seam lets the transport be a different implementation (e.g. collectives or a
shared-memory parameter region) without changing the worker's
`compute → synchronize → apply → broadcast` flow.

The seam is the outer step, not a byte channel: `synchronize` takes a
pseudo-gradient and returns a `SyncResult` carrying the agreed next global
params, so a backend may run the outer optimizer centrally (HTTP), in a
replicated copy per worker, or in place on a shared region — the worker is
agnostic. Backends advertise capability flags (`runs_outer_optimizer`,
`supports_async`, `fault_tolerant`) that callers honor rather than assume. Only
the tensor legs are pluggable; the coordination plane is a separate surface (see
*Coordinator surface* below). The wire-precision casts remain in `ParamView`
(the worker still passes `upload_dtype` / `download_dtype` through to it).

### Coordinator surface

The other half of the role split (#154): coordination — everything that is not
the bulk tensor exchange — is reached through a `CoordinatorClient`
(`coordinator.py`), distinct from the sync backend. The worker holds both a
`backend` (parameter authority / transport) and a `coordinator`, so a future
backend whose parameter authority is not the coordinator (a serverless
collective, a shared-memory region) can still coordinate over the HTTP server
while exchanging params elsewhere.

`CoordinatorClient` is a concrete thin facade over `DiLoCoClient` — coordination
always speaks HTTP regardless of which backend moves the tensors, so it is not an
ABC (unlike `OuterSyncBackend`). It covers the worker-process bring-up
coordination: `heartbeat`, `get_info` (`/info` negotiation), and `fetch_model_def`
(model staging). Two other coordination surfaces keep their own purpose-built
clients and are not folded in: work-unit dispatch (`register_dataset` /
`request_work` / `complete_work`), owned by the dataset layer, and control
(`relay_command` / `save_state` / `shutdown` / `get_status`), owned by the CLI.

---

## Threading Model

### Server threads

The server uses `ThreadingHTTPServer` which spawns a new daemon thread for each
incoming HTTP request. This is required because in synchronous mode, multiple
worker requests block concurrently waiting at the barrier.

`run()` serves on a background thread and blocks the main thread on an event;
SIGTERM/SIGINT just set that event. All stop paths — the signal handlers,
`/control/shutdown`, and `forgather diloco shutdown` — converge on
`graceful_shutdown()`, which relays `save_and_stop` to every worker, then keeps
serving while they finish (including any in-flight sync round), checkpoint, and
deregister. Serving during the drain is essential: a worker parked on the sync
barrier would deadlock if submissions stopped being accepted; as each worker
leaves, the barrier's expected set shrinks via the normal worker-death path.
Once the roster is empty (or a timeout elapses) it saves state and stops. A
re-entrancy guard makes it idempotent so the signal and endpoint paths converge
on one run.

**Critical locking:**

| Lock | Protects | Used by |
|------|----------|---------|
| `_sync_cond` (Condition) | `_pending_pseudograds`, `_completed_rounds`, `_fragment_pending`, `_fragment_rounds`, `_completed_fragment_rounds` | Sync submit, fragment sync submit |
| `_async_lock` (Lock) | All async state, global param reads/writes in async mode | Async submit, async fragment submit, register (async), get_global_params (async) |
| `_workers_lock` (Lock) | `_workers` dict | All handlers that read/update worker info |

**Lock ordering** (always acquire in this order to avoid deadlocks):

1. `_sync_cond` or `_async_lock` (never both at once)
2. `_workers_lock` (acquired inside the above)

In sync mode, `_sync_cond` is used as a Condition (with `wait`/`notify_all`),
not just a lock. In async mode, `_async_lock` is a simple `Lock` (no wait
needed). The two modes are mutually exclusive; the server either uses
`_sync_cond` or `_async_lock`, never both for the same submission type.

### Server health monitor thread

When `heartbeat_timeout > 0`, the server creates a `HealthMonitor` (from
`health.py`) that runs a daemon thread checking worker liveness every
`check_interval` seconds (default: `heartbeat_timeout / 3`). On each check it
reads `_workers` under `_workers_lock`, compares `last_heartbeat` timestamps to
the current time, and calls `_handle_worker_death()` for any worker that
exceeds the timeout. The health monitor is started in `start()` / `run()` and
stopped in `stop()`.

### Worker threads

The worker has up to two background threads:

1. **Heartbeat thread**: sends periodic heartbeats to the server to report
   training speed and maintain liveness. Runs when `heartbeat_interval > 0`
   (default: 30s). Stopped via `_heartbeat_stop` Event. When DyLU is enabled,
   the worker also reads back `recommended_sync_every` from the heartbeat
   response and adjusts its `sync_every`.

2. **Fragment inflight thread** (streaming mode only): submits one fragment's
   pseudo-gradients to the server in the background. At most one inflight thread
   exists at any time. The main training thread joins this thread before starting
   the next fragment submission.

**Invariant:** At most one background fragment is in-flight. Before starting a
new fragment submission, `_wait_and_apply_inflight_fragment()` joins the previous
thread and applies its result to the model. This prevents concurrent model
modifications and simplifies reasoning about parameter consistency.

```
Main thread:     [train]--[join prev, compute pg, launch bg]--[train]--[join, compute, launch]--...
Fragment thread:                                [submit to server]         [submit to server]
```

---

## Synchronization Modes

### Synchronous mode (default)

All workers must submit pseudo-gradients before any receives the updated global
parameters. The server uses a Condition variable as a barrier:

```
Server thread per worker:
  1. Acquire _sync_cond
  2. Store pseudograds in _pending_pseudograds[worker_id]
  3. Record current _sync_round as my_round
  4. If all workers submitted:
     a. Average pseudo-gradients
     b. Apply outer optimizer
     c. Store result in _completed_rounds[my_round]
     d. Increment _sync_round
     e. notify_all()
  5. While my_round not in _completed_rounds: wait()
  6. Return _completed_rounds[my_round]
```

**Per-round result caching:** `_completed_rounds` maps round number to global
params. This prevents a race where a late-waking thread reads the wrong round's
result (the server might already be in the next round). Stale entries are pruned
to keep only the last two rounds.

### Asynchronous mode

Each worker's pseudo-gradients are applied immediately under `_async_lock`. No
barrier, no waiting. The worker receives updated global params in the response.

**Staleness tracking:** When a worker submits, the server computes staleness as
`current_sync_round - worker.last_sync_server_round`. High staleness means the
pseudo-gradient was computed against parameters that are many updates behind.
Staleness is logged but not currently used for weighting or rejection.

### Delayed Nesterov (DN)

A server-side strategy for async mode. When `dn_buffer_size > 0`:

- **Intermediate submissions** (buffer not full): Apply direct gradient descent
  `param -= lr * grad` without calling the optimizer (no momentum update)
- **Buffer-full submissions** (every N-th): Average the buffer, set as `.grad`,
  call `outer_optimizer.step()` (full momentum update), clear buffer

This prevents momentum from tracking stale individual worker directions.

### Dynamic Local Updates (DyLU)

Server-side computation, communicated via heartbeats:

```
H_w = max(1, floor((v_w / v_max) * H_base))
```

Workers report `steps_per_second` in heartbeats. The server computes the
recommended sync interval proportional to the worker's relative speed and returns
it in the heartbeat response. Workers that opt in (`dylu=True`) adjust their
`sync_every` accordingly.

---

## Streaming DiLoCo (Fragments)

### Fragment scheduling

`FragmentManager.get_fragment_schedule(local_step, sync_every)` determines which
fragment syncs at a given step:

```
fragment_interval = sync_every // num_fragments
fragment_idx = (local_step // fragment_interval - 1) % num_fragments
```

Example with `sync_every=600, num_fragments=3`:
- `fragment_interval = 200`
- Step 200: `(200/200 - 1) % 3 = 0` -> fragment 0
- Step 400: `(400/200 - 1) % 3 = 1` -> fragment 1
- Step 600: `(600/200 - 1) % 3 = 2` -> fragment 2

### Background sync flow

```python
# In _post_step_hook, when a fragment is scheduled:
def _sync_fragment(fragment_id):
    # 1. Wait for previous in-flight fragment to complete, apply its result
    self._wait_and_apply_inflight_fragment()

    # 2. Compute pseudo-gradients for this fragment (CPU, main thread)
    pseudograds = self._fragment_manager.compute_fragment_pseudogradients(...)

    # 3. Launch background thread to submit to server
    self._inflight_thread = Thread(target=self._submit_fragment_background, ...)
    self._inflight_thread.start()
    # Main thread returns immediately, training continues
```

### Server-side fragment handling

The server has separate handlers for fragment submissions:

- **Sync fragment:** Per-fragment barrier using `_sync_cond`. Each fragment has
  its own round counter (`_fragment_rounds[frag_id]`). When all workers submit
  the same fragment, the server applies the outer optimizer to just that
  fragment's parameters.

- **Async fragment:** Under `_async_lock`, set `.grad` on fragment parameters
  only, call `step()`, return the updated fragment params.

**Outer optimizer correctness with fragments:** Only the fragment's parameters
have `.grad` set. PyTorch optimizers skip parameters with `None` grad. SGD's
momentum buffers for other parameters remain untouched because `step()` only
processes parameters that have a non-None `.grad`.

### Fragment-standard mode boundary

When `num_fragments=1` (default), the worker's `_fragment_manager` is `None`.
The `_post_step_hook` takes the standard path (full-model sync via `_sync()`)
with zero overhead. No background threads are created. This is a hard branch in
`_post_step_hook`:

```python
if self._fragment_manager is None:
    # Standard: full model sync at sync_every
    if self._local_step >= self.sync_every:
        self._sync()
else:
    # Streaming: check fragment schedule
    frag_id = self._fragment_manager.get_fragment_schedule(...)
    if frag_id is not None:
        self._sync_fragment(frag_id)
```

---

## Outer Optimizer Integration

The outer optimizer is a standard `torch.optim.Optimizer` instance. The server
constructs it by passing `_param_list.parameters()` to a factory function:

```python
factory = outer_optimizer_factory or _default_outer_optimizer_factory
self.outer_optimizer = factory(self._param_list.parameters())
```

Default: `torch.optim.SGD(params, lr=0.7, momentum=0.9, nesterov=True)`

To apply pseudo-gradients:

1. Average pseudo-gradients across workers
2. Set `_param_list[i].grad = avg_grad` for each parameter
3. Call `self.outer_optimizer.step()`
4. Call `self.outer_optimizer.zero_grad()`

This pattern works with any optimizer (Adam, Adafactor, etc.) without code
changes. The optimizer's `state_dict()` is included in server state saves.

For fragments, only the fragment's parameters have `.grad` set. All other
parameters have `None` grad. PyTorch optimizers iterate all parameter groups
but skip parameters where `grad is None`.

**LR extraction for DN:** The server extracts `_outer_lr` from the optimizer's
first param group for use in DN direct gradient steps
(`param -= lr * grad`). This assumes a single learning rate. If different
parameter groups have different LRs, DN would need modification.

---

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
[Checkpoint state selection + empty-meta construction](#checkpoint-state-selection--empty-meta-construction).

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

---

## Server State Persistence

`save_state(path)` saves a dict via `torch.save`:

```python
{
    "global_params": Dict[str, Tensor],    # Current global parameters
    "outer_optimizer": optimizer.state_dict(),
    "sync_round": int,
    "num_workers": int,
    "param_names": List[str],
    "async_mode": bool,
    "total_submissions": int,
    "known_workers": Dict[str, {output_dir, last_registered}],
    "stats": StatsAggregator.state_dict(),   # lifetime counters + loss EMA
}
```

`load_state(path)` restores parameters and optimizer state. Note that
`weights_only=False` is used for loading because the optimizer state dict
contains non-tensor values.

`known_workers` is the roster of every `worker_id` that has ever
registered (see `/known_workers` above). Persisting it here is what lets a
restarted server still offer the previous run's workers for
checkpoint-resuming relaunch; it is snapshotted under `_workers_lock` at
save time and restored on load (absent on pre-feature checkpoints, where it
simply starts empty).

Automatic save: when `save_dir` is set, the server saves every
`save_every_n_rounds` sync rounds. Two files are written: a versioned file
(`diloco_server_state_round{N}.pt`) and a `diloco_server_state_latest.pt`
symlink.

**Fragment state is not persisted.** Fragment round counters and pending
submissions are transient. After a server restart, workers should re-register
and start a fresh sync cycle.

---

## CLI Layer

### diloco_args.py

Builds the argument parser with three subcommands: `server`, `status`, `worker`.
The parser is created by `create_diloco_parser(global_args)` which is called
from `fgcli.py`.

### diloco.py

`diloco_cmd(args)` dispatches to `_server_cmd`, `_status_cmd`, or `_worker_cmd`.

**`_worker_cmd`** does not create a `DiLoCoWorker` directly. Instead, it sets
environment variables and spawns a subprocess running `forgather train`:

```
DILOCO_SERVER       -> server address
DILOCO_HEARTBEAT_INTERVAL -> seconds
DILOCO_WORKER_ID    -> optional worker ID
```

Group-wide settings — `sync_every`, the four wire-precision knobs
(`upload_dtype`, `upload_sr`, `download_dtype`, `download_sr`), `dylu`, and
`num_fragments` — are **not** forwarded via env. They are server-authoritative:
the worker fetches them from `/info` at startup so the whole group shares one
format. (`DILOCO_BF16_COMM` is a legacy single-boolean fallback for pre-#130
servers that don't advertise the four keys.)

The training script reads these environment variables and constructs a
`DiLoCoWorker` internally. This keeps the CLI layer thin and avoids
reimplementing training logic.

---

## Testing

### Test organization

| File | Focus | Approach |
|------|-------|----------|
| `test_server.py` | Outer optimizer math, serialization | Direct method calls, no HTTP |
| `test_server_client.py` | HTTP round-trip | Real `ThreadingHTTPServer`, real `DiLoCoClient` |
| `test_worker.py` | Pseudo-gradient computation, hook lifecycle | Full server + worker integration |
| `test_async.py` | Async mode, DN momentum, DyLU | Multi-threaded workers against real server |
| `test_streaming.py` | FragmentManager, fragment endpoints, streaming worker | Unit + integration |
| `test_fault_tolerance.py` | Health monitor, worker death, barrier release, reconnection | Unit + integration |

### Test patterns

**Server fixture:** Most integration tests create a `DiLoCoServer` with
`start()` (background thread), run their test, then `stop()` the server. The
server auto-selects a port to avoid conflicts.

**TinyModel:** Tests use minimal models (2 `nn.Linear` layers, dim=4-8) to keep
tests fast. The model is the same one used to initialize the server, ensuring
parameter names match.

**Simulated training:** Tests simulate training by directly modifying model
parameters (`p.data.sub_(0.01)`) or running `optimizer.step()` with synthetic
gradients. No actual data loading or forward passes.

**Multi-worker sync:** Tests spawn multiple workers as separate threads or
sequential submissions to the same server. For synchronous tests, threading is
required because each worker's `submit_pseudogradients` blocks until all workers
have submitted.

### Running tests

```bash
# All DiLoCo tests (102 tests)
pytest tests/unit/ml/diloco/ -v

# By phase
pytest tests/unit/ml/diloco/test_server.py tests/unit/ml/diloco/test_server_client.py tests/unit/ml/diloco/test_worker.py -v  # Phase 1 (32)
pytest tests/unit/ml/diloco/test_async.py -v               # Phase 2 (18)
pytest tests/unit/ml/diloco/test_streaming.py -v            # Phase 3 (25)
pytest tests/unit/ml/diloco/test_fault_tolerance.py -v      # Phase 4 (27)

# Quick smoke test
pytest tests/unit/ml/diloco/test_server.py::TestOuterOptimizer::test_single_worker_outer_step -v
```

---

## Troubleshooting

### Worker hangs at sync (synchronous mode)

**Symptom:** One or more workers block at `submit_pseudogradients` for a long
time.

**Cause:** The server barrier waits for all expected workers to submit. If a
worker crashes and health monitoring is disabled (or timeout is too long), the
remaining workers wait until the 600-second HTTP timeout.

**Diagnosis:**
1. Check server status: `forgather diloco status --server host:port`
2. Look at `pending_submissions` in the response. If it lists some workers but
   not all, a worker has failed to submit.
3. Check `total_worker_deaths` to see if the HealthMonitor has already
   evicted the dead worker.
4. Check server logs for health monitor warnings.

**Mitigation:** Ensure `--heartbeat-timeout` is set (default: 120s). The
HealthMonitor will detect dead workers and release the barrier within
approximately one timeout period. The `min_workers` setting prevents the
system from continuing with zero workers.

### Server port already in use

**Symptom:** `OSError: [Errno 98] Address already in use`

**Cause:** Previous server didn't shut down cleanly (socket in TIME_WAIT).

**Fix:** Either wait ~60 seconds, use a different port, or set
`SO_REUSEADDR` (not currently done). When `port=None`, the server
auto-selects an available port starting at 8512.

### Pseudo-gradients all zeros

**Symptom:** Training makes no progress, global params don't change.

**Cause:** The worker model isn't actually training (no gradients flowing).

**Diagnosis:** Check `_local_step` is incrementing. Check that `_sync()` is
being called (look for "starting sync" log messages). Check that the optimizer
hook is installed (verify `len(diloco._hooks) > 0`).

### BFloat16 precision issues

**Symptom:** Numerical differences after sync, especially with very small
parameter values.

**Cause:** BFloat16 has ~3 digits of precision. Very small pseudo-gradients
(difference between global and local params) may be rounded to zero under
round-to-nearest, biasing the cast in a consistent direction across rounds.

**Mitigation (preferred):** enable **stochastic rounding** on the affected leg —
`--upload-sr` for the worker→server pseudo-gradient and/or `--download-sr` for
the server→worker averaged params (only meaningful with the corresponding
`--*-dtype bf16`). SR keeps the fp32→bf16 cast unbiased in expectation, so
sub-ULP signal survives without giving up the bandwidth saving.

**Mitigation (fallback):** drop the affected leg back to full precision —
`--upload-dtype fp32` (the deprecated `--no-bf16` alias) and/or the default
`--download-dtype fp32`. This doubles that leg's bandwidth. All four wire knobs
are server-authoritative (the whole group shares one wire format) and adopted
from `/info`; there are no worker flags.

### Fragment sync deadlock

**Symptom:** Workers hang when the server runs with `--num-fragments > 1` in
sync mode.

**Cause:** Per-fragment barriers require all workers to submit the same fragment
in the same round. Misaligned `sync_every` or `num_fragments` across workers
would break this.

**Requirement:** All workers in synchronous fragment mode must use the same
`sync_every` and `num_fragments`. This is now guaranteed automatically: both
are server-authoritative and adopted by every worker from `/info`, so they
cannot diverge. Set them on the server (`--sync-every`, `--num-fragments`).

### Async staleness drift

**Symptom:** Training loss oscillates or diverges in async mode.

**Diagnosis:** Check staleness values in server logs. Staleness > 5-10 may
indicate that pseudo-gradients are too stale to be useful.

**Mitigation:**
1. Enable DN momentum (`--dn-buffer-size N` where N = num_workers)
2. Enable DyLU to equalize submission rates
3. Increase `sync_every` so each submission is more meaningful

### Memory: server accumulates state

**Symptom:** Server memory grows over time.

**Cause:** `_completed_rounds` and `_completed_fragment_rounds` cache results.
These are pruned to keep only 2 most recent entries per round/fragment, but if
many fragments are in play, the cache can grow.

**Check:** In practice, cache entries are `O(num_fragments)` dicts of parameter
tensors. For a 1B model with 7 fragments, each entry is ~4 GB (float32). The
cache holds at most 2 entries per fragment, so worst case is
`2 * 7 * 4 GB = 56 GB`. For large models, consider reducing num_fragments.

---

## Extension Points

### Adding a new outer optimizer

Pass a custom factory function:

```python
server = DiLoCoServer(
    model_state_dict=sd,
    num_workers=2,
    outer_optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.001),
)
```

The server calls `factory(self._param_list.parameters())` once at init. The
optimizer's `state_dict` is included in saves/loads automatically.

For CLI support, modify `_server_cmd` in `diloco.py` to add new `--outer-*`
flags and build the factory accordingly.

### Adding a new server endpoint

1. Add a handler method `_handle_foo(self, handler)` on `DiLoCoServer`
2. Register it in `DiLoCoRequestHandler.do_POST` or `do_GET` (in
   `_create_handler()`)
3. Add a corresponding method on `DiLoCoClient` using `_request_json` or
   `_request_tensor`

### Adding new communication compression

Currently, bf16 casting happens in the worker (`_compute_pseudogradients`). To
add quantization (e.g., int8, sparse encoding):

1. Modify `_compute_pseudogradients` to apply the compression
2. Modify the server's deserialization to decompress
3. Alternatively, implement as a custom serialization format that replaces
   `torch.save` payloads with a compressed format

### Integrating with Forgather callbacks

A future callback integration would:

1. Create a `DiLoCoCallback` implementing `TrainerCallback`
2. In `on_train_begin`: create and start `DiLoCoWorker`
3. In `on_train_end`: stop the worker
4. In `on_log`: report `diloco.sync_metrics` to the logger

The `DiLoCoWorker` context manager and optimizer hook design makes this
straightforward -- the callback just manages the worker lifecycle.

### Adding P2P allreduce (replacing server)

The current architecture is client-server. To add peer-to-peer allreduce:

1. Create a new sync backend (e.g., `allreduce.py`) that replaces
   `DiLoCoClient.submit_pseudogradients` with a collective allreduce
2. The `DiLoCoWorker` would accept a backend abstraction instead of a
   `DiLoCoClient` directly
3. The outer optimizer would run on each worker locally (all workers compute the
   same average pseudo-gradient, so they'd arrive at the same global params)

---

## Checkpoint state selection + empty-meta construction

A run selects which checkpoint state components it saves/loads via
`TrainingArguments.checkpoint_components`, and a trainer can build the model
**empty on the meta device** when the weights are supplied by an external
authority rather than a checkpoint. DiLoCo uses both: the parameter server
owns the weights, so a worker builds empty-on-meta and checkpoints its
**non-model** training state only.

Why a worker must not checkpoint model weights: the server is the sole
weight authority (workers pull global params at register), so a saved local
copy wastes disk and risks loading a *stale* one from a different sync
round. Trainer progress (step / LR position / RNG) is the state worth
keeping; inner-optimizer state is optional. And building the empty skeleton
on meta is allocation-free versus an on-device build that the sync
immediately overwrites.

### Mechanism

The unifying invariant: *under DiLoCo the server owns the model weights.*
One config knob expresses it and drives both behaviors:

1. **Configurable checkpoint components.** `checkpoint_components`
   (`list[str] | None`, `None` = all) selects which components a run
   saves/loads. `BaseTrainer.get_active_state_components()` calls the
   subclass's `get_state_components()` and filters it by that field. The live
   consumer — `CheckpointManager.__init__` — calls the filtered accessor (via
   `getattr`, so a provider lacking it falls back to the unfiltered set; the
   `checkpoint_coordinator.py` "usage example" is only a docstring). This
   covers all five `get_state_components()` implementations (`base`, `ddp`,
   `fsdp2`, `accel`, `pipeline`) without editing any of them: filtering
   removes the `"model"` component entirely, so `model_state_component` is
   simply `None`. A key outside the known vocabulary
   (`KNOWN_CHECKPOINT_COMPONENTS`) raises — a misspelled `"model"` must not
   quietly convert a normal run into a weights-external one — while a known
   key a given run doesn't produce is allowed and ignored.
2. **Model save/load gated on the component.** With `"model"` excluded,
   `model_state_component is None`, and the CheckpointManager skips both
   `_save_model` and `_load_model_from_checkpoint`, instead dropping a
   `MODEL_EXCLUDED_MARKER` sentinel. `validate_checkpoint` accepts a
   model-less checkpoint **only** when that marker is present, so such
   checkpoints remain discoverable for resume while a model-less *normal*
   checkpoint (missing weights, no marker — a partial/corrupt save) stays
   invalid and discovery falls back to an older complete one.
3. **Construction derives from the component set; external load is a hook.**
   `_model_weights_external()` is true when `"model"` is excluded. The model
   is then built empty on meta (forced meta, no downgrade — no second knob,
   because "model not checkpointed" *is* the "weights come from elsewhere"
   signal), and `_restore_from_checkpoint` runs the **uniform load → init**
   sequence: it loads any resume checkpoint's non-model components, dispatches
   the **`on_load_model_weights`** callback event for the external weights,
   then runs initialize-missing. DiLoCo implements that event — its worker
   registers and applies the server's global params, **flagging** them
   `_is_hf_initialized`, exactly as a checkpoint load flags loaded tensors —
   so initialize-missing fills only what neither source provided (the
   non-persistent buffers, e.g. RoPE `inv_freq`). This is why DiLoCo
   registration moved from `on_train_begin` to the hook: it now happens where
   weights are loaded, before the init pass, so there is no full-init-then-
   overwrite. The pipeline trainer (always meta) skips its rank-0 full-CPU
   build/distribute (`_initialize_params`) in the external case — that
   expensive last-resort path is exactly what loading weights exists to avoid;
   the per-stage initialize-missing recomputes the buffers.
4. **DiLoCo defaults, overridable.** `lm_training_project.yaml` sets, under
   DiLoCo, `construct_model_on: meta` and `checkpoint_components: [optimizer,
   scheduler, trainer, rng]`; a child template / leaf overrides via the
   `checkpoint_components` var. `"model"` is excluded (server-owned weights)
   and so is `"dataset"` — dataset position is tracked by the server via
   work-units, not the local dataloader, so a local dataset checkpoint would
   be stale on resume. The inner-optimizer keep/skip question is therefore a
   config choice (include `"optimizer"` or not), not a hard-coded policy.

### Code map

- `TrainingArguments.checkpoint_components` (`trainer.py`);
  `BaseTrainer.get_active_state_components()` + `KNOWN_CHECKPOINT_COMPONENTS`
  (`base_trainer.py`).
- `CheckpointManager`: filtered accessor at construction; `_save_model` /
  `_load_model_from_checkpoint` gated on `model_state_component is not None`;
  `MODEL_EXCLUDED_MARKER` written when the model is excluded.
- `validate_checkpoint` + `MODEL_EXCLUDED_MARKER` (`sharded_checkpoint.py`).
- `Trainer._model_weights_external()` + `_prepare_model` /
  `_restore_from_checkpoint`; the latter dispatches `on_load_model_weights`
  (guarded by `BaseTrainer._has_event_handler`, fail-loud if no loader) and
  then `_verify_external_weights_loaded()` before initialize-missing.
  `_materialized_modules()` enumerates the on-device module(s) (overridden by
  `PipelineTrainer` → `pipeline_modules`); `PipelineTrainer._prepare_model`
  init condition.
- `on_load_model_weights` event (documented in `TrainerCallback`).
  `DiLoCoCallback.on_load_model_weights` registers the worker, applies the
  server's global params, and flags them; `on_train_begin` is a defensive
  assert that the hook ran (forgather-only callback, so it always should).
- Tests: `tests/unit/ml/test_checkpoint_components.py` (filter, external
  signal, handler-presence, weights-loaded verification) and the empty-meta
  build in `test_meta_checkpoint_load.py`.

## Known Limitations

1. **Single-threaded outer optimizer.** The server applies the outer optimizer
   step in the HTTP handler thread. For very large models, this could delay
   response time.

2. **Fragment split by parameter count, not size.** Two fragments may have very
   different total tensor sizes if parameter dimensions vary (e.g., embedding
   layer vs attention layers). A size-balanced split would improve streaming
   overlap.

3. **No gradient compression beyond bf16.** Int8, sparse, or top-k compression
   could further reduce bandwidth for larger models.

4. **DN direct gradient step uses single LR.** The `_outer_lr` is extracted
   from the first param group. Multiple param groups with different LRs would
   need per-group direct steps.

5. **No per-worker weighting.** All workers' pseudo-gradients are equally
   averaged. Workers with more data or better hardware could be weighted
   proportionally.

6. **`ThreadingHTTPServer` scalability.** One thread per request is fine for
   2-10 workers but would need replacement (asyncio, gRPC) for hundreds.

7. **No fragment-level reconnection.** Worker reconnection (`_reconnect()`)
   re-registers and fetches full global params. If a streaming sync was
   in-flight when the connection dropped, the fragment result is lost and the
   fragment re-syncs from scratch on the next cycle.
