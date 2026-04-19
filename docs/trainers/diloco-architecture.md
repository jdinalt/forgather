# DiLoCo Architecture & Maintainer Guide

This document describes the internal architecture of Forgather's DiLoCo system.
It is intended for developers who need to understand how the system works,
troubleshoot issues, or implement new features.

For user-facing documentation (CLI usage, quick start, API examples), see
[diloco.md](diloco.md).

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
  __init__.py        Exports: DiLoCoServer, DiLoCoClient, DiLoCoWorker, FragmentManager, HealthMonitor
  server.py          HTTP server, outer optimizer, sync barrier, fragment handling, fault tolerance
  client.py          HTTP client, tensor serialization, request construction, retry logic
  worker.py          Optimizer hook, pseudo-gradient computation, streaming, reconnection
  fragments.py       FragmentManager: parameter splitting, scheduling
  health.py          HealthMonitor: background worker liveness detection

src/forgather/cli/
  diloco.py          CLI command handlers (_server_cmd, _status_cmd, _worker_cmd)
  diloco_args.py     Argument parser (create_diloco_parser)

tests/unit/ml/diloco/
  test_server.py          Server: outer optimizer correctness, serialization
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
(server round at last sync), `steps_per_second`, `extra`.

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
| POST | `/heartbeat` | JSON: `{worker_id, steps_per_second}` | JSON: `{status, sync_round, recommended_sync_every?}` |
| POST | `/deregister` | JSON: `{worker_id}` | JSON: `{status: "ok"}` |
| GET | `/status` | (none) | JSON: server state |

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

---

## Threading Model

### Server threads

The server uses `ThreadingHTTPServer` which spawns a new daemon thread for each
incoming HTTP request. This is required because in synchronous mode, multiple
worker requests block concurrently waiting at the barrier.

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
}
```

`load_state(path)` restores parameters and optimizer state. Note that
`weights_only=False` is used for loading because the optimizer state dict
contains non-tensor values.

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
DILOCO_SYNC_EVERY   -> sync interval
DILOCO_BF16_COMM    -> "0" or "1"
DILOCO_DYLU         -> "0" or "1"
DILOCO_HEARTBEAT_INTERVAL -> seconds
DILOCO_NUM_FRAGMENTS -> number of fragments
DILOCO_WORKER_ID    -> optional worker ID
```

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
(difference between global and local params) may be rounded to zero.

**Mitigation:** Disable bf16 communication with `--no-bf16` or
`bf16_comm=False`. This doubles bandwidth usage.

### Fragment sync deadlock

**Symptom:** Workers hang when using `--num-fragments > 1` in sync mode.

**Cause:** Per-fragment barriers require all workers to submit the same fragment
in the same round. If workers have different `sync_every` values (e.g., from
DyLU) or different `num_fragments`, their fragment schedules won't align.

**Requirement:** All workers in synchronous fragment mode must use the same
`sync_every` and `num_fragments`.

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
