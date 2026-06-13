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
- [Lifecycle and Data Flow](diloco-architecture-runtime.md#lifecycle-and-data-flow)
- [Fault Tolerance](diloco-architecture-runtime.md#fault-tolerance)
- [Server State Persistence](diloco-architecture-reference.md#server-state-persistence)
- [CLI Layer](diloco-architecture-reference.md#cli-layer)
- [Testing](diloco-architecture-reference.md#testing)
- [Troubleshooting](diloco-architecture-reference.md#troubleshooting)
- [Extension Points](diloco-architecture-reference.md#extension-points)
- [Checkpoint state selection + empty-meta construction](diloco-architecture-reference.md#checkpoint-state-selection--empty-meta-construction)
- [Known Limitations](diloco-architecture-reference.md#known-limitations)

> This maintainer guide is split across three pages: this **design** page
> (data structures, wire protocol, threading, sync modes, streaming, outer
> optimizer), the [**runtime behavior**](diloco-architecture-runtime.md) page (lifecycle &
> data flow, fault tolerance, statistics), and the
> [**maintainer reference**](diloco-architecture-reference.md) page (state persistence, CLI,
> testing, troubleshooting, extension points, checkpoint/meta-init,
> known limitations).

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
  wire_cast.py       cast_for_upload: the HTTP backend's upload wire cast (fp32/bf16 + SR)
  wire_serialize.py  Bulk payload codec: pickle | safetensors, the shared serialize seam
  bulk_transport.py  BulkBytesTransport seam + HttpBytesTransport (the client's byte round-trip)
  grpc_transport.py  GrpcBytesTransport: the client's gRPC bulk transport
  grpc_bulk.py       Server gRPC servicer + listener + _CapturingHandler (reuses HTTP handlers)
  proto/bulk.proto   gRPC bulk service; generated bulk_pb2*.py (proto/generate.sh)
  shared_memory_region.py  ShmRegion: shared on-disk region mechanics (layout, flock, header, manifest)
  shared_memory_aggregator.py  SharedMemoryAggregator: server-side region owner + outer step
  shared_memory_backend.py  SharedMemoryBackend: worker-side follower (single-host shared-memory OuterSyncBackend)
  collective_backend.py  CollectiveBackend: all-reduce + replicated outer optimizer
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
  test_shared_memory_aggregator.py  SharedMemoryAggregator: server-aggregates-followers, dynamic barrier, crash-reclaim
  test_shared_memory_backend.py  SharedMemoryBackend follower: outer-step equivalence + fail-loud guards
  test_shared_memory_resume.py  shared-memory checkpoint/resume coherence (#197/#198)
  test_collective_backend.py  CollectiveBackend: multi-process all-reduce outer-step + cross-rank identity
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
[Unified statistics](diloco-architecture-runtime.md#unified-statistics)).

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
_async_lock: threading.RLock                           # Serializes async submissions
_total_submissions: int                                # Total submissions received
_dn_delta: List[Tensor]                                # Delayed Nesterov running sum (per param)
_dn_momentum: List[Tensor]                             # Delayed Nesterov momentum (per param)
_dn_count: int                                         # Submissions since the last DN momentum step
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

The control plane uses HTTP/1.1 over TCP — the server runs a
`ThreadingHTTPServer` (one thread per request). The bulk tensor legs use the same
HTTP listener by default, or an optional streaming gRPC listener (`--grpc`); see
[gRPC bulk transport](#grpc-bulk-transport-optional).

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
adopted by every worker. The **upload** cast happens worker-side in the sync
backend (`HttpStarBackend`, via `wire_cast.cast_for_upload`) just before
`/submit_pseudograd`; the **download** cast happens server-side via
`_cast_for_download` on the params returned by `/register`,
`/submit_pseudograd`, and `/submit_fragment_pseudograd`. Each cast optionally
uses stochastic rounding (`fp32_to_bf16_stochastic_round`).
The server's master parameters and outer-optimizer state always stay fp32;
incoming tensors are accumulated in fp32 regardless of wire dtype. The legacy
`bf16_comm` boolean is retained as a deprecated alias for `upload_dtype`.

`/known_workers` returns every `worker_id` the server has ever seen — the
roster `self._known_workers`, persisted with the server's checkpoints (see
[Server state persistence](diloco-architecture-reference.md#server-state-persistence)).
Each entry carries the worker's
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
[remaining bytes: serialized state-dict payload]
```

The JSON header contains `worker_id`, optionally `fragment_id`, and `fmt` — the
wire codec for this request's payload (see below). The payload is a serialized
`Dict[str, torch.Tensor]`.

### Tensor serialization (wire codec)

The bulk payload codec is one of two, negotiated server-authoritatively via
`/info` (`wire_format`, default `pickle`) and shared by both legs;
`wire_serialize.py` is the single seam the client and server both delegate to:

- **`pickle`** — `torch.save` / `torch.load` (`map_location="cpu"`,
  `weights_only=True`). The historical format; dtypes ride implicitly inside the
  pickle.
- **`safetensors`** — `safetensors.torch.save` / `load`. No pickle (no
  arbitrary-code deserialization), an explicit dtype/shape header on the wire,
  zero-copy load, and the *same* format as on-disk checkpoints.

The **upload** stamps its codec in the frame's `fmt` header so the server decodes
each request regardless of its own setting; the **download** response carries no
header, so the worker decodes it with the `wire_format` it adopted from `/info`.
Absent `fmt` / `wire_format` ⇒ `pickle`, so an older peer stays interoperable.
Response payloads use `Content-Type: application/octet-stream`.

### gRPC bulk transport (optional)

With `--grpc` the three bulk legs are served over a streaming gRPC (HTTP/2)
listener instead of the HTTP control port, advertised via `/info`
(`transport: "grpc"` + `grpc_endpoint`). The worker negotiates it
(`GrpcBytesTransport`), falling back to HTTP when a server doesn't offer it; the
control plane stays on HTTP. The same `[len][header][blob]` frame is chunked over
the request/response streams (so the wire-codec negotiation is unchanged), and
the server-side servicer reuses the **unmodified** HTTP submit/barrier handlers
via an in-memory `_CapturingHandler` (`grpc_bulk.py`) — it reassembles the
chunks, drives the handler, and captures the framed response + status, which it
maps back to gRPC (200 → streamed bytes; 4xx/5xx → the corresponding gRPC
status). The blocking barrier works unchanged in gRPC's thread pool. gRPC
**supersedes** the cleartext bulk listener (one bulk fast-path).

**Security (`_grpc_security`)** follows the control-plane TLS posture: a TLS
server builds gRPC `ssl_server_credentials` from the *same* cert/key (plumbed to
the server as file paths — gRPC needs PEM, not a Python `SSLContext`), so the
bulk plane is encrypted and server-authenticated; a cleartext server runs gRPC
cleartext (trusted-LAN). The worker authenticates by **bearer over the TLS
channel** (`authenticate_grpc_context` checks the `authorization` metadata;
`GrpcBytesTransport` sends it only over a secure channel). Unlike the HTTP control
plane's mTLS-or-bearer (`ssl.CERT_OPTIONAL`), gRPC TLS has no `CERT_OPTIONAL`
equivalent — a client cert is only verified/exposed under
`require_client_auth=True`, which would reject every non-cert client at the
handshake — so the worker-only bulk plane authenticates by bearer (the worker
always holds the per-port token). The client mirrors the posture from the control
scheme: an `https` control URL ⇒ a secure gRPC channel (CA bundle as the trust
root); `http` ⇒ cleartext.

The Python stubs (`proto/bulk_pb2*.py`) are generated from `proto/bulk.proto` and
committed; regenerate with `proto/generate.sh`.

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

`HttpStarBackend` is the reference implementation: a thin adapter over
`DiLoCoClient` providing the HTTP central-parameter-server transport, held by the
worker by default. Two non-HTTP siblings exist — `SharedMemoryBackend` (a CPU
master region) and `CollectiveBackend` (an all-reduce + replicated outer
optimizer), both detailed below. The seam lets the transport be any of these
without changing the worker's `compute → synchronize → apply → broadcast` flow.

The seam is the outer step, not a byte channel: `synchronize` takes a raw
pseudo-gradient and returns a `SyncResult` carrying the agreed next global
params (and the on-wire `sent_bytes` / `recv_bytes`), so a backend may run the
outer optimizer centrally (HTTP), in a replicated copy per worker, or in place on
a shared region — the worker is agnostic. Backends advertise capability flags
(`runs_outer_optimizer`, `supports_async`, `fault_tolerant`) that callers honor
rather than assume. Only the tensor legs are pluggable; the coordination plane is
a separate surface (see *Coordinator surface* below).

The backend also owns its **wire representation**: `ParamView` returns the raw
pseudo-gradient (`snapshot - local`, in the live model dtype) and the upload cast
to `upload_dtype` (with optional SR) is applied in `HttpStarBackend`, so a future
backend can use a different representation (no cast for shared-memory, packed
fp8/fp4 for a collective) without touching `ParamView`. The server-side download
cast is unchanged.

### Coordinator surface

Coordination — everything that is not the bulk tensor exchange — is reached
through a `CoordinatorClient` (`coordinator.py`), distinct from the sync backend.
The worker holds both a `backend` (parameter authority / transport) and a
`coordinator`, so a backend whose parameter authority is not the coordinator (a
serverless collective, a shared-memory region) can coordinate over the HTTP
server while exchanging params elsewhere.

`CoordinatorClient` is a concrete thin facade over `DiLoCoClient` — coordination
always speaks HTTP regardless of which backend moves the tensors, so it is not an
ABC (unlike `OuterSyncBackend`). It covers the worker-process bring-up
coordination: `heartbeat`, `get_info` (`/info` negotiation), and `fetch_model_def`
(model staging). Two other coordination surfaces use their own `DiLoCoClient`
instances and are outside this one: work-unit dispatch (`register_dataset` /
`request_work` / `complete_work`), owned by the dataset layer, and control
(`relay_command` / `save_state` / `shutdown` / `get_status`), owned by the CLI.

### Shared-memory backend

The single-host regime that positions DiLoCo as a DDP alternative. Co-located
processes on one machine share a CPU master-weights region (a memory-mapped file
under a per-group dir) instead of round-tripping the HTTP star. Because the region
is single-host and the param server is co-located, the **server maps the same
region and *is* the aggregator** — shared-memory is a transport swap of the HTTP
star, not a second aggregation owner. Three pieces:

- **`ShmRegion`** (`shared_memory_region.py`) — the on-disk mechanics shared by
  both sides so they never disagree on the format: the byte layout, the
  cross-process `flock` (`region.lock`), the int64 control header (magic,
  generation, arrivals, group size, attach count), the manifest, and the fp32
  master/accumulator views.
- **`SharedMemoryAggregator`** (`shared_memory_aggregator.py`, server side) —
  holds the region's ownership lease (an exclusive `flock` on `owner.lock` for the
  server's lifetime), creates + seeds the region from the server's master, and
  each round waits for the followers to contribute, hands the averaged
  pseudo-gradient to the server's outer step (`step_fn`), and publishes the new
  master. The server reuses its own master `ParameterList` + outer optimizer +
  `save_state`/`load_state`, so the optimizer momentum is persisted and
  `sync_round` advances — which is what makes a shared-memory run checkpoint and
  resume coherently. The lease makes re-launch after a crash safe: a region
  orphaned by a dead server has no live lease holder, so the next launch reclaims
  and rebuilds it.
- **`SharedMemoryBackend`** (`shared_memory_backend.py`, worker side) — a pure
  follower built on `ShmRegion`. It never creates a region or self-elects; it
  waits for the server's region, attaches, contributes its raw pseudo-gradient
  (upcast to fp32) into the shared accumulator under the `flock`, and reads back
  the master the server publishes. A worker omitting a name fails loud (the
  average divides by the contributor count); `sent_bytes`/`recv_bytes` are 0.

The barrier is **dynamic**, mirroring the HTTP path's `_round_expected_workers`:
the server aggregates once arrivals reach the live-follower count (the region's
attach count minus the server's own +1), but only *after* the full configured
group has formed (a high-water mark), so a slow-to-arrive worker can't trigger a
partial-group step and a follower that `leave()`s during the drain shrinks the
expectation instead of deadlocking the rest. If the server's aggregation loop
dies it clears the region magic so parked followers fail loud at once. The HTTP
coordinator role (membership, `/info`, heartbeat, work-unit dispatch) is
unchanged. The backend is selected by `DILOCO_BACKEND=shared_memory` (derived at
launch by the scheduler from the server's `/info`); the region `group_dir` +
`group_size` are advertised by the server in `/info` (`shm_group_dir` /
`shm_group_size`, the server's stable configured worker count) and read by the
follower — not derived from a mutable count or a submit-time flag.

### Collective backend

`CollectiveBackend` (`collective_backend.py`) is the first *collective*
`OuterSyncBackend` — the other single-host DDP-alternative path. Every worker is
an **independent DiLoCo replica** (its own data shard, no per-step DDP gradient
all-reduce) sharing one `torch.distributed` process group. `runs_outer_optimizer`
is `"replicated"`: each round, every rank `all_reduce(SUM)`s its raw
pseudo-gradient in a fixed name order, divides by the group size to get the mean,
sets it as the `.grad` on a private CPU fp32 master `ParameterList`, and steps an
**identical outer optimizer** — reproducing `_apply_outer_optimizer` exactly. The
stepped master is the new global; because every rank reduced the same inputs and
stepped an identical optimizer over identical weights, the results are
bit-identical across ranks, so `synchronize` returns them **without a result
broadcast** (a `broadcast_result` flag is a one-line safety valve). `join` has
rank 0 load the init checkpoint and broadcast it so every replica starts
identical — the precondition for the replicated step staying in lockstep. The
collective runs on the group's native device (CUDA for NCCL, the NVLink fast
path; CPU for gloo); the master + optimizer stay on CPU.

The backend borrows its process group from the launcher and never creates or
destroys one. That group is the **`diloco` axis of a device mesh**:
`ForgatherParallelDims` (`distributed_mesh.py`) splits the torchrun world into a
`(diloco, inner)` mesh (`init_device_mesh((diloco, inner), ("diloco",
inner_axis))`), and `DistributedEnvironment._apply_diloco_split` (driven by
`DILOCO_REPLICATE`) reports the **inner** view (`world_size`/`rank`) to the
trainer — so the trainer's per-step collectives span one replica only — while
exposing `diloco_group`/`diloco_rank`/`diloco_size` (and, for `inner > 1`,
`inner_mesh`/`inner_group`) for the `DiLoCoCallback._make_collective_backend` to
hand the backend. With `inner == 1` each replica is a single device and the
trainer sees `world_size == 1`. With `inner > 1` and `DILOCO_INNER_AXIS=
pipeline_parallel` the replica is itself a multi-rank **pipeline** — see *Pipeline
composition* below. Data-parallel inner (`diloco × DDP/FSDP`) is rejected at the
split for now (DiLoCo largely replaces DDP); the guard stays until that
composition lands. Modeled on torchtitan's `ParallelDims`.

`fault_tolerant` is `False` (a dead peer hangs the all-reduce — quorum/skip-step
is a follow-up) and `registers_with_coordinator` is `False` (each replica
registers separately for the coordinator's diagnostics; the tensor path is
off-server). The worker treats a replicated backend as **symmetric**: every rank
is its own leader (like a pipeline rank), computes its own pseudo-gradient, and
participates in the all-reduce — the leader-only sync + post-sync DDP broadcast is
skipped. Per-replica identity (output dir, run logs, work-unit data shard, the
distinct worker-id) all derive from one source: the torchrun entrypoint rewrites
`DILOCO_WORKER_ID` to `{base}_r{diloco_rank}` (`diloco_apply_collective_worker_id`)
before the config is preprocessed. Selected by `DILOCO_BACKEND=collective` +
`DILOCO_REPLICATE`; the model must not be DDP-wrapped (with `inner == 1` the
trainer sees `world_size == 1` and never wraps; the callback also fails loud on a
`DistributedDataParallel` model).

#### Pipeline composition (`diloco × pipeline`)

With `DILOCO_INNER_AXIS=pipeline_parallel` the mesh is `(diloco=R,
pipeline_parallel=P)` over one torchrun world of `R×P` ranks: each replica is a
`P`-rank pipeline, and the `R` replicas at the same pipeline position form a
`diloco` sub-group. Rank `diloco_idx*P + pp_idx` therefore sits in pipeline
`diloco_idx` at stage `pp_idx`; its `diloco_group` strides across replicas
(`[pp_idx, P+pp_idx, 2P+pp_idx, …]`) while its `inner_group` is the contiguous
per-replica pipeline (`[diloco_idx*P … diloco_idx*P + P-1]`).

The pipeline trainer consumes the **inner** sub-mesh: `_init_distributed` takes
`self.dist.inner_mesh`/`inner_group` as its `self.mesh`/`self.pp_group` when the
split is active, and every pipeline-internal collective (stage send/recv, the
loss/token/stop relays, gradient-norm all-reduce, the throughput all-gather)
targets `group=self.pp_group` with group-local `group_src`/`group_dst`/
`group_peer` ranks — so a pipeline runs entirely within its replica and never
crosses into another. The dataloader dispatcher is unchanged from a plain pipeline
run: it broadcasts within the inner pipeline (`dp_mesh_dim=None` over the inner
sub-mesh). The diloco axis is **not** a dataloader dimension — per-replica data
divergence is owned by the DiLoCo work-unit dispatch at the dataset level (keyed
on the per-replica `DILOCO_WORKER_ID`, `{base}_r{diloco_rank}`), so each replica
iterates a distinct shard while its `P` pipeline ranks share it.

Each pipeline rank owns only its parameter **slice** (`PipelineParamView`) and
all-reduces *that slice* across its `diloco` sub-group — the outer step runs
per-slice, in parallel across the `P` positions, and the union is the full model.
`CollectiveBackend.join` filters the rank-0 init broadcast to the slice names the
worker advertises in `worker_info["param_shapes"]`, so a pipeline rank's master
covers exactly the names it reduces (for `inner == 1` that is the whole model, a
no-op filter). The combined worker id composes the two id rules: the entrypoint
rewrite gives the per-replica base `{base}_r{R}`, and the callback's pipeline path
appends `_pp{P}`, yielding `R×P` distinct coordinator cells grouped by
`group_id={base}_r{R}`. Selected by `DILOCO_BACKEND=collective` +
`DILOCO_REPLICATE=R` + `DILOCO_INNER_AXIS=pipeline_parallel`, torchrun
`--nproc-per-node = R×P`.

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

A server-side strategy for async mode that delays the outer Nesterov momentum
(Liu et al. 2024, arXiv:2401.09135, Algorithm 3, `c=0`). When `dn_buffer_size = N > 0`:

- **Every submission**: apply an immediate, momentum-free descent step
  `param -= lr * grad / N` and accumulate the running sum `Delta += grad`.
- **Every N-th submission**: refresh the delayed momentum `m <- beta*m + Delta/N`,
  apply `param -= lr * beta * m`, and reset `Delta`.

Only the running sum `Delta` and the momentum `m` are kept (O(model), not
O(N x model)). For `N = 1` this is exactly the plain Nesterov outer step. This
prevents momentum from tracking the direction of stale individual worker
submissions while still accelerating over the N-submission window.

### Dynamic Local Updates (DyLU)

Server-side computation, communicated via heartbeats:

```
H_w = max(1, floor((v_w / v_max) * H_base))
```

Workers report `steps_per_second` in heartbeats. The server computes the
recommended sync interval proportional to the worker's relative speed and returns
it in the heartbeat response. Workers that opt in (`dylu=True`) adjust their
`sync_every` accordingly.

### Grace Period

When `grace_period > 0` (async only), `_handle_submit_grace` replaces the
immediate apply: it is a **soft barrier with a wall-clock timeout** layered on the
async path, reusing the synchronous barrier's shape (an epoch counter, a
per-epoch results dict, timed waits, `notify_all`).

- A submission parks in `_grace_pending` and the HTTP handler thread waits. The
  deadline is `_grace_tau_sync + grace_period`, where `_grace_tau_sync` is the
  **earliest** arrival — it only ever moves earlier, so a steady trickle of
  arrivals cannot push the window out forever; it provably closes.
- Exactly one parked thread (`_grace_driver_running`) owns the deadline wait and
  triggers `_flush_grace_window`; the window also short-circuits when all live
  workers have submitted. The flush aggregates the batch into one mean
  pseudo-gradient, applies **one** outer step (`_apply_async_pseudograd`,
  `submission_count = len(batch)`), snapshots the post-step params into
  `_grace_results[epoch]`, bumps the epoch, and releases all waiters.
- The grace `Condition` shares the `_async_lock` **RLock**, so the flush's
  periodic `save_state` re-entry is safe (the same property the DN apply relies
  on). Worker death (`_handle_worker_death`) and `stop()` notify the condition so
  a parked batch never strands.
- Layering: the grace period aggregates *within* a round; **one grace batch is
  one DN tick** (DN delays momentum *across* rounds). Batch-size stats are on
  `/status`.

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

