# DiLoCo: Distributed Local-SGD Training

DiLoCo (Distributed Local SGD with Communication) enables distributed training
across multiple heterogeneous machines on a standard LAN. Unlike DDP, which
requires high-bandwidth interconnects (NVLink, InfiniBand), DiLoCo reduces
communication by ~500x, making 1 Gig Ethernet practical for multi-machine
training.

> **Running DiLoCo from the CLI?** The canonical, end-to-end, verified
> walkthrough is
> [`examples/tiny_experiments/diloco/`](../../examples/tiny_experiments/diloco/README.md)
> — start there for a guided run (build the model, start the servers, launch
> workers, monitor, stop, resume). This document is the reference: it explains
> the protocol, every setting, and the advanced modes the example doesn't cover.

The system supports two operating modes:
- **Synchronous**: All workers must submit before the server applies the outer
  optimizer. Simple and deterministic.
- **Asynchronous**: Workers submit independently without waiting. Supports
  heterogeneous hardware (different GPU types, different numbers of GPUs per
  machine) with Delayed Nesterov (DN) momentum and Dynamic Local Updates (DyLU).

## DiLoCo documentation map

This page is the user-facing reference (concepts, CLI, programmatic API,
configuration, wire precision, async mode). The rest of the DiLoCo
documentation lives here:

| Document | What it covers |
|---|---|
| **This page** — `docs/trainers/diloco.md` | Concepts, quick start, CLI, programmatic API, server configuration, [wire precision](#wire-precision), async mode |
| [Architecture & Maintainer Guide](diloco-architecture.md) | Internals: wire protocol, server/worker classes, checkpoint + meta-init, threading model |
| [Work-Unit Dispatch](../design/diloco-work-unit-dispatch.md) | How workers shard the training set via server-issued row ranges |
| [Pipeline Groups](../design/diloco-pipeline-groups.md) | DiLoCo + pipeline parallel: per-rank workers and server-aware groups |
| [Security Model](../design/diloco-security.md) | Auth, mTLS, the endpoint trust split, audit log |
| Example — [`tiny_experiments/diloco`](../../examples/tiny_experiments/diloco/README.md) | Canonical end-to-end CLI walkthrough; DiLoCo vs DDP / PostLocalSGD sweep |
| Example — [`tiny_experiments/diloco_lowprec`](../../examples/tiny_experiments/diloco_lowprec/README.md) | Low-precision wire transport (bf16 ± stochastic rounding) experiment sweep |

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

Each sync round moves the full model twice: workers send their **pseudo-gradient**
up to the server (upload), and the server sends the new averaged **parameters**
back down (download). Either leg can be transported in bfloat16 — halving that
leg's bandwidth — and the fp32→bf16 cast can use **stochastic rounding (SR)** to
remain unbiased in expectation. This is governed by four server-authoritative
knobs (see [Wire precision](#wire-precision)); by default the upload is bf16 and
the download is fp32. With `sync_every=500`, a 1B parameter model transfers ~2 GB
every 500 training steps, achieving >97% compute utilization on 1 Gig Ethernet.

| Model Size | BF16 Size | Transfer Time (1 Gbps) | H=500 steps @ 1s/step | Utilization |
|------------|-----------|------------------------|----------------------|-------------|
| 150M       | 300 MB    | 2.4s                   | 500s compute         | 99.5%       |
| 1B         | 2 GB      | 16s                    | 500s compute         | 97%         |
| 7B         | 14 GB     | 112s                   | 500s compute         | 82%         |

### Wire precision

Both transport legs are controlled by four **server-authoritative** knobs (set on
`forgather diloco server`; every worker adopts them from `/info` at registration,
so the whole group shares one wire format). The defaults reproduce the historical
behavior — bf16 upload, fp32 download.

| Knob | Server flag | Default | Effect |
|---|---|---|---|
| `upload_dtype` | `--upload-dtype {fp32,bf16}` | `bf16` | worker→server pseudo-gradient dtype |
| `upload_sr` | `--upload-sr` | off | stochastic-round the fp32→bf16 upload cast |
| `download_dtype` | `--download-dtype {fp32,bf16}` | `fp32` | server→worker averaged-params dtype (`bf16` halves the return leg) |
| `download_sr` | `--download-sr` | off | stochastic-round the fp32→bf16 download cast |

`--no-bf16` is a deprecated alias for `--upload-dtype fp32`. **Stochastic
rounding** (SR) routes the narrowing cast through the same
`fp32_to_bf16_stochastic_round` the bf16 optimizers use, keeping it unbiased in
expectation so sub-ULP signal survives across many rounds (it only applies to an
fp32→bf16 cast; it is a no-op when the source is already bf16).

**Lineage.** Low-precision communication of the **upload** leg (the
pseudo-gradient / outer gradient) is established prior art: *OpenDiLoCo* first
all-reduced it in FP16 "without noticeable performance hit," and *Streaming
DiLoCo* swept the outer-gradient precision through **bf16/fp8/fp4** with "no sign
of performance regression … even at the billion scale." Both compress only the
upload. The **download** leg — broadcasting the *averaged parameters* back in
bf16 — is not covered by that work; the
[`diloco_lowprec`](../../examples/tiny_experiments/diloco_lowprec/README.md)
experiment finds bf16 download (± SR) essentially lossless on a small Llama at
~1B tokens. See [References](#references).

### Bulk transport

How the bulk legs (pseudo-gradients up, averaged weights down) are serialized and
moved is independent of the wire precision above, and likewise server-authoritative
(advertised via `/info`, adopted by every worker).

| Knob | Server flag | Default | Effect |
|---|---|---|---|
| wire codec | `--wire-format {pickle,safetensors}` | `pickle` | `safetensors` drops pickle for an explicit typed, zero-copy frame; same format as on-disk checkpoints |
| transport | `--grpc` | off (HTTP) | serve the bulk legs over a streaming gRPC listener instead of the HTTP control port |

- **`--wire-format safetensors`** removes pickle from the wire (no arbitrary-code
  deserialization) and makes every tensor's dtype/shape explicit. The codec is
  negotiated, so a mixed old/new fleet stays interoperable; the upload also stamps
  the codec per request.
- **`--grpc`** moves the bulk legs onto an HTTP/2 streaming listener (chunked, with
  backpressure), advertised via `/info`; workers negotiate it and fall back to HTTP
  if a server doesn't offer it. It **supersedes** `--bulk-cleartext` (gRPC is the
  single bulk fast-path). The control plane (register / heartbeat / `/info`) stays
  on HTTP. The gRPC listener is currently cleartext/trusted-LAN (like the cleartext
  bulk listener); TLS/mTLS parity is a follow-up, so prefer it on a trusted network
  for now. Best paid off on large models / slow links, where the streaming + framing
  wins matter; for tiny experiments the HTTP default is fine.

## Quick Start

This section is a condensed inline reference. For a guided, verified, end-to-end
run — building the model, starting the Forgather/dataset/DiLoCo servers, launching
workers, monitoring, stopping, and resuming — follow the canonical CLI example at
[`examples/tiny_experiments/diloco/`](../../examples/tiny_experiments/diloco/README.md).

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

Group-wide worker settings (must match across the group, so they live on the
server; every worker adopts them from `/info` — there are no worker flags):
- `--sync-every N`: Local optimizer steps between syncs (H). Default: 500.
  Under `--dylu`, the DyLU base rate is used instead.
- `--num-fragments N`: Streaming-sync fragments every worker splits the model
  into (1 = no streaming). Default: 1.
- **Wire precision** — `--upload-dtype {fp32,bf16}` / `--upload-sr` /
  `--download-dtype {fp32,bf16}` / `--download-sr` set the dtype and stochastic
  rounding of each transport leg (defaults: bf16 upload, fp32 download).
  `--no-bf16` is a deprecated alias for `--upload-dtype fp32`. See
  [Wire precision](#wire-precision).

```bash
# Load a specific checkpoint and save checkpoints to specified directory.
forgather diloco server -o path/to/output --from-checkpoint output_models/my_model/checkpoint-1000 -n 2
```

**Startup banner.** On start the server prints what a worker needs to connect,
mirroring the dataset server:

- The **bearer auth token** (and a ready-to-run `curl` carrying it), plus the
  per-port token file path. Auth is on by default; `--no-auth` prints a warning
  instead, and `--quiet-tokens` (applied automatically by the webui only in
  `--demo` mode — never an operator checkbox) suppresses the value and the
  file path while still confirming auth is enabled.
- The **server URL**. When bound to a wildcard address (`-H 0.0.0.0`), the
  banner shows the host's primary-interface IP rather than `0.0.0.0`, so the
  address is copy-pasteable straight onto worker `--diloco-server` lines.

### 2. Start Workers

On each machine, launch a worker that wraps the normal training command.
Each worker needs a unique `--worker-id` so its output directory doesn't
collide with the others (the project template appends the worker id to
`ns.output_dir` — applies to both the implicit project default and any
explicit `--output-dir` the operator passes, so workers get distinct
per-worker dirs for free):

```bash
forgather submit --diloco \
    --diloco-server 192.168.1.100:8512 \
    --worker-id w0 \
    -p my_project -t train.yaml \
    train
```

Workers launch via `forgather submit --diloco`, which makes DiLoCo an
opt-in dimension on the regular `submit` verb (mirroring the webui's
single submit modal). Pass `--diloco-server <id>` to pin a specific
param-server; it implies `--diloco`, so it works on its own, but the
clear way to request DiLoCo is `--diloco`. With `--diloco` and no
`--diloco-server`, the single running server is auto-picked.

Worker arguments:
- `--diloco`: Opt into DiLoCo mode for this submission.
- `--diloco-server`: Server address as `host:port` (optional; pins a
  specific server and implies `--diloco`). Omitted = auto-pick the single
  running server.
- `--worker-id`: Unique worker identity. Drives the per-worker output-dir
  suffix the project template appends to `ns.output_dir`, and the
  uniqueness key the server enforces on `/register`. Auto-generated when
  omitted but operators typically set it explicitly so logs / output dirs
  are predictable.
- `--heartbeat-interval`: Seconds between heartbeats for speed reporting
  (default: 30). Client-local; validated against the server's
  `--heartbeat-timeout` at startup.
- `--requested-gpus`: GPUs per worker (the same flag single-node submit
  uses). Per-worker devices are picked by the scheduler from the parent
  env's `CUDA_VISIBLE_DEVICES`; a scheduler-submit command doesn't pin
  CUDA devices itself.

**Resuming a worker from its own checkpoint.** A worker's checkpoints live
under its worker-id-suffixed `output_dir` (`ns.output_dir + "_" +
worker_id`). To resume a stopped worker, relaunch it with the **same**
`--worker-id`: the suffix resolves to the same directory and the trainer
picks up that worker's latest checkpoint. The server remembers the names of
every worker that has registered — persisted with its checkpoints, so the
roster survives a server restart.

**Submitting workers from the webui (worker pool).** When a DiLoCo server is
selected in the submit modal, the DiLoCo section shows a *worker pool* instead
of a single `worker_id` field. The pool has two kinds of chips:

- **Stopped workers** — every not-currently-running name on the server's
  roster, each showing the `output_dir` it would resume from. These are
  **enabled by default**: workers are usually stopped because the server was
  restarted, and the normal intent is to bring them all back, each relaunched
  under its old id to resume from its checkpoint. Toggle a chip off to skip it.
- **New workers** — names you add. Type one and click **Add**, or set a count
  and click **Generate** to mint that many random, mutually-unique names (from
  ~100K adjective-species permutations; generated batches never collide with
  names already in the pool). New chips carry an **×** to remove them.

On **Submit**, one job is spawned per enabled stopped worker plus every new
worker — so "spin up N identical workers" or "resume these three and add two
fresh ones" is a single action. An empty pool submits a single auto-named
worker (its id falls back to the queue id), preserving the simple one-worker
flow.

**Server-authoritative settings.** `sync_every`, the four wire-precision knobs
(`upload_dtype`, `upload_sr`, `download_dtype`, `download_sr`), `dylu`, and
`num_fragments` must match across every worker in the group for the sync
barrier / outer step / fragment barriers to be coherent. The server is
their sole authority: the worker fetches them from the server's `/info` at
startup and logs the values it adopted. There are no worker flags for these —
set them on the **server** (`--sync-every`, `--upload-dtype` / `--upload-sr` /
`--download-dtype` / `--download-sr`, `--dylu`, `--num-fragments`).

Dataset partitioning across workers is handled by the server's **work-unit
dispatch**: each worker registers its train dataset with the DiLoCo server
on first iteration and pulls per-unit row ranges on demand, so no row is
trained on twice within an epoch. There's no operator-facing toggle —
dispatch is active whenever `DILOCO_SERVER` is set on the worker process.
See the *Work-unit dispatch* section below.

### 3. Monitor

```bash
forgather diloco status --diloco-server localhost:8512            # one-shot
forgather diloco status --diloco-server localhost:8512 --watch    # refresh in place
```

Shows sync round, registered workers, their hostnames, training speeds, and
pending sync submissions, plus the outer-optimizer config (`SGD(lr, momentum)`),
the server's save/output dir, and the fault-tolerance thresholds (heartbeat
timeout, min workers). In async mode, also shows total submissions, DN buffer
status, and DyLU configuration. The known-worker line expands into a
**resumable roster** — the not-running worker_ids (with last-seen times) you can
relaunch under to resume from their checkpoints.

Add `--queues` to also show **work-unit dispatch** per `(dataset_id,
shuffle_seed)` queue: a human-readable dataset label (from the worker's
load-args hint, e.g. `wikitext@train`) with the raw `dataset_id` hash kept as a
secondary line, the dataset row count, issued/completed counts with a percent,
and a **per-worker issued/completed breakdown** (the `by_worker` table). The
per-unit heatmap is webui-only — it doesn't translate to a terminal. Pass
`--json` to emit the whole snapshot (status + info + workers [+ queues, with
`by_worker`]) as a single JSON object for scripting / agents.

It also shows **unified training statistics** — a server-level view aggregated
from every worker (see [Unified statistics](#unified-statistics)): total tokens
/ steps / FLOPs, aggregate throughput / MFU / peak memory, and smoothed
train / eval loss. The same block appears in the webui's DiLoCo server view.

`--watch` (`-w`) refreshes the status in place every `--interval` seconds
(default 2.0) until Ctrl-C — like `watch`, but in-process: it reuses the same
connection across ticks (no per-tick subprocess) and works inside the
interactive CLI, where Ctrl-C returns you to the prompt. Not compatible with
`--json`.

## Interacting through the forgather server

These commands route through the forgather server (`forgather server`) by
default — it gives richer, centrally-authenticated access without needing
each parameter server's token on the local machine. The server's proxy
resolves every upstream server's bearer token and TLS verification on your
behalf (the same path the webui uses), so these commands only need the
server's own auth (`~/.config/forgather/server/auth_token`, or
`$FORGATHER_SERVER_TOKEN`). Point at a non-default forgather server with
`--server URL`.

**Picking the DiLoCo server.** The commands that target a DiLoCo server
(`status`, `control`, `shutdown`) take `--diloco-server <id|label|host:port>`,
but it's optional: with exactly one DiLoCo server running it's selected
automatically (the common case). With more than one, you must pass
`--diloco-server` (the error lists the choices). When the forgather server
can't be consulted (e.g. `--local-only`), `--diloco-server` defaults to
`localhost:8512`.

**Locality.** The server is the default, required path: if it isn't
reachable these commands **error** rather than silently doing something
local (a server-coordinated workflow shouldn't be bypassed without asking).
Two opt-outs: `--local-fallback` falls back to a direct/foreground action
when the server is down, and `--local-only` skips the server entirely. There
are only a few corner cases where you want local.

```bash
# Discover every DiLoCo server the forgather server knows. Sources
# surfaced: ``local`` (spawned by this forgather server), ``registered``
# (user-added persistent registry), and ``cluster`` (attested to by a
# peer via the master DiLoCo inventory — automatic cross-node
# discovery in cluster mode). --json for scripting.
forgather diloco servers
forgather diloco servers --json

# Rich status routed through the server (resolves the upstream token for
# you). Falls back to a direct connection if the server isn't running or
# doesn't know this target.
forgather diloco status --diloco-server local:<queue_id> --queues

# Dump or follow the captured TTY of any worker/server job. JOB may be a
# queue_id, a local DiLoCo server id/label, or a worker_id — resolved to
# the underlying job for you.
forgather diloco logs spectacular-fox            # dump
forgather diloco logs spectacular-fox --follow   # live tail
forgather diloco logs spectacular-fox --path     # print the TTY file path
tail -f "$(forgather diloco logs spectacular-fox --path)"  # …or tail it yourself
```

`forgather diloco logs <queue_id>` is a convenience wrapper; the generic
`forgather job tail <queue_id>` / `forgather job dump <queue_id>` work too.

**Cross-node discovery.** When the forgather server is in cluster mode
(every server with `--cluster NAME` reachable on the LAN is a peer), a
DiLoCo server spawned on **any** peer surfaces in `forgather diloco
servers` on **every** peer with `source=cluster`. The master node
aggregates the per-peer view by polling `/api/cluster/diloco_servers_local`
across the cluster, and the webui proxy / CLI resolve the upstream
bearer token from the master snapshot — the operator never copies or
pastes the per-server token. See
`docs/design/diloco-security.md#cross-node-discovery-cluster-inventory`.

```bash
# Register an *external* DiLoCo server — the escape hatch for WAN
# endpoints, SSH tunnels, or anything mDNS can't see on the local
# cluster. The token is stored server-side and used to authenticate
# upstream on your behalf. Servers reachable via the LAN cluster
# don't need this — they surface automatically as ``source=cluster``.
forgather diloco register https://gpu-box:8512 --label prod --auth-token <tok>
forgather diloco unregister registered:<id>     # id shown by `diloco servers`
```

`control` and `shutdown` also route through the forgather server by default
(the relay / save-state / stop actions go through its proxy, so you don't
need the parameter server's token locally). Same locality rules: an
unreachable server errors unless you pass `--local-fallback` / `--local-only`.

### Launching as scheduled jobs

`diloco server` and `forgather submit --diloco` **enqueue scheduled
jobs** by default through the forgather server instead of running in the foreground — the
scheduler picks idle GPUs, captures the TTY, and the jobs show up in the
webui. As above, an unreachable server errors; `--local-fallback` runs
in-process when the server is down, and `--local-only` always runs
in-process (a single foreground worker / server). `--local-only` is also how
the scheduler spawns the actual parameter server, so it doesn't re-enqueue
itself.

```bash
# Enqueue a parameter server (CPU-only); the scheduler starts it.
forgather diloco server -o path/to/model -n 2 --bulk-cleartext

# Launch 4 auto-named workers in one command, each a scheduled training
# job, wired to the cluster's dataset routing. Dynamic/template args work
# exactly like `forgather train` (built from the config's metadata, shown
# in `submit --help`).
forgather -p my_project -t train.yaml submit \
    --diloco --diloco-server local:<queue_id> \
    --diloco-worker-count 4 --dataset auto --max-steps 5000

# Bring a worker set back after a server shutdown / manual stop: re-launch
# every stopped worker the server knows, reusing each id (so each resumes
# from its own checkpoint).
forgather -p my_project -t train.yaml submit --resume-workers --dataset auto
```

Worker launch options (orchestrator path): `--diloco-worker-count N`
(auto-named via the server, guaranteed unique),
`--dataset auto|local|server:<id>`, `--requested-gpus N` (GPUs per
worker — the same flag single-node submit uses), `--priority`. A single
explicit `--worker-id` is honored; `--diloco-worker-count > 1` requires
the server (you can't foreground N). Add `--json` to `server` / `submit`
to get the queue ids back for scripting, or `--dry-run` (a general
`submit` flag — works in single-node, `--global`, and DiLoCo modes) to
print what would be submitted without doing it.

**`--dataset` default is mode-aware**, mirroring the webui Submit-job
modal: when you don't pass `--dataset`, workers default to `auto` (cluster
routing) if the forgather server is in cluster mode, otherwise `local` (the
in-process loader). An explicit value always wins — pass `--dataset local`
to force the in-process loader even in cluster mode. Note that `auto` does
**not** fall back to local: if the cluster has no healthy dataset server for
the requested dataset, the worker retries (server still warming up / none
yet) or fails loudly (warmed up, none can serve it) rather than silently
loading in-process — so a cluster running `auto` must actually have a
dataset server.

`--resume-workers` is a distinct mode: it re-launches every *stopped*
worker the server's known-worker roster reports (deduped on the pipeline
`_pp<N>` suffix), reusing each worker id so it resumes its checkpoint. It
requires the forgather server and can't be combined with `--worker-id` /
`--diloco-worker-count`; it still honors `--dataset` and dynamic args for
the relaunched jobs. (The flag is named `--resume-workers`, not `--resume`,
to avoid clashing with a config's own `--resume` dynamic arg.)

`--resume-workers` currently re-launches *all* stopped workers the server
knows (the original single-set behavior); selectively resuming a subset,
and resume semantics for multi-node worker pools, are future work.

Worker launch and `--resume-workers` currently assume the workers run on the
**same host** as the orchestrator: relaunched jobs are enqueued locally, so a
worker's per-worker checkpoint resume is only correct when it lands back on
the host that holds that checkpoint. Cross-host launch and resume are tracked
in [issue #118](https://github.com/jdinalt/forgather/issues/118).

## Shared-memory backend (single-host)

On a single host, co-located worker processes can exchange the sync tensors
through a **shared CPU master-weights region** instead of the HTTP parameter
server — one shared master per host, no serialization, the outer optimizer
applied in place. This makes DiLoCo a DDP alternative: sync every `H` steps at a
fraction of DDP's per-step all-reduce. The HTTP server stays on as the
**coordinator** (it provides `/info` negotiation and work-unit dispatch); only
the tensor legs move to shared memory, so the workers never submit
pseudo-gradients over the wire (`diloco/last_send_mb` is 0).

Select it per worker with environment variables:

| Variable | Meaning |
|---|---|
| `DILOCO_BACKEND` | `shared_memory` (default `http`) |
| `DILOCO_SHM_GROUP_DIR` | a per-host directory the co-located group shares (the rendezvous) |
| `DILOCO_SHM_GROUP_SIZE` | number of co-located workers in the group |
| `DILOCO_SHM_INIT_CHECKPOINT` | optional; overrides the init checkpoint (default: the dir the coordinator advertises in `/info`) |
| `DILOCO_REPORT_SYNC_STATE` | optional; report per-worker sync-state to the coordinator for diagnostics (default on; set `0`/`false` to omit) |

Shared-memory workers register with the coordinator for membership and report
their sync-state (`sync_count`, send/recv MB, sync time) on the heartbeat, so the
group's progress is exposed in the server's `/status` (for the dashboard/CLI to
surface) even though the tensor exchange is off-server. The server-side per-worker
`sync_round` stays 0 for these workers — they never submit — so the reported
`sync_state.sync_count` is their progress indicator. (`DILOCO_REPORT_SYNC_STATE`
applies to any backend; it is most useful for an off-server one like this, where
the server has no other progress signal.)

The first worker to arrive creates the region and seeds it from the coordinator's
checkpoint; the rest attach. The aggregator also reproduces the coordinator's
outer optimizer (advertised in `/info`), so the group's outer step matches the
server's. The default init checkpoint is the coordinator's local filesystem path,
so the coordinator and workers must share a filesystem (the single-host case);
use `DILOCO_SHM_INIT_CHECKPOINT` if they don't. Each worker is one process (one
GPU) — not a multi-GPU DDP job.

Run a coordinator, then launch the group (one process per GPU):

```bash
# Coordinator (no tensor role for a shared-memory group; provides /info + dispatch)
forgather diloco server --local-only --output-dir <init-checkpoint-dir> \
    --num-workers 2 --sync-every 100 -H 127.0.0.1 --port 8512

# Two co-located workers sharing one region
GROUP=$(mktemp -d)
for w in 0 1; do
  DILOCO_SERVER=https://127.0.0.1:8512 DILOCO_WORKER_ID=shm-w$w \
  DILOCO_BACKEND=shared_memory DILOCO_SHM_GROUP_DIR=$GROUP DILOCO_SHM_GROUP_SIZE=2 \
    forgather -t <config>.yaml train -d $w &
done
wait
```

All co-located workers must agree on `DILOCO_SHM_GROUP_DIR` and
`DILOCO_SHM_GROUP_SIZE`. Streaming-fragment sync (`num_fragments > 1`) is not
supported for this backend. For the internals see
[`diloco-architecture.md`](diloco-architecture.md#shared-memory-backend).

### Via the scheduler (`forgather submit`)

The env-var form above is the manual recipe; the scheduler launches the same
group as a first-class option. Pass `--backend shared_memory` to a DiLoCo
submit and the worker count sizes the group:

```bash
forgather -t <config>.yaml submit --diloco --diloco-worker-count 2 \
    --backend shared_memory
```

The submit mints one group id for the batch; the scheduler derives the per-host
`DILOCO_SHM_GROUP_DIR` (under the host temp dir) and `DILOCO_SHM_GROUP_SIZE` for
every worker, so you don't hand-set the env. The region is created on the first
worker's join and **unlinked when the last worker leaves**, so a completed group
leaves nothing behind. Because the backend is single-host, `--backend
shared_memory` can't be combined with `--global` (the multi-node fan-out).

## Collective backend (single-host DDP alternative)

Where the shared-memory backend exchanges weights through a CPU region, the
**collective** backend exchanges them with a `torch.distributed` **all-reduce**.
Every worker is an *independent DiLoCo replica* — its own data shard, **no
per-step DDP gradient all-reduce** — that, once per `H` steps, all-reduces its
pseudo-gradient with its peers and runs an **identical replicated outer
optimizer** locally. Because every rank reduces the same pseudo-grads to the same
mean and steps an identical optimizer over identical weights, all ranks land on
bit-identical new global params with nothing crossing a central server.

This is DiLoCo as a **DDP alternative**: an all-reduce over NVLink (NCCL) every
`H` steps in place of DDP's per-step all-reduce. As `H` shrinks it approaches DDP
quality at a fraction of DDP's comm. The HTTP server stays on as the
**coordinator** (it provides `/info` negotiation — including the outer-optimizer
config the replicas reproduce — the init-checkpoint reference, and the data
work-unit dispatch that shards rows across replicas; no tensor role).
`diloco/last_send_mb` reflects the all-reduce volume, off-server.

### The replicate (`diloco`) mesh axis

The replicas are a **device-mesh axis**. `DILOCO_REPLICATE=N` splits the torchrun
world into a `(diloco, inner)` mesh: the `diloco` axis is the N replicas (the
collective all-reduces across it); the `inner` axis is whatever the trainer
parallelizes over. The trainer is reported its **inner** view of the world, so its
per-step collectives span one replica only — never across replicas, which is the
whole point. `inner = 1` gives N single-device replicas (the trainer sees
`world_size == 1` and does no gradient sync). `inner > 1` with
`DILOCO_INNER_AXIS=pipeline_parallel` makes each replica a multi-rank **pipeline**
— see [Composing with pipeline parallel](#composing-with-pipeline-parallel).
Data-parallel inner (`diloco × DDP/FSDP`) is rejected for now — DiLoCo largely
replaces DDP. Modeled on torchtitan's `ParallelDims`.

Select it with environment variables (the launch sizes one torchrun world as
`DILOCO_REPLICATE × inner`):

| Variable | Meaning |
|---|---|
| `DILOCO_BACKEND` | `collective` (default `http`) |
| `DILOCO_REPLICATE` | number of replicas on the `diloco` axis (the replicate degree) |
| `DILOCO_INNER_AXIS` | the inner parallelism axis: `data_parallel` (default, `inner` must be 1) or `pipeline_parallel` |
| `DILOCO_WORKER_ID` | base worker id; the entrypoint makes it per-replica (`{base}_r{n}`) so each replica gets its own output dir, run logs, and data shard |
| `DILOCO_INIT_CHECKPOINT` | optional; overrides the init checkpoint (default: the dir the coordinator advertises in `/info`) |
| `DILOCO_REPORT_SYNC_STATE` | optional; report per-worker sync-state to the coordinator (default on) |

The first rank loads the init checkpoint and broadcasts it so every replica
starts from identical weights. Two requirements the regime imposes (both fail
loud where detectable):

- The model must **not** be DDP-wrapped — collective DiLoCo *replaces* DDP's
  gradient sync. (With `inner = 1` the trainer sees `world_size == 1` and never
  wraps, so any `trainer_type` works.)
- The replicas must be **rank-sharded over the data** (each sees different data),
  or they never diverge between syncs and DiLoCo degenerates. This rides on the
  coordinator's work-unit dispatch, keyed by the per-replica `DILOCO_WORKER_ID`.

Run a coordinator, then launch the replicas as a single torchrun world:

```bash
# Coordinator (no tensor role; provides /info + the init checkpoint + dispatch)
forgather diloco server --local-only --output-dir <init-checkpoint-dir> \
    --num-workers 2 --sync-every 100 -H 127.0.0.1 --port 8512

# N replicas in one torchrun world, syncing via all-reduce
DILOCO_SERVER=https://127.0.0.1:8512 DILOCO_BACKEND=collective \
  DILOCO_REPLICATE=2 DILOCO_WORKER_ID=run1 \
  torchrun --standalone --nproc-per-node 2 \
    scripts/train_script.py -p <project-dir> <config>.yaml
```

Every replica syncs at the same `H`-step boundary; the all-reduce is the barrier
(a faster replica waits there for the others).

### Via the scheduler (`forgather submit`)

The env-var form above is the manual recipe; the scheduler launches the same
group as a first-class option. With a coordinator running, pass `--backend
collective` and size the group with `--diloco-replicate`:

```bash
forgather -t <config>.yaml submit --backend collective \
    --diloco-replicate 2 --diloco-server <server-id>
```

Unlike the shared-memory backend (which enqueues N worker jobs), collective is
**one** scheduled job: the scheduler reserves `--diloco-replicate` GPUs, sets
`nproc_per_node` to the same, and derives `DILOCO_BACKEND=collective` +
`DILOCO_REPLICATE` for it (the `DILOCO_WORKER_ID` base is made per-replica at the
entrypoint). Because the backend is single-host, `--backend collective` can't be
combined with `--global`.

### Composing with pipeline parallel

Set `DILOCO_INNER_AXIS=pipeline_parallel` to make each replica a `P`-rank
pipeline. The mesh is `(diloco=R, pipeline_parallel=P)` over one torchrun world of
`R×P` ranks: the `R` replicas at the same pipeline position all-reduce *their
slice* of the model across the `diloco` sub-group, while each replica's `P` ranks
run the pipeline among themselves. Each pipeline rank is its own DiLoCo worker
owning only its parameter slice; the outer step runs per-slice and the union is
the full model. Launch with `--nproc-per-node = R×P`:

```bash
# Coordinator (R*P workers register; each pipeline rank is one worker)
forgather diloco server --local-only --output-dir <init-checkpoint-dir> \
    --num-workers 4 --sync-every 100 -H 127.0.0.1 --port 8512

# R=2 replicas x P=2 pipeline stages = 4 ranks in one torchrun world
DILOCO_SERVER=https://127.0.0.1:8512 DILOCO_BACKEND=collective \
  DILOCO_REPLICATE=2 DILOCO_INNER_AXIS=pipeline_parallel DILOCO_WORKER_ID=run1 \
  torchrun --standalone --nproc-per-node 4 \
    scripts/train_script.py -p <project-dir> <pipeline-config>.yaml
```

The config must select the pipeline trainer (`P` stages). Each replica gets a
distinct data shard (keyed on `{base}_r{replica}`), shared by its `P` pipeline
ranks; the per-replica worker ids are `{base}_r{replica}_pp{stage}`.

The scheduler-driven (`forgather submit`) pipeline sizing, the webui selector, the
streaming-fragment path, and a fault-tolerant quorum (a dead peer currently hangs
the all-reduce) are follow-ups; for the internals see
[`diloco-architecture.md`](diloco-architecture.md#collective-backend).

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
  [Wire precision](#wire-precision).
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
   Each row also has **Save checkpoint** / **Save & Stop** / **Abort**
   controls, and the section header carries **All:** buttons that apply
   the same action to every registered worker at once. All of these go
   through the server's command relay (`/control/command`), so they work
   for every registered worker — including remote ones — without the
   webui needing to reach each worker's trainer-control endpoint.
3. **Server metrics**: outer LR / momentum, worker-death count,
   heartbeat timeout. Sync mode adds a pending-submissions progress
   bar; async mode adds total-submissions, DN buffer status, and DyLU
   state.
4. **Control card**: **Save checkpoint**, **Shutdown** (two-mode
   overlay), live **Optimizer** tuning (LR + momentum + Apply),
   **Workers** expected-count adjustment.

   **Shutdown** is the main path for stopping everything and offers two
   modes (both via the relay, mirroring `forgather diloco shutdown`):
   - **Clean shutdown** (the recommended default): relays **save_and_stop**
     to every worker, waits until they have actually exited (polling the
     server's worker roster), saves a server checkpoint, then stops the
     server. No data loss. The overlay streams progress (a live
     worker-stop count) and, if a worker never stops within the timeout,
     reports it and leaves the server running rather than stranding
     still-live workers. While it is waiting, a **Cancel** button aborts
     the sequence and hands control back immediately (the server is left
     running) so the operator can troubleshoot a worker that won't stop,
     then retry or force.
   - **Force stop**: relays **abort** to all workers (they stop without
     saving) and stops the server without waiting. For "stop it all now,
     don't care about data loss".
5. **Work-unit dispatch**: per-queue heatmap (K cells, three states:
   available / issued / completed), with per-worker counters.

The **same coordinated sequence** runs server-side for every other way a
server is stopped — `forgather diloco shutdown`, a SIGTERM from the scheduler
(stopping/disabling a DiLoCo service, or the Views → DiLoCo run/stop toggle),
and Ctrl-C on a foreground `forgather diloco server`. All of them relay
`save_and_stop` to the workers and let them drain (the server keeps serving so
no worker deadlocks at the barrier) before saving and exiting, rather than
hard-killing the process. Only `force-kill` (SIGKILL) skips this.

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
| `POST /control/command` | `{"command": "save_and_stop", "worker_id": "w0"}` | Relay a trainer-control command to a worker (`worker_id` omitted = all) |
| `POST /control/shutdown` | `{}` | Coordinated shutdown: relay `save_and_stop` to all workers, keep serving while they drain (so none deadlocks on the sync barrier), then save state and stop |

All endpoints return `{"status": "ok", ...}` on success or `{"error": "..."}` on
failure.

#### Command relay

`/control/command` is how the CLI and webui drive a worker's trainer-control
actions (**save_checkpoint** / **save_and_stop** / **abort**) without reaching
each worker's own trainer-control HTTP endpoint. The server queues the command
on the target worker(s) and delivers it on their next **heartbeat**; the DiLoCo
callback then applies it to the trainer loop exactly as the direct
trainer-control endpoint would. Latency is bounded by `--heartbeat-interval`.
For multi-rank workers (DDP / pipeline) only the leader heartbeats, so the
callback `all_reduce(MAX)`-es the command code across ranks each step — every
rank reaches the same save/stop decision and the group can't deadlock on a
divergent stop.

### Controlling workers from the CLI

```bash
# Relay to all registered workers (or one with --worker-id):
forgather diloco control save        --diloco-server host:8512   # checkpoint, keep training
forgather diloco control save-stop   --diloco-server host:8512   # checkpoint, then stop
forgather diloco control abort       --diloco-server host:8512   # stop now, no save

# Stop the whole run. Clean by default: save-stop every worker, wait for them
# to exit, checkpoint the server, then stop it.
forgather diloco shutdown --diloco-server host:8512
forgather diloco shutdown --diloco-server host:8512 --timeout 120   # cap the wait
forgather diloco shutdown --diloco-server host:8512 --force         # stop server now, don't wait
```

A clean `shutdown` that times out waiting for a stuck worker leaves the server
running so you can troubleshoot (re-run it, or use `--force`).

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
forgather submit --diloco --diloco-server localhost:8512 ...
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
forgather diloco server -o ./model -n 4 -H 0.0.0.0
```

(`-H` is the short alias for `--host`, matching `forgather server` and
`forgather dataset-server start`.)

**Warning**: This exposes the server's HTTP control endpoints (including
`/control/shutdown`, `/control/update_optimizer`, etc., which the webui
DiLoCo view's Control card calls into) to any machine on the network.
Only use this on trusted networks with appropriate firewall rules.

## Pipeline parallel + DiLoCo

When the training project uses `trainer_type: pipeline`, each pipeline
rank only holds a slice of the model on-device — the standard
`trainer.model` is a meta-device placeholder. DiLoCo's pre-#84
contract (every worker holds the full model and submits whole-model
pseudo-gradients) doesn't compose; the worker side and server side
both need to be slice-aware.

The current implementation (issue #84) treats each pipeline rank as
its own DiLoCo worker:

- The `DiLoCoCallback` detects the pipeline trainer via
  `trainer.pipeline_modules` and constructs a `PipelineParamView` over
  the rank's on-device stage modules.
- The worker's `--diloco-worker-id` base (e.g. `alpha`) becomes
  `alpha_pp0`, `alpha_pp1`, ... on the server — one entry per
  pipeline rank.
- The ranks form a `WorkerGroup`. The server verifies that the union
  of all members' slices exactly covers its full parameter set, then
  coordinates the per-rank submissions into one logical sync round.
- Each rank submits only its own slice's pseudo-gradients; the server
  applies the contributors per-name. Bandwidth scales as `slice_size
  × pp_world_size`, equal to one full model per group.

Streaming fragments (`num_fragments > 1`) compose with pipeline
groups: each rank partitions its slice into N fragments; the server's
per-fragment barrier waits for every rank's submission of each
`fragment_id`.

Currently out of scope (will fail at startup with a clear error):

- **Async mode + pipeline groups**. Async barrier semantics with
  disjoint slice contributions is fragile. Use sync mode (the default)
  with pipeline trainers.
- **Pipeline + within-stage DDP**. The forgather trainer does not
  currently compose these. When it does, the `PipelineParamView`
  plumbing will gain a `pp_group` argument so post-sync params can be
  broadcast across the within-stage DDP sub-group.

If a member of a pipeline group dies (heartbeat timeout or explicit
deregister), the whole group is evicted atomically — a partial group
can't produce valid pseudo-gradients. Other groups continue
uninterrupted.

See `docs/design/diloco-pipeline-groups.md` for the full design.

## References

**DiLoCo and direct lineage**

- Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models" ([arXiv:2311.08105](https://arxiv.org/abs/2311.08105))
- Douillard et al., "DiPaCo: Distributed Path Composition" (2024)
- Liu et al., "Asynchronous Local-SGD Training for Language Modeling" (2024) — Async DiLoCo, Delayed Nesterov, DyLU
- Jaghouar, Ong & Hagemann, "OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training" (2024, [arXiv:2407.07852](https://arxiv.org/abs/2407.07852)) — first FP16 all-reduce of the pseudo-gradient (the origin of low-precision *upload* communication in the DiLoCo family)
- Douillard et al., "Streaming DiLoCo with Overlapping Communication" (2025, [arXiv:2501.18512](https://arxiv.org/abs/2501.18512)) — fragment-based staggered sync; §2.4 sweeps the outer-gradient (upload) communication precision through bf16/fp8/fp4 with no observed regression. (Neither paper compresses the server→worker *download* of averaged weights — that is what `download_dtype=bf16` adds.)
- Charles et al., "Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo" ([arXiv:2503.09799](https://arxiv.org/abs/2503.09799))
- TorchFt (Meta) — fault-tolerant distributed training library

**Local SGD, slow momentum, and the outer optimizer**

- Wang, Tantia, Ballas & Rabbat, "SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum" (ICLR 2020, [arXiv:1910.00643](https://arxiv.org/abs/1910.00643)) — the slow/outer-momentum update DiLoCo's outer optimizer generalizes
- Lin, Stich, Patel & Jaggi, "Don't Use Large Mini-Batches, Use Local SGD" (ICLR 2020, [arXiv:1808.07217](https://arxiv.org/abs/1808.07217))
- Zhang, Lucas, Ba & Hinton, "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019, [arXiv:1907.08610](https://arxiv.org/abs/1907.08610)) — the single-worker local-SGD analog

**Generalization / flat minima** (why local SGD can train *better*, not just cheaper)

- Gu, Lyu, Huang & Arora, "Why (and When) does Local SGD Generalize Better than SGD?" (ICLR 2023, [arXiv:2303.01215](https://arxiv.org/abs/2303.01215)) — sharpness-reduction drift; needs small LR + long training
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization" (SWA, [arXiv:1803.05407](https://arxiv.org/abs/1803.05407))
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (ICLR 2017, [arXiv:1609.04836](https://arxiv.org/abs/1609.04836))

A worked, reproducible illustration of these effects (DiLoCo overtaking a DDP
baseline at a longer budget; single-worker local-SGD generalizing better) is in
the [canonical example](../../examples/tiny_experiments/diloco/README.md#extended-sweep-budget-sync-interval-and-single-worker-local-sgd).
