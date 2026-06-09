# DiLoCo CLI Reference

> Part of the [DiLoCo documentation](diloco.md). This chapter is the
> command-line reference: low-level (manual) invocation for development and
> debugging, the server-coordinated workflow in depth, monitoring & control,
> and network configuration. New to DiLoCo? Start at the [hub](diloco.md) and
> the [canonical example](../../examples/tiny_experiments/diloco/README.md).

## Manual / low-level invocation (development & debugging)

This section documents the **lower-level foreground path**: starting the
parameter server and each worker by hand with explicit `--diloco-server
host:port` and `--worker-id`. Most users should prefer the
[server-coordinated Quick Start](diloco.md#quick-start-recommended-through-the-forgather-server)
(and the canonical
[example](../../examples/tiny_experiments/diloco/README.md)); reach for these
commands for development, debugging, single-process foreground runs, or
environments without a Forgather server. The flag-reference tables here remain
canonical.

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
- `--verbose-sync`: Log every sync round at INFO — the server's outer step and
  each worker's per-round sync line. **Off by default** (the per-round lines log
  at DEBUG); a targeted DiLoCo diagnostic. Routine progress instead rides the
  per-step training-log columns the worker publishes (see
  [Sync log columns](#sync-log-columns)).
- `--num-fragments N`: Streaming-sync fragments every worker splits the model
  into (1 = no streaming). Default: 1.
- **Wire precision** — `--upload-dtype {fp32,bf16}` / `--upload-sr` /
  `--download-dtype {fp32,bf16}` / `--download-sr` set the dtype and stochastic
  rounding of each transport leg (defaults: bf16 upload, fp32 download).
  `--no-bf16` is a deprecated alias for `--upload-dtype fp32`. See
  [Wire precision](diloco.md#wire-precision).

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
See [Work-unit dispatch](diloco-advanced.md#work-unit-dispatch).

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
from every worker (see [Unified statistics](diloco-advanced.md#unified-statistics)): total tokens
/ steps / FLOPs, aggregate throughput / MFU / peak memory, and smoothed
train / eval loss. The same block appears in the webui's DiLoCo server view.

`--watch` (`-w`) refreshes the status in place every `--interval` seconds
(default 2.0) until Ctrl-C — like `watch`, but in-process: it reuses the same
connection across ticks (no per-tick subprocess) and works inside the
interactive CLI, where Ctrl-C returns you to the prompt. Not compatible with
`--json`.

### Sync log columns

Each worker's training log carries DiLoCo sync metrics inline, in the regular
step table alongside `peak_mem` / `mfu` — so you see sync progress every log
step without per-round log spam:

- `sync` — completed sync rounds.
- `sync_s` — mean wall-time per sync over the log window.
- `up_mb` / `dn_mb` — mean per-sync upload / download over the window. Shown
  **only for backends that move tensors over a wire** (HTTP, collective); a
  shared-memory run omits them (its wire volume is zero).

The rates are the mean over the syncs since the previous log row (windowed, like
`tok/s`), not a single last-sync sample. The rate columns
(`sync_s` / `up_mb` / `dn_mb`) are not populated under streaming-fragment sync
(`--num-fragments > 1`); `sync` still counts rounds. For a deeper, per-round
trace turn on `--verbose-sync` on the server (off by default) — it logs the
server's outer step and each worker's per-round sync line at INFO.

## Running through the Forgather server (detailed reference)

This is the **recommended** way to run DiLoCo (the concise version is the
[Quick Start](diloco.md#quick-start-recommended-through-the-forgather-server) in the hub).
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

