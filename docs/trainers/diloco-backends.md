# DiLoCo Sync Backends

> Part of the [DiLoCo documentation](diloco.md). This chapter covers the
> synchronization backends — HTTP-star (default), shared-memory (single-host),
> and collective (single-host DDP alternative) — how workers agree on a group
> backend, and composition with pipeline parallelism. See the [hub](diloco.md)
> for concepts and quick start.

## Sync backends and group agreement

DiLoCo's sync tensors can move three ways: the default **`http`** parameter
server (independent workers, possibly cross-host), a single-host
**`shared_memory`** region, or an all-reduce **`collective`** group. The backend
is a **group-wide invariant** — a collective group is one all-reduce world, a
shared-memory group is co-located, an HTTP group is independent workers; there is
no valid *mixed* group, and workers that disagree on the backend would deadlock or
corrupt the sync.

So the backend is **set in exactly one place — the server**
(`forgather diloco server --backend {http,shared_memory,collective}`, default
`http`) — and advertised via `/info`. Workers are **not** asked to choose it:
`forgather submit` enqueues them backend-agnostic, and the scheduler queries the
server's `/info` *just before it invokes `torchrun`*, derives the backend, and
shapes the launch (the env, the `shared_memory` region, the `collective` world)
from it. If the server is unreachable at that moment the launch **fails loud** —
there is no default to fall back to. This makes a disagreeing worker
unrepresentable: nobody can launch one with the wrong backend, because nobody
launches one with a chosen backend at all.

A worker can't *adopt* the backend the way it adopts `sync_every`: the backend
fixes the launch topology — GPU count, torchrun world, co-location — which is
decided before the worker runs. Deriving it at launch is what lets that decision
still come from the server. As a second line of defense, each running worker also
validates its own launched backend against `/info` and fails loud on a mismatch
(the safety net for a hand-launched or stale process); an older server that
doesn't declare a backend skips that check.

The **`collective`** topology is the one launch-shape the operator still selects
explicitly, because it's structurally different — one `torchrun` job of N
replicas rather than N independent jobs. Select it with **`--diloco-replicate N`**
(a topology/sizing flag), against a server that declares `--backend collective`;
the scheduler validates the two agree. `--backend` itself is **not** a `submit`
option on the orchestrated path. It survives only as a `--local-only` dev/debug
escape hatch (a direct foreground `torchrun` with no server to query), where the
running worker's own validation still catches a misspecification. The backend
selector lives on the webui's **DiLoCo Server** modal and the agent's
`start_diloco_server` tool — the server side, where it belongs.

## Shared-memory backend (single-host)

On a single host, co-located worker processes exchange the sync tensors through a
**shared CPU master-weights region** instead of moving them over HTTP — one
shared master per host, no serialization, the outer optimizer applied in place.
This makes DiLoCo a DDP alternative: sync every `H` steps at a fraction of DDP's
per-step all-reduce.

Because the region is single-host and the param server is co-located, the
**server maps the same region and *is* the aggregator**. It owns the master and
the outer optimizer, runs the outer step over the shared accumulator each round,
and publishes the new master back into the region; the workers are pure
**followers** that contribute their pseudo-gradient into the region and read the
published master. Shared-memory is a *transport swap* of the HTTP star, not a
second aggregation owner — the workers never submit pseudo-gradients over the
wire (no wire volume — the `up_mb`/`dn_mb` log columns are omitted), and the
server still provides the usual `/info` negotiation and work-unit dispatch.

Because the **server** runs the outer step, its `sync_round` advances normally
and its checkpoints carry the trained weights, named `checkpoint-{round}`. The
outer-optimizer momentum lives in the server and is checkpointed with it, so a
shared-memory run **checkpoints and resumes coherently** — a resume continues the
loss trajectory rather than restarting from the seed.

The region's rendezvous is advertised by the server in `/info`
(`shm_group_dir` + `shm_group_size`, the server's configured worker count), and
each follower reads it from there — so on the orchestrated path you set nothing
per worker. The environment variables below are an optional override (the manual
/ dev recipe):

| Variable | Meaning |
|---|---|
| `DILOCO_BACKEND` | `shared_memory` (default `http`) |
| `DILOCO_SHM_GROUP_DIR` | override the per-host region directory (default: advertised by the server in `/info`) |
| `DILOCO_SHM_GROUP_SIZE` | override the co-located group size (default: the server's configured `--num-workers`, from `/info`) |
| `DILOCO_REPORT_SYNC_STATE` | optional; report per-worker sync-state to the coordinator for diagnostics (default on; set `0`/`false` to omit) |

Shared-memory followers register with the server for membership and report their
sync-state (`sync_count`, send/recv MB, sync time) on the heartbeat, so the
group's progress is exposed in the server's `/status` (for the dashboard/CLI to
surface) even though the tensor exchange is off-server. (`DILOCO_REPORT_SYNC_STATE`
applies to any backend; it is most useful for an off-server one like this.) Each
follower is one process (one GPU) — not a multi-GPU DDP job.

The **server** owns the region's lifecycle: it creates and seeds the region from
its master when it starts (or from the *restored* trained master on resume),
holds the region's ownership lease (an exclusive `flock`) for its lifetime, and
unlinks the region when it stops. The lease makes re-launch after a crash safe —
a region left behind by a dead server has no live lease holder, so the next
launch reclaims and rebuilds it rather than stranding on an ownerless region. The
configured group must fully form (every follower attached) before the first sync
round, so a worker that's slow to launch surfaces as *no progress* rather than a
silently smaller group; a follower that **crashes** mid-round is not tolerated
(shared-memory is single-host, not fault-tolerant) — the server marks the group
dead and the followers fail loud.

Run a server (which is also the aggregator), then launch the followers (one
process per GPU):

```bash
# Server + aggregator: declares the backend, owns the region + the outer step.
# A worker launched with a mismatched backend fails loud at /info.
forgather diloco server --local-only --backend shared_memory \
    --output-dir <checkpoint-dir> \
    --num-workers 2 --sync-every 100 -H 127.0.0.1 --port 8512

# Two co-located followers; they read the region dir + size from the server's
# /info, so no DILOCO_SHM_* env is needed.
for w in 0 1; do
  DILOCO_SERVER=https://127.0.0.1:8512 DILOCO_WORKER_ID=shm-w$w \
  DILOCO_BACKEND=shared_memory \
    forgather -t <config>.yaml train -d $w &
done
wait
```

Streaming-fragment sync (`num_fragments > 1`) is not supported for this backend.
For the internals see
[`diloco-architecture.md`](diloco-architecture.md#shared-memory-backend).

### Via the scheduler (`forgather submit`)

The env-var form above is the manual recipe; the scheduler launches the same
group as a first-class option. Declare `shared_memory` on the **server**, then
submit plain workers — they inherit the backend at launch:

```bash
# Once, on the param server (also the aggregator):
forgather diloco server --backend shared_memory -n 2 ...
# Then submit the followers (no --backend, no region env — derived from the server):
forgather -t <config>.yaml submit --diloco --diloco-worker-count 2
```

The workers carry no backend; the scheduler queries the server's `/info` before
`torchrun`, sees `shared_memory`, and sets only `DILOCO_BACKEND` — the follower
then reads `shm_group_dir` + `shm_group_size` from `/info` at runtime (the server
is the single source of truth for both, sizing the group from its configured
`--num-workers`). The server creates and owns the region and unlinks it on
shutdown, so a completed run leaves nothing behind. Because the backend
is single-host, a `shared_memory` server's workers can't be submitted with
`--global` (the multi-node fan-out).

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
work-unit dispatch that shards rows across replicas; no tensor role). The
`up_mb`/`dn_mb` log columns reflect the all-reduce volume, off-server.

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
# Coordinator (no tensor role; provides /info + the init checkpoint + dispatch).
# --backend collective declares the group backend so the replicas validate
# against it (a worker launched with a mismatched backend fails loud at /info).
forgather diloco server --local-only --backend collective \
    --output-dir <init-checkpoint-dir> \
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
group as a first-class option. With a coordinator that declares
`--backend collective` running, size the group with `--diloco-replicate` — the
replica count is the one launch-shape you pick (the backend itself comes from the
server):

```bash
forgather -t <config>.yaml submit --diloco-replicate 2
```

The scheduler validates that the chosen server actually declares `collective`
before it launches; submitting `--diloco-replicate` against an `http` /
`shared_memory` server fails loud. Unlike the shared-memory backend (which
enqueues N worker jobs), collective is
**one** scheduled job: the scheduler reserves `--diloco-replicate` GPUs, sets
`nproc_per_node` to the same, and derives `DILOCO_BACKEND=collective` +
`DILOCO_REPLICATE` for it (the `DILOCO_WORKER_ID` base is made per-replica at the
entrypoint). Because the backend is single-host, `--diloco-replicate` can't be
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

