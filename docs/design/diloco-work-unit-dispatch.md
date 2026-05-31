# DiLoCo: Work-Unit Dispatch (Design Proposal — Implemented)

**Status:** **implemented** as of the `feature/diloco-webui` work
(May 2026), revised in `feature/diloco-dataset-dispatch` (May 2026,
revision 6) to move the wrap from the backend layer into
``ComposableIterableDataset``. User-facing docs:
[`docs/trainers/diloco.md#work-unit-dispatch`](../trainers/diloco.md).
Related design: [DiLoCo + pipeline parallel](./diloco-pipeline-groups.md)
covers the per-rank-worker / `WorkerGroup` model used when each rank
holds only a slice of the model.
The design proposal below is kept as the rationale-of-record. A few
details diverged from the proposal during implementation; the
**Implementation notes** subsection at the end of each affected
chunk calls them out. The biggest deltas:

- The operator-facing "Use work-unit dispatch" toggle / the
  `DILOCO_WORK_DISPATCH` env var (proposed as opt-in gates) were
  ripped out before merging. Dispatch is unconditional when
  ``shard_dataset.method`` is set to ``"work_units"`` in the
  template, which the project default selects whenever
  ``DILOCO_SERVER`` is set (revision 6).
- Work-queue persistence (Phase 1.3) was dropped after the
  first live bringup — the cross-experiment state-bleed footgun
  outweighed mid-epoch crash recovery value.
- The server's `/register` got a structural fingerprint check
  during bringup, separate from the work-queue work but in the
  same PR — see `docs/trainers/diloco.md#model-fingerprint-check`.
- **Revision 6**: work-unit dispatch now lives **inside**
  ``ComposableIterableDataset`` (state + ``enable_work_dispatch``
  method, applied by ``preprocess_dataset`` after slice/shard are
  settled), not as an ``IterableDatasetBackend`` wrap below the
  composable. The backend-layer placement (revision 5) couldn't see
  the wrapper's slice bounds, so it dispatched over the full backend
  and discarded the first N rows via a yield-counter trick — wasted
  budget and percentage-slice incorrectness. It also couldn't compose
  with DDP/FSDP ``.shard()`` for asymmetric topologies. The new
  placement absorbs the slice into the ``dataset_id`` hash and lets
  all DDP ranks across all DiLoCo hosts share one queue (replacing
  conventional sharding entirely; the two are mutually exclusive via
  the new ``shard_dataset.method`` config). See [Revision 6 details](#revision-6-composable-level-dispatch)
  for the full delta.

**Revision:** 6 (moves the dispatch from the
``IterableDatasetBackend`` layer up into
``ComposableIterableDataset``, applied by ``preprocess_dataset``
after slice/shard are settled. The ``dataset_id`` hash absorbs
slice bounds so two slices of the same source dataset get
separate queues. ``shard_dataset.method`` config makes the
partition choice (conventional vs work_units) explicit and rejects
the two combinations that don't compose — work_units without DiLoCo
and conventional sharding under DiLoCo (asymmetric-DDP row overlap)).
**Supersedes:** the manual `--num-shards N --shard-index I` flow
in `examples/tiny_experiments/diloco/` and
`examples/base_lm_project/templates/configs/diloco.yaml`.

## Scope note (read first)

This design covers **only the train dataset.** The eval / validation
dataset stays as it is today (already gated by `shard_eval=False`-style
flags) — eval sets are small, every worker runs a full pass, and the
trainer averages metrics across workers. Don't apply work-unit dispatch
to eval datasets; it'll just slow things down and add coordination cost
where the existing path is already correct.

## Problem

DiLoCo workers currently partition the dataset by hand. Each worker is
launched with `--num-shards N --shard-index I`, and each worker's
dataloader calls `dataset.shard(N, I)`. Each worker is also assigned a
unique output directory (today: by mixing `shard_index` into the model
name) so the per-worker checkpoint and log streams don't collide.

Failure modes:

- **Duplicate `--shard-index` silently doubles up training on the same
  rows**, biasing the global average.
- **`N` has to be fixed at launch.** A late-joining worker can't expand
  the partition; a dead worker leaves its shard untouched.
- **Static partition is wrong for async / heterogeneous workers.** A 4×
  faster worker chews through its slice and then idles or re-iterates.
- **Per-worker output dir naming is an operator concern.** Mixing
  `shard_index` into `model_name` works but doesn't belong in user-facing
  config.

The DiLoCo implementation predates the **dataset server**, which now
provides arbitrary-row-range seeks
(`GET /v1/datasets/{handle}/iter?seed=&position=&limit=`). This unlocks
a much better dispatch model: the DiLoCo server hands out **work
units** as workers request them, and workers stream the rows for the
unit they were assigned directly from the dataset server. Nothing has
to be known in advance about worker count.

## Goals

1. Workers do **not** declare `num_shards` or `shard_index`. They just
   register with a DiLoCo server and ask for work.
2. Workers can join and leave at any time. A dead worker's in-flight
   unit is silently lost — *not* reissued — so we **never train on
   duplicate rows within an epoch.** Worst case: lose `dead_workers`
   units out of `K` per epoch. At K=1024 that's noise.
3. Async DiLoCo, heterogeneous hardware, and elastic membership all
   "just work" without operator coordination.
4. The unique-per-worker output dir is derivable from `worker_id` (or
   the operator-supplied worker name) at config-preprocessing time —
   no server-side allocation, no runtime renaming. The DiLoCo server
   independently enforces `worker_id` uniqueness on register as
   defense in depth (rejects a second registration of an already-live
   worker_id; the server doesn't need to know how the template uses
   the value).
5. Existing manual `--shard-index` configs keep working, gated by an
   opt-in flag. Migration is per-config.
   _Implementation note: the opt-in flag was removed before merging;
   work-unit dispatch is unconditional when `DILOCO_SERVER` is set.
   The migration ended up being "delete the `--num-shards` /
   `--shard-index` dynamic-args from any custom template that
   declares them." Both first-party examples
   (`examples/tiny_experiments/diloco`,
   `examples/base_lm_project/templates/configs/diloco.yaml`) shipped
   with the change._

## Non-goals

- Replacing the DiLoCo outer-loop algorithm. Untouched.
- Changing the dataset server's wire format. Existing `/v1/load` and
  `/v1/datasets/{handle}/iter` are sufficient.
- Solving training-time data shuffling for non-`dataset_server`
  datasets. Phase 1 requires datasets served by a dataset server.
- Coordination across DiLoCo server instances. One server per training
  group, like today.

## Core concept: work units

A **work unit** is a deterministic, contiguous slice of a dataset,
identified by `(dataset_id, shuffle_seed, unit_id)`.

- `dataset_id` is a stable short hash of the dataset's identity —
  normalized `{path, name, split, data_files, revision}` mirrored from
  the args that `fast_load_iterable_dataset` /
  `_remote_load_iterable_dataset` already accept. Workers compute it
  locally without a server round-trip.
- `shuffle_seed` is the **worker-supplied** seed for `RemoteBackend`'s
  shuffle. Workers pick the seed (typically derived from an epoch
  counter). The DiLoCo server treats it as part of the queue key,
  not as state it owns.
- `unit_id ∈ {0, 1, …, K − 1}`, where `K` is configured on the
  DiLoCo server (CLI flag `--default-work-units`, default **1024**)
  and persists for the lifetime of the server. Per-queue `K` would
  be a footgun — keep one server-wide value.

The mapping `unit_id → row range` is deterministic from
`(unit_id, K, length)`:

```
start = (unit_id      * length) // K
limit = ((unit_id + 1) * length) // K - start
```

The DiLoCo server only ever sees `unit_id` integers; the worker
translates `unit_id → (position, limit)` against the dataset server
using existing endpoints. **Neither server learns the other's internal
data model.**

## State ownership

| Component       | What it owns                                                  |
|-----------------|---------------------------------------------------------------|
| Dataset server  | Datasets themselves: `length`, `column_names`, content, shuffle. Existing endpoints. |
| DiLoCo server   | One bitmap per `(dataset_id, shuffle_seed)` queue, sized K, marking which units have been issued. Optional second bitmap for diagnostic "confirmed completed". `dataset_id → length` snapshot for mismatch detection. |
| Worker          | `worker_id` (operator-supplied or auto), an `output_dir_suffix` derived from `worker_id`, and the current dataloader state. Translates `unit_id` to `(position, limit)` and streams rows from the dataset server. |

## Issuance is one-way

This is the load-bearing simplification in this design:

> **Once a unit is issued, it's consumed from the queue regardless of
> the worker's fate.** No reissue, no abandoned-vs-in-progress state
> machine, no per-unit heartbeat tracking.

Consequences:

- The server's per-queue state is a single bitmap (K bits) plus an
  issued counter — no times, no worker associations.
- A dying worker loses ≤ 1 unit. K=1024 default makes this ≤ 0.1% of
  the epoch. The cost is real but bounded and predictable.
- **Within an epoch we cannot train on the same row twice**, no matter
  how chaotic the worker fleet is. This is worth the tiny data-loss
  tradeoff.

## API surface

### DiLoCo server: new endpoints

```
POST /datasets/register
  body:  {
           worker_id,
           dataset_id,
           shuffle_seed,
           hint: { length, source? }
         }
  reply: { total_units: K }
         | 409 if (dataset_id) was previously registered with a
           different length (catches diverging-config footgun).
```

Idempotent within `(dataset_id, shuffle_seed)`. The first registration
for a `(dataset_id, shuffle_seed)` pair allocates the queue (K bits, all
zero); later registrations from other workers just confirm the plan.
`hint.length` is required so a worker shipping a stale dataset config
fails fast with 409 rather than silently mis-windowing.

```
POST /work/request
  body:  { worker_id, dataset_id, shuffle_seed }
  reply: { unit_id: int } | { exhausted: true }
```

Atomically: scan the bitmap for the lowest-numbered unset bit, set it,
return its index. If every bit is set, return `{exhausted: true}` —
the queue is drained; the worker decides whether to register a new
shuffle_seed (new epoch) or stop.

```
POST /work/complete                # OPTIONAL — diagnostic only
  body:  { worker_id, dataset_id, shuffle_seed, unit_id }
  reply: { ack: true }
```

Workers MAY call this on successful drain of a unit. The server tracks
a second "confirmed completed" bitmap. **Not required for correctness;
nothing about issuance or queue state changes if this call is omitted
entirely.** Useful for the diagnostic surface to show "issued but
unknown" vs "issued and confirmed" — particularly helpful for
visualizing worker liveness.

```
GET /work/queues
  reply: [
    {
      dataset_id,
      shuffle_seed,
      total_units: K,
      issued_count,
      completed_count,
      hint: { length, source? },
    },
    ...
  ]
```

List all active queues. The training-submit UI uses this to render
"3 queues active across 2 datasets" status.

```
GET /work/queue?dataset_id=&shuffle_seed=
  reply: {
    dataset_id,
    shuffle_seed,
    total_units: K,
    issued_count,
    completed_count,
    issued_bitmap_b64,      # K bits, base64-encoded
    completed_bitmap_b64,   # K bits, base64; all-zero if /work/complete unused
    by_worker: { wid: { units_issued, units_completed } }
  }
```

Full single-queue state for the DiLoCo view's per-queue heatmap. K=1024
→ 128 bytes per bitmap; cheap. `by_worker` is populated from in-memory
attribution kept alongside the bitmap (one entry per request / complete)
— it's not necessary for correctness either, just diagnostics.

### Dataset server: no changes in phase 1

Existing `GET /v1/datasets/{handle}/iter?seed=&position=&limit=` is
sufficient. Worker translates `(unit_id, K, length)` to `(position,
limit)` and calls iter as-is.

## Worker-id provisioning (early-bound)

The worker_id is needed at **template-preprocessing time**, not just at
runtime. Forgather configs derive the per-worker output directory leaf
name from `getenv("DILOCO_WORKER_ID")` during preprocessing (see
`examples/base_lm_project/templates/configs/diloco.yaml` `[globals]`).
If the env var were unset at that moment, two workers would compute the
same output dir and clobber each other's checkpoints / logs.

The scheduler (`tools/forgather_server/scheduler.py: _diloco_env_from_job_params`)
therefore **always emits `DILOCO_WORKER_ID` when DiLoCo is enabled**.
Precedence:

1. Operator-supplied `worker_id` (from the submit modal or
   `--diloco-worker-id`), if non-empty after whitespace strip.
2. The `queue_id` — stable per submission, already surfaced as the
   primary identifier in the Jobs view so operators can correlate.

Choice (2) keeps the contract local to the scheduler: every spawned
diloco-enabled worker has a non-empty `DILOCO_WORKER_ID` by
construction. The config template can rely on this without sentinel
values or conditional fallbacks. (The config still guards the
suffix-append on whether the env var is set at all — for the case
where the same config is also used for vanilla, non-DiLoCo finetuning.)

### Output-dir derivation

The config appends `worker_id` to the model name leaf:

```
ns.model_name = ns.model_name + "_" + getenv("DILOCO_WORKER_ID")
ns.output_dir = joinpath(ns.models_dir, ns.model_name)
```

Worker A with `worker_id="alpha"` writes to `models_dir/<base>_alpha/`;
worker B with the auto-generated `queue_id` writes to
`models_dir/<base>_q-7f3a…/`. Collisions can only happen if two
workers were dispatched with the same operator-supplied `worker_id`
(the queue_id fallback is unique by construction).

### Server-side `worker_id` uniqueness check (defense in depth)

The early-bind contract is sufficient when the webui is the only
dispatch path. To catch the manual-CLI footgun (operator forgot to
bump `--diloco-worker-id` when launching the second worker), the
DiLoCo server's `/register` handler enforces:

> **A `worker_id` already present in the registry cannot be
> registered again.**

The server has no view into how a template translates `worker_id`
into an output directory, training-run identifier, log-file name, or
anything else — different configs may do completely different things
with it. The simplest correct rule is to treat the `worker_id` itself
as the uniqueness proxy: if two workers identify themselves the same
way, the *downstream consequences* of that identity collision belong
to the operator's templates, not to the server. Refusing the second
registration breaks the collision cleanly regardless of what those
templates do.

```
POST /register
  body:  { worker_id, hostname, extra: { ... } }
  reply: tensor (existing) | 409 if worker_id is already registered
```

Behavior:

- **`worker_id` not in registry** → register, return initial params
  (existing path).
- **`worker_id` already in registry** → refuse with 409:
  `{"error": "worker_id '<id>' is already registered; if the previous
  worker is dead, wait for heartbeat eviction (default ~120s) or POST
  /deregister"}`.

This **replaces** today's "re-register replaces the existing entry"
semantics. The previous behavior was designed for a single worker
reconnecting after a brief outage, but it doubles as a silent
collision masker — two operators using the same worker_id today
just kick each other out invisibly. The new semantic is honest: brief
outage means the worker waits for heartbeat eviction or the operator
explicitly cleans up via `/deregister`.

Worker-side handling: the `DiLoCoCallback` treats 409 on register as
a fatal clean-exit. The diagnostic from the server lands directly in
the worker's TTY pane via stderr / logger, so the operator sees
"worker_id 'alpha' is already registered; …" in the Jobs view
without poking around server logs.

The cost is one dict membership check per `/register` against a
small in-memory set. The benefit is that no silent on-disk clobber
path remains, regardless of what the operator's config templates do
with the worker_id downstream.

### Eviction interaction

The heartbeat-timeout eviction path (`_handle_worker_death`) already
removes entries from `self._workers`. After eviction, the
`worker_id` becomes available again — a worker restarting via a
supervisor process can re-register once eviction has fired. No
explicit interaction needed; the existing health-monitor cadence is
sufficient.

For operators who want to recover faster than the heartbeat timeout,
`/deregister` provides an explicit clean-up path (already present in
the API).

## Worker integration

The wrap is a **backend-layer** concern: a `WorkUnitBackend` that
implements the same `IterableDatasetBackend` interface as
`ArrowBackend` / `ResilientRemoteBackend` / `InMemoryBackend`, and
gets dependency-injected at backend-construction time from inside
`fast_load_iterable_dataset` (see `src/forgather/ml/datasets/`).

Two reasons it can't live at a higher layer:

- `ComposableIterableDataset` composes map / filter / select / shard /
  shuffle-buffer ops over a backend. Wrapping at *that* layer would
  put work-unit slicing **after** the template's post-processing
  pipeline — the slice would cut the wrong rows.
- `StatefulDataLoader` rejects post-init dataset mutation
  (`ValueError: dataset attribute should not be set after StatefulDataLoader
  is initialized`). Any callback-based wrap can't replace the dataset
  in flight.

The backend layer doesn't have either problem: by the time
`ComposableIterableDataset` is built, the work-dispatch wrap is
already in the backend slot, and the higher-level wrapper sees a
normal `IterableDatasetBackend`.

### The opt-in hook

Inside `fast_load_iterable_dataset` (specifically the remote and
auto-routing paths — phase 1 requires a dataset_server-backed
backend so the worker can seek to arbitrary positions cheaply), after
the inner backend is constructed:

```python
backend = ResilientRemoteBackend(...)
backend = maybe_wrap_for_work_dispatch(
    backend, path=path, name=name, split=split,
    data_files=data_files, revision=revision,
)
ds = ComposableIterableDataset(backend, ...)
```

`maybe_wrap_for_work_dispatch` is env-driven (zero loader-signature
pollution):

- `DILOCO_WORK_DISPATCH` (truthy required) — opt-in gate.
- `DILOCO_SERVER` (required) — DiLoCo server addr.
- `DILOCO_WORKER_ID` (required) — set by the scheduler with a
  queue_id fallback.

When the opt-in gate is off or any prerequisite is missing, the
helper returns the input backend unchanged (no behavior change for
non-DiLoCo runs). Errors during `/datasets/register` are logged at
ERROR; the backend is returned unchanged so a server hiccup disables
work-dispatch for the run but doesn't crash training.

### `WorkUnitBackend` shape

```python
class WorkUnitBackend(IterableDatasetBackend):
    def __init__(self, wrapped, client, worker_id, dataset_id,
                 shuffle_seed, total_units, length): ...

    def __iter__(self):
        while True:
            resp = self.client.request_work(
                self.worker_id, self.dataset_id, self.shuffle_seed
            )
            if resp.get("exhausted"):
                return
            unit_id = resp["unit_id"]
            start, end = unit_range(unit_id, self.total_units, self.length)
            try:
                view = self.wrapped.seek(start)
                yielded = 0
                for row in view:
                    if yielded >= end - start:
                        break
                    yield row
                    yielded += 1
            except Exception as exc:
                # Per-unit drain error — unit is already consumed.
                logger.warning("unit %d drain failed: %s", unit_id, exc)
            try:
                self.client.complete_work(
                    self.worker_id, self.dataset_id,
                    self.shuffle_seed, unit_id,
                )
            except Exception:
                pass  # diagnostic-only

    def __len__(self): return self.length
    def shuffle(self, seed=None): ...  # wraps wrapped.shuffle(seed); queue keying unchanged
    def seek(self, position): return self  # no-op; positions are server-driven
    def position(self): return 0  # composable.state_dict has nothing useful to record
```

### Decoupling from `DiLoCoCallback`

`DiLoCoCallback` (which manages the parameter-sync `DiLoCoWorker`)
does **not** participate in the dataset wrap. The two subsystems are
orthogonal:

- `DiLoCoCallback` owns the optimizer-step hooks and the
  parameter-sync HTTP traffic.
- `WorkUnitBackend` owns the data dispatch HTTP traffic.

They share env vars (`DILOCO_WORKER_ID`, `DILOCO_SERVER`) as common
identity but never call into each other. `WorkUnitBackend` constructs
its own `DiLoCoClient`. This is a deliberate separation — same
worker_id, two independent clients, two orthogonal correctness
properties (param-sync vs no-row-trained-twice).

### Multi-epoch handling

Multi-epoch shuffle rotation is a follow-up; phase 1 keeps
`shuffle_seed = 0` fixed at construction time. The follow-up would
either:
- thread a per-epoch seed through `maybe_wrap_for_work_dispatch` as
  the template re-evaluates the dataset at epoch boundary, or
- have `WorkUnitBackend.set_epoch(n)` re-register against the server
  with a new seed and reset its in-process state.

## Lifecycle scenarios

### Happy path (3 workers, sync DiLoCo)

- W0, W1, W2 register against `(D, seed=42)` with `K=1024`.
- W0 requests work → unit 0. W1 → 1. W2 → 2.
- Each iterates ~`length/1024` rows, requests next.
- Continues until all 1024 bits are set. Next request returns
  `{exhausted: true}`.
- Worker decides: stop, or register `(D, seed=43)` and continue.

### Late worker

- W0, W1 running on units 47, 52.
- W2 starts, registers, requests → gets next unset bit (53).
- W2's training begins from current global params (existing
  dynamic-membership semantics).

### Worker dies mid-unit

- W0 holds unit 47, no heartbeat for `heartbeat_timeout` seconds.
- Existing health monitor evicts W0 from the *worker registry*.
- **Unit 47 stays issued.** It's gone for this epoch.
- Net data loss: one unit. Net data corruption: zero.

### Async DiLoCo, heterogeneous workers

- W_fast does ~10 units per W_slow's 1. Queue naturally routes more
  units to W_fast. No operator action required.

### Multi-epoch

- Workers complete `(D, seed=42)`. The next call to register with
  `(D, seed=43)` creates a fresh queue. Workers can be in different
  epochs concurrently — the diagnostic surface shows
  `[D@42: 100%, D@43: 23%]` until everyone catches up.

### DiLoCo server restart

The server is the authority for which rows have been consumed, so queue
state is persisted with its checkpoint and restored on restart (#105) —
the worker deliberately keeps **no** dataset-progress state of its own,
relying on the server to track and persist it.

- **Persisted in `server_state.pt`:** the per-`(dataset_id, shuffle_seed)`
  `issued`/`completed` bitmaps + counters, and `_dataset_lengths` (the
  first-registered row count per `dataset_id`). Queues are keyed on disk by
  a `"dataset_id|seed"` string (tuple keys don't round-trip) and bitmaps
  are stored as `bytes`.
- **On restart:** `load_state` rehydrates `_work_queues`. A worker
  re-registering its dataset (the wrap calls `/datasets/register` lazily on
  the iterator's first request) hits the *restored* queue —
  `_handle_register_dataset` reuses any existing queue for the key — so
  already-issued units stay issued and issuance resumes at the next
  un-issued unit, not from 0.
- **Cross-experiment safety (the original ghost-queue worry):** a changed
  dataset hashes to a different `dataset_id` → a fresh `(dataset_id, seed)`
  key → any stale queue from a prior dataset is simply never matched and
  sits inert. The 409-on-length-mismatch guard catches a same-id/different-
  length config. An operator wanting a hard reset restarts from the model
  weights and purges the rest of `output_dir`.
- **Save cadence:** queues are written on the normal `save_every_n_rounds`
  cadence and flushed once more on graceful shutdown (SIGINT/SIGTERM), so a
  clean stop doesn't lose units issued since the last autosave.

### Worker dataset mismatch

- W1 registered with `(D, length=1_000_000)`.
- W2 registers with `(D, length=1_000_500)` (stale checkout).
- DiLoCo server returns 409. Operator sees a clear error rather than
  off-by-window training.

## Consequences of "no reset within a queue"

This is the section the reviewer asked to be thorough about. Audit
candidates:

### 1. Pipeline trainer's example-batch fetch — fix required

`PipelineTrainer._prepare_model` calls
`example_dataloader = self.train_dataloader; example_batch = next(iter(example_dataloader))`
purely to extract the shape of `input_ids` (then immediately throws the
batch away and constructs `torch.empty_like(..., device="meta")`).

Under work-unit dispatch each PP rank would issue + consume one unit
just to peek at shape, then immediately throw the data away. **Phase 2
prerequisite:** the actual fix in the current codebase was simpler —
the manual splitter never used the example arg at all, so the whole
``_get_example`` machinery was dead code that just happened to consume
a batch. It's been removed entirely. Cost under work-dispatch: zero.

### 2. Transient dataloader errors mid-unit lose rows

If the dataset_server connection blips mid-unit, the worker
**swallows the error inside `WorkUnitBackend.__iter__`** (logging
the unit_id at WARNING) and moves on to the next unit. The partial
rows are lost for the epoch. The alternative — propagating the error
and crashing the training loop — is much worse, and the unit is
already consumed from the server's bitmap anyway.

### 3. Worker checkpoint resume doesn't replay rows

Today, checkpoint resume restores the dataloader's `state_dict` so
training resumes at the same iteration position. Under work-unit
dispatch, the rows previously consumed are no longer in the queue.
This is actually **better** behavior for DiLoCo (the global params
have moved on; replaying old rows would only add gradient noise).
`WorkUnitBackend.position()` returns `0` and the surrounding
`ComposableIterableDataset.state_dict()` carries that through —
resume just asks the server for the next available unit.

### 4. Workers can be on different epochs simultaneously

Not a bug; an emergent feature. The diagnostic UI should render this
clearly (the queue list is keyed on `(dataset_id, shuffle_seed)`, so
each epoch is a visible row). The training loop's `max_steps` /
`num_train_epochs` semantics keep deciding when each worker stops.

### 5. A misbehaving worker that consumes units without training

…silently shrinks the epoch. Same failure mode as today's static
partitioning with a wrong `--shard-index`, just expressed differently.
Not a new vulnerability.

### 6. A worker that's slow to drain its issued unit ties up rows

…for the duration. With K=1024 and a fleet of fast workers this is
invisible. With a single straggler holding 1/1024 of the dataset for a
very long time, the other workers may hit `exhausted` while the slow
one is still working — moving them into a new shuffle_seed queue while
the slow worker is still in the old one. That's fine semantically,
just worth knowing about.

## Migration plan

Phase 1: backend coordination (testable with `curl`)
- Implement DiLoCo server endpoints: `/datasets/register`,
  `/work/request`, `/work/complete`, `/work/queues`, `/work/queue`.
- Extend `DiLoCoServer.save_state` to persist the bitmaps + the
  `(dataset_id, shuffle_seed) → queue` map.
- Tighten `/register` semantics: refuse a second registration of an
  already-live `worker_id` with 409 + diagnostic. Replaces today's
  silent "re-register replaces" path. The heartbeat-eviction path
  already clears the registry entry, so a worker restarting via a
  supervisor process can re-register once eviction has fired.
- Tests: register, request, exhaustion, dataset_id mismatch, server
  restart preserving bitmaps, two queues for same dataset under
  different seeds, completion accounting, duplicate-worker_id
  register returns 409, register after `/deregister` succeeds,
  register after heartbeat eviction succeeds.
- CLI: `--default-work-units N` (default 1024).

Phase 2: worker integration
- Implement `WorkUnitBackend(IterableDatasetBackend)` in
  `forgather.ml.datasets.work_unit_backend`. Wraps another backend
  (typically `ResilientRemoteBackend`); ``__iter__`` does the
  request → wrapped.seek(start) → iter → complete loop.
- Add `DiLoCoClient.request_work` / `complete_work` /
  `register_dataset` (phase 1 server already speaks these endpoints).
- Add `maybe_wrap_for_work_dispatch(backend, **load_args)` env-driven
  helper that opts in based on `DILOCO_WORK_DISPATCH` + reads
  `DILOCO_SERVER` / `DILOCO_WORKER_ID`. Returns the unwrapped
  backend on any failure (graceful fallback).
- Hook the helper into `fast_load_iterable_dataset`
  (`_remote_load_iterable_dataset` + `_auto_load_iterable_dataset`)
  immediately after the inner backend is constructed. The higher-
  level `ComposableIterableDataset` is then built around the
  wrapped backend and composes its map / filter / shard / state_dict
  ops normally.
- Extend `DiLoCoCallback`: treat 409 on `/register` as a fatal
  clean-exit so the operator sees the server's diagnostic in the TTY
  pane. The callback **does not** wrap the dataset — that's the
  helper's job. The two subsystems are deliberately decoupled.

Pipeline-trainer side-cleanup that needs to land in phase 1 (already
done in this branch): the dead `_get_example` /
`example_args` plumbing in `PipelineTrainer` was removed entirely —
the only remaining splitter never used it, so there's no need to
synthesize a meta tensor at all.

Note: worker_id provisioning at the scheduler layer (always emit
`DILOCO_WORKER_ID`, fall back to `queue_id`) is already implemented
on the `feature/diloco-webui` branch and is independent of Phase 1 —
it shipped with the existing webui DiLoCo radio so the bringup
`diloco.yaml` template's output-dir derivation works today.

Phase 3: webui surface
- Training-submit DiLoCo radio gains a "Use work-unit dispatch"
  checkbox (default ON when the selected server supports it; OFF for
  back-compat).
- DiLoCo view's status panel gains a per-queue progress section with
  the issued/completed bitmap rendered as a heatmap.
- Legacy `num_shards` / `shard_index` dynamic args hidden when
  work-unit dispatch is on.

Phase 4: deprecate manual sharding
- Mark `--num-shards` / `--shard-index` as deprecated in the worker
  CLI after a release of bake time.
- Keep the code path until at least one full release after deprecation
  notice.

## Open questions / non-blocking refinements

- **K auto-tuning.** Fixed 1024 default is fine for the common LM
  dataset sizes (millions of rows). For very small datasets (1k rows)
  K=1024 means 1-row units — silly but not broken. A clamp like
  `effective_K = clip(64, length // 1024, K_configured)` is easy to
  add later if anyone trips over it.
- **Replay / determinism for debugging.** With server-allocated
  issuance order, exact-replay debugging would need the server to
  record `request → unit_id` history. Out of scope for phase 1; the
  bitmap + completion-bitmap pair is enough for "what got trained".
- **Same `worker_id` collision.** Handled by the server-side
  `worker_id` uniqueness check (Phase 1 scope): a second register of
  an already-live `worker_id` returns 409 and the colliding worker
  exits cleanly. The previous "re-register replaces" path is gone —
  it doubled as a silent collision masker. Operators recovering from
  a crashed worker either wait for the heartbeat-timeout eviction
  (~120s default) or POST `/deregister` for an immediate clean-up.
- **Cross-DiLoCo-server dataset sharing.** Out of scope. One DiLoCo
  server per training group.
- **Streaming-DiLoCo (fragment sync).** Fragment sync is
  per-parameter; work-unit dispatch is per-data. Orthogonal; should
  compose without special handling.

## Acceptance criteria

A run with this implemented should be observable as:

1. Start a DiLoCo server with `--default-work-units 1024` pointing at
   a model checkpoint, with a dataset_server reachable on the LAN.
2. Launch any number of workers (1, then 3, then 7, then kill 2)
   without any of them specifying `--num-shards` or `--shard-index`.
3. The DiLoCo view's per-queue heatmap shows units flowing from
   "available" → "issued", with rate roughly proportional to each
   worker's `steps_per_second`.
4. Killing a worker results in its in-flight unit staying issued (not
   recovered). Training continues without intervention. The
   diagnostic UI shows the `(issued - completed)` gap by one unit.
5. Within a single `(dataset_id, shuffle_seed)` queue: no row is
   trained on twice, guaranteed by construction.
6. Worker output directories on disk are distinct per worker. The
   suffix is the operator-supplied `worker_id` (or the scheduler's
   `queue_id` fallback), applied at config-preprocessing time via
   `DILOCO_WORKER_ID`. The directory layout under `models_dir/` is
   directly human-readable (`models_dir/<base>_<worker_id>/`).
7. Attempting to start a second worker with a `worker_id` that's
   already in the registry fails fast: the DiLoCo server returns 409
   on `/register`, the worker exits, and the operator sees a clear
   "worker_id '<id>' is already registered; …" diagnostic in the TTY
   pane. To recover, the operator either bumps `--diloco-worker-id`,
   waits for heartbeat eviction (if the first worker actually died),
   or POSTs `/deregister` explicitly.

---

## Revision 6: composable-level dispatch

Revision 5 placed the wrap as an ``IterableDatasetBackend``
(`WorkUnitBackend`) below ``ComposableIterableDataset``. Two real
problems showed up in operator-facing use:

1. **Slice math was wrong.** Operators routinely use
   `split="train[10000:]"` to reserve the first N rows for eval.
   The backend doesn't know about the wrapper's slice; it carved the
   full N rows into K units. The composable's slice filter then
   *discarded* the first 10000 dispatched rows by misusing
   `WorkUnitBackend.position()` as a yield counter — a kludge
   documented at length in the old code. Workers burned dispatch
   budget on rows they immediately threw away, and percentage slices
   ("train[:25%]") composed even worse.

2. **DDP composition broke for asymmetric topologies.** Templates
   call `ComposableIterableDataset.shard(world_size, rank)` for DDP.
   With one host running DDPx4 and another DDPx8, per-rank shard
   offsets overlap heavily — workers train on the same rows.
   `WorkUnitBackend` couldn't fix this because shard hadn't been
   applied yet when the wrap fired.

The fix moves dispatch into the composable:

- ``ComposableIterableDataset`` carries the load identity
  (`_load_args`, stamped by the loaders) plus dispatch state
  (`_wud_client`, `_wud_worker_id`, `_wud_registered` cache).
- ``_iter_window`` branches: when dispatch is enabled, it lazily
  registers the ``(dataset_id, effective_seed)`` pair with the
  DiLoCo server and drives row emission from the work queue against
  the **post-slice view bounds**. The yield-counter `position()`
  trick is gone.
- ``compute_dataset_id`` absorbs ``slice_start`` / ``slice_end`` so
  two different slices of the same source dataset key separate
  queues. Shard info is **not** absorbed (per the shared-queue model
  below).
- ``shard()`` and ``enable_work_dispatch()`` are mutually exclusive.
  Either ordering raises with a clear message naming the
  asymmetric-DDP failure that would result.

**The shared-queue model.** Under DiLoCo, dispatch IS the
partitioning. All DDP ranks across all DiLoCo hosts compete for
units in one shared queue keyed only by ``(dataset_id, seed)``.
Per-rank queues (the design's first instinct) were rejected because
they fail asymmetric DDP — host A's `shard(4,0)` and host B's
`shard(8,0)` cover overlapping rows, so even with separate queues
the workers would train on the same data.

**Operator config.** ``preprocess_dataset`` accepts a
``shard_dataset.method`` field plus a per-dataset
``partition_purpose: "train" | "eval"`` kwarg. The validity matrix
splits per purpose: train requires cross-host coordination under
DiLoCo (work_units is the only safe choice), eval is replicated
across hosts so conventional sharding is fine and work_units makes
no sense.

`partition_purpose='train'`:

| Config                            | `DILOCO_SERVER` unset | `DILOCO_SERVER` set |
|-----------------------------------|----------------------|---------------------|
| `False`                           | OK                   | OK                  |
| `True` / `{method: conventional}` | OK                   | **error**            |
| `{method: work_units}`            | **error**            | OK                  |

`partition_purpose='eval'`:

| Config                            | `DILOCO_SERVER` unset | `DILOCO_SERVER` set |
|-----------------------------------|----------------------|---------------------|
| `False`                           | OK                   | OK                  |
| `True` / `{method: conventional}` | OK                   | OK (within-host DDP shard) |
| `{method: work_units}`            | **error**            | **error**            |

`lm_training_project.yaml` selects the right value automatically
based on the env / Jinja vars. `load_dataset.yaml` stamps
`partition_purpose` per-singleton (train→"train", eval/test→"eval");
snowflake dataset templates do the same.

**Eval under DiLoCo.** Eval is replicated across DiLoCo hosts (every
host runs the full eval pass; metrics averaged across hosts), but
within each host we want DDP sharding to split the eval workload
across the DDP ranks of that host — otherwise every rank runs the
full eval locally for an identical result, burning W× compute.
``lm_training_project.yaml`` emits ``shard_dataset:
{{ ns.dispatch_batches == False }}`` for eval unconditionally
(True for vanilla DDP, False under dispatch_batches=True), and the
eval-side validity check allows ``conventional + DiLoCo``.

**Multi-epoch and `set_epoch`.** No special handling needed. The
dispatch's lazy-register cache is keyed by
``(dataset_id, effective_seed)``. ``set_epoch(N>0)`` changes the
seed, so the cache misses and the composable registers a fresh
queue with the new seed.

**DataLoader `num_workers > 1`.** Allowed and correct (each forked
worker gets its own ``DiLoCoClient``; server atomicity prevents
double-issuance). Discouraged in operator docs because it
multiplies connection count, reduces shuffle quality, and rarely
helps throughput for iterable datasets in this codebase.

---

Feedback welcome. The phased plan keeps backend correctness work
ahead of UI work — phase 1 can be exercised end-to-end with `curl`
before any client code is touched.
