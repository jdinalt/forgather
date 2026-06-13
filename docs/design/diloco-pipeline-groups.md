# DiLoCo + pipeline parallel: per-rank workers with server-aware groups

## Motivation

DiLoCo's pre-#84 contract assumed every worker held the **full** model.
The server's fingerprint check (`_diff_model_fingerprint` in
`src/forgather/ml/diloco/server.py`) compared each worker's
`param_shapes` to the master `_param_names` set and 422'd on any
mismatch. The sync barrier counted one submission per `worker_id` and
the outer optimizer averaged across all workers per-name.

Under pipeline parallelism each rank holds only a **slice** of the
model — different parameter names, different shapes per rank. The
trainer's `self.model` attribute is a meta-device placeholder kept
around for callbacks that just want to inspect structure; actual
on-device parameters live in `trainer.pipeline_modules`, a per-rank
`List[nn.Module]` (see `pipeline_trainer.py:_prepare_model`). Calling
`.clone()` on the meta-device tensors at `worker.py:_save_global_params_snapshot`
raises `NotImplementedError`.

The "gather to leader" workaround (have stage-0-rank-0 collect a
full-model snapshot from all pipeline ranks via NCCL and talk to the
server alone) was rejected because it requires the leader to hold a
CPU snapshot of the full model — that defeats the primary purpose of
pipeline parallelism (training models too large to fit on a single
device) and doesn't scale to forgather's target workloads.

This document describes the resource-efficient design: each pipeline
rank registers as its own DiLoCo worker, declaring its slice and its
group membership; the server coordinates the per-rank submissions
into one logical DiLoCo job.

## Group data model

A `WorkerGroup` (dataclass in `server.py`) is a set of workers that
together cover the server's full parameter set.

```python
@dataclass
class WorkerGroup:
    group_id: str
    pp_world_size: int
    members: Dict[int, str]           # pp_rank -> worker_id
    member_param_names: Dict[int, set]
    created_at: float
    sealed: bool
```

Solo workers (the pre-#84 common case) form a degenerate group of one:
`group_id == worker_id`, `pp_world_size == 1`, and the seal-time
coverage check immediately enforces equality with the full param set
— exactly the pre-#84 contract.

Pipeline-parallel workers register with `pp_world_size > 1` and each
member declares only its rank's slice. The group is "sealed" once
`pp_world_size` members have registered, at which point the union of
all member slices must exactly cover the server's full param set.

Two indices on the server:

- `_groups: Dict[group_id, WorkerGroup]` — the group registry.
- `_worker_to_group: Dict[worker_id, group_id]` — inverse index used
  by submit/death paths to find the owning group from a `worker_id`.

Both dicts are protected by the existing `_workers_lock`.

## Registration protocol

`POST /register` payload gains an optional `group` block:

```json
{
  "worker_id": "alpha_pp1",
  "hostname": "host42",
  "sync_every": 500,
  "upload_dtype": "bf16",
  "upload_sr": false,
  "download_dtype": "fp32",
  "download_sr": false,
  "bf16_comm": true,
  "param_shapes": {
    "decoder.layers.4.attn.q_proj.weight": [768, 768],
    "decoder.layers.5.attn.q_proj.weight": [768, 768]
  },
  "group": {
    "group_id": "alpha",
    "pp_rank": 1,
    "pp_world_size": 4
  }
}
```

The `group` block is **optional**: workers that omit it form a solo
group of one, preserving the pre-#84 contract for single-GPU / DDP /
FSDP2 workers.

### Server-side flow (`_handle_register`)

1. **Slice fingerprint check** (`_diff_slice_fingerprint`): every name
   in `param_shapes` must appear in the server with matching shape;
   extras (worker has a name the server doesn't) are still a hard
   error. Missing names are *allowed* here — a sliced rank holds only
   its slice. Returns 422 on failure.
2. **Group geometry validation**: `pp_world_size >= 1`,
   `0 <= pp_rank < pp_world_size`, else 400.
3. **Async-mode rejection**: when `pp_world_size > 1` and the server
   runs in `async_mode`, 400 with a clear error. Async barrier
   semantics with disjoint slice contributions is fragile; out of
   scope for #84.
4. **Find or create the group**:
   - If sealed → 409 ("group is sealed; deregister an existing member
     or use a different `group_id`").
   - If existing group declared a different `pp_world_size` → 422.
   - If `pp_rank` slot already filled → 409.
5. **Register the member**: store `slice_shapes` in
   `group.member_param_names[pp_rank]`.
6. **Seal + coverage check**: if `len(members) == pp_world_size`, run
   `_check_group_coverage(group)`. On failure, return 422 with kind
   `group_coverage` AND atomically roll back every member registered
   so far (`_rollback_group`). On success, mark `group.sealed = True`.
7. **Register in `_workers` / `_worker_to_group`** and return the
   current global params.

### Coverage check (`_check_group_coverage`)

The union of every member's `param_shapes.keys()` must equal the
server's `_param_names` set. Names missing from the union → coverage
failure. Duplicate names across slices are **allowed** — this is the
tied-parameter / aliased-weight case (e.g. weight-tied embedding and
lm_head, where the embed lives on stage 0 and the lm_head's
transposed view on the final stage). The per-name averaging on the
apply path handles same-data alias contributions correctly: they're
identical pseudo-gradients and average to the same value.

## Sync barrier

Each submission goes into `_pending_pseudograds[worker_id]`. The
release condition becomes `_round_complete()`:

```python
def _round_complete(self) -> bool:
    if self._round_expected_workers is None:
        return False
    return self._round_expected_workers.issubset(set(self._pending_pseudograds))
```

For solo groups (`pp_world_size == 1`) this collapses to the pre-#84
length check (every registered worker_id has a pending submission).
For pipeline groups every rank of every group must have submitted its
slice.

`_apply_outer_optimizer` switches to **contributors-only** per-name
aggregation:

```python
for i, name in enumerate(self._param_names):
    contributors = [
        wpg[name] for wpg in self._pending_pseudograds.values()
        if name in wpg
    ]
    if not contributors:
        logger.error(...)
        self._param_list[i].grad = None
        continue
    avg = contributors[0].clone()
    for pg in contributors[1:]:
        avg.add_(pg)
    avg.div_(len(contributors))
    self._param_list[i].grad = avg
self.outer_optimizer.step()
```

Each name is averaged over the workers whose slice contained it — for
solo groups that's the whole worker set; for pipeline groups it's
typically one rank per group (G groups → G contributors). Tied
aliases held in multiple ranks contribute identically and average to
the same value.

After the barrier releases, the post-sync response is filtered to the
requesting rank's slice on the way out (via `_params_for_worker`).
Each pp_rank only ever sees its own
`WorkerGroup.member_param_names[pp_rank]` keys in the response — the
worker would discard the rest anyway (`PipelineParamView.apply_global`
silently drops names it doesn't own), but the wire transfer would
otherwise carry the full averaged model. Solo workers and untracked
clients still receive the full state, preserving the pre-#84
ParamView fingerprint contract.

The response is then cast to `download_dtype` (default `fp32`,
configurable to `bf16` with optional stochastic rounding) by
`_cast_for_download` — a separate optimization that halves return-path
bandwidth at the cost of an open research question on convergence
impact (see issue #130). The PP-group slice filter and the dtype
cast compose orthogonally: a PP rank in a bf16-download group gets
its slice in bf16, not the full model in fp32.

## Fragments-within-groups

The same model extends to streaming fragments (`num_fragments > 1`).
Transformer-block boundaries are discovered **globally** from the full
model (`FragmentManager(self.param_view, num_fragments,
boundary_source=self.model)`), assigned to fragments (strided or
sequential), then **filtered to each rank's slice**. The `fragment_id`
is therefore a logical index shared across ranks: rank 0's
`fragment_id_k` and rank 1's `fragment_id_k` cover different names
(each rank's slice intersected with that fragment's blocks) but fire at
the same local step (every rank's pipeline scheduler steps in
lockstep). Doing discovery globally-then-filter — rather than
partitioning each rank's slice independently — is what keeps the
logical ids consistent so the server's per-`fragment_id` barrier
composes.

Server-side per-fragment barrier release:
`_fragment_round_complete(fragment_id)` — every expected worker has
submitted for this fragment. The completed result is now keyed
per-worker: `Dict[(frag_id, round), Dict[worker_id, Dict[name, Tensor]]]`,
so each rank receives back only the names it submitted (its slice's
portion of the fragment), avoiding O(`pp_world_size`) over-fetch.

## Worker-side: `ParamView` abstraction

`src/forgather/ml/diloco/param_view.py` introduces a `ParamView`
protocol with two implementations:

- `SimpleModelParamView(model)` — wraps a single `nn.Module`. Used for
  non-pipeline trainers (single-GPU, DDP, FSDP2). Pre-#84 behavior.
- `PipelineParamView(pipeline_modules, sharing_metadata)` — wraps the
  per-rank stage module list a pipeline trainer stores on
  `trainer.pipeline_modules`. Exposes only this rank's slice.

The worker's five model-touching call sites route through the view:

| Call site | View method |
|---|---|
| `_get_worker_info` (registration `param_shapes`) | `param_shapes()` |
| `_save_global_params_snapshot` | `snapshot()` |
| `_compute_pseudogradients` | `compute_pseudograds(snap)` |
| `_apply_global_params` | `apply_global(global_params)` |
| `_broadcast_params_from_leader` (no-op when `pp_world_size > 1`) | `named_parameters()` |

Both views use `named_parameters(remove_duplicate=False)` so tied
aliases all appear. This matches the server's storage (built from
`model_state_dict.values()` with `.clone()` per slot — aliases get
independent storage on the server) and fixes a latent staleness
issue: under the new contract every alias slot receives updates,
preventing divergence when the server is loaded from a pipeline-
trained checkpoint (which writes aliases via
`make_state_dict(remove_duplicate=False)`).

## Worker death: atomic group eviction

`_handle_worker_death` (called by the health monitor on heartbeat
timeout or by explicit `/deregister`) is group-aware. When the dying
worker belongs to a pipeline group (`pp_world_size > 1`), every
member of the group is evicted atomically. The remaining members
would hold only a partial slice and could not produce valid
pseudo-gradients; the group's only correctness-preserving option is
to die together.

For solo groups (`pp_world_size == 1`) behavior is unchanged from
pre-#84: just the dying worker is removed, its now-empty group entry
is cleaned up.

After eviction, the sync barrier release check is re-evaluated;
surviving (other) groups may now be able to proceed.

## Callback wiring

`DiLoCoCallback.on_train_begin` (`src/forgather/ml/trainer/callbacks/diloco_callback.py`)
detects the pipeline trainer via duck-typing on
`trainer.pipeline_modules`. When present:

- Constructs `PipelineParamView(trainer.pipeline_modules, trainer.sharing_metadata)`.
- Derives `worker_id = f"{base_id}_pp{pp_rank}"` from the operator's
  `--diloco-worker-id` base.
- Passes `group_id, pp_rank, pp_world_size` to `DiLoCoWorker`.

The `trainer` kwarg is already threaded through callbacks by
`BaseTrainer._dispatch_event`.

## Multi-node dispatch (cluster fanout with shared base worker_id)

The pipeline-group semantics above are agnostic to where the per-rank
processes physically run. A single host launching `torchrun
--nproc-per-node N` produces N processes that all read the same
environment (and so the same `DILOCO_WORKER_ID` base, from which the
callback derives `_pp<rank>`). A multi-node bundle reaches the same
end state via a different path:

- `forgather submit --global --diloco-server <id>` (or the webui
  equivalent) builds a cluster bundle of N per-rank training jobs,
  one per participating host.
- The master resolves the base `worker_id` **once** (auto-mints a
  memorable default when the operator didn't supply one) and ships
  the identical `diloco` block to every peer's `training_local`
  enqueue. The peer's scheduler translates it into `DILOCO_*` env
  via `_diloco_env_from_job_params`.
- The DiLoCo bearer token is resolved on the master from a locally-
  spawned `diloco_server` JobRecord and forwarded through every
  peer's `extra_env`. Remote peers don't see that record, so loopback
  auto-discovery would otherwise fail there.
- Each peer's `torchrun` rendezvous via the bundle's shared
  `rdzv_endpoint` / `rdzv_id`. Inside the torchrun world the existing
  `PipelineTrainer` initialization and `DiLoCoCallback` PP detection
  run unchanged — the callback registers each rank under
  `{base}_pp{rank}` and the server collapses them into one group.

Load-bearing invariants the master must hold (and the dispatch tests
enforce):

- Base `worker_id` is identical across every per-peer payload of a
  bundle. If two peers ever auto-generate independently the group
  fragments into N solo workers and parameter averaging silently
  degrades.
- The bearer token reaches every peer's `extra_env` (when the master
  can resolve it locally), so the worker authenticates from the
  routable server address rather than relying on loopback discovery.

The bundle record persists the resolved `diloco` block
(`ClusterJob.diloco`) so both the CLI submit output and the webui's
cluster jobs view can label the bundle as "one DiLoCo worker group"
without peeking into per-peer queue items.

K > 1 multi-node DiLoCo groups (e.g. two 2-node PP bundles averaged
via DiLoCo) is achievable today by submitting K separate `forgather
submit --global` invocations to **non-overlapping** member sets, each
pinning the same `--diloco-server` but a distinct `--worker-id` base.
The non-overlap requirement matters: the per-node GPU pool is FIFO,
so two groups competing for GPUs on the same host serialize, which
defeats the point. A future `--diloco-count K` flag could partition
members and mint distinct bases automatically; until then, do it
manually.

## Out of scope / future work

- **Security model.** The DiLoCo server's HTTP wire is unauthenticated
  (trusted-LAN model). A worker can claim any `group_id` / `pp_rank`
  without verification; pipeline-group membership is operator-
  conventional, not enforced. The unsealed-group and ghost-worker
  guards in this PR protect against accidental misuse but not against
  a network adversary. Authentication, audit logging, and identity-
  bound group membership are tracked in issue #90.
- **Pipeline + within-stage DDP composition.** Each pipeline rank
  could itself wrap its stage modules with `DistributedDataParallel`,
  giving a 2-D rank topology (`pp_world_size × dp_world_size`). The
  `PipelineParamView` plumbing leaves room for a `pp_group:
  ProcessGroup` argument that would broadcast post-sync params across
  the within-stage DDP sub-group only. Not currently composed by the
  forgather trainer.
- **Async mode + groups.** Rejected at worker construction with a
  clear error. Designing per-group submission queues with DN
  buffering semantics is a separate piece of work.
- **DyLU per-group.** `recommended_sync_every` doesn't compose with
  groups (all ranks of a group must sync at the same step).
  Functionally a no-op under groups in this PR; a follow-up could
  average the recommendation across a group and broadcast a single
  value to all members.
- **Webui group affordance.** The per-rank workers (`alpha_pp0`,
  `alpha_pp1`) surface naturally in the existing worker list. A
  future enhancement could add a group column or aggregator.
- **Heartbeat grace window.** Group eviction fires on first member's
  death (operator's chosen policy). A grace window for siblings —
  recognizing that a host crash often takes its ranks down within
  seconds of each other — is a quality-of-life follow-up.

## Related code

- `src/forgather/ml/diloco/server.py` — group registry, barrier,
  apply, death paths.
- `src/forgather/ml/diloco/worker.py` — `ParamView` plumbing,
  registration with the `group` block.
- `src/forgather/ml/diloco/param_view.py` — abstraction definitions.
- `src/forgather/ml/diloco/fragments.py` — fragment partitioning
  duck-types its `model` argument so a `ParamView` is accepted, and discovers
  transformer-block boundaries from the full model via `boundary_source`.
- `src/forgather/ml/trainer/callbacks/diloco_callback.py` — pipeline
  detection + worker-id derivation.
- `src/forgather/ml/trainer/pipeline/pipeline_trainer.py:_prepare_model`
  — produces `self.pipeline_modules` and `self.sharing_metadata`.
- `src/forgather/ml/sharded_checkpoint.py:create_sharing_metadata` —
  source of the tied-name equivalence classes.
- `tests/unit/ml/diloco/test_server_groups.py` — server group
  registration, barrier, eviction tests.
- `tests/unit/ml/diloco/test_param_view.py` — worker view tests.

## See also

- [`diloco-security.md`](diloco-security.md) — auth, TLS, mTLS, audit
  log, and the two-port bulk plane (issue #90). The group-membership
  identity binding deferred there is the natural next step beyond
  this PR.
