# DiLoCo Architecture: Maintainer Reference

> Part of the DiLoCo [Architecture & Maintainer Guide](diloco-architecture.md).
> This page is the maintainer reference: server state persistence, the CLI
> layer, testing, troubleshooting, extension points, checkpoint state selection
> + empty-meta construction, and known limitations. See the
> [design page](diloco-architecture.md) for the architecture and the
> [runtime behavior](diloco-architecture-runtime.md) page for lifecycle and
> fault tolerance.

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
registered (see the `/known_workers` endpoint on the
[design page](diloco-architecture.md#wire-protocol)). Persisting it here is what lets a
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

The upload wire cast lives in the sync backend (`HttpStarBackend`, via
`wire_cast.cast_for_upload`); `_compute_pseudogradients` returns the raw
pseudo-gradient. To add quantization (e.g., int8, sparse encoding):

1. Apply the compression in the backend's `synchronize` / `synchronize_fragment`
   (extend `wire_cast`, or add a backend that owns a different representation)
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

2. **Fragments split by whole blocks, not balanced tensor size.** Fragments are
   groups of whole transformer blocks (Streaming DiLoCo); the non-block params
   (embeddings on the first fragment, final norm + LM head on the last) make
   those fragments somewhat heavier. The no-block-plan *fallback* splits by
   parameter count, which can be even more size-imbalanced. A size-balanced
   split would improve streaming overlap in either mode.

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
