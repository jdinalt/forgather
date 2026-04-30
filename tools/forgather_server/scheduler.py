"""GPU-aware FIFO+priority dispatcher.

Lifecycle:
    queue_store.QueueItem  --(dispatch)-->  job_records.JobRecord
       (waiting for GPUs)                    (starting → running → terminal)

A single asyncio task polls the queue + GPU state every ``TICK_SECONDS``,
reaps any finished JobRecords, attempts to dispatch the head of the queue
when enough idle GPUs exist, and correlates running records with
TrainerControlClient endpoint files.

The scheduler starts **enabled** so a freshly-restarted server picks
up dispatch immediately — the operator can pause it anytime via
``POST /api/queue/scheduler {enabled: false}`` (or the ⏸ button in the
sidebar header). Enqueue is independent and always works regardless of
the scheduler's enabled flag; dispatch only happens when enabled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from forgather import trainer_control

from . import gpu_monitor, job_records, launcher, queue_store
from .job_records import RUNNING_STATUSES, TERMINAL_STATUSES, JobRecord
from .paths import jobs_tty_dir
from .queue_store import LOCAL_NODE, QueueItem

# NOTE (multi-node): today the scheduler directly calls gpu_monitor.snapshot()
# and launcher.spawn_training_process() — both of which always act on the
# local machine. The intended seam for multi-node is to put a
# ``NodeClient`` abstraction between here and those modules, with a pool of
# clients keyed by hostname. Each JobRecord already carries a ``node``
# field; assigning it below is a one-line change away from "pick a node".

log = logging.getLogger("forgather_server.scheduler")

TICK_SECONDS = 2.0


@dataclass
class SchedulerState:
    enabled: bool = True
    # In-memory map of running JobRecord.queue_id -> Popen handle.
    #
    # Persisted state never holds a Popen — only PID/pgid. A ``None`` value
    # means the item was re-attached after a server restart: the trainer is
    # still running in its own session but we no longer own the Popen, so
    # the reap loop has to poll via psutil instead of Popen.poll().
    running: Dict[str, Optional[subprocess.Popen]] = field(default_factory=dict)
    tick_count: int = 0
    last_tick_at: Optional[float] = None
    _lock: Lock = field(default_factory=Lock)


_state = SchedulerState()


def get_state() -> SchedulerState:
    return _state


def set_enabled(enabled: bool) -> None:
    with _state._lock:
        _state.enabled = bool(enabled)
    log.info("scheduler enabled=%s", _state.enabled)


def _idle_gpu_indices() -> List[int]:
    """Indices of GPUs the scheduler is allowed to assign right now.

    A GPU is eligible iff it has no compute processes AND has not been
    excluded by the operator via CUDA_VISIBLE_DEVICES at server start AND
    has not been runtime-disabled by the user via the web UI.
    """
    return [
        g.index
        for g in gpu_monitor.snapshot()
        if not g.processes and not g.excluded and not g.disabled
    ]


def _reserved_gpu_set() -> set[int]:
    """GPU indices currently allocated to JobRecords we are running."""
    reserved: set[int] = set()
    for r in job_records.list_records():
        if r.status in RUNNING_STATUSES:
            reserved.update(r.gpu_indices)
    return reserved


def _pid_is_alive(pid: int) -> bool:
    """True if ``pid`` refers to a still-running (non-zombie) process."""
    try:
        import psutil
    except ImportError:
        # Fall back to kill(0) probing — less precise (zombies look alive)
        # but enough for re-attach.
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _reap_finished() -> None:
    """Transition running items to terminal state when their process exits.

    Items we launched ourselves (``Popen`` handle in ``_state.running``)
    are reaped via ``poll()`` — that gives a real exit code. Items
    *re-attached* after a server restart (``None`` handle) are polled via
    psutil since we're no longer their parent; in that case the exit
    code is unknowable from outside, so terminal status falls back to a
    simple alive/dead check and we record ``exit_code=None``.
    """
    with _state._lock:
        ids = list(_state.running.keys())
    for qid in ids:
        proc = _state.running.get(qid)
        record = job_records.get_record(qid)
        if record is None:
            with _state._lock:
                _state.running.pop(qid, None)
            continue

        rc: Optional[int]
        reattached = proc is None
        if not reattached:
            rc = proc.poll()  # type: ignore[union-attr]
            if rc is None:
                continue  # still alive
        else:
            if record.pid is None:
                # Re-attached item with no pid (shouldn't happen) — clean up.
                rc = None
            else:
                if _pid_is_alive(record.pid):
                    continue  # still alive
                rc = None  # can't retrieve exit code of a non-child process

        if record.status == "aborted":
            new_status = "aborted"
        elif reattached:
            # We can't tell success from failure without an exit code.
            # "done" is the least-wrong default; the UI can distinguish
            # re-attached jobs by the missing exit_code if needed.
            new_status = "done"
        elif rc == 0:
            new_status = "done"
        else:
            new_status = "failed"

        # Re-read the record under the lock right before writing so a
        # concurrent abort (which transitions status="aborted") doesn't
        # get clobbered by our reap path racing on the same record.
        # update_only_if_running narrows the write to non-terminal states.
        job_records.update_if_not_terminal(
            qid,
            status=new_status,
            exit_code=rc,
            finished_at=time.time(),
        )
        with _state._lock:
            _state.running.pop(qid, None)
        if rc is not None:
            log.info("reaped %s: rc=%d status=%s", qid, rc, new_status)
        else:
            log.info(
                "reaped re-attached %s: pid gone, status=%s (no exit code)",
                qid,
                new_status,
            )


def _try_dispatch() -> None:
    """Walk the queue and place jobs onto idle, eligible GPUs.

    Algorithm (see the "Scheduling algorithm" section of the README for
    the operator-facing version):

    1. Sort queue items by priority desc, then by submission time asc
       (FIFO within a priority band).
    2. Build the idle-and-unreserved GPU pool. A GPU is idle when it has
       no compute processes AND is not excluded (via
       ``CUDA_VISIBLE_DEVICES``) AND is not runtime-disabled (via the
       UI toggle).
    3. For each queue item in order, compute the set of GPUs whose
       ``min_priority`` gate the item clears (``gpu.min_priority <=
       item.priority``). If fewer eligible GPUs exist than the item's
       ``requested_gpus``, skip the item — a later item may have a
       lower requested-count or a higher priority and still be placable.
       The queue head is not allowed to block the rest of the queue on
       an over-constrained request.
    4. **Within the eligible set, prefer GPUs with the highest
       ``min_priority`` the item still clears.** That way a
       priority-10 job running in a pool with ``[gpu0 min=0, gpu5
       min=10]`` takes gpu5 first and leaves gpu0 free for a
       priority-0 job that *can't* use gpu5. Sort key is
       ``(-min_priority, index)`` — the index tiebreak keeps allocation
       deterministic.
    5. Commit: remove the chosen GPUs from ``remaining_idle``, launch
       the item (which moves it from queue_store to job_records and
       spawns torchrun), then continue to the next queue item.

    This is "best-fit-to-threshold" placement: each job lands on the
    most-reserved GPU it qualifies for, leaving less-reserved GPUs
    available for jobs that need them.
    """
    if not _state.enabled:
        return

    queued = queue_store.list_items()
    if not queued:
        return
    queued.sort(key=lambda it: (-it.priority, it.submitted_at))

    reserved = _reserved_gpu_set()
    # Build a snapshot so we can filter per-GPU min_priority cheaply.
    gpu_snap = {g.index: g for g in gpu_monitor.snapshot()}
    idle = [i for i in _idle_gpu_indices() if i not in reserved]

    remaining_idle = list(idle)
    for it in queued:
        # Zero-GPU jobs (e.g. tensorboard) bypass the placement search —
        # they need no reservation and dispatch as soon as the queue loop
        # reaches them.
        if it.requested_gpus == 0:
            _launch(it, [])
            continue
        if not remaining_idle:
            # All GPUs allocated to earlier items this tick — but keep
            # scanning the queue in case a later zero-GPU item is still
            # dispatchable.
            continue
        # Best-fit to the priority threshold: prefer GPUs whose
        # min_priority is as close to (but not above) the item's
        # priority as possible. Ties broken by index for determinism.
        eligible = sorted(
            (i for i in remaining_idle if gpu_snap[i].min_priority <= it.priority),
            key=lambda i: (-gpu_snap[i].min_priority, i),
        )
        if len(eligible) < it.requested_gpus:
            # Not enough eligible GPUs for this item — skip it, maybe the
            # next item has a higher priority or needs fewer GPUs.
            continue
        assigned = sorted(eligible[: it.requested_gpus])
        # Remove assigned from the remaining pool (may be a subset of idle).
        assigned_set = set(assigned)
        remaining_idle = [i for i in remaining_idle if i not in assigned_set]
        _launch(it, assigned)


def _launch(item: QueueItem, gpu_indices: List[int]) -> None:
    """Move a queue item to a JobRecord and spawn the appropriate subprocess.

    Training jobs spawn ``scripts/train_script.py`` and correlate with
    TrainerControlClient; eval jobs spawn ``scripts/eval_script.py`` and
    are fire-and-forget. The shared generic lifecycle (TTY capture,
    PID-based reaping, GPU pinning) lives in :func:`_spawn_subprocess`.
    """
    tty_path = jobs_tty_dir() / f"{item.queue_id}.tty"
    record = JobRecord(
        queue_id=item.queue_id,
        project_dir=item.project_dir,
        config=item.config,
        dynamic_args=dict(item.dynamic_args),
        requested_gpus=item.requested_gpus,
        priority=item.priority,
        submitted_at=item.submitted_at,
        job_type=item.job_type,
        job_params=dict(item.job_params),
        node=LOCAL_NODE,
        gpu_indices=gpu_indices,
        status="starting",
        started_at=time.time(),
        tty_log_path=str(tty_path),
    )
    job_records.add_record(record)
    queue_store.remove_item(item.queue_id)
    log.info(
        "launching %s (%s) on GPU %s (config=%s)",
        item.queue_id,
        item.job_type,
        gpu_indices,
        item.config,
    )

    try:
        if item.job_type == "eval":
            params = item.job_params
            result = launcher.spawn_eval_process(
                eval_project=params["eval_project"],
                eval_template=params["eval_template"],
                model_path=params["model_path"],
                checkpoint_path=params.get("checkpoint_path"),
                no_checkpoint=bool(params.get("no_checkpoint", False)),
                trainer=params.get("trainer", "ddp"),
                batch_size=params.get("batch_size"),
                max_length=params.get("max_length"),
                max_steps=int(params.get("max_steps", -1)),
                dtype=params.get("dtype", "bfloat16"),
                attn_implementation=params.get("attn_implementation", "sdpa"),
                compile=bool(params.get("compile", False)),
                output_dir=params.get("output_dir"),
                gpu_indices=gpu_indices,
                tty_log_path=tty_path,
            )
        elif item.job_type == "inference":
            params = item.job_params
            result = launcher.spawn_inference_process(
                model_path=params["model_path"],
                port=int(params["port"]),
                host=params.get("host", "127.0.0.1"),
                dtype=params.get("dtype"),
                attn_implementation=params.get("attn_implementation"),
                checkpoint_path=params.get("checkpoint_path"),
                from_checkpoint=bool(params.get("from_checkpoint", False)),
                compile=bool(params.get("compile", False)),
                disable_kv_cache=bool(params.get("disable_kv_cache", False)),
                ignore_eos=bool(params.get("ignore_eos", False)),
                chat_template=params.get("chat_template"),
                cache_implementation=params.get("cache_implementation"),
                compile_args=params.get("compile_args"),
                log_level=params.get("log_level", "INFO"),
                gpu_indices=gpu_indices,
                tty_log_path=tty_path,
            )
        elif item.job_type == "tensorboard":
            params = item.job_params
            ri = params.get("reload_interval")
            result = launcher.spawn_tensorboard_process(
                logdir=params["logdir"],
                port=int(params["port"]),
                host=params.get("host"),
                bind_all=bool(params.get("bind_all", False)),
                window_title=params.get("window_title"),
                reload_interval=int(ri) if ri is not None else None,
                reload_multifile=bool(params.get("reload_multifile", False)),
                samples_per_plugin=params.get("samples_per_plugin"),
                tty_log_path=tty_path,
            )
        elif item.job_type == "convert":
            params = item.job_params
            cps = params.get("converter_paths")
            if isinstance(cps, str):
                cps = [cps]
            elif not isinstance(cps, list):
                cps = None
            ml = params.get("max_length")
            result = launcher.spawn_convert_process(
                src_model_path=params["src_model_path"],
                dst_model_path=params["dst_model_path"],
                reverse=bool(params.get("reverse", False)),
                model_type=params.get("model_type"),
                dtype=params.get("dtype"),
                max_length=int(ml) if ml is not None else None,
                checkpoint_path=params.get("checkpoint_path"),
                device=params.get("device"),
                generation_test=bool(params.get("generation_test", False)),
                dry_run=bool(params.get("dry_run", False)),
                prompt=params.get("prompt"),
                compare_text_file=params.get("compare_text_file"),
                debug_params=bool(params.get("debug_params", False)),
                chat_template_path=params.get("chat_template_path"),
                add_tokens=params.get("add_tokens"),
                skip_default_tokens=bool(params.get("skip_default_tokens", False)),
                converter_paths=cps,
                log_level=params.get("log_level", "INFO"),
                gpu_indices=gpu_indices,
                tty_log_path=tty_path,
            )
        elif item.job_type == "finalize":
            params = item.job_params
            result = launcher.spawn_finalize_process(
                source=params["source"],
                dest=params["dest"],
                checkpoint=params.get("checkpoint"),
                add_tokens=params.get("add_tokens"),
                skip_default_tokens=bool(params.get("skip_default_tokens", False)),
                chat_template_path=params.get("chat_template_path"),
                no_auto_stop_tokens=bool(params.get("no_auto_stop_tokens", False)),
                stop_tokens=params.get("stop_tokens"),
                generation_config=params.get("generation_config"),
                keep_optimizer=bool(params.get("keep_optimizer", False)),
                root_copy=bool(params.get("root_copy", False)),
                safetensors=bool(params.get("safetensors", False)),
                dtype=params.get("dtype"),
                device=params.get("device"),
                dry_run=bool(params.get("dry_run", False)),
                log_level=params.get("log_level", "INFO"),
                gpu_indices=gpu_indices,
                tty_log_path=tty_path,
            )
        elif item.job_type == "mkdocs":
            params = item.job_params
            watch = params.get("watch")
            if isinstance(watch, str):
                watch = [watch]
            elif not isinstance(watch, list):
                watch = None
            result = launcher.spawn_mkdocs_process(
                config_file=params["config_file"],
                port=int(params["port"]),
                host=params.get("host", "127.0.0.1"),
                strict=bool(params.get("strict", False)),
                livereload=bool(params.get("livereload", True)),
                dirty=bool(params.get("dirty", False)),
                watch=watch,
                tty_log_path=tty_path,
            )
        elif item.job_type == "model":
            params = item.job_params
            cd = params.get("compile_dynamic")
            result = launcher.spawn_model_process(
                project_dir=item.project_dir,
                config_name=item.config,
                subcommand=params.get("subcommand", "construct"),
                dynamic_args=item.dynamic_args,
                device=params.get("device"),
                dtype=params.get("dtype"),
                no_init_weights=bool(params.get("no_init_weights", False)),
                load_from_checkpoint=params.get("load_from_checkpoint"),
                gradient_checkpointing=bool(
                    params.get("gradient_checkpointing", False)
                ),
                fuse_optim_with_backward=bool(
                    params.get("fuse_optim_with_backward", False)
                ),
                refresh_model=bool(params.get("refresh_model", False)),
                save_checkpoint=bool(params.get("save_checkpoint", False)),
                safetensors=bool(params.get("safetensors", False)),
                batch_size=(
                    int(params["batch_size"])
                    if params.get("batch_size") is not None
                    else None
                ),
                sequence_length=(
                    int(params["sequence_length"])
                    if params.get("sequence_length") is not None
                    else None
                ),
                steps=(
                    int(params["steps"]) if params.get("steps") is not None else None
                ),
                lr=(float(params["lr"]) if params.get("lr") is not None else None),
                dataset_project=params.get("dataset_project"),
                dataset_config=params.get("dataset_config"),
                packed=bool(params.get("packed", False)),
                compile=bool(params.get("compile", False)),
                compile_backend=params.get("compile_backend"),
                compile_mode=params.get("compile_mode"),
                compile_dynamic=(None if cd is None else bool(cd)),
                compile_fullgraph=bool(params.get("compile_fullgraph", False)),
                amp=params.get("amp"),
                gpu_indices=gpu_indices,
                tty_log_path=tty_path,
            )
        elif item.job_type == "dataset":
            params = item.job_params
            features = params.get("features")
            if isinstance(features, str):
                features = [features]
            elif not isinstance(features, list):
                features = None
            result = launcher.spawn_dataset_process(
                project_dir=item.project_dir,
                config_name=item.config,
                dynamic_args=item.dynamic_args,
                tokenizer_path=params.get("tokenizer_path"),
                pp=bool(params.get("pp", False)),
                histogram=bool(params.get("histogram", False)),
                target=params.get("target"),
                histogram_samples=(
                    int(params["histogram_samples"])
                    if params.get("histogram_samples") is not None
                    else None
                ),
                examples=(
                    int(params["examples"])
                    if params.get("examples") is not None
                    else None
                ),
                features=features,
                tokenized=bool(params.get("tokenized", False)),
                num_shards=(
                    int(params["num_shards"])
                    if params.get("num_shards") is not None
                    else None
                ),
                shard_index=(
                    int(params["shard_index"])
                    if params.get("shard_index") is not None
                    else None
                ),
                select_range=params.get("select_range"),
                seed=(int(params["seed"]) if params.get("seed") is not None else None),
                example_stride=(
                    int(params["example_stride"])
                    if params.get("example_stride") is not None
                    else None
                ),
                truncate=(
                    int(params["truncate"])
                    if params.get("truncate") is not None
                    else None
                ),
                tty_log_path=tty_path,
            )
        else:
            result = launcher.spawn_training_process(
                project_dir=item.project_dir,
                config_name=item.config,
                dynamic_args=item.dynamic_args,
                gpu_indices=gpu_indices,
                tty_log_path=tty_path,
            )
    except Exception as e:
        log.exception("launch failed for %s", item.queue_id)
        job_records.update_record(
            item.queue_id,
            status="failed",
            error=str(e),
            finished_at=time.time(),
        )
        return

    job_records.update_record(item.queue_id, status="running", pid=result.pid)
    with _state._lock:
        _state.running[item.queue_id] = result.proc


def _pid_ancestors(pid: int) -> List[int]:
    """Return ``pid`` + chain of parent PIDs as far as psutil can walk."""
    try:
        import psutil
    except ImportError:
        return [pid]
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return [pid]
    try:
        return [pid] + [p.pid for p in proc.parents()]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return [pid]


def _correlate_running_records() -> None:
    """Match unattributed running JobRecords to TrainerControlClient endpoints.

    A JobRecord is correlated when an endpoint.json's PID is descended
    from the record's torchrun PID. We also symlink the captured TTY into
    ``logs_dir/tty.log`` so the durable copy lives alongside the trainer's
    other artifacts.
    """
    # Only training jobs register with TrainerControlClient — eval / other
    # fire-and-forget job types never write an endpoint.json, so skip them
    # or the walker wastes work on every tick.
    pending = [
        r
        for r in job_records.list_records()
        if r.status == "running"
        and r.job_id is None
        and r.pid is not None
        and r.job_type == "training"
    ]
    if not pending:
        return

    try:
        endpoints = trainer_control.list_jobs()
    except Exception as e:
        log.debug("list_jobs failed during correlation: %s", e)
        return

    by_launcher_pid: Dict[int, JobRecord] = {r.pid: r for r in pending}  # type: ignore[misc]

    for ep in endpoints:
        if ep.logging_dir is None and ep.output_dir is None:
            continue
        for anc in _pid_ancestors(ep.pid):
            record = by_launcher_pid.get(anc)
            if record is None:
                continue
            updates = {"job_id": ep.job_id}
            if ep.logging_dir and not record.logs_dir:
                updates["logs_dir"] = ep.logging_dir
            if ep.output_dir and not record.output_dir:
                updates["output_dir"] = ep.output_dir
            job_records.update_record(record.queue_id, **updates)
            if ep.logging_dir and record.tty_log_path:
                _try_link_tty(record.tty_log_path, ep.logging_dir)
            break


def _try_link_tty(tty_path: str, logs_dir: str) -> None:
    try:
        logs_path = Path(logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        link = logs_path / "tty.log"
        if link.exists() or link.is_symlink():
            return
        os.symlink(tty_path, link)
    except OSError as e:
        log.warning("could not symlink tty into %s: %s", logs_dir, e)


def cancel_queued(queue_id: str) -> bool:
    """Remove a queued item before it dispatches.

    Returns ``False`` if the queue_id isn't in the queue (it may already
    be a JobRecord — caller should fall through to ``abort_record``).
    """
    return queue_store.remove_item(queue_id)


def abort_record(queue_id: str) -> bool:
    """Abort a running/starting JobRecord — sends SIGTERM to its group."""
    return _kill_record(queue_id, signal.SIGTERM)


def force_kill_record(queue_id: str) -> bool:
    """Force-kill a JobRecord — sends SIGKILL to its group.

    Last-resort path for hung torchrun groups (multiprocess deadlocks,
    rendezvous failures, etc.) that don't respond to SIGTERM. The kernel
    will reap the process(es) regardless of state.
    """
    return _kill_record(queue_id, signal.SIGKILL)


def _kill_record(queue_id: str, sig: int) -> bool:
    record = job_records.get_record(queue_id)
    if record is None:
        return False
    if record.status in TERMINAL_STATUSES:
        return False
    job_records.update_record(queue_id, status="aborted", finished_at=time.time())
    if record.pid:
        launcher.kill_process_group(record.pid, sig)
    with _state._lock:
        _state.running.pop(queue_id, None)
    return True


def abort_or_cancel(queue_id: str) -> bool:
    """Convenience: cancel if queued, abort if running."""
    if cancel_queued(queue_id):
        return True
    return abort_record(queue_id)


def _reattach_or_cleanup_on_startup() -> None:
    """Reconnect to still-alive jobs after a server restart.

    Training subprocesses are spawned with ``start_new_session=True``, so
    they're in their own process session and survive the server exiting
    cleanly. On startup we walk every JobRecord that was mid-flight
    (``status in {starting, running}``) and check whether its PID is
    still alive:

    - **Alive** → re-attach: leave the record in its current status and
      register a ``None`` handle in ``_state.running`` so the reap loop
      polls it via psutil. Abort / control / TTY all keep working because
      they don't need a Popen.
    - **Gone** → mark failed, same as the old orphan-cleanup behaviour.

    PID reuse is guarded by comparing the live process's ``create_time``
    against the record's ``started_at``; a mismatch means the kernel has
    recycled the pid and we treat it as gone.
    """
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    reattached = 0
    cleaned = 0
    for r in job_records.list_records():
        if r.status not in RUNNING_STATUSES:
            continue
        if r.pid is None:
            _mark_failed(r.queue_id, "server restart; no pid recorded")
            cleaned += 1
            continue

        alive = False
        if psutil is not None:
            try:
                p = psutil.Process(r.pid)
                alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                # PID-reuse guard: if the process was created after we
                # launched it, the kernel has recycled the pid.
                if alive and r.started_at is not None:
                    # Allow a few seconds of slack — create_time() is rounded
                    # and started_at is recorded slightly earlier than the
                    # actual fork.
                    if p.create_time() > r.started_at + 10:
                        alive = False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                alive = False
        else:
            alive = _pid_is_alive(r.pid)

        if alive:
            with _state._lock:
                _state.running[r.queue_id] = None
            reattached += 1
            log.info(
                "re-attached to running job %s (pid=%d, config=%s)",
                r.queue_id,
                r.pid,
                r.config,
            )
        else:
            _mark_failed(
                r.queue_id,
                f"server restart; pid {r.pid} no longer running",
            )
            cleaned += 1

    if reattached:
        log.warning(
            "re-attached to %d running job(s) across server restart",
            reattached,
        )
    if cleaned:
        log.warning("marked %d orphaned record(s) as failed on startup", cleaned)


def _mark_failed(queue_id: str, reason: str) -> None:
    job_records.update_record(
        queue_id,
        status="failed",
        error=reason,
        finished_at=time.time(),
    )


async def dispatcher_loop() -> None:
    log.info("dispatcher loop starting (enabled=%s)", _state.enabled)
    _reattach_or_cleanup_on_startup()
    try:
        while True:
            _state.tick_count += 1
            _state.last_tick_at = time.time()
            try:
                _reap_finished()
                _correlate_running_records()
                _try_dispatch()
            except Exception:
                log.exception("dispatcher tick failed")
            await asyncio.sleep(TICK_SECONDS)
    except asyncio.CancelledError:
        log.info("dispatcher loop cancelled")
        raise
