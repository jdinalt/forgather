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
import secrets
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from forgather import trainer_control

from dataset_server.auth import (
    standalone_token_file as dataset_server_standalone_token_file,
    write_standalone_token as dataset_server_write_standalone_token,
)

from . import _atomic, _gc, gpu_monitor, job_records, launcher, queue_store
from .job_records import RUNNING_STATUSES, TERMINAL_STATUSES, JobRecord
from .paths import (
    inference_token_file,
    jobs_tty_dir,
)
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

    A GPU is eligible iff it has not been excluded by the operator via
    ``CUDA_VISIBLE_DEVICES`` at server start AND has not been runtime-
    disabled by the user via the web UI.

    External processes (the user's desktop compositor, an unrelated
    CUDA program, etc.) are *not* consulted. The scheduler tracks its
    own dispatched jobs via ``_reserved_gpu_set()`` (subtracted from
    the idle pool by the dispatch loop), and that's the authoritative
    "Forgather is already using this GPU" signal. Trying to classify
    arbitrary external processes as "real compute work" vs "desktop
    rendering" turns out to be a tar pit (NVIDIA's proprietary driver
    routes graphics-with-CUDA-context daemons through the compute
    list, hybrid C+G processes show up there too, etc.). If you don't
    want Forgather running on a GPU that's already hosting external
    work, click the disable button on the GPU card.
    """
    return [
        g.index for g in gpu_monitor.snapshot() if not g.excluded and not g.disabled
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
        updated = job_records.update_if_not_terminal(
            qid,
            status=new_status,
            exit_code=rc,
            finished_at=time.time(),
        )
        with _state._lock:
            _state.running.pop(qid, None)
        # Move the captured TTY into the run's logs/ dir now that the
        # job is terminal. No-op for non-training jobs (no logs_dir).
        if updated is not None:
            _gc.relocate_tty_to_logs(updated)
            _cleanup_inference_token(updated)
            # Same wake hook as the spawn path — remove the entry
            # from the cluster picker promptly when the server stops.
            if updated.job_type == "inference":
                _wake_inference_inventory()
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


def _build_eval(item, gpu_indices, tty_path):
    # The explicit args below are the script-required ones (eval_project,
    # eval_template, model_path) and the scheduler-owned ones (extra_env).
    # Every other ``--*`` flag is declared once in
    # ``forgather.cli.eval_args._EVAL_SCRIPT_ARGS`` and forwarded by
    # ``forward_eval_script_args_from_params`` deep in ``build_eval_command``;
    # passing the whole ``job_params`` dict through avoids re-listing each
    # key here just to call ``p.get(...)`` on it.
    #
    # Filter to keys the spec recognizes so future webui/queue fields (added
    # for routing, accounting, etc.) don't leak through as kwargs to
    # ``spawn_eval_process``. The forwarder itself already ignores unknown
    # keys, but rejecting them up-front keeps the kwarg surface honest.
    from forgather.cli.eval_args import passthrough_enqueue_keys

    p = dict(item.job_params)
    p.pop("eval_project")
    p.pop("eval_template")
    p.pop("model_path")
    extra_env = p.pop("extra_env", None) or None
    passthrough = {k: v for k, v in p.items() if k in passthrough_enqueue_keys()}
    return launcher.spawn_eval_process(
        eval_project=item.job_params["eval_project"],
        eval_template=item.job_params["eval_template"],
        model_path=item.job_params["model_path"],
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
        extra_env=extra_env,
        **passthrough,
    )


def _build_inference(item, gpu_indices, tty_path):
    p = item.job_params
    # ``no_auth`` opts the spawn out of bearer-token auth entirely. Otherwise
    # the auth_token is generated and persisted in ``_launch`` before this
    # builder runs; pass the per-job token file path so the spawn never sees
    # the token on argv (which would be visible in ``ps``).
    no_auth = bool(p.get("no_auth", False))
    auth_token_file: Optional[str] = None
    if not no_auth:
        auth_token_file = str(inference_token_file(item.queue_id))
    # Multi-model: ``models`` is a list of {name, path}; single-model:
    # ``model_path`` is a string. Exactly one is expected from the
    # enqueue layer (CLI or webui InferenceModal).
    models = p.get("models")
    # ``device`` from job_params is an explicit override (e.g. "auto"
    # for HF device_map sharding across multiple GPUs, "cpu" to force
    # CPU even when GPUs are reserved). When unset / blank, the
    # launcher derives the right value from gpu_indices.
    raw_device = p.get("device")
    device = raw_device.strip() if isinstance(raw_device, str) and raw_device.strip() else None
    return launcher.spawn_inference_process(
        model_path=p.get("model_path") if not models else None,
        models=models,
        port=int(p["port"]),
        host=p.get("host", "127.0.0.1"),
        device=device,
        dtype=p.get("dtype"),
        attn_implementation=p.get("attn_implementation"),
        checkpoint_path=p.get("checkpoint_path"),
        from_checkpoint=bool(p.get("from_checkpoint", False)),
        compile=bool(p.get("compile", False)),
        disable_kv_cache=bool(p.get("disable_kv_cache", False)),
        ignore_eos=bool(p.get("ignore_eos", False)),
        keep_on_gpu=bool(p.get("keep_on_gpu", False)),
        chat_template=p.get("chat_template"),
        cache_implementation=p.get("cache_implementation"),
        compile_args=p.get("compile_args"),
        log_level=p.get("log_level", "INFO"),
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
        auth_token_file=auth_token_file,
        no_auth=no_auth,
    )


def _build_dataset_server(item, gpu_indices, tty_path):
    p = item.job_params
    no_auth = bool(p.get("no_auth", False))
    auth_token_file: Optional[str] = None
    if not no_auth:
        # Use the canonical per-port token path shared with the
        # standalone ``forgather dataset-server start`` CLI so a
        # webui-spawned dataset_server keeps the same token across
        # restarts (rotated only on ``regen_token``).
        port = int(p.get("port", 8766))
        auth_token_file = str(dataset_server_standalone_token_file(port))
    # ``locals`` arrives as a list of [name, path] pairs from the
    # webui (JSON has no tuple type); coerce to tuples for the ops layer.
    raw_locals = p.get("locals") or []
    locals_: list[tuple] = []
    if isinstance(raw_locals, list):
        for entry in raw_locals:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                locals_.append((str(entry[0]), str(entry[1])))
    return launcher.spawn_dataset_server_process(
        host=p.get("host", "127.0.0.1"),
        port=int(p.get("port", 8766)),
        log_level=p.get("log_level", "INFO"),
        no_hf=bool(p.get("no_hf", False)),
        allow_paths=bool(p.get("allow_paths", False)),
        allow_downloads=bool(p.get("allow_downloads", False)),
        locals_=locals_,
        config_file=p.get("config_file"),
        tty_log_path=tty_path,
        auth_token_file=auth_token_file,
        no_auth=no_auth,
    )


def _build_tensorboard(item, gpu_indices, tty_path):
    p = item.job_params
    ri = p.get("reload_interval")
    # ``path_prefix`` is set by ``_launch`` on the JobRecord before this
    # builder runs; mirror it onto the TB CLI so TB's generated links
    # match the proxy's mount path. Falls back to job_params for any
    # legacy callers that pre-computed it themselves.
    record = job_records.get_record(item.queue_id)
    path_prefix = (record.path_prefix if record is not None else None) or p.get(
        "path_prefix"
    )
    return launcher.spawn_tensorboard_process(
        logdir=p["logdir"],
        port=int(p["port"]),
        host=p.get("host"),
        bind_all=bool(p.get("bind_all", False)),
        window_title=p.get("window_title"),
        reload_interval=int(ri) if ri is not None else None,
        reload_multifile=bool(p.get("reload_multifile", False)),
        samples_per_plugin=p.get("samples_per_plugin"),
        path_prefix=path_prefix,
        tty_log_path=tty_path,
    )


def _build_convert(item, gpu_indices, tty_path):
    p = item.job_params
    cps = p.get("converter_paths")
    if isinstance(cps, str):
        cps = [cps]
    elif not isinstance(cps, list):
        cps = None
    ml = p.get("max_length")
    return launcher.spawn_convert_process(
        src_model_path=p["src_model_path"],
        dst_model_path=p["dst_model_path"],
        reverse=bool(p.get("reverse", False)),
        model_type=p.get("model_type"),
        dtype=p.get("dtype"),
        max_length=int(ml) if ml is not None else None,
        checkpoint_path=p.get("checkpoint_path"),
        device=p.get("device"),
        generation_test=bool(p.get("generation_test", False)),
        dry_run=bool(p.get("dry_run", False)),
        prompt=p.get("prompt"),
        compare_text_file=p.get("compare_text_file"),
        debug_params=bool(p.get("debug_params", False)),
        chat_template_path=p.get("chat_template_path"),
        add_tokens=p.get("add_tokens"),
        skip_default_tokens=bool(p.get("skip_default_tokens", False)),
        converter_paths=cps,
        log_level=p.get("log_level", "INFO"),
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
    )


def _build_finalize(item, gpu_indices, tty_path):
    p = item.job_params
    return launcher.spawn_finalize_process(
        source=p["source"],
        dest=p["dest"],
        checkpoint=p.get("checkpoint"),
        add_tokens=p.get("add_tokens"),
        skip_default_tokens=bool(p.get("skip_default_tokens", False)),
        chat_template_path=p.get("chat_template_path"),
        no_auto_stop_tokens=bool(p.get("no_auto_stop_tokens", False)),
        stop_tokens=p.get("stop_tokens"),
        generation_config=p.get("generation_config"),
        keep_optimizer=bool(p.get("keep_optimizer", False)),
        root_copy=bool(p.get("root_copy", False)),
        safetensors=bool(p.get("safetensors", False)),
        dtype=p.get("dtype"),
        device=p.get("device"),
        dry_run=bool(p.get("dry_run", False)),
        log_level=p.get("log_level", "INFO"),
        quantize=p.get("quantize"),
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
    )


def _build_update(item, gpu_indices, tty_path):
    p = item.job_params
    cps = p.get("converter_paths")
    if isinstance(cps, str):
        cps = [cps]
    elif not isinstance(cps, list):
        cps = None
    fv = p.get("from_version")
    tv = p.get("to_version")
    return launcher.spawn_update_process(
        src_model_path=p["src_model_path"],
        dst_model_path=p["dst_model_path"],
        arch=p.get("arch"),
        # Versions are PEP 440 strings; coerce to str so legacy
        # integers from older job_params still flow through.
        from_version=str(fv) if fv is not None else None,
        to_version=str(tv) if tv is not None else None,
        checkpoint=p.get("checkpoint"),
        device=p.get("device"),
        dtype=p.get("dtype"),
        no_strict=bool(p.get("no_strict", False)),
        safetensors=bool(p.get("safetensors", False)),
        converter_paths=cps,
        dry_run=bool(p.get("dry_run", False)),
        log_level=p.get("log_level", "INFO"),
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
    )


def _build_mkdocs(item, gpu_indices, tty_path):
    p = item.job_params
    watch = p.get("watch")
    if isinstance(watch, str):
        watch = [watch]
    elif not isinstance(watch, list):
        watch = None
    return launcher.spawn_mkdocs_process(
        config_file=p["config_file"],
        port=int(p["port"]),
        host=p.get("host", "127.0.0.1"),
        strict=bool(p.get("strict", False)),
        livereload=bool(p.get("livereload", True)),
        dirty=bool(p.get("dirty", False)),
        watch=watch,
        tty_log_path=tty_path,
    )


def _build_model(item, gpu_indices, tty_path):
    p = item.job_params
    cd = p.get("compile_dynamic")
    return launcher.spawn_model_process(
        project_dir=item.project_dir,
        config_name=item.config,
        subcommand=p.get("subcommand", "construct"),
        dynamic_args=item.dynamic_args,
        device=p.get("device"),
        dtype=p.get("dtype"),
        no_init_weights=bool(p.get("no_init_weights", False)),
        load_from_checkpoint=p.get("load_from_checkpoint"),
        gradient_checkpointing=bool(p.get("gradient_checkpointing", False)),
        fuse_optim_with_backward=bool(p.get("fuse_optim_with_backward", False)),
        refresh_model=bool(p.get("refresh_model", False)),
        save_checkpoint=bool(p.get("save_checkpoint", False)),
        safetensors=bool(p.get("safetensors", False)),
        batch_size=int(p["batch_size"]) if p.get("batch_size") is not None else None,
        sequence_length=(
            int(p["sequence_length"]) if p.get("sequence_length") is not None else None
        ),
        steps=int(p["steps"]) if p.get("steps") is not None else None,
        lr=float(p["lr"]) if p.get("lr") is not None else None,
        dataset_project=p.get("dataset_project"),
        dataset_config=p.get("dataset_config"),
        packed=bool(p.get("packed", False)),
        compile=bool(p.get("compile", False)),
        compile_backend=p.get("compile_backend"),
        compile_mode=p.get("compile_mode"),
        compile_dynamic=None if cd is None else bool(cd),
        compile_fullgraph=bool(p.get("compile_fullgraph", False)),
        amp=p.get("amp"),
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
        extra_env=p.get("extra_env") or None,
    )


def _build_dataset(item, gpu_indices, tty_path):
    p = item.job_params
    features = p.get("features")
    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, list):
        features = None
    return launcher.spawn_dataset_process(
        project_dir=item.project_dir,
        config_name=item.config,
        dynamic_args=item.dynamic_args,
        tokenizer_path=p.get("tokenizer_path"),
        pp=bool(p.get("pp", False)),
        histogram=bool(p.get("histogram", False)),
        target=p.get("target"),
        histogram_samples=(
            int(p["histogram_samples"])
            if p.get("histogram_samples") is not None
            else None
        ),
        examples=int(p["examples"]) if p.get("examples") is not None else None,
        features=features,
        tokenized=bool(p.get("tokenized", False)),
        num_shards=int(p["num_shards"]) if p.get("num_shards") is not None else None,
        shard_index=(
            int(p["shard_index"]) if p.get("shard_index") is not None else None
        ),
        select_range=p.get("select_range"),
        seed=int(p["seed"]) if p.get("seed") is not None else None,
        example_stride=(
            int(p["example_stride"]) if p.get("example_stride") is not None else None
        ),
        truncate=int(p["truncate"]) if p.get("truncate") is not None else None,
        tty_log_path=tty_path,
        extra_env=p.get("extra_env") or None,
    )


def _build_construct(item, gpu_indices, tty_path):
    p = item.job_params
    return launcher.spawn_construct_process(
        project_dir=item.project_dir,
        config_name=item.config,
        dynamic_args=item.dynamic_args,
        target=str(p.get("target") or "main"),
        call=bool(p.get("call", False)),
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
        extra_env=p.get("extra_env") or None,
    )


def _build_training(item, gpu_indices, tty_path):
    # Multi-node training jobs (Phase 3 cluster-coordinator submit)
    # carry their torchrun rendezvous args + NCCL env in
    # ``job_params``. Single-node training jobs leave job_params empty
    # and the launcher falls back to ``--standalone``.
    rdzv_args = item.job_params.get("rdzv_args") or None
    extra_env = item.job_params.get("extra_env") or None
    return launcher.spawn_training_process(
        project_dir=item.project_dir,
        config_name=item.config,
        dynamic_args=item.dynamic_args,
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
        extra_env=extra_env,
        rdzv_args=rdzv_args,
    )


# Registry mapping job_type → builder(item, gpu_indices, tty_path) -> LaunchResult.
# Unknown types fall through to _build_training (default Forgather training job).
_LAUNCHERS = {
    "eval": _build_eval,
    "inference": _build_inference,
    "dataset_server": _build_dataset_server,
    "tensorboard": _build_tensorboard,
    "convert": _build_convert,
    "finalize": _build_finalize,
    "update": _build_update,
    "mkdocs": _build_mkdocs,
    "model": _build_model,
    "dataset": _build_dataset,
    "construct": _build_construct,
}


def detect_routable_host() -> Optional[str]:
    """Best-effort LAN-routable address for this host.

    Priority:
      1. Cluster-self address (the one peers already use to peer-pull).
         Most reliable because cluster_discovery already filtered out
         loopback / virtual-interface addresses for us.
      2. First non-loopback IPv4 from psutil.net_if_addrs. Used when
         the server isn't in cluster mode.
      3. None — caller should fall back to whatever it was going to
         display before (typically "localhost").
    """
    try:
        from . import cluster as _cluster

        if _cluster.is_active():
            self_ident = _cluster.self_identity()
            if self_ident:
                m = next(
                    (mm for mm in _cluster.members() if mm.node_id == self_ident.node_id),
                    None,
                )
                if m and m.address and not m.address.startswith("127."):
                    return m.address
    except Exception:
        pass
    try:
        import psutil

        for _iface, entries in psutil.net_if_addrs().items():
            for entry in entries:
                addr = getattr(entry, "address", "")
                if not addr:
                    continue
                # IPv4 only for now; the URL field is a single string,
                # and IPv6 in URLs needs bracket escaping that complicates
                # downstream. IPv4 covers the common-case LAN deployment.
                if entry.family != socket.AF_INET:
                    continue
                if addr.startswith("127."):
                    continue
                # Link-local (169.254.x) and most virtual interfaces fail
                # the "is this the address the operator would type" test.
                if addr.startswith("169.254."):
                    continue
                return addr
    except Exception:
        pass
    return None


def _resolve_inference_server_token(*, port: int, regen: bool) -> str:
    """Return the bearer token an inference_server spawn should use.

    Mirrors the dataset_server token persistence model: the per-port
    standalone token file (the same one a CLI-launched ``forgather inf
    server -p <port>`` writes) is reused across restarts so a remote
    operator who copied the token doesn't have to refetch it every
    time the server bounces. ``regen=True`` is the opt-in rotation.

    Reuse rules:

    - ``regen=True``  -> always mint + persist.
    - File missing    -> mint + persist.
    - File present, non-empty -> reuse its contents.
    - File present, empty / unreadable -> treat as missing.

    Persisting may fail (read-only home, etc.); the spawn still gets
    a valid in-memory token and the JobRecord carries it. Logged at
    WARNING in that case; not raised.
    """
    # Lazy import — the standalone helper lives in tools/inference_server/,
    # which the runtime image puts on the python path but isn't strictly
    # required to import at module load.
    try:
        from inference_server.auth_paths import (
            standalone_token_file as _inf_token_path,
            write_standalone_token as _inf_token_write,
        )
    except ImportError:
        # Fall back to ephemeral if the helper isn't reachable (shouldn't
        # happen in any supported install). The operator just loses
        # persistence; functionality is unaffected.
        return secrets.token_hex(32)

    token_path = _inf_token_path(port)
    if not regen and token_path.is_file():
        try:
            existing = token_path.read_text().strip()
        except OSError:
            existing = ""
        if existing:
            return existing
    token = secrets.token_hex(32)
    try:
        _inf_token_write(port, token)
    except OSError as exc:
        log.warning(
            "could not persist inference-server token to %s: %s "
            "(spawn still works; CLI auto-discovery for this port disabled "
            "until the next successful write).",
            token_path,
            exc,
        )
    return token


def _resolve_dataset_server_token(*, port: int, regen: bool) -> str:
    """Return the bearer token a dataset_server spawn should use.

    Unlike inference (per-queue ephemeral tokens), dataset_server shares
    the same per-port persisted token file the standalone CLI uses, so
    long-running clients (training peers) keep working across server
    restarts. ``regen=True`` is the operator-opt-in to rotate.

    Reuse rules:

    - ``regen=True``  -> always mint + persist.
    - File missing    -> mint + persist.
    - File present, non-empty -> reuse its contents.
    - File present, empty / unreadable -> treat as missing.

    Persisting to the per-port file may fail (e.g. read-only home);
    the spawn still gets a valid token in memory and the JobRecord
    carries it, but the next start won't auto-discover. Logged at
    WARNING; not raised.
    """
    token_path = dataset_server_standalone_token_file(port)
    if not regen and token_path.is_file():
        try:
            existing = token_path.read_text().strip()
        except OSError:
            existing = ""
        if existing:
            return existing
    token = secrets.token_hex(32)
    try:
        dataset_server_write_standalone_token(port, token)
    except OSError as e:
        log.warning(
            "could not persist dataset_server token at %s: %s",
            token_path,
            e,
        )
    return token


def _launch(item: QueueItem, gpu_indices: List[int]) -> None:
    """Move a queue item to a JobRecord and spawn the appropriate subprocess.

    add_record + remove_item happen before the try block so the queue item
    is always promoted to a record (even if spawn fails). On spawn failure
    the record transitions to "failed"; on success the ordering is:

      1. spawn (result = builder(...))
      2. _state.running[queue_id] = proc   — in-memory handle registered first
      3. update_record(status="running")   — durable state written after

    Registering the handle before the disk write means the reap loop can
    collect the process even if the server crashes between steps 2 and 3.
    """
    tty_path = jobs_tty_dir() / f"{item.queue_id}.tty"
    # TB jobs are reachable through the auth-gated reverse proxy at
    # ``/api/tb/<queue_id>/...``; record the prefix here so ``--path_prefix``
    # propagates to the spawned tensorboard process and the proxy route can
    # look it up by ``queue_id`` later.
    path_prefix: Optional[str] = None
    if item.job_type == "tensorboard":
        path_prefix = f"/api/tb/{item.queue_id}"

    # Generate the bearer token here (before the builder runs) so the
    # JobRecord persists it and the spawn reads it from the same 0600 file.
    # ``no_auth`` in job_params opts out. Same pattern is used for both
    # inference and dataset_server jobs.
    auth_token: Optional[str] = None
    if not bool(item.job_params.get("no_auth", False)):
        if item.job_type == "inference":
            # Per-port persistent token (matches dataset_server model)
            # so restarts don't invalidate the token a remote operator
            # already copied. The per-queue file at
            # inference_token_file(queue_id) stays the path
            # spawn_inference_process reads via --auth-token-file
            # (operator-managed tokens shouldn't appear in argv) — write
            # the resolved-persistent value there too.
            auth_token = _resolve_inference_server_token(
                port=int(item.job_params.get("port", 8137)),
                regen=bool(item.job_params.get("regen_token", False)),
            )
            _atomic.atomic_write_text(
                inference_token_file(item.queue_id), auth_token, mode=0o600
            )
        elif item.job_type == "dataset_server":
            auth_token = _resolve_dataset_server_token(
                port=int(item.job_params.get("port", 8766)),
                regen=bool(item.job_params.get("regen_token", False)),
            )

    # Stamp the actual URL scheme into job_params for inference and
    # dataset_server jobs. The spawned child picks TLS up from the
    # shared config (forgather.tls), but the webui has no view into
    # that config — without this stamp the Job card and the Inference
    # panel would always show http:// even when the upstream is
    # actually HTTPS.
    # TensorBoard and MkDocs are intentionally not stamped: those
    # services don't read forgather's TLS config and always serve HTTP.
    finalized_params = dict(item.job_params)
    if item.job_type in ("inference", "dataset_server"):
        try:
            from forgather.tls import client_scheme as _client_scheme

            host_for_scheme = finalized_params.get("host", "127.0.0.1")
            finalized_params.setdefault(
                "scheme", _client_scheme(host_for_scheme)
            )
        except Exception:
            finalized_params.setdefault("scheme", "http")

    # Stamp a routable host for cross-machine URL display. When the
    # spawned service binds 0.0.0.0, "localhost" in the rendered URL
    # is correct for a browser on the same host but useless for any
    # other machine the operator is browsing from. Pick the
    # cluster-routable address when available (same one peers use),
    # or fall back to the first non-loopback psutil-detected IP.
    # Leave unset for explicit bind hosts (operator knows what they
    # typed). Applies to every job type that exposes a clickable URL
    # on its Job card — currently inference, dataset_server, and
    # mkdocs. TensorBoard renders its own URL with its bind_all
    # toggle in mind and is left alone.
    if item.job_type in ("inference", "dataset_server", "mkdocs"):
        if finalized_params.get("host") in ("0.0.0.0", "::", ""):
            routable = detect_routable_host()
            if routable:
                finalized_params["routable_host"] = routable

    record = JobRecord(
        queue_id=item.queue_id,
        project_dir=item.project_dir,
        config=item.config,
        dynamic_args=dict(item.dynamic_args),
        requested_gpus=item.requested_gpus,
        priority=item.priority,
        submitted_at=item.submitted_at,
        job_type=item.job_type,
        job_params=finalized_params,
        node=LOCAL_NODE,
        gpu_indices=gpu_indices,
        status="starting",
        started_at=time.time(),
        tty_log_path=str(tty_path),
        path_prefix=path_prefix,
        auth_token=auth_token,
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

    build = _LAUNCHERS.get(item.job_type, _build_training)
    try:
        result = build(item, gpu_indices, tty_path)
    except Exception as e:
        log.exception("launch failed for %s", item.queue_id)
        job_records.update_record(
            item.queue_id,
            status="failed",
            error=str(e),
            finished_at=time.time(),
        )
        return

    with _state._lock:
        _state.running[item.queue_id] = result.proc
    job_records.update_record(item.queue_id, status="running", pid=result.pid)
    # Inference jobs are tracked in the cluster picker inventory.
    # Waking the collect loop on spawn drops the picker-convergence
    # latency from up to ``COLLECT_INTERVAL_SECONDS`` (10s) to ~1s
    # so an operator who just clicked "Start server" sees the new
    # entry before they can switch tabs.
    if item.job_type == "inference":
        _wake_inference_inventory()


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
    # Use the CAS variant so a reap that lands between our status check
    # above and this write can't be clobbered. The kill path losing the
    # race is benign: both ends are terminal and the process is gone.
    updated = job_records.update_if_not_terminal(
        queue_id, status="aborted", finished_at=time.time()
    )
    if record.pid:
        launcher.kill_process_group(record.pid, sig)
        # Verify the process actually died before declaring success.
        # Without this we silently leave orphan processes consuming
        # GPUs while the JobRecord disappears from the UI — the
        # operator hit exactly this on muthur+wopr after a hung
        # save-stop. Workers stuck in a CUDA driver call (uninter-
        # ruptible D state) won't die immediately even on SIGKILL,
        # so we poll briefly and stamp the record's ``error`` field
        # if it's still alive at the end. The CAS keeps the status
        # at "aborted" but the error makes the orphan visible to the
        # operator instead of silently disappearing.
        if not _wait_for_pid_exit(record.pid, timeout=2.0):
            log.warning(
                "kill of %s (pid=%d, sig=%d) did not exit within "
                "timeout — process may be stuck in a CUDA driver "
                "call or torch.distributed deadlock",
                queue_id,
                record.pid,
                sig,
            )
            job_records.update_record(
                queue_id,
                error=(
                    f"PID {record.pid} did not exit within 2s of "
                    f"signal {sig}. Process may be stuck in a CUDA "
                    f"driver call (uninterruptible sleep). Check "
                    f"GPU panel for the lingering PID; retry "
                    f"force-kill, or kill from the host if the "
                    f"container can't see the PID."
                ),
            )
    with _state._lock:
        _state.running.pop(queue_id, None)
    # Same TTY relocation as the reap path so an aborted run still ends
    # up with its tty.log materialized inside the run's logs/ dir.
    if updated is not None:
        _gc.relocate_tty_to_logs(updated)
        _cleanup_inference_token(updated)
        if updated.job_type == "inference":
            _wake_inference_inventory()
    return updated is not None


def _wait_for_pid_exit(pid: int, timeout: float) -> bool:
    """Poll until ``pid`` is dead-or-zombie, or ``timeout`` seconds.

    Treats zombies as exited — once the process has hit zombie state
    its actual work is done and the parent will reap it momentarily
    via Popen.poll(). Without this, a child that exits cleanly but
    hasn't been waited on yet still passes ``psutil.pid_exists`` and
    we'd spuriously time out on every successful kill.

    Returns True on confirmed exit / zombie, False on timeout. We
    don't escalate signals here — the caller already chose SIGTERM
    vs SIGKILL based on whether the operator hit "abort" or
    "force-kill".
    """
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_is_alive(pid)


def _wake_inference_inventory() -> None:
    """Signal the cluster inference-server inventory to re-poll.

    Lazy import so the scheduler stays usable in tests + standalone
    setups that don't load the cluster machinery. Wake is a latency
    hint, not correctness-critical — silently swallow failures so a
    missing/broken cluster module never breaks the scheduler.
    """
    try:
        from . import cluster_inference_inventory

        cluster_inference_inventory.wake_loops()
    except Exception:
        pass


def _cleanup_inference_token(record: JobRecord) -> None:
    """Best-effort delete of the per-job auth token file.

    Inference servers use per-queue ephemeral tokens — those get tidied
    up here once the spawn exits. Dataset servers intentionally share a
    per-port persistent token with the standalone CLI (so restarts
    don't invalidate every remote client); that file is NOT deleted on
    reap. Errors are swallowed so a missing/already-removed file never
    breaks reap.
    """
    if record.job_type != "inference":
        return
    path = inference_token_file(record.queue_id)
    try:
        if path.exists():
            path.unlink()
    except OSError as e:
        log.debug("could not unlink token for %s: %s", record.queue_id, e)


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

    # Sweep orphan TTYs left over from previous server runs. Files
    # younger than the TTL are protected to avoid racing an in-flight
    # dispatch that hasn't persisted its JobRecord yet.
    try:
        swept = _gc.sweep_orphan_ttys()
        if swept:
            log.info("startup orphan-tty sweep removed %d file(s)", swept)
    except Exception:
        log.exception("startup orphan-tty sweep failed")


def _mark_failed(queue_id: str, reason: str) -> None:
    job_records.update_record(
        queue_id,
        status="failed",
        error=reason,
        finished_at=time.time(),
    )


# Long-running servers also do a periodic sweep so accumulation between
# restarts is bounded. Daily is plenty given the per-job relocation already
# eats the common case.
GC_SWEEP_INTERVAL_SECONDS = 24 * 3600


async def dispatcher_loop() -> None:
    log.info("dispatcher loop starting (enabled=%s)", _state.enabled)
    _reattach_or_cleanup_on_startup()
    last_gc_at = time.time()
    try:
        while True:
            _state.tick_count += 1
            _state.last_tick_at = time.time()
            try:
                _reap_finished()
                _correlate_running_records()
                _try_dispatch()
                if time.time() - last_gc_at > GC_SWEEP_INTERVAL_SECONDS:
                    last_gc_at = time.time()
                    try:
                        swept = _gc.sweep_orphan_ttys()
                        if swept:
                            log.info(
                                "periodic orphan-tty sweep removed %d file(s)",
                                swept,
                            )
                    except Exception:
                        log.exception("periodic orphan-tty sweep failed")
            except Exception:
                log.exception("dispatcher tick failed")
            await asyncio.sleep(TICK_SECONDS)
    except asyncio.CancelledError:
        log.info("dispatcher loop cancelled")
        raise
