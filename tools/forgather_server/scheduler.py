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
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from dataset_server.auth import (
    standalone_token_file as dataset_server_standalone_token_file,
)
from dataset_server.auth import (
    write_standalone_token as dataset_server_write_standalone_token,
)

from forgather import trainer_control
from forgather.ml.diloco.auth import (
    standalone_token_file as diloco_server_standalone_token_file,
)
from forgather.ml.diloco.auth import (
    write_standalone_token as diloco_server_write_standalone_token,
)

from . import _atomic, _gc, gpu_monitor, job_records, launcher, queue_store
from .auth import demo_mode_enabled
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
                # Use the PID-reuse-guarded check (create_time vs started_at) —
                # same as startup re-attach and the GC sweep — so a recycled pid
                # can't keep a re-attached / promoted-external record "running"
                # forever.
                if trainer_control.is_endpoint_pid_alive(record.pid, record.started_at):
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
            elif updated.job_type == "diloco_server":
                _wake_diloco_inventory()
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


def _coerce_nproc_override(raw):
    """Normalise a job_params['nproc'] value for the launcher.

    The webui sends a free-form string (typed into the modal's nproc
    input); other callers might pass an integer. Trim whitespace,
    treat empty as "not set" (return None so the launcher's
    gpu_indices-derived default kicks in), pass everything else
    through as a string (torchrun accepts an integer or one of
    ``gpu`` / ``cpu`` / ``auto``).
    """
    if isinstance(raw, str):
        trimmed = raw.strip()
        return trimmed if trimmed else None
    if isinstance(raw, int):
        return str(raw)
    return None


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
    # ``nproc`` is a server-side knob (torchrun --nproc-per-node
    # override from the EvalModal), not a flag for eval_script.py
    # itself -- extract it separately so it doesn't have to live in
    # the _EVAL_SCRIPT_ARGS spec.
    nproc_override = _coerce_nproc_override(p.pop("nproc", None))
    if not gpu_indices:
        log.info(
            "eval job %s dispatching with 0 GPUs reserved (CPU mode); "
            "trainer=%r nproc_override=%r",
            item.queue_id,
            p.get("trainer", "ddp"),
            nproc_override,
        )
    passthrough = {k: v for k, v in p.items() if k in passthrough_enqueue_keys()}
    return launcher.spawn_eval_process(
        eval_project=item.job_params["eval_project"],
        eval_template=item.job_params["eval_template"],
        model_path=item.job_params["model_path"],
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
        extra_env=extra_env,
        nproc_override=nproc_override,
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
    device = (
        raw_device.strip()
        if isinstance(raw_device, str) and raw_device.strip()
        else None
    )
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
        # Token redaction in the spawned server's TTY is a demo-mode
        # concern, not an operator choice: suppress the bearer in the
        # launch banner only when this webui is running --demo (public
        # TTY pane). In normal operation the token is shown so it can be
        # copied onto clients, Jupyter-style.
        quiet_tokens=demo_mode_enabled(),
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
        # Token redaction in the spawned server's TTY is a demo-mode
        # concern, not an operator choice: suppress the bearer in the
        # launch banner only when this webui is running --demo (public
        # TTY pane). In normal operation the token is shown so it can be
        # copied onto clients, Jupyter-style.
        quiet_tokens=demo_mode_enabled(),
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


def _build_diloco_server(item, gpu_indices, tty_path):
    p = item.job_params
    # Auth-token file path (per-port). The scheduler's _launch resolves
    # and persists the token before invoking this builder; here we
    # just hand the spawn the path to read from.
    no_auth = bool(p.get("no_auth", False))
    port = int(p.get("port", 8512))
    auth_token_file = (
        None if no_auth else str(diloco_server_standalone_token_file(port))
    )
    return launcher.spawn_diloco_server_process(
        output_dir=p["output_dir"],
        num_workers=int(p["num_workers"]),
        port=port,
        host=p.get("host", "127.0.0.1"),
        async_mode=bool(p.get("async_mode", False)),
        dn_buffer_size=int(p.get("dn_buffer_size", 0) or 0),
        dylu=bool(p.get("dylu", False)),
        dylu_base_sync_every=int(p.get("dylu_base_sync_every", 500) or 500),
        sync_every=int(p.get("sync_every", 500) or 500),
        # Wire precision (issue #130). Pre-#130 job_params only carry
        # ``bf16_comm`` (Bool); post-#130 carry the four explicit keys.
        # When neither is present, falls through to launcher defaults.
        bf16_comm=p.get("bf16_comm"),
        upload_dtype=p.get("upload_dtype"),
        upload_sr=bool(p.get("upload_sr", False)),
        download_dtype=str(p.get("download_dtype", "fp32") or "fp32"),
        download_sr=bool(p.get("download_sr", False)),
        # Bulk transport (issue #154): wire codec + optional gRPC listener.
        wire_format=str(p.get("wire_format", "pickle") or "pickle"),
        grpc_enabled=bool(p.get("grpc_enabled", False)),
        # Declared group backend (issue #154), advertised via /info for workers
        # to validate against.
        backend=str(p.get("backend", "http") or "http"),
        num_fragments=int(p.get("num_fragments", 1) or 1),
        from_checkpoint=p.get("from_checkpoint") or None,
        save_every=int(p.get("save_every", 10) or 0),
        save_total_limit=int(p.get("save_total_limit", 3) or 0),
        outer_lr=(float(p["outer_lr"]) if p.get("outer_lr") is not None else None),
        outer_momentum=(
            float(p["outer_momentum"]) if p.get("outer_momentum") is not None else None
        ),
        no_nesterov=bool(p.get("no_nesterov", False)),
        heartbeat_timeout=(
            float(p["heartbeat_timeout"])
            if p.get("heartbeat_timeout") is not None
            else None
        ),
        min_workers=(
            int(p["min_workers"]) if p.get("min_workers") is not None else None
        ),
        auth_token_file=auth_token_file,
        no_auth=no_auth,
        # Token redaction in the spawned server's TTY is a demo-mode
        # concern, not an operator choice: suppress the bearer in the
        # launch banner only when this webui is running --demo (public
        # TTY pane). In normal operation the token is shown so it can be
        # copied onto clients, Jupyter-style.
        quiet_tokens=demo_mode_enabled(),
        bulk_cleartext=bool(p.get("bulk_cleartext", False)),
        run_name=(p.get("run_name") or None),
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


#: Loopback host aliases for matching a diloco_server JobRecord to a
#: ``server_addr``. Mirrors ``forgather.ml.diloco.auth._LOCAL_HOSTS``.
_DILOCO_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

#: Env var the worker's DiLoCoClient consults for its bearer token
#: (see ``forgather.ml.diloco.client.TOKEN_ENV_VAR``). Hardcoded rather
#: than imported to keep the scheduler free of the torch-importing
#: client module at load time.
_DILOCO_TOKEN_ENV_VAR = "FORGATHER_DILOCO_SERVER_TOKEN"


def _diloco_token_for_server_addr(server_addr: str) -> Optional[str]:
    """Bearer token for a locally-spawned diloco_server at ``server_addr``.

    A training worker discovers its token from the per-port loopback
    file only when the URL is loopback (see
    ``forgather.ml.diloco.auth.read_standalone_token``). When the webui
    spawns a server bound to ``0.0.0.0``, ``server_addr`` carries the
    stamped *routable* (non-loopback) host, so loopback auto-discovery
    never fires and the worker would hit a 401. Resolve the token here
    by matching the URL against running ``diloco_server`` JobRecords —
    the same host/port logic the webui proxy uses in
    ``routes/diloco.py:_token_for_local`` — so the scheduler can inject
    it into the worker's environment.

    Returns ``None`` for genuinely remote servers (no matching local
    record), in which case the worker relies on an explicitly configured
    token / env var.
    """
    from urllib.parse import urlparse

    addr = str(server_addr).strip()
    if "://" not in addr:
        # Bare host:port — give urlparse a scheme to parse against.
        addr = "http://" + addr
    try:
        parsed = urlparse(addr)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        return None
    host_is_loopback = host in _DILOCO_LOOPBACK_HOSTS
    for r in job_records.list_records():
        if r.job_type != "diloco_server":
            continue
        if r.status not in RUNNING_STATUSES:
            continue
        if not r.auth_token:
            continue
        params = r.job_params or {}
        rec_port = params.get("port")
        try:
            rec_port = int(rec_port) if rec_port is not None else None
        except (TypeError, ValueError):
            continue
        if rec_port != port:
            continue
        rec_host = (params.get("host") or "127.0.0.1").lower()
        rec_routable = (params.get("routable_host") or "").lower()
        if host_is_loopback and (
            rec_host in _DILOCO_LOOPBACK_HOSTS or rec_host == "0.0.0.0"
        ):
            return r.auth_token
        if rec_routable and host == rec_routable:
            return r.auth_token
        if rec_host == host:
            return r.auth_token
    return None


def _diloco_server_token(server_addr: str) -> Optional[str]:
    """Bearer token for a param server at ``server_addr``.

    Resolves a locally-spawned server's token from its JobRecord first (that
    path also handles the routable-host stamping), then falls back to the same
    inventory the webui proxy uses — the user registry + the cluster snapshot —
    so a **registered remote** server's token is found too. Without this
    fallback the launch-time ``/info`` query (and the worker's own token) would
    401 against any server this scheduler didn't itself spawn. Returns ``None``
    when no source knows the address (a ``--no-auth`` server, or genuinely
    unknown).
    """
    token = _diloco_token_for_server_addr(server_addr)
    if token:
        return token
    try:
        from . import cluster_diloco_inventory as cdi

        norm = cdi._normalize(server_addr)
        for s in cdi.local_servers():
            if cdi._normalize(s.base_url) == norm and s.auth_token:
                return s.auth_token
        t = cdi.master_inventory.token_for_url(server_addr)
        if t:
            return t
    except Exception:
        pass
    return None


def _diloco_server_verify_tls(server_addr: str) -> bool:
    """``verify_tls`` for a known param server at ``server_addr``.

    Resolved from the same source the webui proxy uses
    (``routes/diloco.py:_verify_for``): the local-server inventory (which folds
    in user-registry entries) then the cluster snapshot. Defaults to ``True`` —
    a verified handshake — when the server isn't in any inventory (e.g. a bare
    loopback dev server, where TLS is moot anyway).
    """
    try:
        from . import cluster_diloco_inventory as cdi

        norm = cdi._normalize(server_addr)
        for s in cdi.local_servers():
            if cdi._normalize(s.base_url) == norm:
                return bool(s.verify_tls)
        v = cdi.master_inventory.verify_tls_for_url(server_addr)
        if v is not None:
            return bool(v)
    except Exception:
        pass
    return True


def _diloco_query_info(server_addr: str, queue_id: str) -> Dict[str, Any]:
    """Fetch ``<server>/info`` to derive the launch backend.

    The sync backend is server-authoritative (issue #154): an orchestrated
    worker is enqueued *without* one, and the value is read here — at the moment
    we're about to spawn ``torchrun`` — from the server's ``/info``. There is no
    safe default, so an unreachable server **raises**; ``_launch`` wraps the
    builder in ``try/except`` and marks the job ``failed`` with this message.

    Implemented as a minimal ``urllib`` GET (not ``DiLoCoClient``) to keep torch
    off the long-lived server's import path. Auth/TLS reuse the same primitives
    the worker's client and the webui proxy use: the per-port bearer token and a
    cluster-CA ``SSLContext`` (which also presents this host's cert for an mTLS
    server).
    """
    import json
    import urllib.request

    base = str(server_addr).strip().rstrip("/")
    if "://" not in base:
        # Bare host:port — a loopback dev server. Orchestrated server_addrs
        # always carry a scheme (the inventory normalizes them), so this only
        # fires for the cleartext-localhost case.
        base = "http://" + base
    url = base + "/info"
    headers: Dict[str, str] = {}
    token = _diloco_server_token(server_addr)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    ctx = None
    if base.lower().startswith("https"):
        from forgather.tls.runtime import urllib_ssl_context

        ctx = urllib_ssl_context(verify=_diloco_server_verify_tls(server_addr))
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(
            f"DiLoCo backend could not be derived for job {queue_id}: the param "
            f"server at {base} is unreachable ({e}). Start it, or launch the "
            f"worker directly with 'submit --local-only --backend <kind>'."
        ) from e


def _diloco_env_from_job_params(
    diloco: Dict[str, Any],
    queue_id: str,
) -> Dict[str, str]:
    """Translate a ``job_params.diloco`` dict into ``DILOCO_*`` env vars.

    The DiLoCoCallback constructor reads these as the fallback when its
    template args are None. Only keys the operator actually set get
    forwarded so the rest fall back to whatever the callback's
    constructor default / project template specifies.

    Special case for ``worker_id``: the env var is **always set** when
    DiLoCo is enabled, even if the operator didn't supply one — config
    preprocessing reads ``DILOCO_WORKER_ID`` to derive a unique output
    directory per worker, so a missing value at preprocessing time
    would cause two workers to share an output dir and clobber each
    other's checkpoints. The queue route at ``routes/queue.py:enqueue``
    fills in a memorable two-word default for any DiLoCo training
    submission that arrives without one (matches the pool-style
    submit-modal behavior); this fallback covers the remaining edge
    case where a queue item somehow reaches dispatch with an empty
    worker_id (e.g. a direct ``queue_store.add_item`` from a future
    code path that bypasses the route).

    Expected input shape (all keys optional except ``server_addr``):
        {
          "server_addr": "host:port",
          "heartbeat_interval": float,
          "worker_id": str,
          # Sync backend (issue #154). Orchestrated worker submissions omit it —
          # the backend is derived here from the server's /info at launch (see
          # below). ``backend`` is present only for a --local-only re-enqueue or
          # an older queued job, in which case it's honored verbatim and the
          # group-coordination keys (``shm_group_id`` / ``shm_group_size``) come
          # inline. ``diloco_replicate`` marks a collective job — one torchrun
          # world whose size rides the separate ``job_params.nproc`` path.
          "backend": "http" | "shared_memory" | "collective",
          "shm_group_id": str,
          "shm_group_size": int,
          "diloco_replicate": int,
        }

    The server-authoritative settings (``sync_every`` / ``num_fragments`` /
    ``dylu`` / ``bf16_comm``) are deliberately NOT forwarded — the worker reads
    them from the server's ``/info`` so the whole group agrees; a stale
    submission carrying them is ignored here.
    """
    env: Dict[str, str] = {}
    server = diloco.get("server_addr")
    if not server:
        return env
    env["DILOCO_SERVER"] = str(server)
    # Forward the bearer token so the worker can authenticate even when
    # server_addr is routable (non-loopback) and the per-port loopback file
    # auto-discovery in DiLoCoClient wouldn't fire. Resolves a locally-spawned
    # server's token from its JobRecord and a registered remote's from the
    # inventory. An explicit token in extra_env / the worker's own env still
    # wins (we never overwrite a set value).
    token = _diloco_server_token(str(server))
    if token:
        env[_DILOCO_TOKEN_ENV_VAR] = token
    # sync_every / num_fragments / dylu / bf16_comm are server-authoritative
    # (they must match across the group); the worker reads them from the
    # server's /info at startup, so we no longer forward them from the
    # submission. Only client-local knobs are forwarded.
    if diloco.get("heartbeat_interval") is not None:
        env["DILOCO_HEARTBEAT_INTERVAL"] = str(float(diloco["heartbeat_interval"]))
    # Always-set: operator-supplied value if present, the route-
    # filled memorable default otherwise. Last-resort fallback to a
    # freshly-minted memorable name covers any path that bypasses the
    # route (no queue_id leak into the worker identity).
    wid = (diloco.get("worker_id") or "").strip()
    if not wid:
        from forgather.utils import generate_name

        wid = generate_name()
    env["DILOCO_WORKER_ID"] = wid
    # Sync backend (issue #154). The server is the single declarer: an
    # orchestrated worker is enqueued *without* a backend and the value is
    # derived here, at launch, from the server's /info — so a worker can never be
    # spawned disagreeing with its group. An explicit backend in the submission
    # is honored only for the --local-only dev path (and for back-compat with a
    # job queued by an older CLI), which skips the query.
    explicit_backend = (diloco.get("backend") or "").strip().lower()
    is_collective_job = bool(diloco.get("diloco_replicate"))
    server_info: Dict[str, Any] = {}
    if explicit_backend:
        # Trusted/back-compat path (--local-only re-enqueue, or a job from an
        # older CLI): honor the submitted backend verbatim, no query, no topology
        # cross-check — the operator set it directly.
        backend_kind = explicit_backend
    else:
        # Derived path (the orchestrated default): the backend comes from the
        # server's /info at launch.
        server_info = _diloco_query_info(str(server), queue_id)
        ecs = server_info.get("expected_client_settings") or {}
        backend_kind = (ecs.get("backend") or "http").strip().lower()
        log.info(
            "diloco job %s: derived backend=%s from server %s",
            queue_id,
            backend_kind,
            server,
        )
        # The derived backend must match the job's launch topology, which was
        # fixed at enqueue and can't be reconciled now: a collective job is one
        # torchrun world (sized by job_params.nproc), a worker job is one of N
        # independent processes. A mismatch is fatal (fail the launch).
        if is_collective_job and backend_kind != "collective":
            raise RuntimeError(
                f"DiLoCo backend mismatch for job {queue_id}: this is a "
                f"collective job (one torchrun world of "
                f"{diloco.get('diloco_replicate')} replicas) but the server at "
                f"{server} declares backend={backend_kind!r}. Start the param "
                f"server with --backend collective."
            )
        if backend_kind == "collective" and not is_collective_job:
            raise RuntimeError(
                f"DiLoCo backend mismatch for job {queue_id}: the server at "
                f"{server} declares backend='collective' (one torchrun world) "
                f"but this was submitted as an independent worker. Launch it "
                f"with --diloco-replicate N instead."
            )
    if backend_kind == "shared_memory":
        env["DILOCO_BACKEND"] = "shared_memory"
        if explicit_backend:
            # Back-compat path: an older queued job carried the group
            # coordination inline (one dir per submit, group size = worker count).
            group_id = (str(diloco.get("shm_group_id") or "")).strip()
            if group_id:
                env["DILOCO_SHM_GROUP_DIR"] = os.path.join(
                    tempfile.gettempdir(), f"diloco_shm_{group_id}"
                )
            size = diloco.get("shm_group_size")
            if size:
                env["DILOCO_SHM_GROUP_SIZE"] = str(int(size))
        else:
            # Server-derived: the group dir is one per *server* (stable,
            # base_url-derived) so every co-located worker computes the same
            # region without a submit-time uuid; the group size is the server's
            # declared worker count (single source of truth, /info num_workers).
            # The dir being stable per server (vs the old per-submit uuid) is
            # safe: SharedMemoryBackend decides the aggregator role with an OS
            # ownership lease, so a region orphaned by a crashed group is
            # reclaimed and rebuilt by the next launch rather than attached to.
            from . import cluster_diloco_inventory as cdi

            group_id = cdi.server_id_for(cdi._normalize(str(server)))
            env["DILOCO_SHM_GROUP_DIR"] = os.path.join(
                tempfile.gettempdir(), f"diloco_shm_{group_id}"
            )
            num_workers = server_info.get("num_workers")
            if num_workers:
                env["DILOCO_SHM_GROUP_SIZE"] = str(int(num_workers))
    elif backend_kind == "collective":
        # Collective backend (issue #154): N replicas in one torchrun job
        # all-reduce pseudo-grads. The torchrun world is sized by job_params.nproc
        # (the separate nproc_override path); here we only set the backend +
        # replicate degree the DistributedEnvironment reads to build the diloco
        # mesh axis. DILOCO_WORKER_ID (the base) is already set above; the
        # train_script rewrites it per-rank.
        env["DILOCO_BACKEND"] = "collective"
        replicate = diloco.get("diloco_replicate")
        if replicate:
            env["DILOCO_REPLICATE"] = str(int(replicate))
    # http (the request/response default) emits no DILOCO_BACKEND: the trainer
    # defaults to http when it's unset, and the worker's own /info check (PR1)
    # validates the (default-http) value against the server regardless.
    return env


def _build_training(item, gpu_indices, tty_path):
    # Multi-node training jobs (Phase 3 cluster-coordinator submit)
    # carry their torchrun rendezvous args + NCCL env in
    # ``job_params``. Single-node training jobs leave job_params empty
    # and the launcher falls back to ``--standalone``.
    rdzv_args = item.job_params.get("rdzv_args") or None
    extra_env = dict(item.job_params.get("extra_env") or {})
    # DiLoCo opt-in: a non-empty ``job_params.diloco.server_addr``
    # means this worker should join the named DiLoCo server. The
    # DiLoCoCallback wired into the training config reads DILOCO_*
    # env vars when its template args are None; the webui sets the
    # env via this channel so no scheduler-side surgery to dynamic
    # args is needed.
    diloco = item.job_params.get("diloco") or {}
    if isinstance(diloco, dict):
        diloco_env = _diloco_env_from_job_params(diloco, item.queue_id)
        # The derived DILOCO_* values win over any inherited extra_env,
        # EXCEPT the bearer token: an operator who set it explicitly in
        # extra_env knows their server better than our JobRecord match,
        # so that value takes precedence.
        for k, v in diloco_env.items():
            if k == _DILOCO_TOKEN_ENV_VAR and k in extra_env:
                continue
            extra_env[k] = v
    extra_env = extra_env or None
    # ``nproc`` from job_params is an explicit single-node override
    # (typed into the SubmitModal nproc field, or supplied by other
    # callers that want to bypass the config's nproc_per_node).
    # Falls back to either the config value or the CPU "gpu"->1
    # dispatch fallback inside build_command when unset. Cluster
    # dispatches ignore this in favor of rdzv_args's per-peer nproc.
    nproc_override = _coerce_nproc_override(item.job_params.get("nproc"))
    if not gpu_indices and rdzv_args is None:
        log.info(
            "training job %s dispatching with 0 GPUs reserved (CPU mode); "
            "nproc_override=%r",
            item.queue_id,
            nproc_override,
        )
    return launcher.spawn_training_process(
        project_dir=item.project_dir,
        config_name=item.config,
        dynamic_args=item.dynamic_args,
        gpu_indices=gpu_indices,
        tty_log_path=tty_path,
        extra_env=extra_env,
        rdzv_args=rdzv_args,
        nproc_override=nproc_override,
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
    "diloco_server": _build_diloco_server,
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
                    (
                        mm
                        for mm in _cluster.members()
                        if mm.node_id == self_ident.node_id
                    ),
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
        from inference_server.auth_paths import standalone_token_file as _inf_token_path
        from inference_server.auth_paths import (
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


def _resolve_diloco_server_token(*, port: int, regen: bool) -> str:
    """Return the bearer token a diloco_server spawn should use.

    Mirrors ``_resolve_dataset_server_token``: per-port persisted token
    file, reused across restarts until the operator passes
    ``--regen-token``. The same file the standalone ``forgather diloco
    server`` CLI reads, so a webui-spawned server and a manually-run
    server are interchangeable from the worker's point of view.

    Persistence may fail (e.g. read-only home); the spawn still gets
    a valid token in memory and the JobRecord carries it, but the next
    start won't auto-discover. Logged at WARNING; not raised.
    """
    token_path = diloco_server_standalone_token_file(port)
    if not regen and token_path.is_file():
        try:
            existing = token_path.read_text().strip()
        except OSError:
            existing = ""
        if existing:
            return existing
    token = secrets.token_hex(32)
    try:
        diloco_server_write_standalone_token(port, token)
    except OSError as e:
        log.warning(
            "could not persist diloco_server token at %s: %s",
            token_path,
            e,
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
        elif item.job_type == "diloco_server":
            # Per-port persisted token (matches dataset_server). Worker
            # plumbing in forgather.ml.diloco.client auto-discovers it
            # from this same per-port file when the URL is loopback;
            # the webui proxy reads it from the JobRecord and attaches
            # the bearer header for cross-host browsers.
            auth_token = _resolve_diloco_server_token(
                port=int(item.job_params.get("port", 8512)),
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
    if item.job_type in ("inference", "dataset_server", "diloco_server"):
        try:
            from forgather.tls import client_scheme as _client_scheme

            host_for_scheme = finalized_params.get("host", "127.0.0.1")
            finalized_params.setdefault("scheme", _client_scheme(host_for_scheme))
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
    # on its Job card — inference, dataset_server, diloco_server,
    # and mkdocs. TensorBoard renders its own URL with its bind_all
    # toggle in mind and is left alone.
    if item.job_type in (
        "inference",
        "dataset_server",
        "diloco_server",
        "mkdocs",
    ):
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
    elif item.job_type == "diloco_server":
        _wake_diloco_inventory()


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


# Endpoints that looked external on the previous tick. We only promote an
# endpoint after it survives a tick unclaimed, so a scheduler-spawned trainer
# whose record hasn't correlated yet (and whose PID lineage momentarily can't be
# walked) gets a tick for _correlate_running_records to claim it first.
_promote_grace: set[str] = set()


def _promote_external_endpoints() -> None:
    """Promote externally-launched trainer endpoints into JobRecords.

    A trainer started outside the scheduler (e.g. a foreground
    ``forgather train`` with ``TrainerControlCallback``) writes a control
    endpoint but has no JobRecord. Synthesize one (marked
    ``externally_launched``) so it appears in ``forgather job`` with
    first-class status/lifecycle and is reaped by PID liveness like a
    re-attached job. Scheduler-spawned trainers are excluded two ways — by
    job_id (once correlated) and by PID lineage — plus a one-tick grace so a
    not-yet-correlated scheduler job is never promoted as a duplicate.

    GPU accounting is intentionally deferred: external records carry
    ``gpu_indices=[]`` (so they don't reserve the scheduler's pool). Wiring
    real, ``CUDA_VISIBLE_DEVICES``-correct accounting is a follow-up.
    """
    global _promote_grace
    try:
        endpoints = trainer_control.list_jobs()
    except Exception as e:
        log.debug("list_jobs failed during promotion: %s", e)
        return
    if not endpoints:
        _promote_grace = set()
        return

    records = job_records.list_records()
    known_job_ids = {r.job_id for r in records if r.job_id}
    # Launcher pids of records we're running: an endpoint whose pid descends
    # from one of these is a scheduler-spawned trainer that _correlate owns.
    launcher_pids = {
        r.pid for r in records if r.status in RUNNING_STATUSES and r.pid is not None
    }

    seen_unclaimed: set[str] = set()
    for ep in endpoints:
        if ep.job_id in known_job_ids:
            continue  # already represented (correlated or already promoted)
        if not trainer_control.is_endpoint_pid_alive(ep.pid, ep.started_at):
            continue  # dead endpoint; the GC sweep removes the dir
        if launcher_pids and any(
            anc in launcher_pids for anc in _pid_ancestors(ep.pid)
        ):
            continue  # scheduler-spawned; _correlate attaches it to its record
        # Genuine external candidate. Wait one tick before promoting so a
        # not-yet-correlated scheduler job isn't briefly double-counted.
        seen_unclaimed.add(ep.job_id)
        if ep.job_id not in _promote_grace:
            continue
        queue_id = f"ext_{ep.job_id}"
        if job_records.get_record(queue_id) is not None:
            continue
        record = JobRecord(
            queue_id=queue_id,
            job_id=ep.job_id,
            externally_launched=True,
            job_type="training",
            status="running",
            node=job_records.LOCAL_NODE,
            pid=ep.pid,
            started_at=ep.started_at,
            requested_gpus=0,  # not scheduler-allocated
            gpu_indices=[],  # GPU accounting deferred
            logs_dir=ep.logging_dir,
            output_dir=ep.output_dir,
        )
        job_records.add_record(record)
        with _state._lock:
            _state.running[queue_id] = None  # reaped via PID liveness
        log.info(
            "promoted external trainer endpoint %s (pid=%s) to a job record",
            ep.job_id,
            ep.pid,
        )

    # Carry this tick's unclaimed candidates into the next tick's grace set.
    _promote_grace = seen_unclaimed


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
        # A DiLoCo server traps SIGTERM into a *coordinated* shutdown — it
        # relays save_and_stop to its workers and keeps serving until they
        # drain, which legitimately takes far longer than 2s. So for a
        # diloco_server + SIGTERM, a non-exit at 2s is expected, not a hang:
        # don't stamp the misleading "stuck in CUDA" error. (SIGKILL, or any
        # other job type, keeps the original fast-exit expectation.)
        graceful_diloco = record.job_type == "diloco_server" and sig == signal.SIGTERM
        if graceful_diloco:
            log.info(
                "Signalled %s (pid=%d) to shut down; the DiLoCo server is "
                "draining its workers (save_and_stop) and will exit once they "
                "finish.",
                queue_id,
                record.pid,
            )
        elif not _wait_for_pid_exit(record.pid, timeout=2.0):
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
        elif updated.job_type == "diloco_server":
            _wake_diloco_inventory()
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


def _wake_diloco_inventory() -> None:
    """Signal the cluster DiLoCo-server inventory to re-poll.

    Same shape as :func:`_wake_inference_inventory` — lazy import,
    swallow failures, wake is a latency hint not a correctness
    requirement.
    """
    try:
        from . import cluster_diloco_inventory

        cluster_diloco_inventory.wake_loops()
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
    recycled the pid and we treat it as gone. Delegated to
    :func:`trainer_control.is_endpoint_pid_alive` so the guard and its
    slack constant (:data:`trainer_control.PID_REUSE_SLACK_SECONDS`)
    stay aligned with the Jobs API and the ``forgather job``
    cleanup/list paths.
    """
    reattached = 0
    cleaned = 0
    for r in job_records.list_records():
        if r.status not in RUNNING_STATUSES:
            continue
        if r.pid is None:
            _mark_failed(r.queue_id, "server restart; no pid recorded")
            cleaned += 1
            continue

        alive = trainer_control.is_endpoint_pid_alive(r.pid, r.started_at)

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
    try:
        swept_dirs = _gc.sweep_dead_endpoint_dirs()
        if swept_dirs:
            log.info("startup endpoint-dir sweep removed %d dir(s)", swept_dirs)
    except Exception:
        log.exception("startup endpoint-dir sweep failed")


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
                _promote_external_endpoints()
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
                    try:
                        swept_dirs = _gc.sweep_dead_endpoint_dirs()
                        if swept_dirs:
                            log.info(
                                "periodic endpoint-dir sweep removed %d dir(s)",
                                swept_dirs,
                            )
                    except Exception:
                        log.exception("periodic endpoint-dir sweep failed")
            except Exception:
                log.exception("dispatcher tick failed")
            await asyncio.sleep(TICK_SECONDS)
    except asyncio.CancelledError:
        log.info("dispatcher loop cancelled")
        raise
