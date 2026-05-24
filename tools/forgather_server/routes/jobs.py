"""Unified jobs API.

A "job" here is a training run we know about, from one of two sources:

    1. ``JobRecord`` — a run *we* dispatched via the queue. Carries config,
       dynamic_args, GPU assignment, captured TTY path, exit code, etc.
    2. ``TrainerControlClient.JobInfo`` — discovery from a
       ``~/.config/forgather/jobs/{job_id}/endpoint.json`` file. Carries the
       trainer's host:port for control commands and (for newer callbacks)
       its ``logging_dir`` / ``output_dir``.

These are merged by PID lineage: an endpoint whose PID is descended from a
JobRecord's torchrun PID is "the same job"; otherwise it's an external
discovery (e.g. a job started via the CLI).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from forgather import trainer_control

from .. import _gc
from .. import auth as auth_mod
from .. import job_records, scheduler

log = logging.getLogger("forgather_server.jobs")

router = APIRouter(tags=["jobs"])

TTY_POLL_INTERVAL_SECONDS = 0.5
TTY_TAIL_AFTER_EXIT_SECONDS = 3.0
# Caps to bound memory on TTY reads. A long-running training job's tty.log
# can grow to multiple GB; without bounds, both the dump endpoint and the
# initial WebSocket backlog OOM the server.
TTY_DUMP_MAX_BYTES = 32 * 1024 * 1024  # 32 MiB tail by default
TTY_BACKLOG_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MiB per WS send during backlog


class JobModel(BaseModel):
    """Unified view of a job (server-launched or externally discovered)."""

    # Stable identifier the UI keys on. Equal to queue_id for our jobs;
    # equal to job_id for externally-discovered ones (no queue_id).
    id: str

    queue_id: Optional[str] = None
    job_id: Optional[str] = None  # trainer's id from endpoint.json

    # Run metadata (only populated for our jobs)
    project_dir: Optional[str] = None
    config: Optional[str] = None
    dynamic_args: Optional[Dict[str, Any]] = None
    requested_gpus: Optional[int] = None
    priority: Optional[int] = None
    submitted_at: Optional[float] = None
    node: Optional[str] = None
    gpu_indices: Optional[List[int]] = None
    # Job-type dispatch. "training" is the historical default and is what
    # externally-discovered endpoints always are (only the trainer writes
    # endpoint.json). "eval" is a fire-and-forget subprocess with no
    # trainer control; ``job_params`` then carries eval-specific fields
    # (eval_project, eval_template, model_path, checkpoint_path, etc.).
    job_type: str = "training"
    job_params: Optional[Dict[str, Any]] = None

    # Lifecycle
    status: str  # starting | running | done | failed | aborted
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None

    # IO + control
    pid: Optional[int] = None
    host: Optional[str] = None
    port: Optional[int] = None
    alive: bool = False
    tty_log_path: Optional[str] = None
    logs_dir: Optional[str] = None
    output_dir: Optional[str] = None
    # For path-prefixed sub-services (e.g. TensorBoard spawned with
    # ``--path_prefix /api/tb/<queue_id>``). The webui appends this to the
    # host:port link so users SSH-forwarding the upstream port get a URL
    # that actually serves content (the spawned TB returns 404 on ``/``).
    path_prefix: Optional[str] = None
    # Per-spawn bearer token for inference jobs. Surfaced to the webui so
    # the inference panel can auto-populate the Auth Token field when the
    # user picks a known local server. Already inside an auth-gated
    # endpoint; an authenticated user can use the token via the proxy
    # auto-lookup anyway, so exposing it adds no new attack surface.
    auth_token: Optional[str] = None

    # Source: where did this entry come from
    source: str  # "record" | "endpoint" | "merged"


class ControlResponseModel(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


def _pid_alive(pid: Optional[int]) -> bool:
    """True iff ``pid`` refers to a running, *non-zombie* process.

    Thin wrapper that handles ``None`` and delegates to the shared
    helper used by the scheduler's reap path, so "is this PID alive"
    has a single answer regardless of which subsystem is asking.
    Pre-consolidation, this module had its own copy with a different
    psutil-less fallback (returned True) than the scheduler's (used
    ``os.kill(pid, 0)``); operators who hit the no-psutil path saw
    inconsistent live-ness reporting between the Jobs view and the
    scheduler's reap loop.

    Note: this is *not* PID-reuse aware. Callers that have a known
    process-start timestamp (e.g. an endpoint's ``started_at``) should
    use :func:`_endpoint_is_live` so a recycled pid from a prior boot
    doesn't masquerade as the original owner.
    """
    if pid is None:
        return False
    return scheduler._pid_is_alive(pid)


def _endpoint_is_live(ep: trainer_control.JobInfo) -> bool:
    """PID-reuse-aware liveness for an externally-discovered endpoint.

    Thin wrapper over :func:`trainer_control.is_endpoint_pid_alive` —
    see that function for the rationale. The wrapper exists so call
    sites here read as "is the endpoint live" rather than two attribute
    accesses on every line.
    """
    return trainer_control.is_endpoint_pid_alive(ep.pid, ep.started_at)


def _record_to_model(
    r: job_records.JobRecord,
    matched_endpoint: Optional[trainer_control.JobInfo] = None,
) -> JobModel:
    # Backfill `scheme` for inference/dataset_server records that
    # pre-date the scheduler-side stamp (or were created on a host
    # whose TLS state has since flipped). The current TLS state is
    # the right answer because the upstream child read the same
    # shared config when it started; if the operator just enabled
    # TLS, they also need to restart the spawned child, so the
    # backfill matches what the server *currently* serves.
    job_params_out = dict(r.job_params)
    if r.job_type in ("inference", "dataset_server") and "scheme" not in job_params_out:
        try:
            from forgather.tls import client_scheme as _client_scheme

            host_for_scheme = job_params_out.get("host", "127.0.0.1")
            job_params_out["scheme"] = _client_scheme(host_for_scheme)
        except Exception:
            job_params_out["scheme"] = "http"
    # Backfill routable_host the same way — pre-existing inference/
    # dataset_server records bound to 0.0.0.0 want a LAN-routable URL
    # displayed in the Job card. Computed at API-response time because
    # the routable address depends on the *current* network state (mDNS
    # discovery may have come up since the record was written).
    if (
        r.job_type in ("inference", "dataset_server")
        and job_params_out.get("host") in ("0.0.0.0", "::", "")
        and "routable_host" not in job_params_out
    ):
        try:
            from .. import scheduler as _scheduler

            routable = _scheduler.detect_routable_host()
            if routable:
                job_params_out["routable_host"] = routable
        except Exception:
            pass

    # In demo mode strip bearer tokens — both the dedicated field and
    # any token-shaped key buried in job_params — so the webui can't
    # display, and curl /api/jobs can't exfiltrate, credentials minted
    # for spawned inference / dataset jobs.
    auth_token_out = None if auth_mod.demo_mode_enabled() else r.auth_token
    job_params_out = auth_mod.redact_sensitive_in_demo(job_params_out)
    return JobModel(
        id=r.queue_id,
        queue_id=r.queue_id,
        job_id=r.job_id or (matched_endpoint.job_id if matched_endpoint else None),
        project_dir=r.project_dir,
        config=r.config,
        dynamic_args=r.dynamic_args,
        requested_gpus=r.requested_gpus,
        priority=r.priority,
        submitted_at=r.submitted_at,
        node=r.node,
        gpu_indices=list(r.gpu_indices),
        job_type=r.job_type,
        job_params=job_params_out,
        status=r.status,
        started_at=r.started_at,
        finished_at=r.finished_at,
        exit_code=r.exit_code,
        error=r.error,
        pid=r.pid,
        host=matched_endpoint.host if matched_endpoint else None,
        port=matched_endpoint.port if matched_endpoint else None,
        alive=_pid_alive(r.pid) and r.status in ("starting", "running"),
        tty_log_path=r.tty_log_path,
        path_prefix=r.path_prefix,
        auth_token=auth_token_out,
        logs_dir=r.logs_dir,
        output_dir=r.output_dir,
        source="merged" if matched_endpoint else "record",
    )


def _endpoint_to_model(ep: trainer_control.JobInfo) -> JobModel:
    alive = _endpoint_is_live(ep)
    return JobModel(
        id=ep.job_id,
        job_id=ep.job_id,
        status="running" if alive else "failed",
        started_at=ep.started_at,
        pid=ep.pid,
        host=ep.host,
        port=ep.port,
        alive=alive,
        logs_dir=ep.logging_dir,
        output_dir=ep.output_dir,
        source="endpoint",
    )


def _build_unified_list(include_dead_endpoints: bool) -> List[JobModel]:
    """Merge JobRecords + TrainerControlClient endpoints.

    Pairing key: an endpoint whose ``job_id`` matches a JobRecord's
    correlated ``job_id`` is the "same" job and shows up once with
    source="merged". Endpoints not matched by job_id stay as their own
    entries (source="endpoint"). JobRecords always appear, even if their
    correlation hasn't found an endpoint yet (source="record").
    """
    records = job_records.list_records()
    try:
        endpoints = trainer_control.list_jobs()
    except Exception as e:
        log.warning("list_jobs (endpoint discovery) failed: %s", e)
        endpoints = []

    by_jobid: Dict[str, trainer_control.JobInfo] = {ep.job_id: ep for ep in endpoints}
    matched_jobids: set[str] = set()

    out: List[JobModel] = []
    for r in records:
        ep = by_jobid.get(r.job_id) if r.job_id else None
        if ep is not None:
            matched_jobids.add(ep.job_id)
        out.append(_record_to_model(r, ep))

    for ep in endpoints:
        if ep.job_id in matched_jobids:
            continue
        model = _endpoint_to_model(ep)
        if not model.alive and not include_dead_endpoints:
            continue
        out.append(model)

    # Newest first overall, with active items above terminal ones.
    def sort_key(m: JobModel):
        terminal = m.status not in ("starting", "running")
        ts = m.started_at or m.submitted_at or 0.0
        return (terminal, -ts)

    out.sort(key=sort_key)
    return out


@router.get("/jobs", response_model=List[JobModel])
def list_jobs(include_dead_endpoints: bool = False):
    """List all known jobs.

    By default omits externally-discovered endpoint files whose process is
    no longer alive (those accumulate as ``~/.config/forgather/jobs/`` cruft).
    JobRecords are always returned so users can see their own history.
    """
    return _build_unified_list(include_dead_endpoints=include_dead_endpoints)


@router.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    """Proxy GET /status to the trainer's HTTP control endpoint.

    Accepts either a queue_id (our id) or a trainer job_id. If a queue_id
    is given we look up its correlated job_id first.
    """
    target = job_id
    rec = job_records.get_record(job_id)
    if rec is not None:
        if rec.job_id is None:
            raise HTTPException(
                status_code=409,
                detail="not yet correlated to a trainer endpoint",
            )
        target = rec.job_id
    try:
        return trainer_control.get_job_status(target)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


_ACTION_DISPATCH = {
    "save": trainer_control.save_checkpoint,
    "stop": trainer_control.graceful_stop,
    "save-stop": trainer_control.save_and_stop,
    "abort": trainer_control.abort,
}


@router.post("/jobs/{job_id}/control/{action}", response_model=ControlResponseModel)
def job_control(job_id: str, action: str):
    """Forward a control command to the trainer's HTTP endpoint.

    For server-launched jobs, ``abort`` also kills the local process group
    via the scheduler — that handles the case where the trainer hasn't
    registered an endpoint yet.
    """
    LOCAL_ACTIONS = {"kill", "force-kill"}
    if action not in _ACTION_DISPATCH and action not in LOCAL_ACTIONS:
        raise HTTPException(status_code=400, detail=f"unknown action: {action}")

    rec = job_records.get_record(job_id)
    target_jobid = rec.job_id if rec else job_id

    # Non-training jobs don't register trainer HTTP endpoints, so
    # save/stop/save-stop/abort have no subscriber. kill/force-kill work
    # on any subprocess-style job because they signal our local group.
    if rec is not None and rec.job_type != "training" and action not in LOCAL_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"action {action!r} is not supported for {rec.job_type} jobs; "
                "use kill or force-kill instead"
            ),
        )

    # Local kill paths are server-only — they signal our own process group
    # rather than going through the trainer's HTTP endpoint. Useful when
    # the trainer is hung and not servicing /control any more.
    if action in LOCAL_ACTIONS:
        if rec is None:
            raise HTTPException(
                status_code=400,
                detail=f"{action} is only supported on server-launched jobs",
            )
        if action == "force-kill":
            ok = scheduler.force_kill_record(rec.queue_id)
            msg = "force-killed (SIGKILL on process group)"
        else:
            ok = scheduler.abort_record(rec.queue_id)
            msg = "aborted (SIGTERM on process group)"
        return ControlResponseModel(
            success=ok, message=msg if ok else "nothing to kill"
        )

    if target_jobid is None:
        raise HTTPException(
            status_code=409,
            detail="no trainer endpoint yet; use action=kill for a hard abort",
        )
    fn = _ACTION_DISPATCH[action]
    try:
        resp = fn(target_jobid)
    except Exception as e:
        # Mirror job_status — surface trainer connectivity issues as 502
        # rather than letting them bubble up as a generic 500 stack.
        raise HTTPException(status_code=502, detail=str(e))
    return ControlResponseModel(
        success=resp.success, message=resp.message, data=resp.data
    )


# ---------------------------------------------------------------------- TTY


def _tty_path_for(job_id: str) -> str:
    rec = job_records.get_record(job_id)
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{job_id} is not a server-launched job; TTY capture only "
                "exists for jobs we spawned"
            ),
        )
    if not rec.tty_log_path:
        raise HTTPException(status_code=404, detail="no TTY log recorded yet")
    return rec.tty_log_path


def _is_terminal(job_id: str) -> bool:
    rec = job_records.get_record(job_id)
    return rec is not None and rec.status in job_records.TERMINAL_STATUSES


@router.get("/jobs/{job_id}/tty", response_class=PlainTextResponse)
def tty_dump(job_id: str):
    """Captured stdout/stderr (tail; no follow).

    Returns at most :data:`TTY_DUMP_MAX_BYTES` from the end of the log.
    Long-running training jobs accumulate hundreds of MB of TTY output;
    a bare ``f.read()`` would OOM the server on each click.
    """
    path = _tty_path_for(job_id)
    try:
        import os as _os

        with open(path, "rb") as f:
            try:
                size = _os.fstat(f.fileno()).st_size
            except OSError:
                size = 0
            if size > TTY_DUMP_MAX_BYTES:
                f.seek(size - TTY_DUMP_MAX_BYTES)
                # Drop the leading partial line for cleanliness.
                f.readline()
            data = f.read()
            return PlainTextResponse(
                data.decode("utf-8", errors="replace"),
                media_type="text/plain; charset=utf-8",
            )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"tty file missing: {path}")


@router.websocket("/jobs/{job_id}/tty")
async def tty_stream(ws: WebSocket, job_id: str, follow: bool = True):
    """Stream the TTY log: backlog then poll-follow.

    The TTY path is re-read from the JobRecord on every poll iteration so
    the stream survives a relocation (e.g. terminal-time relocation from
    ``~/.config/forgather/server/jobs/q_*.tty`` into the run's ``logs/tty.log``).
    The file content is preserved across the move, so the maintained byte
    ``offset`` remains valid against the new path.
    """
    await ws.accept()

    offset = 0
    exited_at: Optional[float] = None
    try:
        while True:
            try:
                path = _tty_path_for(job_id)
            except HTTPException as e:
                await ws.send_json({"type": "error", "detail": e.detail})
                break
            sent_any = False
            try:
                # Read in bounded chunks so a multi-GB backlog doesn't
                # materialize the whole file in memory before sending.
                # Loop until the file's current EOF, sending each chunk as
                # we read it.
                with open(path, "rb") as f:
                    f.seek(offset)
                    while True:
                        chunk = f.read(TTY_BACKLOG_CHUNK_BYTES)
                        if not chunk:
                            break
                        offset += len(chunk)
                        await ws.send_bytes(chunk)
                        sent_any = True
            except FileNotFoundError:
                # Stale path — relocation may have updated tty_log_path
                # between our resolve and our open. Let the next iteration
                # re-resolve. If the path stays missing past the
                # post-terminal grace window, the terminal-exit branch
                # below will close the stream cleanly.
                pass
            if sent_any:
                exited_at = None
            if not follow:
                break
            if _is_terminal(job_id):
                if exited_at is None:
                    exited_at = time.time()
                elif time.time() - exited_at > TTY_TAIL_AFTER_EXIT_SECONDS:
                    break
            await asyncio.sleep(TTY_POLL_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        return
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- mgmt


@router.delete("/jobs/{job_id}")
def remove_job(job_id: str):
    """Remove a job entry from the list.

    Two paths depending on what backs the entry:

    * **JobRecord** — must be in a terminal status. We unlink the
      central TTY (if not already relocated into a run's logs/ dir)
      and drop the JobRecord.
    * **Endpoint-only** (``source="endpoint"`` in the merged list,
      no JobRecord) — must be a *dead* endpoint (PID gone, zombie,
      or recycled). We rmtree the trainer-control directory so the
      stale ``endpoint.json`` and ``status.json`` files stop
      surfacing in the Jobs list. Live endpoint-only entries are
      refused: they belong to an actively-running trainer that
      isn't ours to evict.

    The endpoint-cleanup branch is what closes the "phantom running
    job from a previous server instance" loop the operator hit on
    muthur — without it, killing the zombie process did nothing
    visible, and the Forgather control CLI on each peer was the
    only escape.
    """
    rec = job_records.get_record(job_id)
    if rec is not None:
        if rec.status not in job_records.TERMINAL_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"cannot remove an active record (status={rec.status})",
            )
        _gc.delete_central_tty_for(rec)
        job_records.remove_record(job_id)
        return {"removed": job_id, "source": "record"}

    # Endpoint-only path. Look up the trainer-control entry, refuse
    # to evict a live one, otherwise rmtree its directory.
    ep = _find_endpoint_by_id(job_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"no record for {job_id}")
    if _endpoint_is_live(ep):
        raise HTTPException(
            status_code=409,
            detail=(
                f"endpoint {job_id} (pid={ep.pid}) is still alive. "
                "Refusing to remove a running trainer's endpoint dir."
            ),
        )
    removed_dir = _remove_endpoint_dir(job_id)
    if not removed_dir:
        raise HTTPException(
            status_code=500,
            detail=f"could not remove endpoint directory for {job_id}",
        )
    return {"removed": job_id, "source": "endpoint"}


def _find_endpoint_by_id(job_id: str) -> Optional[trainer_control.JobInfo]:
    """Locate a TrainerControl endpoint entry by its job_id."""
    try:
        eps = trainer_control.list_jobs()
    except Exception as e:
        log.warning("endpoint enumeration failed during remove: %s", e)
        return None
    for ep in eps:
        if ep.job_id == job_id:
            return ep
    return None


def _remove_endpoint_dir(job_id: str) -> bool:
    """Delete the trainer-control directory for ``job_id``.

    Mirrors the directory layout the trainer's HTTP control endpoint
    writes into (``<forgather_config_dir>/jobs/<job_id>/``). Returns
    True on successful removal, False if the directory is missing or
    rmtree raises. Errors are logged but swallowed so a partial
    removal still produces a usable response.
    """
    import shutil
    from pathlib import Path

    from forgather.preprocess import forgather_config_dir

    job_dir = Path(forgather_config_dir()) / "jobs" / job_id
    if not job_dir.exists():
        return False
    try:
        shutil.rmtree(job_dir)
        return True
    except OSError as e:
        log.warning("failed to rmtree %s: %s", job_dir, e)
        return False


@router.post("/jobs/cleanup")
def cleanup_jobs():
    """Remove every terminal JobRecord in one shot.

    Bulk counterpart of ``DELETE /api/jobs/{id}``. Walks the full record
    list, removes anything whose status is terminal (``done`` / ``failed``
    / ``aborted``), and returns the removed ids so the UI can report a
    count. Active records are left untouched — no race with running jobs.
    Same TTY-deletion policy as ``remove_job``: central files are
    unlinked, run-relocated files are kept.
    """
    removed: List[str] = []
    for r in job_records.list_records():
        if r.status in job_records.TERMINAL_STATUSES:
            _gc.delete_central_tty_for(r)
            if job_records.remove_record(r.queue_id):
                removed.append(r.queue_id)
    return {"removed": removed, "count": len(removed)}


@router.post("/jobs/gc")
def gc_jobs():
    """Sweep orphan TTY files from the central jobs_tty_dir.

    Reaps ``q_*.tty`` whose ``queue_id`` is not referenced by any
    JobRecord or queued item, mtime older than the TTL configured by
    ``$FORGATHER_ORPHAN_TTY_TTL_SECONDS`` (default 1h). Best-effort:
    per-file errors are logged and swallowed.
    """
    swept = _gc.sweep_orphan_ttys()
    return {"swept": swept}
