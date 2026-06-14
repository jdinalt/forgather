"""GPU snapshot + WebSocket stream endpoints."""

import asyncio
import logging
import os
import signal
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .. import gpu_monitor, gpu_policy, job_records
from ..job_records import RUNNING_STATUSES

log = logging.getLogger("forgather_server.gpus")

router = APIRouter(tags=["gpus"])

# Push interval for the live stream. 2 s is frequent enough to feel live and
# slow enough that NVML polling stays a rounding-error of CPU cost.
STREAM_INTERVAL_SECONDS = 2.0


class GpuProcessModel(BaseModel):
    pid: int
    used_mem_bytes: int
    name: Optional[str] = None
    kind: str = "compute"


class GpuInfoModel(BaseModel):
    index: int
    name: str
    total_mem_bytes: int
    used_mem_bytes: int
    util_pct: Optional[int] = None
    mem_util_pct: Optional[int] = None
    power_w: Optional[float] = None
    temp_c: Optional[int] = None
    fan_pct: Optional[int] = None
    processes: List[GpuProcessModel] = []
    source: str = "nvml"
    node: str = ""
    excluded: bool = False
    disabled: bool = False
    min_priority: int = 0
    # True when a running JobRecord on the owning peer has reserved this
    # GPU. Stamped server-side so the cluster Nodes panel can mark peer
    # GPUs busy without needing cross-node job visibility — each peer
    # is authoritative for its own reservations.
    reserved: bool = False


class GpuPolicyModel(BaseModel):
    disabled: bool = False
    min_priority: int = 0


class SetGpuPolicyRequest(BaseModel):
    disabled: Optional[bool] = None
    min_priority: Optional[int] = None


def local_reserved_gpu_indices() -> set[int]:
    """GPU indices reserved by this peer's running JobRecords.

    Mirrors ``scheduler._reserved_gpu_set`` but lives here so the routes
    layer doesn't have to import the scheduler module (which pulls in
    dataset_server / trainer_control / etc.). Both paths read the same
    job_records file, so the result is identical.
    """
    reserved: set[int] = set()
    for r in job_records.list_records():
        if r.status in RUNNING_STATUSES:
            reserved.update(r.gpu_indices)
    return reserved


def _to_model(
    g: gpu_monitor.GpuInfo,
    reserved_indices: Optional[set[int]] = None,
) -> GpuInfoModel:
    return GpuInfoModel(
        index=g.index,
        name=g.name,
        total_mem_bytes=g.total_mem_bytes,
        used_mem_bytes=g.used_mem_bytes,
        util_pct=g.util_pct,
        mem_util_pct=g.mem_util_pct,
        power_w=g.power_w,
        temp_c=g.temp_c,
        fan_pct=g.fan_pct,
        processes=[
            GpuProcessModel(
                pid=p.pid,
                used_mem_bytes=p.used_mem_bytes,
                name=p.name,
                kind=p.kind,
            )
            for p in g.processes
        ],
        source=g.source,
        node=g.node,
        excluded=g.excluded,
        disabled=g.disabled,
        min_priority=g.min_priority,
        reserved=(reserved_indices is not None and g.index in reserved_indices),
    )


@router.get("/gpus", response_model=List[GpuInfoModel])
def list_gpus():
    reserved = local_reserved_gpu_indices()
    return [_to_model(g, reserved) for g in gpu_monitor.snapshot()]


@router.websocket("/gpus/stream")
async def stream_gpus(ws: WebSocket):
    """Push GPU snapshots on a ~2 s cadence until the client disconnects.

    ``gpu_monitor.snapshot()`` is a synchronous NVML call. Running it
    directly in the asyncio event loop blocks every other request for
    the duration of the call; almost always microseconds, but pynvml
    can occasionally hang briefly under driver contention. Hop to a
    thread so that a slow snapshot never stalls the rest of the
    server.

    Any exception other than the normal ``WebSocketDisconnect`` is
    logged with a traceback before the connection drops — without
    this, prior intermittent disconnects left no forensic trail. The
    client reconnects with exponential backoff, so a one-shot
    failure heals on its own.
    """
    await ws.accept()
    try:
        while True:
            snap = await asyncio.to_thread(gpu_monitor.snapshot)
            reserved = await asyncio.to_thread(local_reserved_gpu_indices)
            payload = [_to_model(g, reserved).model_dump() for g in snap]
            await ws.send_json(payload)
            await asyncio.sleep(STREAM_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("gpus/stream loop crashed; client will reconnect")


class GpuKillRequest(BaseModel):
    # Required acknowledgement, mirrors the /api/fs/delete-dir guard. Belt
    # and suspenders for an irreversibly-destructive action.
    confirmed: bool = False


class GpuKillResponse(BaseModel):
    gpu_index: int
    pids: List[int]
    killed: List[int]
    failed: List[int]


def _self_ancestors() -> set[int]:
    """PIDs we must never kill: this server process and its ancestors.

    We exclude only ancestors (self → parents → init), not descendants — a
    scheduler-launched GPU worker can be a descendant of this process and is a
    legitimate kill target. Walking ``/proc/<pid>/stat`` for the ppid keeps the
    set to the process tree above us.
    """
    keep = {1}
    cur = os.getpid()
    while cur and cur not in keep:
        keep.add(cur)
        try:
            with open(f"/proc/{cur}/stat") as f:
                # "pid (comm) state ppid ..." — comm may contain spaces/parens,
                # so split after the final ") ".
                ppid = int(f.read().rsplit(") ", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
        if ppid <= 0:
            break
        cur = ppid
    return keep


def _in_container_pids_for_gpu(gpu_index: int) -> List[int]:
    """Container-local PIDs attached to ``gpu_index`` via CUDA_VISIBLE_DEVICES.

    The scheduler launches every GPU job with ``CUDA_VISIBLE_DEVICES`` set to
    the assigned physical indices (launcher.py). That env survives in
    ``/proc/<pid>/environ`` and is the one reliable, container-local signal for
    "which process is attached to which GPU": NVML reports *host*-namespace PIDs
    (unkillable, and untranslatable from inside a child PID namespace), and a
    CUDA process opens *every* ``/dev/nvidiaN`` node regardless of its assigned
    device, so an fd scan can't attribute per-GPU. Matching the env instead
    reaps wedged workers and their children (inductor compile workers inherit
    the same env) by their real container PIDs.

    Only catches processes *inside this container* — by design (an out-of-
    container process holding the card can't be signalled from here anyway).
    """
    keep = _self_ancestors()
    matches: List[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return matches
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in keep:
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                raw = f.read()
        except OSError:
            # Process gone, or environ unreadable (different uid) — can't kill
            # it either way, so skip.
            continue
        cvd = None
        for kv in raw.split(b"\0"):
            if kv.startswith(b"CUDA_VISIBLE_DEVICES="):
                cvd = kv[len(b"CUDA_VISIBLE_DEVICES=") :].decode("utf-8", "replace")
                break
        if not cvd:
            continue
        indices = {int(t) for t in (s.strip() for s in cvd.split(",")) if t.isdigit()}
        if gpu_index in indices:
            matches.append(pid)
    return matches


@router.post("/gpus/{gpu_index}/kill", response_model=GpuKillResponse)
def kill_gpu_processes(gpu_index: int, req: GpuKillRequest):
    """SIGKILL every in-container process attached to ``gpu_index``.

    Last-resort cleanup for wedged GPU jobs — torchrun ranks stuck after a
    botched ^C, an orphan whose JobRecord already went terminal, a worker
    deadlocked in a CUDA driver call. They hold CUDA contexts (and GPU memory)
    until killed with SIGKILL; we don't try SIGTERM first because if they were
    responsive they'd already be gone.

    Primary reaping is by ``CUDA_VISIBLE_DEVICES`` over ``/proc`` (see
    ``_in_container_pids_for_gpu``) so it works inside a container, where NVML
    reports host-namespace PIDs that ``os.kill`` can't touch. A direct
    ``os.kill`` over the NVML-reported PIDs runs as a backstop — it reaps the
    card on a ``--pid=host`` / bare-metal deployment and, inside a container,
    surfaces any out-of-container holder in ``failed`` (which is fine: those
    live outside this container and aren't ours to kill).

    The endpoint kills *every* matching process, including ones the server
    didn't directly launch, so ``confirmed`` is required to make accidental
    hits hard. The UI also surfaces a confirm() dialog on top.
    """
    if not req.confirmed:
        raise HTTPException(status_code=400, detail="kill requires confirmed=true")

    snapshot = gpu_monitor.snapshot()
    target = next((g for g in snapshot if g.index == gpu_index), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"no GPU at index {gpu_index}")

    # NVML's view of compute processes on the card. Inside a container these are
    # host-namespace PIDs; we report them for visibility and use them only for
    # the os.kill backstop below. Graphics processes (X server, compositor) are
    # excluded so we never log the operator's desktop out.
    pids = [p.pid for p in target.processes if p.kind == "compute"]

    killed: List[int] = []
    failed: List[int] = []

    # Primary: reap in-container processes assigned to this GPU by their real
    # container PIDs (CUDA_VISIBLE_DEVICES scan). This is the path that actually
    # frees the card inside the container.
    for pid in _in_container_pids_for_gpu(gpu_index):
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            log.warning("SIGKILL sent to in-container pid %d on GPU %d", pid, gpu_index)
        except ProcessLookupError:
            killed.append(pid)  # gone between scan and now
        except Exception as e:
            log.warning("kill in-container pid %d failed: %s", pid, e)
            failed.append(pid)

    # Backstop: directly SIGKILL the NVML-reported PIDs. Reaps the card on a
    # --pid=host / bare-metal deployment (there the NVML PID == container PID,
    # and the scan above already got it -> ProcessLookupError here). Inside a
    # container these are host-namespace PIDs os.kill() can't reach; they land
    # in `failed`, flagging an out-of-container holder.
    for pid in pids:
        if pid in killed or pid in failed:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            log.warning("SIGKILL sent to pid %d on GPU %d", pid, gpu_index)
        except ProcessLookupError:
            killed.append(pid)
        except PermissionError as e:
            log.warning("cannot kill pid %d on GPU %d: %s", pid, gpu_index, e)
            failed.append(pid)
        except Exception as e:
            log.warning("kill pid %d failed: %s", pid, e)
            failed.append(pid)

    return GpuKillResponse(
        gpu_index=gpu_index,
        pids=pids,
        killed=killed,
        failed=failed,
    )


@router.get("/gpus/policy", response_model=Dict[str, GpuPolicyModel])
def get_all_gpu_policies():
    """Return the scheduling policy for every GPU that has a stored entry."""
    return {
        str(idx): GpuPolicyModel(
            disabled=pol.disabled,
            min_priority=pol.min_priority,
        )
        for idx, pol in gpu_policy.all_policies().items()
    }


@router.post("/gpus/{gpu_index}/policy", response_model=GpuPolicyModel)
def set_gpu_policy(gpu_index: int, req: SetGpuPolicyRequest):
    """Set the scheduling policy for a GPU.

    Only the fields included in the request body are updated; omitted fields
    retain their current values.  Setting policy for an index that has no
    physical GPU is allowed (e.g. for a card that will be plugged in later).
    """
    result = gpu_policy.set_policy(
        gpu_index,
        disabled=req.disabled,
        min_priority=req.min_priority,
    )
    return GpuPolicyModel(disabled=result.disabled, min_priority=result.min_priority)
