"""GPU snapshot + WebSocket stream endpoints."""

import asyncio
import logging
import os
import signal
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .. import gpu_monitor, gpu_policy

log = logging.getLogger("forgather_server.gpus")

router = APIRouter(tags=["gpus"])

# Push interval for the live stream. 2 s is frequent enough to feel live and
# slow enough that NVML polling stays a rounding-error of CPU cost.
STREAM_INTERVAL_SECONDS = 2.0


class GpuProcessModel(BaseModel):
    pid: int
    used_mem_bytes: int


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


class GpuPolicyModel(BaseModel):
    disabled: bool = False
    min_priority: int = 0


class SetGpuPolicyRequest(BaseModel):
    disabled: Optional[bool] = None
    min_priority: Optional[int] = None


def _to_model(g: gpu_monitor.GpuInfo) -> GpuInfoModel:
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
            GpuProcessModel(pid=p.pid, used_mem_bytes=p.used_mem_bytes)
            for p in g.processes
        ],
        source=g.source,
        node=g.node,
        excluded=g.excluded,
        disabled=g.disabled,
        min_priority=g.min_priority,
    )


@router.get("/gpus", response_model=List[GpuInfoModel])
def list_gpus():
    return [_to_model(g) for g in gpu_monitor.snapshot()]


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
            payload = [_to_model(g).model_dump() for g in snap]
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


@router.post("/gpus/{gpu_index}/kill", response_model=GpuKillResponse)
def kill_gpu_processes(gpu_index: int, req: GpuKillRequest):
    """SIGKILL every compute process currently attached to ``gpu_index``.

    Last-resort cleanup for the case where torchrun ranks are wedged on a
    GPU after a botched ^C — they hold CUDA contexts open until killed
    individually with SIGKILL. We don't try SIGTERM first because if these
    processes were responsive they'd have been gone already; sending a
    handleable signal just delays the inevitable.

    The endpoint kills *every* process on the GPU, including ones the
    server didn't launch, so the ``confirmed`` flag is required to make
    accidental hits hard. The UI also surfaces a confirm() dialog on top.
    """
    if not req.confirmed:
        raise HTTPException(status_code=400, detail="kill requires confirmed=true")

    snapshot = gpu_monitor.snapshot()
    target = next((g for g in snapshot if g.index == gpu_index), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"no GPU at index {gpu_index}")

    pids = [p.pid for p in target.processes]
    killed: List[int] = []
    failed: List[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            log.warning("SIGKILL sent to pid %d on GPU %d", pid, gpu_index)
        except ProcessLookupError:
            # Already gone between snapshot and now — treat as success.
            killed.append(pid)
        except PermissionError as e:
            log.warning("cannot kill pid %d on GPU %d: %s", pid, gpu_index, e)
            failed.append(pid)
        except Exception as e:
            log.warning("kill pid %d failed: %s", pid, e)
            failed.append(pid)

    return GpuKillResponse(gpu_index=gpu_index, pids=pids, killed=killed, failed=failed)


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
