"""GPU enumeration and live status for the dashboard.

Tries NVML (via ``pynvml``) first for the full info set — per-device util,
power draw, temperature, and compute-process list — and silently falls back
to ``torch.cuda`` if NVML is unavailable. The fallback path gives name and
memory only; util/power/temp come back as ``None``.

Both paths are safe to call from async handlers (they do light synchronous
work per call). For the WebSocket stream, poll this module once every ~2 s.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Set

from . import gpu_policy

log = logging.getLogger("forgather_server.gpu")

# Hostname this process sees itself as. Recorded on every GpuInfo so the UI
# and scheduler can tell cards apart when multi-node support lands — until
# then every snapshot is tagged with the local host.
LOCAL_NODE = platform.node()


def _parse_visible_devices() -> Optional[Set[int]]:
    """Parse CUDA_VISIBLE_DEVICES at module import.

    Returns ``None`` when the env var is unset (= no filter, all GPUs allowed)
    and a possibly-empty set otherwise. UUID syntax is not supported — those
    tokens are ignored with a warning.

    The point: respecting this lets the operator hide a GPU that's
    misbehaving (e.g. a card with thermal/PCIe issues) by simply launching
    the server with ``CUDA_VISIBLE_DEVICES=0,1,3,4,5`` — the scheduler will
    never pick the excluded indices but the UI still shows them so they can
    be monitored.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        log.warning("CUDA_VISIBLE_DEVICES is empty — all GPUs excluded")
        return set()
    out: Set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            log.warning(
                "CUDA_VISIBLE_DEVICES contains non-integer token %r "
                "(UUID syntax not supported); ignoring",
                tok,
            )
    if not out:
        log.warning("CUDA_VISIBLE_DEVICES yielded no usable indices")
    return out


# Computed once at import. The scheduler reads this via gpu_monitor.snapshot()
# rather than checking the env var directly, so the policy stays in one place.
_ALLOWED_INDICES: Optional[Set[int]] = _parse_visible_devices()


def allowed_indices() -> Optional[Set[int]]:
    """Public read of the parsed allow-list (None = no filter)."""
    return _ALLOWED_INDICES


@dataclass
class GpuProcess:
    pid: int
    used_mem_bytes: int
    # Best-effort process name. ``None`` when NVML / /proc lookup failed.
    name: Optional[str] = None
    # "compute" for processes from nvmlDeviceGetComputeRunningProcesses,
    # "graphics" for processes from nvmlDeviceGetGraphicsRunningProcesses.
    # Surfaced only for UI display and to gate the "kill GPU process" endpoint
    # (which restricts itself to compute processes so it can't terminate the
    # user's desktop session). The scheduler does NOT use this — see
    # scheduler._idle_gpu_indices for the dispatch rule.
    kind: str = "compute"


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mem_bytes: int
    used_mem_bytes: int
    util_pct: Optional[int] = None
    mem_util_pct: Optional[int] = None
    power_w: Optional[float] = None
    temp_c: Optional[int] = None
    fan_pct: Optional[int] = None
    processes: List[GpuProcess] = field(default_factory=list)
    source: str = "nvml"  # "nvml" or "torch" — lets the UI hint on missing fields
    # Hostname of the node these stats came from. Defaults to the local
    # machine; once multi-node lands, remote NodeClients will fill this in
    # for their aggregated snapshots.
    node: str = LOCAL_NODE
    # True when the operator has filtered this GPU out via
    # CUDA_VISIBLE_DEVICES at server start. Excluded GPUs still appear in
    # the snapshot (so their telemetry remains visible) but the scheduler
    # refuses to assign them.
    excluded: bool = False
    # Runtime user toggle — distinct from excluded. Set via the web UI or
    # API; persists across server restarts via gpu_policy.json.
    disabled: bool = False
    # Minimum queue priority required to schedule on this GPU (inclusive).
    # 0 means no restriction. Persists alongside disabled in gpu_policy.json.
    min_priority: int = 0
    # True when total_mem_bytes is the host's system RAM rather than a
    # discrete VRAM pool. Set for GPUs like GB10 / Jetson where NVML returns
    # "Not Supported" for memory info and the device shares system memory.
    # The UI can use this to render a "shared with host" hint instead of a
    # discrete-VRAM bar.
    unified_memory: bool = False


_nvml_state: Optional[bool] = None  # True = ready, False = unavailable, None = untried
_nvml_lock = Lock()


def _lookup_process_name(pid: int) -> Optional[str]:
    """Best-effort process name resolution.

    Tries ``nvmlSystemGetProcessName`` first (cheap, works regardless of
    /proc visibility), falls back to ``/proc/<pid>/comm``, returns ``None``
    on failure. Used to recognise desktop graphics processes so they don't
    block scheduler dispatch.
    """
    try:
        import pynvml  # type: ignore

        raw = pynvml.nvmlSystemGetProcessName(pid)
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
        # NVML returns the absolute path of the executable; strip dirname.
        if "/" in name:
            name = name.rsplit("/", 1)[-1]
        if name:
            return name
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/comm", "r") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _ensure_nvml() -> bool:
    """Initialize NVML lazily. Idempotent and thread-safe."""
    global _nvml_state
    with _nvml_lock:
        if _nvml_state is not None:
            return _nvml_state
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            atexit.register(_shutdown_nvml)
            _nvml_state = True
        except Exception as e:
            log.info("NVML unavailable, falling back to torch.cuda: %s", e)
            _nvml_state = False
    return _nvml_state


def _shutdown_nvml() -> None:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlShutdown()
    except Exception:
        pass


def _snapshot_nvml() -> Optional[List[GpuInfo]]:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None

    result: List[GpuInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception as e:
        log.warning("nvmlDeviceGetCount failed: %s", e)
        return None

    # NVML always sees physical indices regardless of CUDA_VISIBLE_DEVICES,
    # which is exactly what we want — the scheduler picks physical indices
    # and the child's CUDA_VISIBLE_DEVICES gets set to the chosen set.
    for i in range(count):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            raw_name = pynvml.nvmlDeviceGetName(h)
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            # nvmlDeviceGetMemoryInfo is best-effort: DGX Spark / GB10 has
            # unified system memory and NVML returns "Not Supported" here.
            # We still want to surface the GPU with name + policy + util,
            # so a missing memory reading must not drop the whole device.
            # When NVML can't tell us, fall back to host RAM: on unified-
            # memory platforms (GB10, Jetson) that's literally the GPU's
            # memory pool. Mark the result with ``unified_memory=True`` so
            # the UI can render it as shared rather than dedicated VRAM.
            total_mem_bytes = 0
            used_mem_bytes = 0
            unified_memory = False
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                total_mem_bytes = int(mem.total)
                used_mem_bytes = int(mem.used)
            except Exception:
                try:
                    import psutil  # type: ignore

                    vm = psutil.virtual_memory()
                    total_mem_bytes = int(vm.total)
                    used_mem_bytes = int(vm.total - vm.available)
                    unified_memory = True
                except Exception:
                    pass
            excluded = _ALLOWED_INDICES is not None and i not in _ALLOWED_INDICES
            policy = gpu_policy.get_policy(i)
            info = GpuInfo(
                index=i,
                name=name,
                total_mem_bytes=total_mem_bytes,
                used_mem_bytes=used_mem_bytes,
                source="nvml",
                excluded=excluded,
                disabled=policy.disabled,
                min_priority=policy.min_priority,
                unified_memory=unified_memory,
            )
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                info.util_pct = int(util.gpu)
                info.mem_util_pct = int(util.memory)
            except Exception:
                pass
            try:
                info.power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
            try:
                info.temp_c = int(
                    pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass
            try:
                # Cards without a fan (datacenter passive coolers, mobile parts
                # on shared chassis fans) raise NotSupported here — keep None
                # in that case so the UI can hide the field.
                info.fan_pct = int(pynvml.nvmlDeviceGetFanSpeed(h))
            except Exception:
                pass
            try:
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                    # p.usedGpuMemory is None for some drivers/older NVML
                    used = (
                        int(p.usedGpuMemory) if getattr(p, "usedGpuMemory", None) else 0
                    )
                    info.processes.append(
                        GpuProcess(
                            pid=int(p.pid),
                            used_mem_bytes=used,
                            name=_lookup_process_name(int(p.pid)),
                            kind="compute",
                        )
                    )
            except Exception:
                pass
            # Also surface graphics-only processes (X server, compositor, …)
            # so the UI can show them. The scheduler doesn't gate on these
            # at all — it only refuses to dispatch when the GPU is excluded
            # via CUDA_VISIBLE_DEVICES, runtime-disabled in the UI, or
            # already reserved by another Forgather job.
            try:
                seen_pids = {p.pid for p in info.processes}
                for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(h):
                    pid = int(p.pid)
                    if pid in seen_pids:
                        continue
                    used = (
                        int(p.usedGpuMemory) if getattr(p, "usedGpuMemory", None) else 0
                    )
                    info.processes.append(
                        GpuProcess(
                            pid=pid,
                            used_mem_bytes=used,
                            name=_lookup_process_name(pid),
                            kind="graphics",
                        )
                    )
            except Exception:
                pass
            result.append(info)
        except Exception as e:
            log.warning("NVML failure on GPU %d: %s", i, e)
            continue

    return result


def _snapshot_torch() -> List[GpuInfo]:
    """Minimal fallback. Gives name + memory only; util/power/temp stay None.

    Caveat: when ``CUDA_VISIBLE_DEVICES`` is set, ``torch.cuda`` only sees
    the visible subset and renumbers them 0..N-1. We can't safely map the
    torch indices back to physical indices from here. The fallback is rare
    in practice (pynvml ships with most CUDA installs); the NVML path uses
    physical indices throughout. If you actually rely on the fallback and
    have devices excluded, install pynvml.
    """
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    if _ALLOWED_INDICES is not None:
        log.warning(
            "torch.cuda fallback active while CUDA_VISIBLE_DEVICES is set: "
            "GPU indices in the UI may not match physical indices. "
            "Install pynvml for accurate enumeration."
        )

    result: List[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            policy = gpu_policy.get_policy(i)
            result.append(
                GpuInfo(
                    index=i,
                    name=props.name,
                    total_mem_bytes=int(total),
                    used_mem_bytes=int(total - free),
                    source="torch",
                    disabled=policy.disabled,
                    min_priority=policy.min_priority,
                )
            )
        except Exception as e:
            log.warning("torch.cuda failure on GPU %d: %s", i, e)
            continue
    return result


def snapshot() -> List[GpuInfo]:
    """Return a best-effort GPU snapshot; never raises."""
    if _ensure_nvml():
        data = _snapshot_nvml()
        if data is not None:
            return data
    return _snapshot_torch()
