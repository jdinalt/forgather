"""Auto-started services persisted in the server config.

Services are long-running spawned processes the operator wants the
forgather server to bring up automatically on start: dataset_server,
inference, tensorboard, mkdocs. Each service has a stable identifier
(``<type>:<name>``) and a bag of args matching the job_params shape its
modal would have submitted.

At boot, ``autostart`` walks every enabled service and enqueues a
QueueItem unless an equivalent one is already running. Equivalence is
defined by an args signature — sha256 over a canonical JSON encoding of
``(job_type, normalized_args)`` — so manually-launched jobs with the
same parameters are detected as the running instance, and a restart
that happens to land the same service twice doesn't double-spawn.

The persistent representation lives under ``services:`` in
``server_config.yaml``:

    services:
      dataset:
        my_dataset:
          enabled: true
          port: 8766
      inference:
        llama:
          enabled: false
          model_path: /models/llama
          port: 8137

Writes regenerate the file body and lose user-added inline comments —
the file carries a fixed documentation preamble that survives.

Service args are *almost* the job_params shape the corresponding modal
submits, with three operator-level meta keys lifted out:

  enabled (bool)        — auto-start at boot.
  priority (int)        — queue priority. Default 0.
  requested_gpus (int)  — gpu count. Default 1 for inference; 0 for
                          dataset / tensorboard / mkdocs.

Everything else is forwarded to ``job_params`` verbatim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from . import job_records, queue_store, server_config
from ._atomic import atomic_write_text
from .job_records import RUNNING_STATUSES, TERMINAL_STATUSES

log = logging.getLogger("forgather_server.services")


# Allowed service types and the queue ``job_type`` each maps to.
SERVICE_TYPES: Dict[str, str] = {
    "dataset": "dataset_server",
    "inference": "inference",
    "tensorboard": "tensorboard",
    "mkdocs": "mkdocs",
    "diloco": "diloco_server",
}

# Reverse map for signature computation when we observe a queue item /
# job record (which carries the queue-side job_type).
_JOB_TYPE_TO_SERVICE: Dict[str, str] = {v: k for k, v in SERVICE_TYPES.items()}

# Keys that are operator-level metadata, not part of the spawned
# process's args. Stripped before building the QueueItem's job_params.
_META_KEYS = ("enabled", "priority", "requested_gpus")

# Keys the scheduler injects into ``job_params`` when it dispatches a
# QueueItem into a JobRecord — derived from the runtime environment,
# not the operator's intent. Stripped before signature computation so
# pre- and post-dispatch signatures for the same logical service
# match. Without this the configured-service status flips from green
# back to red the moment the queue item becomes a JobRecord.
_DISPATCH_INJECTED_KEYS = ("scheme", "routable_host")

# Everything excluded from the signature hash.
_SIG_EXCLUDED_KEYS = _META_KEYS + _DISPATCH_INJECTED_KEYS


_write_lock = threading.Lock()


@dataclass
class Service:
    type: str
    name: str
    enabled: bool = False
    args: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.type}:{self.name}"

    def signature(self) -> str:
        return compute_signature(self.type, self.args)


def compute_signature(svc_type: str, args: Dict[str, Any]) -> str:
    """Stable signature for dedupe.

    Strips operator-meta keys and dispatch-injected keys so the same
    logical service produces the same signature whether we're looking
    at the YAML entry, the QueueItem before dispatch, or the JobRecord
    after.
    """
    params = _signature_params(args)
    canonical = json.dumps(
        {"type": svc_type, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _job_params_from_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Drop meta fields; everything else is the spawned process's args."""
    return {k: v for k, v in args.items() if k not in _META_KEYS}


def _signature_params(args: Dict[str, Any]) -> Dict[str, Any]:
    """Drop both meta and dispatch-injected keys for signature use."""
    return {k: v for k, v in args.items() if k not in _SIG_EXCLUDED_KEYS}


def list_services() -> List[Service]:
    """Read the services section of the server config.

    Order is preserved as it appears in the YAML — within a type,
    services come out in the order the operator wrote them.
    """
    _, data = server_config.load(None)
    return _parse(data)


def _parse(data: Dict[str, Any]) -> List[Service]:
    raw = data.get("services") or {}
    if not isinstance(raw, dict):
        log.warning("services: in config is not a mapping; ignoring")
        return []
    out: List[Service] = []
    for svc_type, instances in raw.items():
        if svc_type not in SERVICE_TYPES:
            log.warning("unknown service type %r in config; skipping", svc_type)
            continue
        if not isinstance(instances, dict):
            log.warning(
                "services.%s must be a mapping of name -> args; skipping",
                svc_type,
            )
            continue
        for name, body in instances.items():
            if not isinstance(body, dict):
                log.warning(
                    "services.%s.%s must be a mapping; skipping",
                    svc_type,
                    name,
                )
                continue
            enabled = bool(body.get("enabled", False))
            args = {k: v for k, v in body.items() if k != "enabled"}
            out.append(
                Service(
                    type=svc_type,
                    name=str(name),
                    enabled=enabled,
                    args=args,
                )
            )
    return out


def get_service(svc_type: str, name: str) -> Optional[Service]:
    for s in list_services():
        if s.type == svc_type and s.name == name:
            return s
    return None


def upsert_service(
    svc_type: str,
    name: str,
    enabled: bool,
    args: Dict[str, Any],
) -> Service:
    """Write or replace a service entry in the server config file."""
    if svc_type not in SERVICE_TYPES:
        raise ValueError(f"unknown service type: {svc_type!r}")
    if not _is_valid_name(name):
        raise ValueError(
            f"service name must be non-empty and may only contain "
            f"letters, digits, dash, underscore: {name!r}"
        )
    body = dict(args)
    body.pop("enabled", None)
    body = {"enabled": bool(enabled), **body}
    with _write_lock:
        path, data = server_config.load(None)
        services = _ensure_services_dict(data)
        per_type = services.setdefault(svc_type, {})
        if not isinstance(per_type, dict):
            per_type = {}
            services[svc_type] = per_type
        per_type[name] = body
        _write_back(path, data)
    return Service(
        type=svc_type,
        name=name,
        enabled=bool(enabled),
        args={k: v for k, v in body.items() if k != "enabled"},
    )


def delete_service(svc_type: str, name: str) -> bool:
    with _write_lock:
        path, data = server_config.load(None)
        services = data.get("services")
        if not isinstance(services, dict):
            return False
        per_type = services.get(svc_type)
        if not isinstance(per_type, dict) or name not in per_type:
            return False
        per_type.pop(name)
        if not per_type:
            # Drop empty type buckets so the file stays tidy.
            services.pop(svc_type, None)
        _write_back(path, data)
    return True


def set_enabled(svc_type: str, name: str, enabled: bool) -> Optional[Service]:
    with _write_lock:
        path, data = server_config.load(None)
        services = data.get("services")
        if not isinstance(services, dict):
            return None
        per_type = services.get(svc_type)
        if not isinstance(per_type, dict) or name not in per_type:
            return None
        body = per_type[name]
        if not isinstance(body, dict):
            body = {}
        body["enabled"] = bool(enabled)
        per_type[name] = body
        _write_back(path, data)
    args = {k: v for k, v in body.items() if k != "enabled"}
    return Service(type=svc_type, name=name, enabled=bool(enabled), args=args)


def _ensure_services_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    cur = data.get("services")
    if not isinstance(cur, dict):
        cur = {}
        data["services"] = cur
    return cur


def _is_valid_name(name: str) -> bool:
    if not name:
        return False
    return all(c.isalnum() or c in "-_" for c in name)


def _write_back(path: Path, data: Dict[str, Any]) -> None:
    """Re-emit the entire config file with the documentation preamble.

    PyYAML doesn't round-trip inline comments, so any user-added
    comments inside the YAML body are lost on programmatic write. The
    fixed preamble at the top of the file is preserved unconditionally
    so the file always opens with a brief description of what each
    section is for.

    Atomic: writes to a sibling .tmp, fsyncs, then os.replaces into
    place. A crash mid-write can't leave a truncated / empty
    ``server_config.yaml`` — either the previous contents or the new
    contents are visible, never something in between. ``mode=0o600``
    sets the tmp file's mode at ``os.open`` time and re-asserts it
    via fchmod, so the file is never readable at the umask default,
    even momentarily.
    """
    # Normalize: empty top-level sections render as ``key: {}`` rather
    # than ``key: null`` — easier to read and to extend by hand later.
    normalized = dict(data)
    for k in ("args", "services"):
        if normalized.get(k) is None:
            normalized[k] = {}
    body = yaml.safe_dump(normalized, sort_keys=False, default_flow_style=False)
    text = _PROGRAMMATIC_HEADER + body
    atomic_write_text(path, text, mode=0o600)


_PROGRAMMATIC_HEADER = """\
# Forgather server config (auto-managed; comments below are stripped on
# programmatic writes by the webui — keep durable notes elsewhere).
#
# args:     defaults for the ``forgather server`` CLI; any value passed
#           on the command line still wins. See ``forgather server
#           --help`` for the full set of supported keys.
# services: long-running spawned processes auto-started at boot. Each
#           entry under <type>.<name> is enabled=true|false plus the
#           same args its modal would have submitted as ``job_params``.
#           Supported types: dataset, inference, tensorboard, mkdocs,
#           diloco.

"""


# ---------------------------------------------------------------------------
# Status / dedupe helpers
# ---------------------------------------------------------------------------


@dataclass
class ServiceStatus:
    service: Service
    running: bool
    queue_id: Optional[str] = None
    status: Optional[str] = None  # 'queued' | 'starting' | 'running' | None


def active_signatures() -> Dict[str, Tuple[str, Optional[str]]]:
    """Map of signature -> (queue_id, status) for active service-shaped jobs.

    "Active" means in the queue (not yet dispatched) or running as a
    JobRecord with a non-terminal status. Used both at boot to skip
    auto-starting a service whose instance is already up, and at status
    query time to colour the indicator green.
    """
    out: Dict[str, Tuple[str, Optional[str]]] = {}
    for item in queue_store.list_items():
        svc_type = _JOB_TYPE_TO_SERVICE.get(item.job_type)
        if svc_type is None:
            continue
        sig = compute_signature(svc_type, item.job_params or {})
        out.setdefault(sig, (item.queue_id, "queued"))
    for rec in job_records.list_records():
        if rec.status in TERMINAL_STATUSES:
            continue
        svc_type = _JOB_TYPE_TO_SERVICE.get(rec.job_type)
        if svc_type is None:
            continue
        sig = compute_signature(svc_type, rec.job_params or {})
        # Job-record entries are more authoritative than queue entries
        # (a dispatched item left the queue), so let them overwrite.
        out[sig] = (rec.queue_id, rec.status)
    return out


def status_for_each(services: Iterable[Service]) -> List[ServiceStatus]:
    """Status snapshot, polled by the webui for the red/green dot.

    ``running`` is true only when an actual JobRecord with status
    ``"running"`` matches the service signature. "queued" and
    "starting" both stay red — the dot is meant to reflect the
    spawned process being live, not "the operator intended to start
    this", so it doesn't flip until the process is really up and
    doesn't flip back until the process has finished exiting.
    """
    sigs = active_signatures()
    out: List[ServiceStatus] = []
    for svc in services:
        sig = svc.signature()
        active = sigs.get(sig)
        if active:
            qid, st = active
            running = st == "running"
            out.append(
                ServiceStatus(service=svc, running=running, queue_id=qid, status=st)
            )
        else:
            out.append(ServiceStatus(service=svc, running=False))
    return out


# ---------------------------------------------------------------------------
# Build / submit
# ---------------------------------------------------------------------------


def build_queue_item(svc: Service) -> queue_store.QueueItem:
    """Translate a service entry into a QueueItem the scheduler can dispatch."""
    raw = dict(svc.args)
    priority = int(raw.pop("priority", 0) or 0)
    # GPU defaults match the modal defaults: inference needs >=1, the
    # rest are CPU-only.
    default_gpus = 1 if svc.type == "inference" else 0
    requested_gpus = int(raw.pop("requested_gpus", default_gpus) or default_gpus)
    job_type = SERVICE_TYPES[svc.type]
    job_params = _job_params_from_args(raw)
    # Type-specific project_dir/config defaults — these are the same
    # conventions the existing modals use so jobs look identical.
    if svc.type == "dataset":
        port = job_params.get("port", 8766)
        project_dir = "/"
        config = f"dataset:{port}"
    elif svc.type == "inference":
        port = job_params.get("port", 8137)
        project_dir = str(job_params.get("model_path") or "/")
        config = f"inference:{port}"
    elif svc.type == "tensorboard":
        port = job_params.get("port", 6006)
        project_dir = str(job_params.get("logdir") or "/")
        config = f"tensorboard:{port}"
    elif svc.type == "mkdocs":
        port = job_params.get("port", 9999)
        project_dir = str(job_params.get("config_file") or "/")
        config = f"mkdocs:{port}"
    elif svc.type == "diloco":
        port = job_params.get("port", 8512)
        # ``output_dir`` is the model checkpoint dir the server holds
        # global params for — same shape as inference's ``model_path``.
        # Falls back to "/" so the JobRecord shows *something* even when
        # the operator didn't fill it in (the spawn will then fail loud
        # with a clear error from the diloco CLI).
        project_dir = str(job_params.get("output_dir") or "/")
        config = f"diloco:{port}"
    else:  # unreachable — validated upstream
        raise ValueError(svc.type)
    return queue_store.QueueItem.new(
        project_dir=project_dir,
        config=config,
        dynamic_args={},
        requested_gpus=requested_gpus,
        priority=priority,
        job_type=job_type,
        job_params=job_params,
    )


def autostart() -> List[Service]:
    """Enqueue every enabled service whose signature isn't already active.

    Returns the list of services that were actually enqueued (mostly
    useful for logging). Safe to call multiple times — idempotent by
    signature.
    """
    services = list_services()
    if not services:
        return []
    sigs = active_signatures()
    started: List[Service] = []
    for svc in services:
        if not svc.enabled:
            continue
        sig = svc.signature()
        if sig in sigs:
            log.info(
                "service %s already running (signature %s); skipping autostart",
                svc.id,
                sig,
            )
            continue
        try:
            item = build_queue_item(svc)
            queue_store.add_item(item)
            started.append(svc)
            log.info(
                "autostarted service %s as queue %s (sig %s)",
                svc.id,
                item.queue_id,
                sig,
            )
        except Exception:
            log.exception("failed to autostart service %s", svc.id)
    return started
