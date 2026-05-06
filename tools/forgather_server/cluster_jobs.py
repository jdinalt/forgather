"""Multi-node training bundle records.

A "cluster job" is a single user-level training submission that spans
multiple nodes. The master holds the bundle record; each participating
peer has a regular per-rank queue/job item correlated to the bundle by
``cluster_job_id``. This module is the in-memory + journal-backed
store for the bundle records themselves; per-rank queue items live in
the existing ``queue_store`` / ``job_records`` modules on each peer.

Why a separate store: the bundle has cross-node state (rendezvous id,
endpoint, the per-peer assignments) that no single node owns. The
master is the source of truth; on master failover (Phase 5) the
backup will reconstruct from the journal.

State shape: in-memory dict keyed by ``cluster_job_id`` (UUID). Every
mutation is journaled via ``cluster_journal`` so Phase 4's replication
seam already covers cluster-job lifecycle without further changes here.

Status field: "submitted" → "running" → terminal ("done", "cancelled",
"failed"). For Phase 3 we infer the high-level status from per-peer
queue/record state when the master assembles a list response, rather
than tracking it explicitly here. The bundle record itself only stores
"submitted" and "cancelled".
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("forgather_server.cluster.jobs")


@dataclass
class MemberAssignment:
    node_id: str
    hostname: str
    address: str
    port: int
    queue_id: str
    nproc_per_node: int
    node_rank: int
    nccl_socket_ifname: Optional[str] = None


@dataclass
class ClusterJob:
    cluster_job_id: str
    project_dir: str
    config: str
    submitted_at: float
    rdzv_endpoint: str
    rdzv_id: str
    rdzv_node_id: str
    members: List[MemberAssignment] = field(default_factory=list)
    # Lifecycle marker independent of per-peer queue/record status.
    # "submitted" means the bundle has been fanned out; "cancelled"
    # means the master has issued cancel-fanout. Per-peer terminal
    # status is reported separately and aggregated for the UI.
    status: str = "submitted"
    cancelled_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["members"] = [asdict(m) for m in self.members]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterJob":
        members = [
            MemberAssignment(**m) if isinstance(m, dict) else m
            for m in data.get("members", [])
        ]
        return cls(
            cluster_job_id=data["cluster_job_id"],
            project_dir=data["project_dir"],
            config=data["config"],
            submitted_at=data["submitted_at"],
            rdzv_endpoint=data["rdzv_endpoint"],
            rdzv_id=data["rdzv_id"],
            rdzv_node_id=data["rdzv_node_id"],
            members=members,
            status=data.get("status", "submitted"),
            cancelled_at=data.get("cancelled_at"),
        )


_lock = threading.RLock()
_jobs: Dict[str, ClusterJob] = {}


def _journal_append(event_type: str, payload: Dict[str, Any]) -> None:
    """Best-effort journal write; never let a logging failure break the
    submit path. Phase 4 will turn this into the source of truth."""
    try:
        from . import cluster_journal

        cluster_journal.append(event_type, payload)
    except Exception:
        log.exception("journal append failed for %s", event_type)


def new_cluster_job_id() -> str:
    return f"cj_{uuid.uuid4().hex[:16]}"


def new_rdzv_id() -> str:
    # torchrun's c10d backend is happy with anything; keep it readable
    # in logs by using a hex prefix tied to the cluster job id.
    return uuid.uuid4().hex[:12]


def add_job(job: ClusterJob) -> ClusterJob:
    with _lock:
        if job.cluster_job_id in _jobs:
            raise ValueError(
                f"cluster_job_id collision: {job.cluster_job_id}"
            )
        _jobs[job.cluster_job_id] = job
    _journal_append("multinode_job_submitted", job.to_dict())
    log.info(
        "cluster job %s submitted: project=%s config=%s nnodes=%d rdzv=%s",
        job.cluster_job_id,
        job.project_dir,
        job.config,
        len(job.members),
        job.rdzv_endpoint,
    )
    return job


def get_job(cluster_job_id: str) -> Optional[ClusterJob]:
    with _lock:
        return _jobs.get(cluster_job_id)


def list_jobs() -> List[ClusterJob]:
    with _lock:
        # Deterministic order, newest-first — mirrors what the UI
        # wants for its top-down list. Sorting here rather than in the
        # route keeps the API consistent across callers.
        return sorted(
            _jobs.values(), key=lambda j: -j.submitted_at
        )


def set_terminal_status(
    cluster_job_id: str, status: str
) -> Optional[ClusterJob]:
    """Promote a bundle to a terminal status ("done" or "failed").

    Read-path optimisation: once every member's queue item is in a
    terminal local state, the master writes the rolled-up value back
    onto the bundle record so future status reads can short-circuit
    without fanning out to every peer. Idempotent — re-setting the
    same terminal status is a no-op.
    """
    if status not in ("done", "failed"):
        raise ValueError(
            f"set_terminal_status only accepts done/failed, got {status}"
        )
    with _lock:
        job = _jobs.get(cluster_job_id)
        if job is None:
            return None
        if job.status in ("done", "failed", "cancelled"):
            return job
        job.status = status
    _journal_append(
        f"multinode_job_{status}",
        {"cluster_job_id": cluster_job_id, "at": time.time()},
    )
    log.info("cluster job %s reached terminal status %s", cluster_job_id, status)
    return job


def mark_cancelled(cluster_job_id: str) -> Optional[ClusterJob]:
    with _lock:
        job = _jobs.get(cluster_job_id)
        if job is None:
            return None
        if job.status == "cancelled":
            return job
        job.status = "cancelled"
        job.cancelled_at = time.time()
    _journal_append(
        "multinode_job_cancelled",
        {"cluster_job_id": cluster_job_id, "cancelled_at": job.cancelled_at},
    )
    log.info("cluster job %s cancelled", cluster_job_id)
    return job


def remove_job(cluster_job_id: str) -> bool:
    """Drop a bundle from the in-memory store. Used for cleanup of
    long-since-terminal records; not currently exposed via the API.
    """
    with _lock:
        if cluster_job_id not in _jobs:
            return False
        del _jobs[cluster_job_id]
    return True


def _reset_for_tests() -> None:
    with _lock:
        _jobs.clear()
