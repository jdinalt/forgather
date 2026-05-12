"""
Cluster dataset-server inventory.

Phase 2 of the multi-node dataset-server work: per-peer enumeration
of the dataset_servers each forgather_server instance knows about,
exposed via ``GET /api/cluster/dataset_servers_local``.

A peer's "known" dataset_servers come from two sources:

  1. JobRecords with ``job_type == "dataset_server"`` (i.e., a server
     this forgather_server spawned via the webui's Tools menu).
  2. The user-added registry persisted at
     ``<config>/server/dataset_server_registry.json``.

Loopback-only entries are excluded — a 127.0.0.1 URL is not reachable
from other cluster members, so reporting it would just produce dead
inventory entries on the master. JobRecord servers bound to
``0.0.0.0`` are rewritten to use the node's cluster-visible hostname
so other peers can route to the right machine.

The returned :class:`LocalServer` records include the bearer token —
they are intended for the master's aggregator (Phase 3) and the
cluster carve-out auth gates the endpoint accordingly. Anything
exposed to a browser must strip ``auth_token`` first; see Phase 3 / 6.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

from . import cluster, dataset_server_registry, job_records

log = logging.getLogger("forgather_server.cluster_dataset_inventory")

# Hostnames that count as "this machine" — excluded from the cluster
# inventory because they're not reachable from other peers.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


@dataclass
class LocalServer:
    """A dataset_server this peer attests to.

    ``base_url`` is normalized (no trailing slash) and rewritten when
    needed (0.0.0.0 -> cluster hostname) so it is consumable by other
    peers' HTTP clients.

    ``auth_token`` may be empty for servers running ``--no-auth``.
    """

    server_id: str
    base_url: str
    auth_token: str
    label: str
    source: str  # "local" (JobRecord) or "user" (registry)
    peer_node_id: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def server_id_for(base_url: str) -> str:
    """Stable, ``base_url``-derived identifier.

    Used by the master aggregator to deduplicate the same URL when two
    peers happen to both know about it (e.g., both registered the same
    user entry). 12 hex chars is plenty — the input universe is small
    (one entry per URL the operator has typed).
    """
    return hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:12]


def _normalize(base_url: str) -> str:
    return base_url.rstrip("/")


def _routable_jobrecord_base_url(
    host: str, port: int, *, tls: bool
) -> Optional[str]:
    """Build a peer-visible base URL for a JobRecord-spawned server.

    - 127.0.0.1 / localhost: returns ``None`` (other peers can't reach
      it; reporting would just clutter the master's inventory with
      dead entries).
    - 0.0.0.0: rewritten to the cluster identity's hostname so other
      peers can route to this node. Falls back to ``None`` if cluster
      identity is unset (single-node mode).
    - Anything else (a routable hostname or IP the operator picked):
      used as-is.
    """
    h = (host or "").lower()
    if h in _LOOPBACK_HOSTS:
        return None
    if h == "0.0.0.0":
        ident = cluster.self_identity()
        if ident is None or not ident.hostname:
            return None
        h = ident.hostname
    scheme = "https" if tls else "http"
    return _normalize(f"{scheme}://{h}:{int(port)}")


def _local_jobrecord_servers(peer_node_id: Optional[str]) -> List[LocalServer]:
    """Dataset servers this forgather_server has spawned and which are
    currently in the ``starting``/``running`` state.

    Mirrors the JobRecord scan in
    :func:`tools.forgather_server.routes.dataset_server._local_servers`
    but additionally:

    - skips loopback binds (not useful cross-cluster),
    - rewrites 0.0.0.0 to the cluster hostname,
    - includes the auth_token (the local-routes scan deliberately
      strips it; the cluster carve-out gates this surface instead).
    """
    out: List[LocalServer] = []
    for r in job_records.list_records():
        if r.job_type != "dataset_server":
            continue
        if r.status not in {"starting", "running"}:
            continue
        params = r.job_params or {}
        try:
            port = int(params.get("port") or 0)
        except (TypeError, ValueError):
            port = 0
        if port <= 0:
            continue
        host = str(params.get("host") or "127.0.0.1")
        # Whether the spawned server is serving HTTPS: stored on the
        # JobRecord params when known (post-Phase-1 spawn path); fall
        # back to the forgather_server-wide TLS setting otherwise so
        # the scheme matches what the dataset_server's auto-discovery
        # would produce for a clean spawn.
        tls = bool(params.get("tls"))
        if "tls" not in params:
            try:
                from forgather.tls import client_scheme

                tls = client_scheme("0.0.0.0") == "https"
            except Exception:
                tls = False
        base_url = _routable_jobrecord_base_url(host, port, tls=tls)
        if base_url is None:
            continue
        out.append(
            LocalServer(
                server_id=server_id_for(base_url),
                base_url=base_url,
                auth_token=r.auth_token or "",
                label=f"{r.config or 'dataset_server'}:{port}",
                source="local",
                peer_node_id=peer_node_id,
            )
        )
    return out


def _user_registry_servers(peer_node_id: Optional[str]) -> List[LocalServer]:
    """User-registered dataset_server entries that point at a peer-
    reachable address (loopback entries are skipped — same rationale
    as the JobRecord side)."""
    out: List[LocalServer] = []
    for e in dataset_server_registry.list_entries():
        try:
            parsed = urlparse(e.base_url)
        except Exception:
            continue
        host = (parsed.hostname or "").lower()
        if host in _LOOPBACK_HOSTS:
            continue
        base_url = _normalize(e.base_url)
        out.append(
            LocalServer(
                server_id=server_id_for(base_url),
                base_url=base_url,
                auth_token=e.auth_token or "",
                label=e.label or e.base_url,
                source="user",
                peer_node_id=peer_node_id,
            )
        )
    return out


def local_servers() -> List[LocalServer]:
    """All dataset_servers this peer attests to.

    Sources (in order of priority on `server_id` collision):

    1. JobRecord-spawned servers that are currently
       ``starting``/``running``.
    2. User-registered entries from the persistent registry.

    A duplicate ``base_url`` from both sources is reported once; the
    JobRecord entry wins so the locally-spawned label / source is
    preserved.

    Records include the bearer token. Callers exposing this list to a
    browser must strip ``auth_token`` before serialization.
    """
    ident = cluster.self_identity()
    peer_node_id = ident.node_id if ident else None
    seen: Dict[str, LocalServer] = {}
    for entry in _local_jobrecord_servers(peer_node_id):
        seen.setdefault(entry.server_id, entry)
    for entry in _user_registry_servers(peer_node_id):
        seen.setdefault(entry.server_id, entry)
    return list(seen.values())
