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

import asyncio
import hashlib
import logging
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from forgather.tls import httpx_peer_kwargs, httpx_verify

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

    ``verify_tls`` is False when the operator registered this URL
    with chain validation off (typical for SSH-tunneled remotes).
    Default True so JobRecord-spawned and standard user-registry
    entries keep the secure-by-default posture.
    """

    server_id: str
    base_url: str
    auth_token: str
    label: str
    source: str  # "local" (JobRecord) or "user" (registry)
    peer_node_id: Optional[str]
    verify_tls: bool = True
    # Source-side identifier on the *owning peer* (the JobRecord
    # queue_id for "local"; the registry entry id for "user").
    # Propagated so the webui can target a DELETE at the right
    # peer's local endpoint without round-tripping through the
    # master. ``None`` when not applicable.
    source_id: Optional[str] = None
    # True when the URL's host is loopback. Cluster routing skips
    # these (no other peer can reach them) but the UI still shows
    # them so operators can register node-local datasets without
    # losing visibility.
    loopback: bool = False

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


def _is_loopback_url(url: str) -> bool:
    """True iff the URL's host is loopback (127.0.0.1 / localhost / ::1).

    Used by the inventory to mark entries that other cluster peers
    can't route to. They still appear in the per-node Servers view —
    operators can register node-local datasets for non-cluster use —
    but ``resolve()`` excludes them from cluster auto-routing.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host in _LOOPBACK_HOSTS


def _jobrecord_base_url(
    host: str, port: int, *, tls: bool, routable_host: Optional[str] = None
) -> Optional[str]:
    """Build a base URL for a JobRecord-spawned server.

    Priority order for the host portion:
      1. ``routable_host`` from job_params if provided (the scheduler
         records the auto-detected LAN address there).
      2. Cluster identity's hostname (when ``host == "0.0.0.0"`` and
         we're in cluster mode).
      3. ``host`` as-is (operator picked an explicit address; this
         includes loopback binds, which the inventory now keeps
         instead of dropping).

    Returns ``None`` only when ``host`` is ``0.0.0.0`` and no
    routable address can be inferred — at that point the URL truly
    isn't usable by anyone, including the local node, so dropping
    is correct.
    """
    if routable_host:
        scheme = "https" if tls else "http"
        return _normalize(
            f"{scheme}://{_bracket_ipv6(routable_host.lower())}:{int(port)}"
        )
    h = (host or "").lower()
    if h == "0.0.0.0":
        ident = cluster.self_identity()
        if ident is not None and ident.hostname:
            h = ident.hostname
        else:
            return None
    scheme = "https" if tls else "http"
    return _normalize(f"{scheme}://{_bracket_ipv6(h)}:{int(port)}")


def _bracket_ipv6(host: str) -> str:
    """Wrap IPv6 literals in ``[…]`` so they parse correctly as URL
    netlocs. Pass-through for hostnames and IPv4 literals."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


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
        # Whether the spawned server is serving HTTPS. Newer spawns
        # store ``scheme`` (preferred) or ``tls``; fall back to the
        # forgather_server-wide setting otherwise so the scheme
        # matches what dataset_server auto-discovery would produce.
        if "scheme" in params:
            tls = str(params.get("scheme") or "http").lower() == "https"
        else:
            tls = bool(params.get("tls"))
            if "tls" not in params:
                try:
                    from forgather.tls import client_scheme

                    tls = client_scheme("0.0.0.0") == "https"
                except Exception:
                    tls = False
        routable = params.get("routable_host")
        base_url = _jobrecord_base_url(
            host,
            port,
            tls=tls,
            routable_host=str(routable) if routable else None,
        )
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
                source_id=r.queue_id,
                loopback=_is_loopback_url(base_url),
            )
        )
    return out


def _user_registry_servers(peer_node_id: Optional[str]) -> List[LocalServer]:
    """User-registered dataset_server entries.

    Loopback entries are *included* (so the Servers panel shows
    everything the operator has registered, even node-local datasets
    that aren't cluster-routable). The ``loopback`` flag marks them
    so ``MasterInventory.resolve()`` can exclude them from cluster
    auto-routing without hiding them from the UI.
    """
    out: List[LocalServer] = []
    for e in dataset_server_registry.list_entries():
        base_url = _normalize(e.base_url)
        out.append(
            LocalServer(
                server_id=server_id_for(base_url),
                base_url=base_url,
                auth_token=e.auth_token or "",
                label=e.label or e.base_url,
                source="user",
                peer_node_id=peer_node_id,
                verify_tls=e.verify_tls,
                source_id=e.id,
                loopback=_is_loopback_url(base_url),
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


# ===========================================================================
# Master-side aggregation (Phase 3)
# ===========================================================================
#
# The master node aggregates the cluster-wide view by:
#   1. ``master_collect_servers_loop`` — every ``COLLECT_INTERVAL_SECONDS``,
#      GET ``/api/cluster/dataset_servers_local`` from every reachable peer
#      and merge the results into the master inventory's server set.
#   2. ``master_health_loop`` — every ``HEALTH_INTERVAL_SECONDS``, GET
#      ``/v1/health`` on every known server and flip its ``healthy`` flag.
#   3. ``master_dataset_refresh_loop`` — every ``REFRESH_INTERVAL_SECONDS``,
#      GET ``/v1/datasets`` + ``/v1/local`` on every healthy server and
#      refresh the dataset listing + the local-name routing index.
#
# All three loops run on every node but self-gate on
# ``cluster.is_self_master()``. On a master transition the new master
# clears its inventory and starts fresh — the 503 cold-start window
# in ``/api/cluster/dataset_router/resolve`` covers the time before
# the first full pass completes.

COLLECT_INTERVAL_SECONDS = 10.0
HEALTH_INTERVAL_SECONDS = 10.0
REFRESH_INTERVAL_SECONDS = 60.0

# Per-call HTTP timeouts.
PEER_TIMEOUT_SECONDS = 5.0
HEALTH_TIMEOUT_SECONDS = 5.0
DATASETS_TIMEOUT_SECONDS = 10.0


@dataclass
class MasterServerEntry:
    """One dataset_server known to the master, with health + content.

    The ``available_keys`` list is the routing-side index for
    ``local/<name>`` lookups. ``handles`` is a snapshot of
    ``/v1/datasets`` (already-loaded handles + their resolved
    ``load_args``) for the Cluster-tab UI. ``locals_info`` snapshots
    ``/v1/local`` for the same UI surface.
    """

    server_id: str
    base_url: str
    auth_token: str
    label: str
    source: str  # "local" or "user"
    peer_node_id: Optional[str]
    # Per-entry TLS verification policy. False = chain + hostname
    # validation off (SSH-tunneled / out-of-band-secured upstreams).
    # Propagates from the user-registry entry through to every
    # outbound call the master makes against this server.
    verify_tls: bool = True
    # See ``LocalServer.source_id`` / ``loopback``. Carried through
    # so the webui can target DELETE at the owning peer and so the
    # cluster router can exclude loopback URLs from auto-routing
    # while keeping them visible in the Servers panel.
    source_id: Optional[str] = None
    loopback: bool = False
    healthy: bool = False
    last_health_check: float = 0.0
    last_health_error: str = ""
    # Cumulative + streak counters. ``consecutive_*_failures`` resets
    # to 0 on every successful poll, so a non-zero value is a clear
    # "this server is currently in trouble" signal independent of how
    # many ticks ago the first failure occurred.
    total_health_polls: int = 0
    health_failures: int = 0
    consecutive_health_failures: int = 0
    total_dataset_polls: int = 0
    dataset_failures: int = 0
    consecutive_dataset_failures: int = 0
    available_keys: List[str] = field(default_factory=list)
    handles: List[Dict[str, Any]] = field(default_factory=list)
    locals_info: List[Dict[str, Any]] = field(default_factory=list)
    # HF cache snapshot from /v1/cache/hf. Lets the cluster inventory
    # surface "this HF repo is *available* to load on this server"
    # without requiring a client to have already triggered /v1/load.
    # Same shape as ``HFCacheResponse``: a list of repo dicts with
    # ``repo``, ``configs``, ``size_bytes``.
    hf_cache: List[Dict[str, Any]] = field(default_factory=list)
    last_dataset_refresh: float = 0.0
    last_dataset_error: str = ""


class MasterInventory:
    """Master-side aggregation of cluster-wide dataset_server state.

    Thread-safe — the loops mutate from asyncio tasks, the route
    handlers read synchronously from the FastAPI worker threads.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._servers: Dict[str, MasterServerEntry] = {}
        self._master_become_ts: Optional[float] = None
        self._last_servers_collect_ts: Optional[float] = None
        self._last_health_pass_ts: Optional[float] = None
        self._last_dataset_pass_ts: Optional[float] = None
        # Set once the master has observed at least one healthy server
        # via a successful dataset-refresh pass. Resets to False on
        # role transition. ``is_warmed_up()`` gates on this so a
        # transient zero-server window doesn't flip warm-up True with
        # an empty inventory — which would convert the router from
        # ``503 try-again`` to ``410 give-up``.
        self._ever_observed_healthy: bool = False
        # ``local/<name>`` collisions already warned about this
        # tenure. Stops the WARNING log from spamming on every
        # 60s dataset-refresh tick.
        self._warned_local_collisions: set = set()

    # ----- master role transitions -----

    def set_master_state(self, is_master: bool) -> bool:
        """Update internal "am I master" state. Returns True if a
        transition occurred (caller can use this to wake loops).

        Becoming master clears the inventory so a stale snapshot from
        a prior tenure can't be served. Stopping clears state too —
        ``/resolve`` returns 503 (not stale data) until a new master
        rebuilds it.
        """
        with self._lock:
            currently_master = self._master_become_ts is not None
            if is_master and not currently_master:
                self._servers = {}
                self._master_become_ts = time.time()
                self._last_servers_collect_ts = None
                self._last_health_pass_ts = None
                self._last_dataset_pass_ts = None
                self._ever_observed_healthy = False
                self._warned_local_collisions = set()
                log.info(
                    "became master; cleared dataset inventory and starting fresh"
                )
                return True
            if not is_master and currently_master:
                self._servers = {}
                self._master_become_ts = None
                self._last_servers_collect_ts = None
                self._last_health_pass_ts = None
                self._last_dataset_pass_ts = None
                self._ever_observed_healthy = False
                self._warned_local_collisions = set()
                log.info("no longer master; dataset inventory cleared")
                return True
            return False

    def is_master(self) -> bool:
        with self._lock:
            return self._master_become_ts is not None

    # ----- server set merge -----

    def merge_servers(self, fresh: Dict[str, MasterServerEntry]) -> None:
        """Replace the server set, preserving health + content for
        servers that survived the round."""
        with self._lock:
            merged: Dict[str, MasterServerEntry] = {}
            for sid, new_entry in fresh.items():
                old = self._servers.get(sid)
                if old is not None:
                    # Carry forward health + dataset data; identity
                    # fields are overwritten (peer_node_id may have
                    # changed if a different peer started reporting it).
                    new_entry.healthy = old.healthy
                    new_entry.last_health_check = old.last_health_check
                    new_entry.last_health_error = old.last_health_error
                    new_entry.available_keys = list(old.available_keys)
                    new_entry.handles = list(old.handles)
                    new_entry.locals_info = list(old.locals_info)
                    new_entry.hf_cache = list(old.hf_cache)
                    new_entry.last_dataset_refresh = old.last_dataset_refresh
                    new_entry.last_dataset_error = old.last_dataset_error
                    # verify_tls is set on the fresh entry by
                    # _to_master_entry → LocalServer.verify_tls
                    # already; nothing to carry over (the new value
                    # is the authoritative one from the peer's
                    # registry).
                    # Polling counters too — these accumulate across
                    # the master's whole tenure and would falsely
                    # reset every 10s collect tick otherwise.
                    new_entry.total_health_polls = old.total_health_polls
                    new_entry.health_failures = old.health_failures
                    new_entry.consecutive_health_failures = (
                        old.consecutive_health_failures
                    )
                    new_entry.total_dataset_polls = old.total_dataset_polls
                    new_entry.dataset_failures = old.dataset_failures
                    new_entry.consecutive_dataset_failures = (
                        old.consecutive_dataset_failures
                    )
                merged[sid] = new_entry
            self._servers = merged
            self._last_servers_collect_ts = time.time()

    def update_health(
        self,
        server_id: str,
        *,
        healthy: bool,
        error: str = "",
    ) -> None:
        with self._lock:
            s = self._servers.get(server_id)
            if s is None:
                return
            s.healthy = healthy
            s.last_health_check = time.time()
            s.last_health_error = error
            s.total_health_polls += 1
            if healthy:
                s.consecutive_health_failures = 0
            else:
                s.health_failures += 1
                s.consecutive_health_failures += 1

    def mark_health_pass_complete(self) -> None:
        with self._lock:
            self._last_health_pass_ts = time.time()

    def update_datasets(
        self,
        server_id: str,
        *,
        handles: List[Dict[str, Any]],
        locals_info: List[Dict[str, Any]],
        hf_cache: Optional[List[Dict[str, Any]]] = None,
        error: str = "",
    ) -> None:
        with self._lock:
            s = self._servers.get(server_id)
            if s is None:
                return
            s.handles = list(handles)
            s.locals_info = list(locals_info)
            if hf_cache is not None:
                s.hf_cache = list(hf_cache)
            # The routing-side key set: "local/<name>" for each entry
            # the server advertises in /v1/local. HF / path requests
            # don't need an index entry — any healthy server is a
            # candidate (server loads on demand).
            keys: List[str] = []
            for item in locals_info:
                name = item.get("name") if isinstance(item, dict) else None
                if isinstance(name, str) and name:
                    keys.append(f"local/{name}")
            s.available_keys = keys
            s.last_dataset_refresh = time.time()
            s.last_dataset_error = error
            s.total_dataset_polls += 1
            if error:
                s.dataset_failures += 1
                s.consecutive_dataset_failures += 1
            else:
                s.consecutive_dataset_failures = 0

    def mark_dataset_pass_complete(self) -> None:
        with self._lock:
            self._last_dataset_pass_ts = time.time()

    def mark_observed_healthy(self) -> None:
        """Sticky flag: ≥1 healthy server has been observed during
        this master tenure. Resets only on master role change."""
        with self._lock:
            self._ever_observed_healthy = True

    def local_collisions(self) -> Dict[str, List[str]]:
        """Detect ``local/<name>`` entries reported with different
        ``meta_hash`` values across the cluster.

        Returns a ``{name: [hash, ...]}`` mapping containing only
        names with more than one distinct hash observed. Empty when
        every advertised ``local/<name>`` is content-equivalent
        across the servers that expose it — the common case for
        intentional redundancy.

        A non-empty result indicates the operator has named two
        genuinely distinct datasets the same thing on different
        nodes; the router will silently load-balance between them,
        and training jobs will see arbitrary content depending on
        which replica won the random pick. Caller logs a one-shot
        WARNING for each new collision (see
        ``_dataset_refresh_tick``).
        """
        from collections import defaultdict

        with self._lock:
            by_name: Dict[str, set] = defaultdict(set)
            for s in self._servers.values():
                for entry in s.locals_info:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name")
                    meta = entry.get("meta_hash")
                    if isinstance(name, str) and isinstance(meta, str):
                        by_name[name].add(meta)
            return {
                name: sorted(hashes)
                for name, hashes in by_name.items()
                if len(hashes) > 1
            }

    def warn_new_collisions(self) -> None:
        """Log a WARNING for each ``local/<name>`` collision not yet
        reported this tenure. Idempotent — already-warned collisions
        are silently skipped, so the loop can call this on every
        refresh tick without spamming the log."""
        new_collisions = self.local_collisions()
        with self._lock:
            for name, hashes in new_collisions.items():
                if name in self._warned_local_collisions:
                    continue
                log.warning(
                    "local/%s collision: %d distinct meta_hashes across "
                    "the cluster (%s). Servers advertising this name "
                    "are NOT content-equivalent; training jobs auto-"
                    "routed here will see arbitrary content. Either "
                    "rename one of the conflicting datasets or "
                    "re-register them with matching content.",
                    name,
                    len(hashes),
                    ", ".join(hashes),
                )
                self._warned_local_collisions.add(name)

    # ----- reads -----

    def servers_snapshot(self) -> List[MasterServerEntry]:
        with self._lock:
            return [
                MasterServerEntry(**asdict(s)) for s in self._servers.values()
            ]

    def get_server(self, server_id: str) -> Optional[MasterServerEntry]:
        with self._lock:
            s = self._servers.get(server_id)
            return None if s is None else MasterServerEntry(**asdict(s))

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_master": self._master_become_ts is not None,
                "master_become_ts": self._master_become_ts,
                "last_servers_collect_ts": self._last_servers_collect_ts,
                "last_health_pass_ts": self._last_health_pass_ts,
                "last_dataset_pass_ts": self._last_dataset_pass_ts,
                "server_count": len(self._servers),
            }

    def resolve(self, dataset_id: str) -> Optional[MasterServerEntry]:
        """Pick a healthy server for the given ``dataset_id``.

        ``dataset_id`` is the logical name of the dataset:

        * ``local/<name>``: filter by servers that advertise that name
          in their ``/v1/local`` snapshot. Two servers advertising
          ``local/stories`` are treated as interchangeable replicas
          (the operator's intent — global naming).
        * Any other ``dataset_id`` (HF id, filesystem path): every
          healthy server is a candidate. The server loads on demand;
          the resilient client retries elsewhere on failure.

        Returns a deep-copy ``MasterServerEntry`` of the chosen
        server (or ``None`` if no candidate is available). Returning
        the entry rather than just ``(base_url, token)`` lets the
        route handler emit ``server_id`` without re-walking the
        snapshot — the resolve path is on the hot client retry loop.

        Selection is uniform random across the candidate set — crude
        load balancing across replicas.
        """
        with self._lock:
            # Loopback entries are visible in the Servers panel
            # (node-local datasets are a legit thing) but never
            # cluster-routable — other peers can't reach them. Filter
            # them out at the routing decision rather than at the
            # inventory level.
            if dataset_id.startswith("local/"):
                candidates = [
                    s
                    for s in self._servers.values()
                    if s.healthy
                    and not s.loopback
                    and dataset_id in s.available_keys
                ]
            else:
                candidates = [
                    s
                    for s in self._servers.values()
                    if s.healthy and not s.loopback
                ]
            if not candidates:
                return None
            chosen = random.choice(candidates)
            # Defensive copy so the caller can read the entry outside
            # the lock without racing the next merge.
            return MasterServerEntry(**asdict(chosen))

    def is_warmed_up(self) -> bool:
        """True once we have completed a dataset-refresh pass AND
        have observed at least one healthy server in the cluster.

        Both conditions matter:

        - Without ``_last_dataset_pass_ts``, ``available_keys`` is
          unpopulated and ``local/*`` lookups would falsely report
          "no server has this."
        - Without ``_ever_observed_healthy``, a transiently empty
          cluster (no servers in the inventory yet, or every server
          DOWN simultaneously) would mark warm-up complete after
          the first vacuous pass — and the router would then return
          ``410`` ("no candidate") to clients, which the resilient
          client treats as fatal. With the gate, the router keeps
          returning ``503 Retry-After`` until at least one healthy
          server has ever been seen.
        """
        with self._lock:
            return (
                self._last_dataset_pass_ts is not None
                and self._ever_observed_healthy
            )


# Module-level singleton — the loops mutate it, the routes read it.
master_inventory = MasterInventory()


# ----- conversion helpers -----


def _to_master_entry(local: LocalServer) -> MasterServerEntry:
    return MasterServerEntry(
        server_id=local.server_id,
        base_url=local.base_url,
        auth_token=local.auth_token,
        label=local.label,
        source=local.source,
        peer_node_id=local.peer_node_id,
        verify_tls=local.verify_tls,
        source_id=local.source_id,
        loopback=local.loopback,
    )


def _local_server_from_dict(raw: Dict[str, Any]) -> Optional[LocalServer]:
    try:
        sid = str(raw["server_id"])
        base_url = str(raw["base_url"])
        return LocalServer(
            server_id=sid,
            base_url=base_url,
            auth_token=str(raw.get("auth_token") or ""),
            label=str(raw.get("label") or base_url),
            source=str(raw.get("source") or "user"),
            peer_node_id=raw.get("peer_node_id"),
            # Default True so older peers that don't ship the field
            # keep their secure-by-default posture.
            verify_tls=bool(raw.get("verify_tls", True)),
            source_id=(
                str(raw["source_id"]) if raw.get("source_id") else None
            ),
            loopback=bool(raw.get("loopback", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


# ----- peer-side fetcher -----


async def _fetch_peer_servers(
    client: httpx.AsyncClient, member: cluster.MemberInfo
) -> List[LocalServer]:
    """Pull ``/api/cluster/dataset_servers_local`` from one peer.

    Returns an empty list on any error; the master tolerates partial
    failures and re-tries on the next collect tick.
    """
    scheme = "https" if getattr(member, "tls", False) else "http"
    url = (
        f"{scheme}://{member.address}:{member.port}"
        "/api/cluster/dataset_servers_local"
    )
    try:
        r = await client.get(url, timeout=PEER_TIMEOUT_SECONDS)
    except (httpx.HTTPError, OSError) as e:
        log.debug("dataset_servers_local fetch failed: %s -> %s", member.hostname, e)
        return []
    if r.status_code != 200:
        log.debug(
            "dataset_servers_local non-200: %s status=%d",
            member.hostname,
            r.status_code,
        )
        return []
    try:
        body = r.json()
    except ValueError:
        log.debug("dataset_servers_local non-JSON from %s", member.hostname)
        return []
    raw_servers = body.get("servers") if isinstance(body, dict) else None
    if not isinstance(raw_servers, list):
        return []
    out: List[LocalServer] = []
    for raw in raw_servers:
        if not isinstance(raw, dict):
            continue
        parsed = _local_server_from_dict(raw)
        if parsed is None:
            continue
        out.append(parsed)
    return out


# ----- collect / health / refresh implementations -----


async def _collect_servers_tick(client: httpx.AsyncClient) -> None:
    """One round of "ask every peer for their local server list."""
    self_id = cluster.self_identity()
    if self_id is None:
        return
    # Local node's own servers — direct call, no HTTP needed.
    fresh: Dict[str, MasterServerEntry] = {}
    for s in local_servers():
        fresh.setdefault(s.server_id, _to_master_entry(s))
    # Fan out to peers (excluding self).
    peers = [
        m
        for m in cluster.members()
        if m.node_id != self_id.node_id and m.reachable and m.address
    ]
    if peers:
        results = await asyncio.gather(
            *[_fetch_peer_servers(client, m) for m in peers],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, BaseException):
                continue
            for s in r:
                fresh.setdefault(s.server_id, _to_master_entry(s))
    master_inventory.merge_servers(fresh)


def _verify_for_entry(entry: MasterServerEntry) -> object:
    """Pick the ``verify=`` argument for an outbound call to ``entry``.

    ``False`` when the operator registered the URL with chain
    validation off (SSH-tunneled / out-of-band-secured); otherwise
    the standard forgather.tls verify policy (CA bundle, optional
    hostname check) applies.
    """
    if not entry.verify_tls:
        return False
    return httpx_verify()


async def _check_one_health(
    client: httpx.AsyncClient, entry: MasterServerEntry
) -> None:
    """GET ``/v1/health`` on one server; update inventory.

    Default-verify entries reuse the loop's shared ``client``;
    ``verify_tls=False`` entries get a one-shot client so chain
    validation can be skipped without breaking pooling for the
    secure-by-default majority.
    """
    if not entry.verify_tls:
        async with httpx.AsyncClient(verify=False) as c:
            return await _check_one_health_inner(c, entry)
    return await _check_one_health_inner(client, entry)


async def _check_one_health_inner(
    client: httpx.AsyncClient, entry: MasterServerEntry
) -> None:
    url = entry.base_url.rstrip("/") + "/v1/health"
    try:
        r = await client.get(url, timeout=HEALTH_TIMEOUT_SECONDS)
    except (httpx.HTTPError, OSError) as e:
        master_inventory.update_health(
            entry.server_id, healthy=False, error=f"{type(e).__name__}: {e}"
        )
        return
    if r.status_code != 200:
        master_inventory.update_health(
            entry.server_id, healthy=False, error=f"HTTP {r.status_code}"
        )
        return
    master_inventory.update_health(entry.server_id, healthy=True, error="")


async def _health_tick(client: httpx.AsyncClient) -> None:
    servers = master_inventory.servers_snapshot()
    if not servers:
        master_inventory.mark_health_pass_complete()
        return
    await asyncio.gather(
        *[_check_one_health(client, s) for s in servers],
        return_exceptions=True,
    )
    master_inventory.mark_health_pass_complete()


def _auth_headers(token: str) -> Dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def _refresh_one_dataset_listing(
    client: httpx.AsyncClient, entry: MasterServerEntry
) -> None:
    """GET ``/v1/datasets``, ``/v1/local``, and ``/v1/cache/hf`` on
    one server, then update the inventory's dataset listing + routing
    index.

    The HF cache snapshot is the answer to "what HF datasets are
    *available* on this server" — the inventory surfaces it so the
    Cluster tab can list cached repos even before any client has
    issued a /v1/load. Without the cache poll, the unified cluster
    view would only show currently-loaded handles + ``local/<name>``
    entries, which understates what the cluster can actually serve.

    ``verify_tls=False`` entries get a one-shot client so chain
    validation can be skipped without affecting the secure-by-
    default pool.
    """
    if not entry.verify_tls:
        async with httpx.AsyncClient(verify=False) as c:
            return await _refresh_one_dataset_listing_inner(c, entry)
    return await _refresh_one_dataset_listing_inner(client, entry)


async def _refresh_one_dataset_listing_inner(
    client: httpx.AsyncClient, entry: MasterServerEntry
) -> None:
    headers = _auth_headers(entry.auth_token)
    base = entry.base_url.rstrip("/")
    handles: List[Dict[str, Any]] = []
    locals_info: List[Dict[str, Any]] = []
    hf_cache_repos: List[Dict[str, Any]] = []
    error_parts: List[str] = []

    try:
        r = await client.get(
            base + "/v1/datasets",
            headers=headers or None,
            timeout=DATASETS_TIMEOUT_SECONDS,
        )
        if r.status_code == 200:
            body = r.json()
            if isinstance(body, list):
                handles = body
            elif isinstance(body, dict):
                # The server returns ``{"handles": [...]}`` today; the
                # older ``{"datasets": [...]}`` shape is tolerated for
                # forward compatibility with mixed-version clusters.
                for key in ("handles", "datasets"):
                    items = body.get(key)
                    if isinstance(items, list):
                        handles = items
                        break
        else:
            error_parts.append(f"datasets HTTP {r.status_code}")
    except (httpx.HTTPError, OSError, ValueError) as e:
        error_parts.append(f"datasets {type(e).__name__}: {e}")

    try:
        r = await client.get(
            base + "/v1/local",
            headers=headers or None,
            timeout=DATASETS_TIMEOUT_SECONDS,
        )
        if r.status_code == 200:
            body = r.json()
            if isinstance(body, list):
                locals_info = body
            elif isinstance(body, dict):
                # Current server returns ``{"local": [...]}`` —
                # tolerate ``entries`` and ``{"local": {<name>: {...}}}``
                # for forward + backward compatibility.
                for key in ("local", "entries"):
                    items = body.get(key)
                    if isinstance(items, list):
                        locals_info = items
                        break
                    if isinstance(items, dict):
                        locals_info = [
                            {"name": k, **(v if isinstance(v, dict) else {})}
                            for k, v in items.items()
                        ]
                        break
        else:
            error_parts.append(f"local HTTP {r.status_code}")
    except (httpx.HTTPError, OSError, ValueError) as e:
        error_parts.append(f"local {type(e).__name__}: {e}")

    try:
        r = await client.get(
            base + "/v1/cache/hf",
            headers=headers or None,
            timeout=DATASETS_TIMEOUT_SECONDS,
        )
        if r.status_code == 200:
            body = r.json()
            if isinstance(body, dict) and isinstance(body.get("datasets"), list):
                hf_cache_repos = body["datasets"]
            elif isinstance(body, list):
                hf_cache_repos = body
        else:
            error_parts.append(f"cache HTTP {r.status_code}")
    except (httpx.HTTPError, OSError, ValueError) as e:
        error_parts.append(f"cache {type(e).__name__}: {e}")

    master_inventory.update_datasets(
        entry.server_id,
        handles=handles,
        locals_info=locals_info,
        hf_cache=hf_cache_repos,
        error="; ".join(error_parts),
    )


async def _dataset_refresh_tick(client: httpx.AsyncClient) -> None:
    healthy = [s for s in master_inventory.servers_snapshot() if s.healthy]
    if not healthy:
        # No healthy servers — mark the pass complete so ``is_warmed_up``
        # can advance once we've also observed a healthy server, but
        # do NOT flip ``_ever_observed_healthy``. A transient empty
        # window keeps the router returning 503 (retry) instead of 410
        # (give up).
        master_inventory.mark_dataset_pass_complete()
        return
    await asyncio.gather(
        *[_refresh_one_dataset_listing(client, s) for s in healthy],
        return_exceptions=True,
    )
    # ≥1 healthy server seen on this pass — the warm-up gate can flip
    # True now. Once set, it stays True for the rest of this master
    # tenure, so a later all-DOWN window still serves 503 (transient)
    # rather than 410 (fatal) to in-flight training jobs.
    master_inventory.mark_observed_healthy()
    master_inventory.mark_dataset_pass_complete()
    # Surface ``local/<name>`` collisions once per tenure — operator
    # config bug otherwise hidden by the random load-balance pick.
    master_inventory.warn_new_collisions()


# ----- public loop entry points -----

# Per-loop wake events. Each loop owns one event so a single
# ``wake_loops()`` call from the membership listener fans out to all
# three; using a single shared event was a bug — whichever loop woke
# first cleared it, and the other two slept through the transition.
#
# Each event is created lazily on first ``wake_loops()`` so this
# module is importable before any asyncio loop is running (the unit
# tests rely on it).
_wake_events: List[asyncio.Event] = []


def _register_wake_event() -> asyncio.Event:
    """Each loop calls this at startup to claim its wake event."""
    ev = asyncio.Event()
    _wake_events.append(ev)
    return ev


def wake_loops() -> None:
    """Signal every registered master loop to run one immediate tick.

    Called from the membership loop on a master-role transition so a
    newly-elected master populates its inventory in seconds instead of
    waiting up to ``REFRESH_INTERVAL_SECONDS``. Each loop owns its own
    asyncio.Event; setting them all means *every* loop wakes, not just
    whichever clears the shared event first.
    """
    for ev in _wake_events:
        try:
            ev.set()
        except RuntimeError:
            # The event's loop isn't running yet (very-early init /
            # certain test fixtures). Wake-up is a latency hint, not
            # correctness-critical — skip silently.
            pass


async def _await_or_wake(event: asyncio.Event, seconds: float) -> None:
    """Sleep up to ``seconds`` or until ``event`` is set.

    The event is cleared after wake so subsequent ticks resume on
    their normal cadence. Per-loop event means a wake on this loop
    doesn't accidentally skip another loop's sleep.
    """
    try:
        await asyncio.wait_for(event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return
    finally:
        event.clear()


def _sync_master_state() -> bool:
    """Read the authoritative ``cluster.is_self_master()`` and reflect
    it into the inventory's cached state. Returns the live value.

    Every master loop calls this at the top of every tick — that way
    if one loop dies the others keep the inventory's cached
    ``is_master()`` consistent with reality. ``set_master_state`` is
    idempotent on no-change.
    """
    if not cluster.is_active():
        if master_inventory.is_master():
            master_inventory.set_master_state(False)
        return False
    is_now = cluster.is_self_master()
    master_inventory.set_master_state(is_now)
    return is_now


async def master_collect_servers_loop(
    *, interval_seconds: Optional[float] = None
) -> None:
    """Run the peer-fanout collect loop until cancelled."""
    interval = (
        interval_seconds if interval_seconds is not None else COLLECT_INTERVAL_SECONDS
    )
    log.info("master collect-servers loop starting (interval=%.1fs)", interval)
    wake = _register_wake_event()
    async with httpx.AsyncClient(**httpx_peer_kwargs()) as client:
        try:
            while True:
                try:
                    if _sync_master_state():
                        await _collect_servers_tick(client)
                except Exception:
                    log.exception("collect-servers tick failed")
                await _await_or_wake(wake, interval)
        except asyncio.CancelledError:
            log.info("master collect-servers loop cancelled")
            raise


async def master_health_loop(
    *, interval_seconds: Optional[float] = None
) -> None:
    """Run the per-server /v1/health probe loop until cancelled."""
    interval = (
        interval_seconds if interval_seconds is not None else HEALTH_INTERVAL_SECONDS
    )
    log.info("master health loop starting (interval=%.1fs)", interval)
    wake = _register_wake_event()
    async with httpx.AsyncClient(**httpx_peer_kwargs()) as client:
        try:
            while True:
                try:
                    if _sync_master_state():
                        await _health_tick(client)
                except Exception:
                    log.exception("health tick failed")
                await _await_or_wake(wake, interval)
        except asyncio.CancelledError:
            log.info("master health loop cancelled")
            raise


async def master_dataset_refresh_loop(
    *, interval_seconds: Optional[float] = None
) -> None:
    """Run the per-server /v1/datasets + /v1/local refresh loop until cancelled.

    During the cold-start window (no completed pass yet), runs at
    ``HEALTH_INTERVAL_SECONDS`` cadence so the router warms up faster
    than the steady-state ``REFRESH_INTERVAL_SECONDS``.
    """
    fast = (
        interval_seconds
        if interval_seconds is not None
        else HEALTH_INTERVAL_SECONDS
    )
    steady = (
        interval_seconds
        if interval_seconds is not None
        else REFRESH_INTERVAL_SECONDS
    )
    log.info(
        "master dataset-refresh loop starting (fast=%.1fs steady=%.1fs)",
        fast,
        steady,
    )
    wake = _register_wake_event()
    async with httpx.AsyncClient(**httpx_peer_kwargs()) as client:
        try:
            while True:
                interval = steady if master_inventory.is_warmed_up() else fast
                try:
                    if _sync_master_state():
                        await _dataset_refresh_tick(client)
                except Exception:
                    log.exception("dataset-refresh tick failed")
                await _await_or_wake(wake, interval)
        except asyncio.CancelledError:
            log.info("master dataset-refresh loop cancelled")
            raise


def _reset_master_state_for_tests() -> None:
    """Wipe the module-level singleton + wake events between tests.

    Clears in place rather than reassigning so test code that
    imported ``master_inventory`` at module load (a common idiom)
    keeps pointing at the live singleton across resets.
    """
    with master_inventory._lock:
        master_inventory._servers = {}
        master_inventory._master_become_ts = None
        master_inventory._last_servers_collect_ts = None
        master_inventory._last_health_pass_ts = None
        master_inventory._last_dataset_pass_ts = None
        master_inventory._ever_observed_healthy = False
    # Loops register wake events on startup. Between tests the loops
    # are not running, so accumulated stale events have nothing to
    # tell — clear them so the next test's loops start with a clean
    # list.
    _wake_events.clear()
