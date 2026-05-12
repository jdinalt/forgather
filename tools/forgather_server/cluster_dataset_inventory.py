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

from forgather.tls import httpx_verify

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
    healthy: bool = False
    last_health_check: float = 0.0
    last_health_error: str = ""
    available_keys: List[str] = field(default_factory=list)
    handles: List[Dict[str, Any]] = field(default_factory=list)
    locals_info: List[Dict[str, Any]] = field(default_factory=list)
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
                    new_entry.last_dataset_refresh = old.last_dataset_refresh
                    new_entry.last_dataset_error = old.last_dataset_error
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

    def mark_health_pass_complete(self) -> None:
        with self._lock:
            self._last_health_pass_ts = time.time()

    def update_datasets(
        self,
        server_id: str,
        *,
        handles: List[Dict[str, Any]],
        locals_info: List[Dict[str, Any]],
        error: str = "",
    ) -> None:
        with self._lock:
            s = self._servers.get(server_id)
            if s is None:
                return
            s.handles = list(handles)
            s.locals_info = list(locals_info)
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

    def mark_dataset_pass_complete(self) -> None:
        with self._lock:
            self._last_dataset_pass_ts = time.time()

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

    def resolve(self, path: str) -> Optional[Tuple[str, str]]:
        """Pick a healthy server for the given dataset request.

        * ``local/<name>``: filter by servers that advertise that name
          in their ``/v1/local`` snapshot. Two servers advertising
          ``local/stories`` are treated as interchangeable replicas
          (the operator's intent — global naming).
        * Any other path (HF id, filesystem path): every healthy
          server is a candidate. The server loads on demand; the
          resilient client retries elsewhere on failure.

        Returns ``(base_url, auth_token)`` or ``None`` if no candidate
        is available. Auth-token may be empty for ``--no-auth``
        servers.

        Selection is uniform random across the candidate set — crude
        load balancing across replicas.
        """
        with self._lock:
            if path.startswith("local/"):
                candidates = [
                    s
                    for s in self._servers.values()
                    if s.healthy and path in s.available_keys
                ]
            else:
                candidates = [s for s in self._servers.values() if s.healthy]
            if not candidates:
                return None
            chosen = random.choice(candidates)
            return chosen.base_url, chosen.auth_token

    def is_warmed_up(self) -> bool:
        """True once we have at least one completed dataset-refresh
        pass. The router uses this to gate the 503 cold-start window:
        until a refresh has filled ``available_keys``, ``local/...``
        lookups would falsely report "no server has this"."""
        with self._lock:
            return self._last_dataset_pass_ts is not None


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


async def _check_one_health(
    client: httpx.AsyncClient, entry: MasterServerEntry
) -> None:
    """GET ``/v1/health`` on one server; update inventory."""
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
    """GET ``/v1/datasets`` and ``/v1/local`` on one server, then
    update the inventory's dataset listing + routing index."""
    headers = _auth_headers(entry.auth_token)
    base = entry.base_url.rstrip("/")
    handles: List[Dict[str, Any]] = []
    locals_info: List[Dict[str, Any]] = []
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
            elif isinstance(body, dict) and isinstance(body.get("datasets"), list):
                handles = body["datasets"]
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
            elif isinstance(body, dict) and isinstance(body.get("entries"), list):
                locals_info = body["entries"]
            elif isinstance(body, dict) and isinstance(body.get("local"), dict):
                # Tolerate ``{"local": {"<name>": {...}}}`` shapes from
                # older server builds — normalize to a list of dicts.
                locals_info = [
                    {"name": k, **(v if isinstance(v, dict) else {})}
                    for k, v in body["local"].items()
                ]
        else:
            error_parts.append(f"local HTTP {r.status_code}")
    except (httpx.HTTPError, OSError, ValueError) as e:
        error_parts.append(f"local {type(e).__name__}: {e}")

    master_inventory.update_datasets(
        entry.server_id,
        handles=handles,
        locals_info=locals_info,
        error="; ".join(error_parts),
    )


async def _dataset_refresh_tick(client: httpx.AsyncClient) -> None:
    healthy = [s for s in master_inventory.servers_snapshot() if s.healthy]
    if not healthy:
        master_inventory.mark_dataset_pass_complete()
        return
    await asyncio.gather(
        *[_refresh_one_dataset_listing(client, s) for s in healthy],
        return_exceptions=True,
    )
    master_inventory.mark_dataset_pass_complete()


# ----- public loop entry points -----

# A shared event the membership loop sets when this node's master
# status changes, so the loops can act on the transition without
# waiting out their normal sleep cadence. Set during a tick = "do
# the work now"; the loops `.wait()` on this with a timeout.
_wake_event = asyncio.Event()


def wake_loops() -> None:
    """Signal the master loops to run one immediate tick.

    Called from the membership loop on a master-role transition so a
    newly-elected master populates its inventory in seconds instead of
    waiting up to ``REFRESH_INTERVAL_SECONDS``.
    """
    try:
        _wake_event.set()
    except RuntimeError:
        # The default loop hasn't started yet (testing / very-early
        # init). Safe to ignore — wake-up is just a latency hint, not
        # correctness-critical.
        pass


async def _await_or_wake(seconds: float) -> None:
    """Sleep up to ``seconds`` or until ``wake_loops`` fires.

    Implemented with ``wait_for`` so an early wake cuts the sleep
    short. The event is cleared after wake so subsequent ticks resume
    on their normal cadence.
    """
    try:
        await asyncio.wait_for(_wake_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return
    finally:
        _wake_event.clear()


async def master_collect_servers_loop(
    *, interval_seconds: Optional[float] = None
) -> None:
    """Run the peer-fanout collect loop until cancelled."""
    interval = (
        interval_seconds if interval_seconds is not None else COLLECT_INTERVAL_SECONDS
    )
    log.info("master collect-servers loop starting (interval=%.1fs)", interval)
    async with httpx.AsyncClient(verify=httpx_verify()) as client:
        try:
            while True:
                try:
                    if cluster.is_active():
                        was = master_inventory.is_master()
                        is_now = cluster.is_self_master()
                        master_inventory.set_master_state(is_now)
                        if is_now:
                            await _collect_servers_tick(client)
                        elif was:
                            # Transition out of master role — log was
                            # already emitted in set_master_state.
                            pass
                except Exception:
                    log.exception("collect-servers tick failed")
                await _await_or_wake(interval)
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
    async with httpx.AsyncClient(verify=httpx_verify()) as client:
        try:
            while True:
                try:
                    if cluster.is_active() and master_inventory.is_master():
                        await _health_tick(client)
                except Exception:
                    log.exception("health tick failed")
                await asyncio.sleep(interval)
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
    async with httpx.AsyncClient(verify=httpx_verify()) as client:
        try:
            while True:
                interval = steady if master_inventory.is_warmed_up() else fast
                try:
                    if cluster.is_active() and master_inventory.is_master():
                        await _dataset_refresh_tick(client)
                except Exception:
                    log.exception("dataset-refresh tick failed")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            log.info("master dataset-refresh loop cancelled")
            raise


def _reset_master_state_for_tests() -> None:
    """Wipe the module-level singleton + wake event between tests.

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
    try:
        _wake_event.clear()
    except RuntimeError:
        pass
