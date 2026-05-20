"""
Cluster inference-server inventory.

Mirror of :mod:`cluster_dataset_inventory` for the inference proxy /
Analyze tab. Without this, the "Running inference servers" picker only
listed JobRecords on the local peer, so a multi-node cluster could host
inference jobs on several peers but the webui would never surface them.

What each peer attests to (``LocalInference``):

  - JobRecords with ``job_type == "inference"`` in the ``starting`` /
    ``running`` state.
  - 0.0.0.0 binds rewritten to the peer's cluster hostname so other
    peers can route to it.
  - Pure-loopback binds are kept (operators sometimes pin a job to
    loopback for local-only inference) but flagged ``loopback=True``;
    the webui surfaces them with a "node-local" badge and the cluster
    UI hides them from off-host pickers.

What the master adds (``MasterServerEntry``):

  - Periodic ``/health`` polling so the webui can show a status dot
    without every browser hammering every server.
  - Aggregation across peers via ``master_collect_servers_loop``.

Tokens are carried in ``/api/cluster/inference_servers_local`` but
stripped from ``/api/cluster/inference_servers`` (browser-facing).
The same cluster-bearer trust model as
:mod:`cluster_dataset_inventory` applies.

Difference from the dataset side: no ``user`` registry (no
"add a remote inference URL" surface today) and no dataset-refresh
loop (inference servers have no equivalent of ``/v1/datasets``). The
``models`` list per entry is populated synchronously from
``job_params`` — each Forgather inference job records the configured
model routing names at spawn time, so no network call is needed for
that part.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from forgather.tls import httpx_peer_kwargs

from . import cluster, job_records

log = logging.getLogger("forgather_server.cluster_inference_inventory")

# Hostnames that count as "this machine" — excluded from cluster
# routing because they're not reachable from other peers. Kept in
# the inventory with ``loopback=True`` so single-node operators still
# see them in the Servers panel.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


@dataclass
class LocalInference:
    """An inference server this peer attests to.

    Same shape as ``cluster_dataset_inventory.LocalServer`` plus an
    inference-specific ``models`` field listing the OpenAI routing
    names each server is configured to host.
    """

    server_id: str
    base_url: str
    auth_token: str
    label: str
    peer_node_id: Optional[str]
    source_id: Optional[str]  # JobRecord queue_id
    loopback: bool = False
    # Configured model routing names (the OpenAI ``model`` field
    # clients send). For single-model jobs this is a one-element
    # list derived from ``job_params["model_path"]``; for multi-model
    # it's the names from ``job_params["models"]``. Empty when the
    # JobRecord doesn't carry any usable hint (legacy / partial
    # writes) — the webui falls back to the model_path basename.
    models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def server_id_for(base_url: str) -> str:
    """Stable, ``base_url``-derived identifier (12 hex chars).

    Used by the master aggregator to dedup the same URL when two
    peers happen to know about it. Same construction as the dataset
    side; the resulting ids live in different namespaces but the
    algorithm doesn't need to differ.
    """
    return hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:12]


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize(base_url: str) -> str:
    """Canonical form for URL string comparison.

    Inventory entries and lookup queries both flow through here so a
    URL written as ``HTTP://Host:8137`` matches one stored as
    ``http://host:8137``. Specifically:

      - scheme + hostname lowercased;
      - default ports stripped (``http://x:80`` ↔ ``http://x``);
      - IPv6 literals re-bracketed via :func:`_bracket_ipv6`;
      - trailing slash dropped.

    Falls back to a plain ``rstrip("/")`` if ``urlparse`` can't make
    sense of the input — never raises.
    """
    raw = (base_url or "").rstrip("/")
    try:
        p = urlparse(raw)
    except Exception:
        return raw
    if not p.scheme or not p.hostname:
        return raw
    scheme = p.scheme.lower()
    host = _bracket_ipv6(p.hostname.lower())
    port = p.port
    if port is None or port == _DEFAULT_PORTS.get(scheme):
        netloc = host
    else:
        netloc = f"{host}:{port}"
    # Preserve userinfo if any. URLs reaching the inventory don't
    # carry credentials today, but a defensive pass-through is cheap.
    if p.username is not None:
        userinfo = p.username
        if p.password is not None:
            userinfo += f":{p.password}"
        netloc = f"{userinfo}@{netloc}"
    rebuilt = f"{scheme}://{netloc}"
    # Drop the path entirely — base URLs are netloc-only in this module.
    return rebuilt


def _is_loopback_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host in _LOOPBACK_HOSTS


def _bracket_ipv6(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _jobrecord_base_url(
    host: str, port: int, *, tls: bool, routable_host: Optional[str]
) -> Optional[str]:
    """Build a base URL for a JobRecord-spawned inference server.

    Same priority order as the dataset side:
      1. ``routable_host`` from job_params (the scheduler stamps the
         auto-detected LAN address there).
      2. Cluster identity's hostname (for ``host == "0.0.0.0"`` binds).
      3. ``host`` as-is (explicit binds, including loopback).
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


def _extract_model_names(params: Dict[str, Any]) -> List[str]:
    """Pull the configured model routing names out of an inference job's
    ``job_params``.

    Multi-model jobs carry ``models: [{name, path}, ...]``;
    single-model jobs carry ``model_path: PATH`` (name derived from
    the basename, matching the inference server's CLI parsing).
    Returns an empty list when neither shape is recognizable; the
    webui falls back to a generic label in that case.
    """
    models_field = params.get("models")
    if isinstance(models_field, list) and models_field:
        out: List[str] = []
        for entry in models_field:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name:
                out.append(name)
                continue
            # If the entry has a path but no explicit name, derive it
            # from the path basename — same convention the inference
            # server's argv parser uses.
            path = entry.get("path") or entry.get("model_path")
            if isinstance(path, str) and path:
                base = path.rstrip("/").rsplit("/", 1)[-1]
                if base:
                    out.append(base)
        return out
    path = params.get("model_path")
    if isinstance(path, str) and path:
        base = path.rstrip("/").rsplit("/", 1)[-1]
        if base:
            return [base]
    return []


def _local_jobrecord_servers(peer_node_id: Optional[str]) -> List[LocalInference]:
    """Inference servers this forgather_server has spawned and which
    are currently ``starting`` / ``running``.

    Mirrors the JobRecord scan in :mod:`cluster_dataset_inventory`,
    adapted for ``job_type == "inference"`` and with the configured
    model-name list extracted from ``job_params``.
    """
    out: List[LocalInference] = []
    for r in job_records.list_records():
        if r.job_type != "inference":
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
        # ``scheme`` is the canonical hint; older records may only
        # carry ``tls``. Fall back to the forgather_server-wide TLS
        # state so the scheme matches what the spawned child actually
        # serves.
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
        models = _extract_model_names(params)
        out.append(
            LocalInference(
                server_id=server_id_for(base_url),
                base_url=base_url,
                auth_token=r.auth_token or "",
                label=f"{r.config or 'inference'}:{port}",
                peer_node_id=peer_node_id,
                source_id=r.queue_id,
                loopback=_is_loopback_url(base_url),
                models=models,
            )
        )
    return out


def local_servers() -> List[LocalInference]:
    """All inference servers this peer attests to.

    Currently only JobRecord-spawned servers; no user-registry analog
    exists for inference. Callers exposing this to a browser must
    strip ``auth_token`` first.
    """
    ident = cluster.self_identity()
    peer_node_id = ident.node_id if ident else None
    seen: Dict[str, LocalInference] = {}
    for entry in _local_jobrecord_servers(peer_node_id):
        seen.setdefault(entry.server_id, entry)
    return list(seen.values())


# ===========================================================================
# Master-side aggregation
# ===========================================================================

COLLECT_INTERVAL_SECONDS = 10.0
HEALTH_INTERVAL_SECONDS = 10.0

PEER_TIMEOUT_SECONDS = 5.0
HEALTH_TIMEOUT_SECONDS = 5.0


@dataclass
class MasterServerEntry:
    """One inference server known to the master, with health state.

    Token is preserved on the master so the proxy can attach it
    server-side when the webui dials the URL — the browser never
    sees it.
    """

    server_id: str
    base_url: str
    auth_token: str
    label: str
    peer_node_id: Optional[str]
    source_id: Optional[str] = None
    loopback: bool = False
    models: List[str] = field(default_factory=list)
    healthy: bool = False
    last_health_check: float = 0.0
    last_health_error: str = ""
    total_health_polls: int = 0
    health_failures: int = 0
    consecutive_health_failures: int = 0


class MasterInventory:
    """Master-side aggregation of cluster-wide inference-server state.

    Thread-safe — async tasks mutate, FastAPI route handlers read from
    worker threads.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._servers: Dict[str, MasterServerEntry] = {}
        self._master_become_ts: Optional[float] = None
        self._last_servers_collect_ts: Optional[float] = None
        self._last_health_pass_ts: Optional[float] = None

    def set_master_state(self, is_master: bool) -> bool:
        with self._lock:
            currently_master = self._master_become_ts is not None
            if is_master and not currently_master:
                self._servers = {}
                self._master_become_ts = time.time()
                self._last_servers_collect_ts = None
                self._last_health_pass_ts = None
                log.info(
                    "became master; cleared inference inventory and starting fresh"
                )
                return True
            if not is_master and currently_master:
                self._servers = {}
                self._master_become_ts = None
                self._last_servers_collect_ts = None
                self._last_health_pass_ts = None
                log.info("no longer master; inference inventory cleared")
                return True
            return False

    def is_master(self) -> bool:
        with self._lock:
            return self._master_become_ts is not None

    def merge_servers(self, fresh: Dict[str, MasterServerEntry]) -> None:
        """Replace the server set, preserving health state for entries
        that survived the round."""
        with self._lock:
            merged: Dict[str, MasterServerEntry] = {}
            for sid, new_entry in fresh.items():
                old = self._servers.get(sid)
                if old is not None:
                    new_entry.healthy = old.healthy
                    new_entry.last_health_check = old.last_health_check
                    new_entry.last_health_error = old.last_health_error
                    new_entry.total_health_polls = old.total_health_polls
                    new_entry.health_failures = old.health_failures
                    new_entry.consecutive_health_failures = (
                        old.consecutive_health_failures
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

    def servers_snapshot(self) -> List[MasterServerEntry]:
        with self._lock:
            return [
                MasterServerEntry(**asdict(s)) for s in self._servers.values()
            ]

    def get_server(self, server_id: str) -> Optional[MasterServerEntry]:
        with self._lock:
            s = self._servers.get(server_id)
            return None if s is None else MasterServerEntry(**asdict(s))

    def token_for_url(self, base_url: str) -> Optional[str]:
        """Look up the bearer token for a base URL across the cluster.

        Used by the inference proxy's token-attach path when the webui
        asks to talk to an off-host server. Match is on the canonical
        URL form produced by :func:`_normalize` — scheme + host
        lowercased, IPv6 brackets restored, default ports stripped —
        so ``HTTP://Host:8137`` and ``http://host:8137`` both find an
        entry stored as ``http://host:8137``. No DNS / alias
        resolution.
        """
        norm = _normalize(base_url)
        with self._lock:
            for s in self._servers.values():
                if _normalize(s.base_url) == norm:
                    return s.auth_token or None
        return None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_master": self._master_become_ts is not None,
                "master_become_ts": self._master_become_ts,
                "last_servers_collect_ts": self._last_servers_collect_ts,
                "last_health_pass_ts": self._last_health_pass_ts,
                "server_count": len(self._servers),
            }


# Module-level singleton.
master_inventory = MasterInventory()


# ----- conversion helpers -----


def _to_master_entry(local: LocalInference) -> MasterServerEntry:
    return MasterServerEntry(
        server_id=local.server_id,
        base_url=local.base_url,
        auth_token=local.auth_token,
        label=local.label,
        peer_node_id=local.peer_node_id,
        source_id=local.source_id,
        loopback=local.loopback,
        models=list(local.models),
    )


def _local_server_from_dict(raw: Dict[str, Any]) -> Optional[LocalInference]:
    try:
        sid = str(raw["server_id"])
        base_url = str(raw["base_url"])
        models_raw = raw.get("models")
        models = (
            [str(m) for m in models_raw if isinstance(m, str)]
            if isinstance(models_raw, list)
            else []
        )
        return LocalInference(
            server_id=sid,
            base_url=base_url,
            auth_token=str(raw.get("auth_token") or ""),
            label=str(raw.get("label") or base_url),
            peer_node_id=raw.get("peer_node_id"),
            source_id=(
                str(raw["source_id"]) if raw.get("source_id") else None
            ),
            loopback=bool(raw.get("loopback", False)),
            models=models,
        )
    except (KeyError, TypeError, ValueError):
        return None


# ----- peer-side fetcher -----


async def _fetch_peer_servers(
    client: httpx.AsyncClient, member: cluster.MemberInfo
) -> List[LocalInference]:
    """Pull ``/api/cluster/inference_servers_local`` from one peer.

    Tolerates partial failures: returns an empty list on any error.
    The master retries on the next collect tick.
    """
    scheme = "https" if getattr(member, "tls", False) else "http"
    url = (
        f"{scheme}://{member.address}:{member.port}"
        "/api/cluster/inference_servers_local"
    )
    try:
        r = await client.get(url, timeout=PEER_TIMEOUT_SECONDS)
    except (httpx.HTTPError, OSError) as e:
        log.debug(
            "inference_servers_local fetch failed: %s -> %s", member.hostname, e
        )
        return []
    if r.status_code != 200:
        log.debug(
            "inference_servers_local non-200: %s status=%d",
            member.hostname,
            r.status_code,
        )
        return []
    try:
        body = r.json()
    except ValueError:
        return []
    raw_servers = body.get("servers") if isinstance(body, dict) else None
    if not isinstance(raw_servers, list):
        return []
    out: List[LocalInference] = []
    for raw in raw_servers:
        if not isinstance(raw, dict):
            continue
        parsed = _local_server_from_dict(raw)
        if parsed is None:
            continue
        out.append(parsed)
    return out


async def _collect_servers_tick(client: httpx.AsyncClient) -> None:
    self_id = cluster.self_identity()
    if self_id is None:
        return
    fresh: Dict[str, MasterServerEntry] = {}
    for s in local_servers():
        fresh.setdefault(s.server_id, _to_master_entry(s))
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
    """GET ``/health`` on one inference server; update inventory.

    The Forgather inference server's health endpoint is unauthenticated
    and lives at the server root, not under ``/v1``.
    """
    url = entry.base_url.rstrip("/") + "/health"
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


# ----- public loop entry points -----

_wake_events: List[asyncio.Event] = []


def _register_wake_event() -> asyncio.Event:
    ev = asyncio.Event()
    _wake_events.append(ev)
    return ev


def wake_loops() -> None:
    """Signal every registered loop to run one immediate tick.

    Called from:

      - the membership role-change listener so a newly-elected master
        populates its inventory within seconds of the transition;
      - the inference scheduler's spawn / reap / abort paths via
        ``scheduler._wake_inference_inventory`` so the picker reflects
        new + finished jobs in ~1s rather than waiting on the
        ``COLLECT_INTERVAL_SECONDS`` cadence;
      - the ``POST /api/cluster/inference_servers/refresh`` endpoint
        for an explicit operator-driven refresh.
    """
    for ev in _wake_events:
        try:
            ev.set()
        except RuntimeError:
            pass


async def _await_or_wake(event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return
    finally:
        event.clear()


def _sync_master_state() -> bool:
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
        interval_seconds
        if interval_seconds is not None
        else COLLECT_INTERVAL_SECONDS
    )
    log.info(
        "master inference collect-servers loop starting (interval=%.1fs)", interval
    )
    wake = _register_wake_event()
    async with httpx.AsyncClient(**httpx_peer_kwargs()) as client:
        try:
            while True:
                try:
                    if _sync_master_state():
                        await _collect_servers_tick(client)
                except Exception:
                    log.exception("inference collect-servers tick failed")
                await _await_or_wake(wake, interval)
        except asyncio.CancelledError:
            log.info("master inference collect-servers loop cancelled")
            raise


async def master_health_loop(
    *, interval_seconds: Optional[float] = None
) -> None:
    """Run the per-server /health probe loop until cancelled."""
    interval = (
        interval_seconds
        if interval_seconds is not None
        else HEALTH_INTERVAL_SECONDS
    )
    log.info("master inference health loop starting (interval=%.1fs)", interval)
    wake = _register_wake_event()
    # ``httpx_peer_kwargs`` already carries the shared CA bundle that
    # Forgather-spawned inference servers serve with under TLS. Reusing
    # it here means the master polls children with the same trust
    # config as it polls peer dataset_servers — no special-case logic.
    async with httpx.AsyncClient(**httpx_peer_kwargs()) as client:
        try:
            while True:
                try:
                    if _sync_master_state():
                        await _health_tick(client)
                except Exception:
                    log.exception("inference health tick failed")
                await _await_or_wake(wake, interval)
        except asyncio.CancelledError:
            log.info("master inference health loop cancelled")
            raise


def _reset_master_state_for_tests() -> None:
    """Wipe the singleton + wake events between tests."""
    with master_inventory._lock:
        master_inventory._servers = {}
        master_inventory._master_become_ts = None
        master_inventory._last_servers_collect_ts = None
        master_inventory._last_health_pass_ts = None
    _wake_events.clear()
