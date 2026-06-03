"""
Cluster DiLoCo-server inventory.

Cross-node discovery for DiLoCo parameter servers, modelled after
:mod:`cluster_dataset_inventory` and :mod:`cluster_inference_inventory`.
Without this, the forgather server on host ``B`` had no way to learn
that a DiLoCo server was running on host ``A``: each forgather server
only knew about its own ``JobRecord`` rows and its own user-registry.
The ``forgather diloco register <url> --auth-token <tok>`` command
remains as the escape hatch for endpoints mDNS can't see
(WAN, SSH tunnel), but on a normal LAN cluster a DiLoCo server
spawned on any peer surfaces in every webui without operator action.

Two sources per peer (``LocalDiLoCo``):

  1. JobRecords with ``job_type == "diloco_server"`` in the
     ``starting`` / ``running`` state.
  2. The user-added registry at
     ``<config>/server/diloco_server_registry.json``.

Loopback-only binds are kept in the inventory with ``loopback=True``
so the operator still sees node-local servers in the panel, but they
are excluded from cross-node candidate-selection (a remote peer can't
reach another node's 127.0.0.1). JobRecord servers bound to
``0.0.0.0`` are rewritten to the cluster identity's hostname (or to
the scheduler-stamped ``routable_host`` when present) so other peers
can dial them.

What the master adds (``MasterServerEntry``):

  - Periodic ``/health`` polling so the webui can show a status dot
    without every browser hammering every server.
  - Aggregation across peers via ``master_collect_servers_loop``.

Tokens are carried in ``/api/cluster/diloco_servers_local`` (peer-
mTLS gate via ``auth._PEER_ALLOWED_PATHS``) and propagated to the
master-aggregated ``/api/cluster/diloco_servers``; same trust model
as the dataset and inference inventories. The webui proxy at
``routes/diloco.py`` consults ``master_inventory.token_for_url(base)``
to dial off-host upstreams without operator token handling.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from forgather.tls import httpx_peer_kwargs, httpx_verify

from . import cluster, diloco_server_registry, job_records

log = logging.getLogger("forgather_server.cluster_diloco_inventory")

# Hostnames that count as "this machine" — kept in the inventory
# with ``loopback=True`` so single-node operators still see them in
# the DiLoCo panel, but excluded from cross-node routing because
# another peer can't reach them.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


@dataclass
class LocalDiLoCo:
    """A DiLoCo server this peer attests to.

    ``base_url`` is normalized (no trailing slash) and rewritten
    (0.0.0.0 -> cluster hostname) so it is consumable by another
    peer's HTTP client.

    ``auth_token`` may be empty for servers running ``--no-auth``.

    ``verify_tls`` is False when the operator registered the URL with
    chain validation off (SSH-tunneled remotes). Default True so
    JobRecord-spawned and standard registry entries keep the secure-
    by-default posture.
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
    # Propagated so the webui can target a DELETE at the right peer
    # without round-tripping through the master.
    source_id: Optional[str] = None
    loopback: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def server_id_for(base_url: str) -> str:
    """Stable, ``base_url``-derived identifier (12 hex chars).

    Used by the master aggregator to deduplicate when the same URL
    arrives from multiple peers (e.g., several operators independently
    registered the same external server). Matches the construction in
    :mod:`cluster_dataset_inventory` / :mod:`cluster_inference_inventory`;
    ids live in different namespaces but the algorithm doesn't need
    to differ.
    """
    return hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:12]


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _bracket_ipv6(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _normalize(base_url: str) -> str:
    """Canonical form for URL string comparison.

    Same shape as :func:`cluster_inference_inventory._normalize`: lower
    scheme + host, strip default port, drop trailing slash. Inventory
    entries and ``token_for_url`` lookups both flow through here so a
    URL written as ``HTTPS://Host:8512`` matches one stored as
    ``https://host:8512``. Falls back to a plain ``rstrip("/")`` if
    ``urlparse`` can't make sense of the input.
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
    return f"{scheme}://{netloc}"


def _is_loopback_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host in _LOOPBACK_HOSTS


def _jobrecord_base_url(
    host: str, port: int, *, tls: bool, routable_host: Optional[str]
) -> Optional[str]:
    """Build a base URL for a JobRecord-spawned DiLoCo server.

    Priority order for the host portion:
      1. ``routable_host`` from job_params (the scheduler records the
         auto-detected LAN address there).
      2. Cluster identity's hostname (for ``host == "0.0.0.0"`` binds).
      3. ``host`` as-is (explicit binds, including loopback).

    Returns ``None`` only when ``host`` is ``0.0.0.0`` and no routable
    address can be inferred — the URL truly isn't usable.
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


def _local_jobrecord_servers(peer_node_id: Optional[str]) -> List[LocalDiLoCo]:
    """DiLoCo servers this forgather_server has spawned and which are
    currently in the ``starting`` / ``running`` state.

    Mirrors the JobRecord scan in
    :func:`tools.forgather_server.routes.diloco._local_servers` but:

    - rewrites 0.0.0.0 to the cluster hostname (so other peers can
      route to it);
    - keeps loopback binds with ``loopback=True`` (single-node
      operators still see them; cross-node selection skips them);
    - includes ``auth_token`` (the browser-facing local route strips
      it; the cluster carve-out gates this surface instead).
    """
    out: List[LocalDiLoCo] = []
    for r in job_records.list_records():
        if r.job_type != "diloco_server":
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
        # Scheme stamped at dispatch time (issue #90); fall back to
        # the forgather_server-wide TLS state so the URL matches what
        # the spawned child actually serves.
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
            LocalDiLoCo(
                server_id=server_id_for(base_url),
                base_url=base_url,
                auth_token=r.auth_token or "",
                label=f"{r.config or 'diloco_server'}:{port}",
                source="local",
                peer_node_id=peer_node_id,
                source_id=r.queue_id,
                loopback=_is_loopback_url(base_url),
            )
        )
    return out


def _user_registry_servers(peer_node_id: Optional[str]) -> List[LocalDiLoCo]:
    """User-registered DiLoCo server entries (the escape-hatch path).

    Loopback entries are kept (operators sometimes register node-local
    servers from a fixed URL); ``loopback=True`` excludes them from
    cross-node selection while leaving them visible in the panel.
    """
    out: List[LocalDiLoCo] = []
    for e in diloco_server_registry.list_entries():
        base_url = _normalize(e.base_url)
        out.append(
            LocalDiLoCo(
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


def local_servers() -> List[LocalDiLoCo]:
    """All DiLoCo servers this peer attests to.

    Sources, in priority order on ``server_id`` collision:

    1. JobRecord-spawned servers currently ``starting`` / ``running``.
    2. User-registered entries from the persistent registry.

    A duplicate ``base_url`` from both sources is reported once; the
    JobRecord entry wins so the locally-spawned label / source is
    preserved.

    Records include the bearer token. Callers exposing this list to a
    browser must strip ``auth_token`` before serialization — the
    intended consumer of this list is the master collect loop, which
    receives it over the mTLS-gated peer endpoint.
    """
    ident = cluster.self_identity()
    peer_node_id = ident.node_id if ident else None
    seen: Dict[str, LocalDiLoCo] = {}
    for entry in _local_jobrecord_servers(peer_node_id):
        seen.setdefault(entry.server_id, entry)
    for entry in _user_registry_servers(peer_node_id):
        seen.setdefault(entry.server_id, entry)
    return list(seen.values())


# ===========================================================================
# Master-side aggregation
# ===========================================================================
#
# The master node aggregates the cluster-wide view by:
#   1. ``master_collect_servers_loop`` — every ``COLLECT_INTERVAL_SECONDS``,
#      GET ``/api/cluster/diloco_servers_local`` from every reachable peer
#      and merge the results into the master inventory.
#   2. ``master_health_loop`` — every ``HEALTH_INTERVAL_SECONDS``, GET
#      ``/health`` on every known server and flip its ``healthy`` flag.
#
# Both loops run on every node but self-gate on ``cluster.is_self_master()``;
# failover is automatic.

COLLECT_INTERVAL_SECONDS = 10.0
HEALTH_INTERVAL_SECONDS = 10.0

# Per-call HTTP timeouts.
PEER_TIMEOUT_SECONDS = 5.0
HEALTH_TIMEOUT_SECONDS = 5.0


@dataclass
class MasterServerEntry:
    """One DiLoCo server known to the master, with cluster identity +
    health. No content fields (cf. the dataset side's ``handles`` /
    ``locals_info``): a DiLoCo server's only "content" is the live
    training state, which the webui surfaces via the per-server
    ``/status`` and ``/info`` proxies — not the right granularity for
    a polling inventory.
    """

    server_id: str
    base_url: str
    auth_token: str
    label: str
    source: str  # "local" or "user"
    peer_node_id: Optional[str]
    verify_tls: bool = True
    source_id: Optional[str] = None
    loopback: bool = False
    healthy: bool = False
    last_health_check: float = 0.0
    last_health_error: str = ""
    total_health_polls: int = 0
    health_failures: int = 0
    consecutive_health_failures: int = 0


class MasterInventory:
    """Master-side aggregation of cluster-wide DiLoCo server state.

    Thread-safe — the loops mutate from asyncio tasks, the route
    handlers read synchronously from FastAPI worker threads.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._servers: Dict[str, MasterServerEntry] = {}
        self._master_become_ts: Optional[float] = None
        self._last_servers_collect_ts: Optional[float] = None
        self._last_health_pass_ts: Optional[float] = None

    # ----- master role transitions -----

    def set_master_state(self, is_master: bool) -> bool:
        """Update internal "am I master" state. Returns True iff a
        transition occurred."""
        with self._lock:
            currently_master = self._master_become_ts is not None
            if is_master and not currently_master:
                self._servers = {}
                self._master_become_ts = time.time()
                self._last_servers_collect_ts = None
                self._last_health_pass_ts = None
                log.info("became master; cleared diloco inventory and starting fresh")
                return True
            if not is_master and currently_master:
                self._servers = {}
                self._master_become_ts = None
                self._last_servers_collect_ts = None
                self._last_health_pass_ts = None
                log.info("no longer master; diloco inventory cleared")
                return True
            return False

    def is_master(self) -> bool:
        with self._lock:
            return self._master_become_ts is not None

    # ----- server set merge -----

    def merge_servers(self, fresh: Dict[str, MasterServerEntry]) -> None:
        """Replace the server set, preserving health for surviving
        entries so a 10s collect tick doesn't reset a flapping
        server's failure streak."""
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

    # ----- reads -----

    def servers_snapshot(self) -> List[MasterServerEntry]:
        with self._lock:
            return [MasterServerEntry(**asdict(s)) for s in self._servers.values()]

    def get_server(self, server_id: str) -> Optional[MasterServerEntry]:
        with self._lock:
            s = self._servers.get(server_id)
            return None if s is None else MasterServerEntry(**asdict(s))

    def token_for_url(self, base_url: str) -> Optional[str]:
        """Bearer token for a known ``base_url``, or ``None``.

        The webui proxy uses this so an operator's browser action
        against a remote-peer DiLoCo server doesn't require the
        operator to have the upstream token — same shape as
        :meth:`cluster_inference_inventory.MasterInventory.token_for_url`.
        """
        normalized = _normalize(base_url)
        with self._lock:
            for s in self._servers.values():
                if _normalize(s.base_url) == normalized and s.auth_token:
                    return s.auth_token
        return None

    def verify_tls_for_url(self, base_url: str) -> Optional[bool]:
        """Per-entry ``verify_tls`` for a known ``base_url``.

        Returns ``None`` if the URL isn't in the master inventory —
        callers should fall back to their own default (which is True
        / secure-by-default) in that case.
        """
        normalized = _normalize(base_url)
        with self._lock:
            for s in self._servers.values():
                if _normalize(s.base_url) == normalized:
                    return s.verify_tls
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


# Module-level singleton — the loops mutate it, the routes read it.
master_inventory = MasterInventory()


# ----- conversion helpers -----


def _to_master_entry(local: LocalDiLoCo) -> MasterServerEntry:
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


def _local_server_from_dict(raw: Dict[str, Any]) -> Optional[LocalDiLoCo]:
    """Parse one peer-supplied inventory entry, rejecting malformed URLs.

    The validation here is defense-in-depth against a credentialed
    peer that ships a structurally-broken or hostile base URL — the
    cluster bearer trust boundary concedes peer compromise can submit
    code, but does not extend to letting peers inject arbitrary
    Basic-auth credentials into this node's outbound httpx calls or
    bypass the http/https scheme guard the operator-facing proxy
    enforces (``routes/diloco._validate_base``).

    Rejects:
      * Non-http/https schemes.
      * Empty hostnames.
      * URLs with embedded userinfo (``http://user:pass@host``) —
        httpx would forward those credentials on every outbound call.
      * ``verify_tls=False`` on entries that aren't user-registry-
        sourced: a peer's local SSH-tunnel escape hatch shouldn't
        downgrade *our* outbound TLS posture for the same URL. The
        local operator can still override by registering the URL
        locally with ``verify_tls=False``.
    """
    try:
        sid = str(raw["server_id"])
        base_url = str(raw["base_url"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        parsed = urlparse(base_url)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    source = str(raw.get("source") or "user")
    raw_verify = bool(raw.get("verify_tls", True))
    # Honor ``verify_tls=False`` only when the operator on *some* node
    # explicitly registered the URL with that opt-out (source=user).
    # Locally-spawned servers always run with the cluster CA chain;
    # accepting verify_tls=False from a "local" peer entry would let
    # one peer downgrade every other peer's outbound TLS.
    verify_tls = raw_verify if source == "user" else True
    try:
        return LocalDiLoCo(
            server_id=sid,
            base_url=base_url,
            auth_token=str(raw.get("auth_token") or ""),
            label=str(raw.get("label") or base_url),
            source=source,
            peer_node_id=raw.get("peer_node_id"),
            verify_tls=verify_tls,
            source_id=(str(raw["source_id"]) if raw.get("source_id") else None),
            loopback=bool(raw.get("loopback", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


# ----- peer-side fetcher -----


async def _fetch_peer_servers(
    client: httpx.AsyncClient, member: cluster.MemberInfo
) -> List[LocalDiLoCo]:
    """Pull ``/api/cluster/diloco_servers_local`` from one peer.

    Returns an empty list on any error; the master tolerates partial
    failures and re-tries on the next collect tick.
    """
    scheme = "https" if getattr(member, "tls", False) else "http"
    url = (
        f"{scheme}://{member.address}:{member.port}"
        "/api/cluster/diloco_servers_local"
    )
    try:
        r = await client.get(url, timeout=PEER_TIMEOUT_SECONDS)
    except (httpx.HTTPError, OSError) as e:
        log.debug("diloco_servers_local fetch failed: %s -> %s", member.hostname, e)
        return []
    if r.status_code != 200:
        log.debug(
            "diloco_servers_local non-200: %s status=%d",
            member.hostname,
            r.status_code,
        )
        return []
    try:
        body = r.json()
    except ValueError:
        log.debug("diloco_servers_local non-JSON from %s", member.hostname)
        return []
    raw_servers = body.get("servers") if isinstance(body, dict) else None
    if not isinstance(raw_servers, list):
        return []
    out: List[LocalDiLoCo] = []
    for raw in raw_servers:
        if not isinstance(raw, dict):
            continue
        parsed = _local_server_from_dict(raw)
        if parsed is None:
            continue
        out.append(parsed)
    return out


# ----- collect / health implementations -----


async def _collect_servers_tick(client: httpx.AsyncClient) -> None:
    """One round of "ask every peer for their local server list."""
    self_id = cluster.self_identity()
    if self_id is None:
        return
    fresh: Dict[str, MasterServerEntry] = {}
    # Local node first — direct call, no HTTP needed.
    for s in local_servers():
        fresh.setdefault(s.server_id, _to_master_entry(s))
    # Fan out to peers.
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
    validation off; otherwise the standard forgather.tls verify policy.
    """
    if not entry.verify_tls:
        return False
    return httpx_verify()


def _auth_headers(token: str) -> Dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def _check_one_health(
    client: httpx.AsyncClient, entry: MasterServerEntry
) -> None:
    """GET ``/health`` on one server; update inventory.

    ``/health`` is unauthenticated on the DiLoCo server (matches the
    dataset/inference equivalents), so no bearer is attached. A one-
    shot client handles the ``verify_tls=False`` case so chain
    validation can be skipped without breaking pooling for the
    secure-by-default majority.

    Note: the ``verify_tls=False`` branch creates a bare
    ``httpx.AsyncClient(verify=False)`` and therefore omits the
    cluster CA bundle and the client cert from ``httpx_peer_kwargs``.
    Intentional — the opt-out exists precisely for SSH-tunneled or
    out-of-band-secured upstreams where the cluster's CA chain
    doesn't apply on either side. Routing to a cluster-internal
    server with ``verify_tls=False`` would silently degrade mTLS;
    that combination is operator-error, not a code path to guard.
    """
    if not entry.verify_tls:
        async with httpx.AsyncClient(verify=False) as c:
            return await _check_one_health_inner(c, entry)
    return await _check_one_health_inner(client, entry)


async def _check_one_health_inner(
    client: httpx.AsyncClient, entry: MasterServerEntry
) -> None:
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

# Per-loop wake events — see the matching comment in
# :mod:`cluster_dataset_inventory`. A single shared event was a bug:
# whichever loop woke first cleared it and the others slept through.
_wake_events: List[asyncio.Event] = []


def _register_wake_event() -> asyncio.Event:
    ev = asyncio.Event()
    _wake_events.append(ev)
    return ev


def wake_loops() -> None:
    """Signal every registered master loop to run one immediate tick.

    Called from the scheduler on DiLoCo server spawn/reap, from the
    ``/diloco_servers/refresh`` endpoint, and from the membership
    role-change listener so a newly-elected master populates its
    inventory in seconds instead of waiting up to
    ``COLLECT_INTERVAL_SECONDS``.
    """
    for ev in _wake_events:
        try:
            ev.set()
        except RuntimeError:
            # Event's loop isn't running yet (very-early init or
            # certain test fixtures). Wake-up is a latency hint, not
            # correctness-critical.
            pass


async def _await_or_wake(event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return
    finally:
        event.clear()


def _sync_master_state() -> bool:
    """Read authoritative ``cluster.is_self_master()`` into the
    inventory's cached state and return the live value."""
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
    log.info("master diloco collect-servers loop starting (interval=%.1fs)", interval)
    wake = _register_wake_event()
    async with httpx.AsyncClient(**httpx_peer_kwargs()) as client:
        try:
            while True:
                try:
                    if _sync_master_state():
                        await _collect_servers_tick(client)
                except Exception:
                    log.exception("diloco collect-servers tick failed")
                await _await_or_wake(wake, interval)
        except asyncio.CancelledError:
            log.info("master diloco collect-servers loop cancelled")
            raise


async def master_health_loop(*, interval_seconds: Optional[float] = None) -> None:
    """Run the per-server /health probe loop until cancelled."""
    interval = (
        interval_seconds if interval_seconds is not None else HEALTH_INTERVAL_SECONDS
    )
    log.info("master diloco health loop starting (interval=%.1fs)", interval)
    wake = _register_wake_event()
    async with httpx.AsyncClient(**httpx_peer_kwargs()) as client:
        try:
            while True:
                try:
                    if _sync_master_state():
                        await _health_tick(client)
                except Exception:
                    log.exception("diloco health tick failed")
                await _await_or_wake(wake, interval)
        except asyncio.CancelledError:
            log.info("master diloco health loop cancelled")
            raise


def _reset_master_state_for_tests() -> None:
    """Wipe the module-level singleton + wake events between tests."""
    with master_inventory._lock:
        master_inventory._servers = {}
        master_inventory._master_become_ts = None
        master_inventory._last_servers_collect_ts = None
        master_inventory._last_health_pass_ts = None
    _wake_events.clear()
