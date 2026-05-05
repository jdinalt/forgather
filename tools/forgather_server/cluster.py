"""Cluster identity and member-table state for multi-node operation.

This module is the single source of truth for "is this server part of a
cluster, and if so, who else is in it." It deliberately holds *no*
network logic — discovery (mDNS) and liveness (peer-pull) are separate
modules that feed updates in here.

Lifecycle:

    server.py --cluster <name>
        -> cluster.activate(cluster_name, port) at startup
        -> cluster_discovery.start() advertises and browses
        -> cluster_membership.start() pulls peer member tables
    Both feed updates via update_member()/mark_unreachable().

The node identity (UUID) persists at ``~/.forgather/cluster/node_id``
and is reused across restarts. A node's UUID survives hostname changes,
NIC swaps, and IP renumbering — it is the only stable handle the rest
of the cluster has on us. The cluster *name* is per-invocation (CLI
flag) so a single host can move between clusters without rewriting
state.

Master selection is deterministic over the live member set: lowest UUID
wins. With UUIDs as 128-bit randoms, ties are not a concern. Unreachable
members are excluded from the master computation but kept in the table
(for the union-of-ever-seen view the user agreed to in design).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Dict, List, Optional

from . import paths
from ._atomic import atomic_write_text

log = logging.getLogger("forgather_server.cluster")


# Treat a member as unreachable if we have not heard from it (or about
# it via a peer's table) within this many seconds. 30s gives roughly
# six missed peer-pull cycles at the default 5s cadence — long enough
# to ride out a transient network blip, short enough to be useful for
# master selection.
DEFAULT_UNREACHABLE_AFTER_SECONDS = 30.0


def _forgather_version() -> str:
    try:
        return _pkg_version("forgather")
    except PackageNotFoundError:
        # Editable installs without dist-info — fall back to the
        # source-of-truth in pyproject.toml. We don't read pyproject at
        # runtime; an "unknown" string is honest and only affects the
        # version-mismatch UI in Phase 2.
        return "unknown"


@dataclass
class NodeIdentity:
    """This server's stable identity within a cluster."""

    node_id: str
    hostname: str
    cluster_name: str
    port: int
    forgather_version: str
    started_at: float


@dataclass
class MemberInfo:
    """What we know (or have been told) about another node."""

    node_id: str
    hostname: str
    address: str  # routable host or IP — last seen
    port: int
    cluster_name: str
    forgather_version: str
    first_seen: float
    last_seen: float
    reachable: bool = True
    # Source of the most recent update — useful for debugging which
    # mechanism is keeping a stale entry alive. One of "discovery",
    # "peer_pull", "self".
    last_source: str = "discovery"


class _ClusterState:
    """Module-private container; access through the public functions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._self: Optional[NodeIdentity] = None
        self._members: Dict[str, MemberInfo] = {}
        # Tunable for tests; production uses the module-level default.
        self._unreachable_after: float = DEFAULT_UNREACHABLE_AFTER_SECONDS

    # ------------------------------------------------------------
    # Activation / identity
    # ------------------------------------------------------------

    def activate(self, cluster_name: str, port: int) -> NodeIdentity:
        if not cluster_name:
            raise ValueError("cluster_name must be a non-empty string")
        node_id = _load_or_create_node_id()
        import platform

        identity = NodeIdentity(
            node_id=node_id,
            hostname=platform.node() or "unknown",
            cluster_name=cluster_name,
            port=int(port),
            forgather_version=_forgather_version(),
            started_at=time.time(),
        )
        with self._lock:
            self._self = identity
            # Self always present in the members table so callers don't
            # have to special-case it. Address "127.0.0.1" is a
            # placeholder; the discovery layer will replace it once the
            # outward-facing IP is known.
            self._members[identity.node_id] = MemberInfo(
                node_id=identity.node_id,
                hostname=identity.hostname,
                address="127.0.0.1",
                port=identity.port,
                cluster_name=identity.cluster_name,
                forgather_version=identity.forgather_version,
                first_seen=identity.started_at,
                last_seen=identity.started_at,
                reachable=True,
                last_source="self",
            )
        log.info(
            "cluster activated: name=%s node_id=%s hostname=%s port=%d",
            identity.cluster_name,
            identity.node_id,
            identity.hostname,
            identity.port,
        )
        return identity

    def is_active(self) -> bool:
        with self._lock:
            return self._self is not None

    def self_identity(self) -> Optional[NodeIdentity]:
        with self._lock:
            return self._self

    # ------------------------------------------------------------
    # Member table
    # ------------------------------------------------------------

    def update_member(
        self,
        node_id: str,
        *,
        hostname: str,
        address: str,
        port: int,
        cluster_name: str,
        forgather_version: str = "unknown",
        source: str = "discovery",
        now: Optional[float] = None,
    ) -> MemberInfo:
        """Insert or refresh a member entry. Idempotent."""
        if not self.is_active():
            raise RuntimeError("cluster is not active; cannot update members")
        self_id = self.self_identity()
        if self_id is not None and cluster_name != self_id.cluster_name:
            # Same-LAN cross-cluster contamination — different `--cluster
            # <name>` advertisements should never mix. Discovery is
            # responsible for filtering, but defend in depth here too.
            raise ValueError(
                f"refusing to add member from cluster {cluster_name!r}; "
                f"this node is in cluster {self_id.cluster_name!r}"
            )
        ts = now if now is not None else time.time()
        with self._lock:
            existing = self._members.get(node_id)
            if existing is None:
                member = MemberInfo(
                    node_id=node_id,
                    hostname=hostname,
                    address=address,
                    port=port,
                    cluster_name=cluster_name,
                    forgather_version=forgather_version,
                    first_seen=ts,
                    last_seen=ts,
                    reachable=True,
                    last_source=source,
                )
                self._members[node_id] = member
                log.info(
                    "cluster member discovered: %s (%s) at %s:%d via %s",
                    hostname,
                    node_id,
                    address,
                    port,
                    source,
                )
                return member
            existing.hostname = hostname
            existing.address = address
            existing.port = port
            existing.forgather_version = forgather_version
            existing.last_seen = ts
            existing.last_source = source
            if not existing.reachable:
                log.info(
                    "cluster member back online: %s (%s)", hostname, node_id
                )
            existing.reachable = True
            return existing

    def mark_unreachable(self, node_id: str) -> None:
        """Flag a member as unreachable without removing it."""
        with self._lock:
            m = self._members.get(node_id)
            if m is None or not m.reachable:
                return
            self_id = self._self
            if self_id is not None and node_id == self_id.node_id:
                # Refuse to flag self unreachable; otherwise master
                # selection becomes degenerate during transient peer-pull
                # failures from our own node.
                return
            m.reachable = False
            log.warning(
                "cluster member unreachable: %s (%s)", m.hostname, m.node_id
            )

    def sweep_unreachable(self, *, now: Optional[float] = None) -> List[str]:
        """Mark members unreachable if last_seen is older than the
        configured threshold. Returns the list of node_ids transitioned.

        Called from the membership task on each tick. Keeping it here
        (rather than in the task module) makes the threshold testable in
        isolation and lets the unit tests fast-forward time.
        """
        if not self.is_active():
            return []
        ts = now if now is not None else time.time()
        threshold = self._unreachable_after
        transitioned: List[str] = []
        with self._lock:
            self_id = self._self
            for node_id, m in self._members.items():
                if self_id is not None and node_id == self_id.node_id:
                    # Refresh self.last_seen on every sweep so the
                    # member table stays internally consistent without a
                    # separate self-heartbeat path.
                    m.last_seen = ts
                    continue
                if not m.reachable:
                    continue
                if ts - m.last_seen > threshold:
                    m.reachable = False
                    transitioned.append(node_id)
        for node_id in transitioned:
            m = self._members[node_id]
            log.warning(
                "cluster member silent > %.0fs, marking unreachable: %s (%s)",
                threshold,
                m.hostname,
                node_id,
            )
        return transitioned

    def members(self) -> List[MemberInfo]:
        with self._lock:
            # Stable order by node_id so callers (and tests) see
            # deterministic output. The HTTP layer can re-sort if it
            # wants something else.
            return sorted(self._members.values(), key=lambda m: m.node_id)

    def reachable_members(self) -> List[MemberInfo]:
        return [m for m in self.members() if m.reachable]

    # ------------------------------------------------------------
    # Master selection
    # ------------------------------------------------------------

    def master_node_id(self) -> Optional[str]:
        """Lowest UUID among currently reachable members."""
        live = self.reachable_members()
        if not live:
            return None
        return min(m.node_id for m in live)

    def is_self_master(self) -> bool:
        s = self.self_identity()
        if s is None:
            return False
        return self.master_node_id() == s.node_id

    def is_peer_address(self, address: str) -> bool:
        """Return True if ``address`` belongs to a known cluster member.

        Used by the auth middleware to allow unauthenticated GETs from
        peers on the cluster API surface. Self is treated as a peer
        too: it makes loopback testing simpler and there is no
        privilege difference (anything reaching loopback already has
        the server's uid).
        """
        if not address:
            return False
        with self._lock:
            for m in self._members.values():
                if m.address == address:
                    return True
        return False

    # ------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------

    def _reset_for_tests(self) -> None:
        with self._lock:
            self._self = None
            self._members.clear()
            self._unreachable_after = DEFAULT_UNREACHABLE_AFTER_SECONDS

    def _set_unreachable_after_for_tests(self, seconds: float) -> None:
        with self._lock:
            self._unreachable_after = float(seconds)


_state = _ClusterState()


# ---------------------------------------------------------------------------
# Module-level public API. Wraps the singleton so callers don't carry a
# handle around — matches the style used elsewhere in this server
# (queue_store, gpu_policy, etc.).
# ---------------------------------------------------------------------------


def activate(cluster_name: str, port: int) -> NodeIdentity:
    return _state.activate(cluster_name, port)


def is_active() -> bool:
    return _state.is_active()


def self_identity() -> Optional[NodeIdentity]:
    return _state.self_identity()


def update_member(
    node_id: str,
    *,
    hostname: str,
    address: str,
    port: int,
    cluster_name: str,
    forgather_version: str = "unknown",
    source: str = "discovery",
    now: Optional[float] = None,
) -> MemberInfo:
    return _state.update_member(
        node_id,
        hostname=hostname,
        address=address,
        port=port,
        cluster_name=cluster_name,
        forgather_version=forgather_version,
        source=source,
        now=now,
    )


def mark_unreachable(node_id: str) -> None:
    _state.mark_unreachable(node_id)


def sweep_unreachable(*, now: Optional[float] = None) -> List[str]:
    return _state.sweep_unreachable(now=now)


def members() -> List[MemberInfo]:
    return _state.members()


def reachable_members() -> List[MemberInfo]:
    return _state.reachable_members()


def master_node_id() -> Optional[str]:
    return _state.master_node_id()


def is_self_master() -> bool:
    return _state.is_self_master()


def is_peer_address(address: str) -> bool:
    return _state.is_peer_address(address)


def _reset_for_tests() -> None:
    _state._reset_for_tests()


def _set_unreachable_after_for_tests(seconds: float) -> None:
    _state._set_unreachable_after_for_tests(seconds)


# ---------------------------------------------------------------------------
# Persistent UUID
# ---------------------------------------------------------------------------


def _load_or_create_node_id() -> str:
    """Read ``~/.forgather/cluster/node_id`` or generate one.

    Mode 0600 like the auth token. The UUID has no privacy weight by
    itself, but the directory holds other secrets so we keep modes
    consistent.
    """
    path = paths.cluster_node_id_file()
    if path.exists():
        try:
            text = path.read_text().strip()
        except OSError as e:
            log.warning("could not read %s: %s", path, e)
            text = ""
        if text:
            try:
                # Validate the on-disk value; a corrupted file becomes
                # an explicit error rather than a silent re-roll that
                # would split-brain the node from the cluster's POV.
                uuid.UUID(text)
                return text
            except ValueError:
                log.error(
                    "invalid UUID in %s; refusing to overwrite. "
                    "Inspect the file and remove it manually if you "
                    "intend to mint a new identity.",
                    path,
                )
                raise
    new_id = str(uuid.uuid4())
    atomic_write_text(path, new_id + "\n", mode=0o600)
    try:
        os.chmod(path, 0o600)
    except OSError as e:
        log.warning("could not chmod %s to 0600: %s", path, e)
    log.info("minted new cluster node_id %s at %s", new_id, path)
    return new_id
