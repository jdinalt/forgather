"""Peer-pull liveness loop for cluster membership.

mDNS tells us *who exists*. This module tells us *who is currently
answering HTTP*. Every ``TICK_SECONDS``:

  1. For each known peer (excluding self), GET
     ``http://<address>:<port>/api/cluster/members`` with a short
     timeout.
  2. Merge the peer's reported members into the local view (so a
     node we have not seen via mDNS but a peer has, becomes known).
  3. Refresh the peer's ``last_seen`` on success; on failure, leave it
     alone so the sweep below can mark it unreachable when its
     window elapses.
  4. ``cluster.sweep_unreachable`` flips silent peers to unreachable.

Unauthenticated peer-pull is fine in v1: the cluster is on a trusted
LAN (per the design contract). The receiving end (`routes/cluster.py`)
gates the carve-out on source IP belonging to a known cluster member,
so an unrelated process on the same host can't fake-poll the API.

The loop runs in the FastAPI lifespan task group alongside the
existing scheduler dispatcher.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from . import cluster

log = logging.getLogger("forgather_server.cluster.membership")


TICK_SECONDS = 5.0
PEER_TIMEOUT_SECONDS = 3.0


async def _pull_one_peer(
    client: httpx.AsyncClient, member: cluster.MemberInfo
) -> bool:
    """Fetch ``/api/cluster/members`` from one peer and merge.

    Returns True on success (peer answered with a usable payload),
    False otherwise. The caller uses this to refresh ``last_seen`` for
    the polled peer.
    """
    url = f"http://{member.address}:{member.port}/api/cluster/members"
    try:
        r = await client.get(url, timeout=PEER_TIMEOUT_SECONDS)
    except (httpx.HTTPError, OSError) as e:
        log.debug("peer pull failed: %s -> %s: %s", member.hostname, url, e)
        return False
    if r.status_code != 200:
        log.debug(
            "peer pull non-200: %s -> %s status=%d",
            member.hostname,
            url,
            r.status_code,
        )
        return False
    try:
        body = r.json()
    except ValueError:
        log.debug("peer pull non-JSON from %s", url)
        return False
    items = body.get("members") if isinstance(body, dict) else None
    if not isinstance(items, list):
        log.debug("peer pull missing 'members' list from %s", url)
        return False
    self_id = cluster.self_identity()
    self_node_id = self_id.node_id if self_id else None
    self_cluster = self_id.cluster_name if self_id else None
    # Snapshot existing member addresses so we can refuse loopback
    # downgrades. Pre-fix peers had ``self.address == "127.0.0.1"``
    # in their member tables (placeholder from activate()); pulling
    # that and merging it would clobber a perfectly good
    # mDNS-discovered address for the same node. After both ends are
    # on the fix this defense is redundant, but it costs nothing and
    # protects mixed-version clusters during rollout.
    existing_addrs = {m.node_id: m.address for m in cluster.members()}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        node_id = entry.get("node_id")
        peer_cluster = entry.get("cluster_name")
        if not node_id or not peer_cluster:
            continue
        if node_id == self_node_id:
            continue
        if peer_cluster != self_cluster:
            # Belt-and-suspenders: cluster.update_member would reject
            # this anyway, but skipping early avoids a noisy raise.
            continue
        new_address = str(entry.get("address") or "")
        prior = existing_addrs.get(node_id, "")
        if (
            prior
            and not prior.startswith("127.")
            and new_address.startswith("127.")
        ):
            # A peer is telling us node X is at 127.0.0.1 but we
            # already have a real address for X. Don't downgrade.
            log.debug(
                "ignoring loopback downgrade for %s: %s -> %s",
                node_id,
                prior,
                new_address,
            )
            continue
        # Probe is optional: a peer running pre-fix code won't send
        # one. Pass None in that case so cluster.update_member
        # preserves whatever we already have for this node.
        peer_probe = entry.get("probe")
        if not isinstance(peer_probe, dict):
            peer_probe = None
        try:
            cluster.update_member(
                node_id,
                hostname=str(entry.get("hostname") or ""),
                address=new_address,
                port=int(entry.get("port") or 0),
                cluster_name=peer_cluster,
                forgather_version=str(
                    entry.get("forgather_version") or "unknown"
                ),
                source="peer_pull",
                probe=peer_probe,
            )
        except Exception:
            # Logged at debug — a bad single entry shouldn't abort the
            # whole peer's update batch.
            log.debug(
                "ignoring bad member entry from %s: %r", member.hostname, entry
            )
    return True


async def _tick(client: httpx.AsyncClient) -> None:
    """One round of pulls + sweep."""
    if not cluster.is_active():
        return
    self_id = cluster.self_identity()
    self_node_id = self_id.node_id if self_id else None
    targets = [
        m
        for m in cluster.members()
        if m.node_id != self_node_id and m.address
    ]
    if not targets:
        cluster.sweep_unreachable()
        return
    results = await asyncio.gather(
        *[_pull_one_peer(client, m) for m in targets],
        return_exceptions=True,
    )
    for member, ok in zip(targets, results):
        if isinstance(ok, BaseException):
            log.debug("peer pull task raised: %s", ok)
            continue
        if ok:
            # Re-stamp last_seen via update_member so the entry stays
            # fresh even if the peer's response did not include itself
            # in its own member list (defensive — it should).
            try:
                cluster.update_member(
                    member.node_id,
                    hostname=member.hostname,
                    address=member.address,
                    port=member.port,
                    cluster_name=member.cluster_name,
                    forgather_version=member.forgather_version,
                    source="peer_pull",
                )
            except Exception:
                log.exception(
                    "failed to refresh peer %s after successful pull",
                    member.node_id,
                )
    cluster.sweep_unreachable()


async def membership_loop(
    *, tick_seconds: Optional[float] = None
) -> None:
    """Run the peer-pull loop until cancelled.

    Started from ``app.py:lifespan`` when cluster mode is active.
    """
    interval = tick_seconds if tick_seconds is not None else TICK_SECONDS
    log.info("cluster membership loop starting (tick=%.1fs)", interval)
    async with httpx.AsyncClient() as client:
        try:
            while True:
                try:
                    await _tick(client)
                except Exception:
                    # The loop must survive any single-tick failure —
                    # otherwise a transient httpx bug silently kills
                    # liveness for the rest of the server's uptime.
                    log.exception("cluster membership tick failed")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            log.info("cluster membership loop cancelled")
            raise
