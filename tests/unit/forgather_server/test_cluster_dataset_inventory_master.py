"""
Tests for the master-side aggregation layer of
:mod:`forgather_server.cluster_dataset_inventory`.

Three concerns:

1. **MasterInventory state machine** — set_master_state transitions,
   merge_servers preserves health/dataset data, resolve() picks
   correctly across local/<name> vs HF/path requests.
2. **The tick functions** — collect/health/refresh against an httpx
   transport whose responses we control byte-for-byte.
3. **The role-change hook in cluster_membership** — listeners are
   fired on master_id transitions.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import httpx
import pytest

from forgather_server import cluster, cluster_dataset_inventory, cluster_membership
from forgather_server.cluster_dataset_inventory import (
    LocalServer,
    MasterInventory,
    MasterServerEntry,
    master_inventory,
)


@pytest.fixture(autouse=True)
def isolated_cluster_state(tmp_path, monkeypatch):
    from forgather_server import paths

    cluster_dir = tmp_path / "cluster"
    cluster_dir.mkdir()
    monkeypatch.setattr(paths, "cluster_state_dir", lambda: cluster_dir)
    monkeypatch.setattr(
        paths, "cluster_node_id_file", lambda: cluster_dir / "node_id"
    )
    cluster._reset_for_tests()
    cluster_dataset_inventory._reset_master_state_for_tests()
    cluster_membership._reset_role_listeners_for_tests()
    yield


# ---------------------------------------------------------------------------
# MasterInventory state machine
# ---------------------------------------------------------------------------


class TestMasterInventoryStateMachine:
    def test_initial_state(self):
        inv = MasterInventory()
        assert inv.is_master() is False
        assert inv.is_warmed_up() is False
        assert inv.servers_snapshot() == []
        assert inv.resolve("local/x") is None

    def test_become_master_clears_state(self):
        inv = MasterInventory()
        # Seed some stale state (simulating a prior tenure).
        inv.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://x:8766",
                    auth_token="",
                    label="x",
                    source="local",
                    peer_node_id=None,
                    healthy=True,
                )
            }
        )
        inv.set_master_state(False)  # we weren't actually master yet
        # Now do the actual transition:
        inv.set_master_state(True)
        assert inv.is_master() is True
        assert inv.servers_snapshot() == []
        assert inv.is_warmed_up() is False

    def test_lose_master_role_clears_state(self):
        inv = MasterInventory()
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://x:8766",
                    auth_token="",
                    label="x",
                    source="local",
                    peer_node_id=None,
                )
            }
        )
        inv.set_master_state(False)
        assert inv.is_master() is False
        assert inv.servers_snapshot() == []

    def test_merge_preserves_health_and_datasets(self):
        inv = MasterInventory()
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://x:8766",
                    auth_token="t",
                    label="x",
                    source="local",
                    peer_node_id="peer-1",
                )
            }
        )
        inv.update_health("s1", healthy=True)
        inv.update_datasets(
            "s1",
            handles=[{"handle": "h1", "source": "hf"}],
            locals_info=[{"name": "stories"}],
        )
        # Second merge with the same server set — should preserve.
        inv.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://x:8766",
                    auth_token="t",
                    label="x",
                    source="local",
                    peer_node_id="peer-1",
                )
            }
        )
        s = inv.get_server("s1")
        assert s is not None
        assert s.healthy is True
        assert s.available_keys == ["local/stories"]
        assert s.handles == [{"handle": "h1", "source": "hf"}]

    def test_merge_carries_polling_counters(self):
        """The poll counters are owned by the master's loops, not the
        peer collect tick. A naive merge that drops them would reset
        them to 0 every COLLECT_INTERVAL_SECONDS, making them useless
        for diagnosis. Regression: caught while smoke-testing the
        Phase 7 metrics surface — verbose CLI was always printing
        ``polls: health=0/0 failed`` even on a long-running master."""
        inv = MasterInventory()
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://x:8766",
                    auth_token="",
                    label="x",
                    source="local",
                    peer_node_id=None,
                )
            }
        )
        # Simulate a few health passes + one dataset pass.
        inv.update_health("s1", healthy=False, error="boom")
        inv.update_health("s1", healthy=False, error="boom")
        inv.update_health("s1", healthy=True)
        inv.update_datasets("s1", handles=[], locals_info=[])
        s_before = inv.get_server("s1")
        assert s_before.total_health_polls == 3
        assert s_before.health_failures == 2
        assert s_before.consecutive_health_failures == 0  # last poll healthy
        assert s_before.total_dataset_polls == 1

        # Next collect tick replays the server set with a fresh
        # MasterServerEntry (matching what _collect_servers_tick does).
        # The counters must survive.
        inv.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://x:8766",
                    auth_token="",
                    label="x",
                    source="local",
                    peer_node_id=None,
                )
            }
        )
        s_after = inv.get_server("s1")
        assert s_after.total_health_polls == 3
        assert s_after.health_failures == 2
        assert s_after.total_dataset_polls == 1

    def test_is_warmed_up_requires_observed_healthy_server(self):
        """CQ-7: warm-up gating must require ≥1 healthy server seen.

        Without this, a transient zero-server window (no servers in
        the inventory yet, or every server simultaneously DOWN) flips
        warm-up True with empty available_keys, and the router then
        returns 410 ("no candidate") to clients — which the resilient
        client treats as fatal and aborts training. The right behavior
        is to keep returning 503 (transient) until at least one
        healthy server has been observed during this master tenure."""
        inv = MasterInventory()
        inv.set_master_state(True)
        # Pass with zero healthy servers: should NOT count as warm.
        inv.mark_dataset_pass_complete()
        assert inv.is_warmed_up() is False
        # Now observe a healthy server. Warm-up flips True.
        inv.mark_observed_healthy()
        assert inv.is_warmed_up() is True
        # And stays True even after a subsequent pass that finds zero
        # healthy servers — once we've seen one work, "everything DOWN"
        # is transient, not the cluster being unprovisioned.
        inv.mark_dataset_pass_complete()
        assert inv.is_warmed_up() is True

    def test_warm_up_resets_on_role_transition(self):
        """Becoming master after losing the role must clear the
        observed-healthy latch — the new tenure starts fresh."""
        inv = MasterInventory()
        inv.set_master_state(True)
        inv.mark_observed_healthy()
        inv.mark_dataset_pass_complete()
        assert inv.is_warmed_up() is True
        # Lose master.
        inv.set_master_state(False)
        # Become master again.
        inv.set_master_state(True)
        assert inv.is_warmed_up() is False

    def test_merge_drops_servers_no_longer_reported(self):
        inv = MasterInventory()
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1", base_url="http://x:8766",
                    auth_token="", label="x", source="local",
                    peer_node_id=None,
                ),
                "s2": MasterServerEntry(
                    server_id="s2", base_url="http://y:8766",
                    auth_token="", label="y", source="local",
                    peer_node_id=None,
                ),
            }
        )
        inv.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1", base_url="http://x:8766",
                    auth_token="", label="x", source="local",
                    peer_node_id=None,
                )
            }
        )
        ids = {s.server_id for s in inv.servers_snapshot()}
        assert ids == {"s1"}


# ---------------------------------------------------------------------------
# resolve() — routing strategy
# ---------------------------------------------------------------------------


def _seed_inventory(servers):
    inv = MasterInventory()
    inv.set_master_state(True)
    inv.merge_servers({s.server_id: s for s in servers})
    return inv


def _make_entry(*, server_id, base_url, healthy=True, locals_names=()):
    e = MasterServerEntry(
        server_id=server_id,
        base_url=base_url,
        auth_token="tok-" + server_id,
        label=server_id,
        source="local",
        peer_node_id=None,
        healthy=healthy,
    )
    e.available_keys = [f"local/{n}" for n in locals_names]
    return e


class TestResolve:
    """``resolve()`` now returns a MasterServerEntry (or None) so the
    route handler can read ``server_id`` without re-walking the
    snapshot. Tests assert on entry fields rather than the old
    ``(base_url, token)`` tuple shape."""

    def test_local_path_only_servers_with_that_name(self):
        a = _make_entry(server_id="a", base_url="http://a:8766",
                        locals_names=["stories"])
        b = _make_entry(server_id="b", base_url="http://b:8766",
                        locals_names=["other"])
        inv = _seed_inventory([a, b])
        pick = inv.resolve("local/stories")
        assert pick is not None
        assert pick.base_url == "http://a:8766"
        assert pick.auth_token == "tok-a"
        assert pick.server_id == "a"

    def test_local_returns_none_when_no_match(self):
        a = _make_entry(server_id="a", base_url="http://a:8766",
                        locals_names=["other"])
        inv = _seed_inventory([a])
        assert inv.resolve("local/missing") is None

    def test_local_picks_at_random_across_replicas(self):
        """Two servers advertising the same local/<name> should be
        treated as interchangeable replicas (the documented redundancy
        story). Verify the resolver picks both over many calls."""
        a = _make_entry(server_id="a", base_url="http://a:8766",
                        locals_names=["stories"])
        b = _make_entry(server_id="b", base_url="http://b:8766",
                        locals_names=["stories"])
        inv = _seed_inventory([a, b])
        urls = {inv.resolve("local/stories").base_url for _ in range(40)}
        # Statistically vanishingly unlikely to be wrong; >40 calls of
        # uniform-random pick from {a, b} hits both with overwhelming
        # probability. The test's failure mode (single URL) is the bug.
        assert urls == {"http://a:8766", "http://b:8766"}

    def test_local_skips_unhealthy(self):
        a = _make_entry(server_id="a", base_url="http://a:8766",
                        locals_names=["stories"], healthy=False)
        b = _make_entry(server_id="b", base_url="http://b:8766",
                        locals_names=["stories"], healthy=True)
        inv = _seed_inventory([a, b])
        for _ in range(20):
            pick = inv.resolve("local/stories")
            assert pick is not None
            assert pick.base_url == "http://b:8766"

    def test_hf_path_picks_any_healthy_server(self):
        """For non-local requests, the master picks any healthy server
        — the server loads on demand. The client's resilient backend
        retries elsewhere if the chosen server can't actually serve
        the dataset."""
        a = _make_entry(server_id="a", base_url="http://a:8766",
                        locals_names=["foo"])
        b = _make_entry(server_id="b", base_url="http://b:8766",
                        locals_names=["bar"])
        inv = _seed_inventory([a, b])
        urls = {inv.resolve("allenai/c4").base_url for _ in range(40)}
        assert urls == {"http://a:8766", "http://b:8766"}

    def test_hf_path_returns_none_when_no_healthy_server(self):
        a = _make_entry(server_id="a", base_url="http://a:8766", healthy=False)
        inv = _seed_inventory([a])
        assert inv.resolve("allenai/c4") is None

    def test_loopback_entries_excluded_from_routing(self):
        """Loopback entries appear in the inventory (the Servers
        panel shows them) but ``resolve()`` skips them — other
        cluster peers can't reach a node-local URL even if the
        server is healthy from the master's POV."""
        # Build a loopback entry and a normal one, both with
        # local/stories. Only the non-loopback should be returned.
        loopback = _make_entry(
            server_id="lb",
            base_url="http://127.0.0.1:8766",
            locals_names=["stories"],
        )
        loopback.loopback = True
        normal = _make_entry(
            server_id="lan",
            base_url="http://10.0.0.5:8766",
            locals_names=["stories"],
        )
        inv = _seed_inventory([loopback, normal])
        for _ in range(20):
            pick = inv.resolve("local/stories")
            assert pick is not None
            assert pick.base_url == "http://10.0.0.5:8766"

    def test_loopback_only_returns_none(self):
        """If the only healthy server is loopback, the router returns
        None — i.e. surfaces 410 to clients. Loopback entries are
        never cluster-routable, regardless of how many of them there
        are."""
        loopback = _make_entry(
            server_id="lb",
            base_url="http://127.0.0.1:8766",
            locals_names=["stories"],
        )
        loopback.loopback = True
        inv = _seed_inventory([loopback])
        assert inv.resolve("local/stories") is None
        assert inv.resolve("allenai/c4") is None


# ---------------------------------------------------------------------------
# Tick functions exercised via mocked httpx transport
# ---------------------------------------------------------------------------


def _mock_transport(handler):
    """Build a httpx.AsyncClient whose every request is dispatched to
    ``handler(request) -> httpx.Response``."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _activate_cluster_as_master(node_id="self-node"):
    """Activate cluster with us as the only (and therefore master)
    member. ``cluster.activate`` generates a UUID node_id; we patch
    the lookup so it's deterministic for assertions."""
    # cluster.activate persists a node_id on disk; the
    # isolated_cluster_state fixture relocates the storage dir so
    # this stays a clean slate.
    cluster.activate("c", port=8765)


class TestCollectServersTick:
    def test_includes_local_servers(self, monkeypatch):
        _activate_cluster_as_master()
        cluster_dataset_inventory.master_inventory.set_master_state(True)

        local = LocalServer(
            server_id="abc",
            base_url="http://node-a:8766",
            auth_token="tok",
            label="x",
            source="local",
            peer_node_id="self",
        )
        monkeypatch.setattr(
            cluster_dataset_inventory, "local_servers", lambda: [local]
        )
        # No peers — only the local server should land in inventory.

        def handler(req):
            return httpx.Response(404)

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_dataset_inventory._collect_servers_tick(client)

        asyncio.run(go())
        entries = master_inventory.servers_snapshot()
        assert len(entries) == 1
        assert entries[0].server_id == "abc"
        assert entries[0].auth_token == "tok"

    def test_fans_to_peers(self, monkeypatch):
        _activate_cluster_as_master()
        master_inventory.set_master_state(True)

        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer-1",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
        )
        monkeypatch.setattr(
            cluster_dataset_inventory, "local_servers", lambda: []
        )

        def handler(req):
            assert req.url.path == "/api/cluster/dataset_servers_local"
            return httpx.Response(
                200,
                json={
                    "self_node_id": peer_id,
                    "servers": [
                        {
                            "server_id": "remote-srv",
                            "base_url": "http://peer-host:8766",
                            "auth_token": "remote-tok",
                            "label": "peer-cfg",
                            "source": "local",
                            "peer_node_id": peer_id,
                        }
                    ],
                },
            )

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_dataset_inventory._collect_servers_tick(client)

        asyncio.run(go())
        entries = master_inventory.servers_snapshot()
        assert len(entries) == 1
        assert entries[0].server_id == "remote-srv"
        assert entries[0].auth_token == "remote-tok"
        assert entries[0].peer_node_id == peer_id


class TestHealthTick:
    def test_marks_unreachable_on_network_error(self):
        _activate_cluster_as_master()
        master_inventory.set_master_state(True)
        master_inventory.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://dead:8766",
                    auth_token="",
                    label="x",
                    source="local",
                    peer_node_id=None,
                    healthy=True,  # starts healthy
                )
            }
        )

        def handler(req):
            raise httpx.ConnectError("simulated unreachable")

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_dataset_inventory._health_tick(client)

        asyncio.run(go())
        s = master_inventory.get_server("s1")
        assert s.healthy is False
        assert "simulated unreachable" in s.last_health_error

    def test_marks_healthy_on_200(self):
        master_inventory.set_master_state(True)
        master_inventory.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://alive:8766",
                    auth_token="",
                    label="x",
                    source="local",
                    peer_node_id=None,
                )
            }
        )

        def handler(req):
            assert req.url.path == "/v1/health"
            return httpx.Response(200, json={"status": "ok"})

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_dataset_inventory._health_tick(client)

        asyncio.run(go())
        s = master_inventory.get_server("s1")
        assert s.healthy is True
        assert s.last_health_error == ""

    def test_health_pass_complete_flag_set(self):
        master_inventory.set_master_state(True)

        def handler(req):
            return httpx.Response(200, json={"status": "ok"})

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_dataset_inventory._health_tick(client)

        asyncio.run(go())
        # Even with zero servers, the pass should be marked complete.
        assert master_inventory.status()["last_health_pass_ts"] is not None


class TestDatasetRefreshTick:
    def test_collects_datasets_and_locals(self):
        master_inventory.set_master_state(True)
        master_inventory.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://alive:8766",
                    auth_token="tok",
                    label="x",
                    source="local",
                    peer_node_id=None,
                    healthy=True,
                )
            }
        )

        def handler(req):
            assert req.headers["authorization"] == "Bearer tok"
            if req.url.path == "/v1/datasets":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "handle": "h1",
                            "source": "hf",
                            "load_args": {"path": "allenai/c4"},
                            "length": 100,
                        }
                    ],
                )
            if req.url.path == "/v1/local":
                return httpx.Response(
                    200,
                    json=[{"name": "stories", "length": 20}],
                )
            if req.url.path == "/v1/cache/hf":
                # Master also polls /v1/cache/hf so the Cluster tab
                # can list HF repos that are *available* (cached on
                # disk) even before any client has triggered /v1/load.
                return httpx.Response(
                    200,
                    json={
                        "cache_root": "/cache",
                        "datasets": [
                            {
                                "repo": "allenai/c4",
                                "size_bytes": 1024,
                                "configs": [
                                    {
                                        "config": "en",
                                        "splits": [
                                            {
                                                "name": "train",
                                                "num_examples": 100,
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    },
                )
            return httpx.Response(404)

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_dataset_inventory._dataset_refresh_tick(client)

        asyncio.run(go())
        s = master_inventory.get_server("s1")
        assert s.last_dataset_error == ""
        assert s.available_keys == ["local/stories"]
        assert s.handles == [
            {
                "handle": "h1",
                "source": "hf",
                "load_args": {"path": "allenai/c4"},
                "length": 100,
            }
        ]
        # The /v1/cache/hf poll populates ``hf_cache`` with the cached
        # repo set — surfaced under each repo's path in the cluster
        # inventory response.
        assert len(s.hf_cache) == 1
        assert s.hf_cache[0]["repo"] == "allenai/c4"
        assert master_inventory.is_warmed_up() is True

    def test_skips_unhealthy_servers(self):
        master_inventory.set_master_state(True)
        master_inventory.merge_servers(
            {
                "s1": MasterServerEntry(
                    server_id="s1",
                    base_url="http://alive:8766",
                    auth_token="",
                    label="x",
                    source="local",
                    peer_node_id=None,
                    healthy=False,
                )
            }
        )

        called = []

        def handler(req):
            called.append(req.url.path)
            return httpx.Response(200, json=[])

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_dataset_inventory._dataset_refresh_tick(client)

        asyncio.run(go())
        assert called == []  # no HTTP issued
        # Warm-up gate stays False until ≥1 healthy server has been
        # observed. The dataset-pass timestamp updates (so loops
        # advance), but ``is_warmed_up()`` is gated on
        # ``_ever_observed_healthy`` — keeps the router returning
        # 503 (try again later) instead of 410 (no candidate) when
        # the cluster is transiently empty.
        assert master_inventory.is_warmed_up() is False
        # The pass timestamp DID advance so the next refresh tick
        # sleeps on the steady-state interval rather than the cold-
        # start one (decoupling of "pass complete" from "warmed up").
        assert master_inventory.status()["last_dataset_pass_ts"] is not None


# ---------------------------------------------------------------------------
# Role-change listener hook
# ---------------------------------------------------------------------------


class TestLocalCollisions:
    """``local/<name>`` collision detection: two servers reporting
    the same name with different meta_hash values are NOT
    content-equivalent. The master surfaces this so operators can
    fix the config — the router otherwise silently load-balances
    between distinct datasets, which is the worst kind of bug."""

    def test_no_collision_when_hashes_match(self):
        inv = MasterInventory()
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "a": MasterServerEntry(
                    server_id="a", base_url="http://a", auth_token="",
                    label="a", source="local", peer_node_id=None,
                ),
                "b": MasterServerEntry(
                    server_id="b", base_url="http://b", auth_token="",
                    label="b", source="local", peer_node_id=None,
                ),
            }
        )
        # Both servers report local/stories with the SAME meta_hash —
        # genuine content-equivalent replicas, no collision.
        inv.update_datasets(
            "a",
            handles=[],
            locals_info=[{"name": "stories", "meta_hash": "h1"}],
        )
        inv.update_datasets(
            "b",
            handles=[],
            locals_info=[{"name": "stories", "meta_hash": "h1"}],
        )
        assert inv.local_collisions() == {}

    def test_collision_detected_across_servers(self):
        inv = MasterInventory()
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "a": MasterServerEntry(
                    server_id="a", base_url="http://a", auth_token="",
                    label="a", source="local", peer_node_id=None,
                ),
                "b": MasterServerEntry(
                    server_id="b", base_url="http://b", auth_token="",
                    label="b", source="local", peer_node_id=None,
                ),
            }
        )
        # Same name, DIFFERENT meta_hashes — operators named distinct
        # datasets the same thing on different nodes.
        inv.update_datasets(
            "a",
            handles=[],
            locals_info=[{"name": "stories", "meta_hash": "h1"}],
        )
        inv.update_datasets(
            "b",
            handles=[],
            locals_info=[{"name": "stories", "meta_hash": "h2"}],
        )
        collisions = inv.local_collisions()
        assert "stories" in collisions
        assert sorted(collisions["stories"]) == ["h1", "h2"]

    def test_warn_is_one_shot_per_tenure(self, caplog):
        import logging as _logging

        caplog.set_level(_logging.WARNING)
        inv = MasterInventory()
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "a": MasterServerEntry(
                    server_id="a", base_url="http://a", auth_token="",
                    label="a", source="local", peer_node_id=None,
                ),
                "b": MasterServerEntry(
                    server_id="b", base_url="http://b", auth_token="",
                    label="b", source="local", peer_node_id=None,
                ),
            }
        )
        inv.update_datasets(
            "a",
            handles=[],
            locals_info=[{"name": "stories", "meta_hash": "h1"}],
        )
        inv.update_datasets(
            "b",
            handles=[],
            locals_info=[{"name": "stories", "meta_hash": "h2"}],
        )
        # First call: emits one WARNING.
        inv.warn_new_collisions()
        first = sum(
            1
            for r in caplog.records
            if r.levelno == _logging.WARNING and "stories collision" in r.getMessage()
        )
        assert first == 1
        # Second call with no new collisions: silent.
        inv.warn_new_collisions()
        second = sum(
            1
            for r in caplog.records
            if r.levelno == _logging.WARNING and "stories collision" in r.getMessage()
        )
        assert second == 1


class TestMasterStateSync:
    """CQ-2 regression: every loop must reconcile the inventory's
    cached ``is_master()`` with the authoritative
    ``cluster.is_self_master()`` at the top of every tick, so a crash
    in one loop doesn't desync the others. ``_sync_master_state`` is
    the shared helper."""

    def test_sync_reflects_cluster_state(self):
        _activate_cluster_as_master()
        # We are master (lowest UUID among reachable). Sync should
        # flip the inventory cache to is_master=True.
        assert cluster_dataset_inventory._sync_master_state() is True
        assert master_inventory.is_master() is True

    def test_sync_clears_when_no_longer_master(self):
        _activate_cluster_as_master()
        cluster_dataset_inventory._sync_master_state()
        assert master_inventory.is_master() is True
        # Add a peer with a smaller UUID — we lose master.
        smaller = "00000000-0000-0000-0000-000000000000"
        cluster.update_member(
            smaller,
            hostname="peer",
            address="10.0.0.5",
            port=8765,
            cluster_name="c",
        )
        assert cluster_dataset_inventory._sync_master_state() is False
        assert master_inventory.is_master() is False

    def test_sync_clears_when_cluster_inactive(self):
        # Set up: we're master.
        _activate_cluster_as_master()
        cluster_dataset_inventory._sync_master_state()
        assert master_inventory.is_master() is True
        # Now deactivate the cluster (e.g., partition recovery).
        cluster._reset_for_tests()
        assert cluster_dataset_inventory._sync_master_state() is False
        assert master_inventory.is_master() is False


class TestWakeEvents:
    """The per-loop wake-event registry: each loop owns its own
    event so a single ``wake_loops()`` call from the membership
    listener fans out to all of them, not just whichever one
    consumed a shared event first."""

    def test_wake_loops_sets_all_registered_events(self):
        # Simulate a few loops registering their events at startup.
        import asyncio

        # asyncio.Event needs a running loop to set/get in Python
        # 3.10+. Use asyncio.run to provide one.
        async def go():
            cluster_dataset_inventory._wake_events.clear()
            e1 = cluster_dataset_inventory._register_wake_event()
            e2 = cluster_dataset_inventory._register_wake_event()
            e3 = cluster_dataset_inventory._register_wake_event()
            # None set yet.
            assert not e1.is_set()
            assert not e2.is_set()
            assert not e3.is_set()
            # Fire once.
            cluster_dataset_inventory.wake_loops()
            # All set — that's the fix; a shared single event would
            # only set one and any subsequent loop would miss the
            # signal.
            assert e1.is_set()
            assert e2.is_set()
            assert e3.is_set()

        asyncio.run(go())


class TestRoleChangeListener:
    def test_listener_fired_on_master_id_change(self):
        _activate_cluster_as_master()

        events: list = []
        cluster_membership.register_role_change_listener(
            lambda prev, new: events.append((prev, new))
        )

        # First tick observes us as the only (and therefore master)
        # member. That's a transition from None -> our_id.
        cluster_membership._notify_role_change_if_needed()
        assert len(events) == 1
        prev, new = events[0]
        assert prev is None
        assert new == cluster.self_identity().node_id

        # Calling again with no change: no event.
        cluster_membership._notify_role_change_if_needed()
        assert len(events) == 1

        # Introduce a peer with a "smaller" UUID so master flips.
        peer_id = "00000000-0000-0000-0000-000000000000"
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.0.0.5",
            port=8765,
            cluster_name="c",
        )
        cluster_membership._notify_role_change_if_needed()
        assert len(events) == 2
        prev, new = events[1]
        assert new == peer_id  # smaller UUID wins

    def test_listener_exception_does_not_break_membership(self):
        _activate_cluster_as_master()

        def angry(prev, new):
            raise RuntimeError("listener exploded")

        cluster_membership.register_role_change_listener(angry)
        # Should not raise — exceptions are caught + logged.
        cluster_membership._notify_role_change_if_needed()
