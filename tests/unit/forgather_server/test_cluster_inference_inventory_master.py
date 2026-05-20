"""
Tests for the master-side aggregation layer of
:mod:`forgather_server.cluster_inference_inventory`.

Parallel to ``test_cluster_dataset_inventory_master.py``. Three
concerns:

1. **Tick functions** — ``_collect_servers_tick`` and ``_health_tick``
   driven against a mocked httpx transport so we can assert exactly
   which URLs the master polls and how it reacts to each response.
2. **State synchronization** — ``_sync_master_state`` follows
   ``cluster.is_self_master()``.
3. **Wake events** — registered events fire on ``wake_loops()``.
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from forgather_server import cluster, cluster_inference_inventory, cluster_membership
from forgather_server.cluster_inference_inventory import (
    LocalInference,
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
    cluster_inference_inventory._reset_master_state_for_tests()
    cluster_membership._reset_role_listeners_for_tests()
    yield


def _mock_transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _activate_cluster_as_master():
    cluster.activate("c", port=8765)


# ---------------------------------------------------------------------------
# _collect_servers_tick
# ---------------------------------------------------------------------------


class TestCollectServersTick:
    def test_includes_local_servers(self, monkeypatch):
        _activate_cluster_as_master()
        master_inventory.set_master_state(True)
        local = LocalInference(
            server_id="abc",
            base_url="http://node-a:8137",
            auth_token="tok",
            label="inference:8137",
            peer_node_id="self",
            source_id="q1",
            models=["llama"],
        )
        monkeypatch.setattr(
            cluster_inference_inventory, "local_servers", lambda: [local]
        )

        def handler(req):
            return httpx.Response(404)  # no peers to poll

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_inference_inventory._collect_servers_tick(client)

        asyncio.run(go())
        entries = master_inventory.servers_snapshot()
        assert len(entries) == 1
        e = entries[0]
        assert e.server_id == "abc"
        assert e.auth_token == "tok"
        assert e.models == ["llama"]

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
            source="peer_pull",
        )
        monkeypatch.setattr(
            cluster_inference_inventory, "local_servers", lambda: []
        )

        seen_paths: list = []

        def handler(req):
            seen_paths.append(req.url.path)
            assert req.url.path == "/api/cluster/inference_servers_local"
            return httpx.Response(
                200,
                json={
                    "self_node_id": peer_id,
                    "servers": [
                        {
                            "server_id": "remote-srv",
                            "base_url": "http://peer-host:8137",
                            "auth_token": "remote-tok",
                            "label": "inference:8137",
                            "peer_node_id": peer_id,
                            "source_id": "q-remote",
                            "loopback": False,
                            "models": ["qwen"],
                        }
                    ],
                },
            )

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_inference_inventory._collect_servers_tick(client)

        asyncio.run(go())
        assert seen_paths  # peer was polled
        entries = master_inventory.servers_snapshot()
        assert len(entries) == 1
        e = entries[0]
        assert e.server_id == "remote-srv"
        assert e.auth_token == "remote-tok"
        assert e.peer_node_id == peer_id
        assert e.models == ["qwen"]

    def test_peer_error_is_tolerated(self, monkeypatch):
        """A flaky peer returning 500 / connection error must not
        prevent local servers from landing in the inventory."""
        _activate_cluster_as_master()
        master_inventory.set_master_state(True)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer-x",
            address="10.0.0.99",
            port=8765,
            cluster_name="c",
            source="peer_pull",
        )
        local = LocalInference(
            server_id="local-srv",
            base_url="http://node-a:8137",
            auth_token="tok",
            label="x",
            peer_node_id="self",
            source_id="q1",
            models=[],
        )
        monkeypatch.setattr(
            cluster_inference_inventory, "local_servers", lambda: [local]
        )

        def handler(req):
            raise httpx.ConnectError("simulated")

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_inference_inventory._collect_servers_tick(client)

        asyncio.run(go())
        entries = master_inventory.servers_snapshot()
        assert [e.server_id for e in entries] == ["local-srv"]


# ---------------------------------------------------------------------------
# _health_tick
# ---------------------------------------------------------------------------


def _seed(server_id="s1", base_url="http://x:8137", healthy=False):
    master_inventory.merge_servers(
        {
            server_id: MasterServerEntry(
                server_id=server_id,
                base_url=base_url,
                auth_token="",
                label="x",
                peer_node_id=None,
                healthy=healthy,
            )
        }
    )


class TestHealthTick:
    def test_marks_healthy_on_200(self):
        master_inventory.set_master_state(True)
        _seed(base_url="http://alive:8137")

        seen_paths: list = []

        def handler(req):
            seen_paths.append(req.url.path)
            return httpx.Response(200, json={"status": "ok"})

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_inference_inventory._health_tick(client)

        asyncio.run(go())
        # Inference servers expose ``/health`` at the root, NOT
        # ``/v1/health`` (that's the dataset side). Verify the
        # tick targets the right path — getting this wrong would
        # silently mark every server unhealthy in production.
        assert seen_paths == ["/health"]
        s = master_inventory.get_server("s1")
        assert s.healthy is True
        assert s.last_health_error == ""
        assert s.total_health_polls == 1
        assert s.consecutive_health_failures == 0

    def test_marks_unhealthy_on_non_200(self):
        master_inventory.set_master_state(True)
        _seed(base_url="http://broken:8137", healthy=True)

        def handler(req):
            return httpx.Response(500, text="boom")

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_inference_inventory._health_tick(client)

        asyncio.run(go())
        s = master_inventory.get_server("s1")
        assert s.healthy is False
        assert "500" in s.last_health_error
        assert s.consecutive_health_failures == 1

    def test_marks_unreachable_on_network_error(self):
        master_inventory.set_master_state(True)
        _seed(base_url="http://dead:8137", healthy=True)

        def handler(req):
            raise httpx.ConnectError("simulated")

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_inference_inventory._health_tick(client)

        asyncio.run(go())
        s = master_inventory.get_server("s1")
        assert s.healthy is False
        assert "simulated" in s.last_health_error

    def test_consecutive_failures_reset_on_recovery(self):
        master_inventory.set_master_state(True)
        _seed(base_url="http://flaky:8137")
        # Two failed ticks.
        for _ in range(2):

            def handler(req):
                raise httpx.ConnectError("down")

            async def go():
                async with _mock_transport(handler) as client:
                    await cluster_inference_inventory._health_tick(client)

            asyncio.run(go())
        assert master_inventory.get_server("s1").consecutive_health_failures == 2

        # Recovery — counter resets, totals keep climbing.
        def good_handler(req):
            return httpx.Response(200)

        async def go():
            async with _mock_transport(good_handler) as client:
                await cluster_inference_inventory._health_tick(client)

        asyncio.run(go())
        s = master_inventory.get_server("s1")
        assert s.consecutive_health_failures == 0
        assert s.health_failures == 2  # cumulative kept
        assert s.total_health_polls == 3

    def test_pass_complete_flag_set_even_with_zero_servers(self):
        master_inventory.set_master_state(True)

        def handler(req):
            return httpx.Response(200)

        async def go():
            async with _mock_transport(handler) as client:
                await cluster_inference_inventory._health_tick(client)

        asyncio.run(go())
        assert master_inventory.status()["last_health_pass_ts"] is not None


# ---------------------------------------------------------------------------
# _sync_master_state — every loop calls this at the top of every tick
# ---------------------------------------------------------------------------


class TestMasterStateSync:
    def test_sync_true_when_self_is_master(self):
        _activate_cluster_as_master()
        assert cluster_inference_inventory._sync_master_state() is True
        assert master_inventory.is_master() is True

    def test_sync_false_when_cluster_inactive(self):
        # cluster.is_active() is False before activate(). The sync
        # path must report False AND clear any stale master state.
        master_inventory.set_master_state(True)
        master_inventory.merge_servers({})  # no-op but exercises the API
        assert cluster_inference_inventory._sync_master_state() is False
        assert master_inventory.is_master() is False

    def test_sync_clears_when_no_longer_master(self, monkeypatch):
        _activate_cluster_as_master()
        cluster_inference_inventory._sync_master_state()
        assert master_inventory.is_master() is True
        # Simulate another node being elected master.
        monkeypatch.setattr(cluster, "is_self_master", lambda: False)
        assert cluster_inference_inventory._sync_master_state() is False
        assert master_inventory.is_master() is False


# ---------------------------------------------------------------------------
# Wake events
# ---------------------------------------------------------------------------


class TestWakeEvents:
    def test_wake_loops_sets_all_registered_events(self):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            ev_a = cluster_inference_inventory._register_wake_event()
            ev_b = cluster_inference_inventory._register_wake_event()
            assert not ev_a.is_set()
            assert not ev_b.is_set()
            cluster_inference_inventory.wake_loops()
            assert ev_a.is_set()
            assert ev_b.is_set()
        finally:
            cluster_inference_inventory._wake_events.clear()
            loop.close()
            asyncio.set_event_loop(None)
