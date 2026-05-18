"""Tests for tools/forgather_server/cluster_membership.py.

Real HTTP hits would make these tests flaky and slow. We mock the
``httpx.AsyncClient.get`` call to return controlled JSON payloads and
verify that the merge + sweep logic does the right thing. We use
``asyncio.run`` rather than pytest-asyncio so no test-only dependency
needs to land on every dev's machine.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import forgather_server.cluster as cluster
import forgather_server.cluster_membership as cm
import httpx
import pytest
from forgather_server import paths


@pytest.fixture(autouse=True)
def isolated_cluster(tmp_path, monkeypatch):
    cluster_dir = tmp_path / "cluster"
    cluster_dir.mkdir()
    monkeypatch.setattr(paths, "cluster_state_dir", lambda: cluster_dir)
    monkeypatch.setattr(
        paths, "cluster_node_id_file", lambda: cluster_dir / "node_id"
    )
    cluster._reset_for_tests()
    yield cluster_dir
    cluster._reset_for_tests()


def _make_response(status_code=200, json_body=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body)
    return resp


def _client_with(get_impl):
    client = MagicMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=get_impl)
    return client


class TestPullOnePeer:
    def test_merges_peer_member_list(self):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
        )
        peer = next(m for m in cluster.members() if m.node_id == peer_id)
        third_id = str(uuid.uuid4())
        body = {
            "members": [
                {
                    "node_id": peer_id,
                    "hostname": "peer1",
                    "address": "10.0.0.7",
                    "port": 8765,
                    "cluster_name": "c",
                    "forgather_version": "1.1.0",
                },
                {
                    "node_id": third_id,
                    "hostname": "peer2",
                    "address": "10.0.0.8",
                    "port": 8765,
                    "cluster_name": "c",
                    "forgather_version": "1.1.0",
                },
            ]
        }

        async def fake_get(url, timeout=None):
            return _make_response(200, body)

        async def go():
            client = _client_with(fake_get)
            return await cm._pull_one_peer(client, peer)

        ok = asyncio.run(go())
        assert ok is True
        ids = {m.node_id for m in cluster.members()}
        assert third_id in ids
        third = next(m for m in cluster.members() if m.node_id == third_id)
        # Transitive entries from another peer's member list are
        # tagged ``peer_report`` — they tell us about identities the
        # polled peer knows about, but the polled peer is the only
        # one we actually heard from. The next peer-pull tick will
        # try to GET from this address directly and stamp a real
        # ``peer_pull`` source on success.
        assert third.last_source == "peer_report"
        assert third.address == "10.0.0.8"

    def test_returns_false_on_http_error(self):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
        )
        peer = next(m for m in cluster.members() if m.node_id == peer_id)

        async def fake_get(url, timeout=None):
            raise httpx.ConnectError("refused")

        async def go():
            client = _client_with(fake_get)
            return await cm._pull_one_peer(client, peer)

        assert asyncio.run(go()) is False

    def test_returns_false_on_non_200(self):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
        )
        peer = next(m for m in cluster.members() if m.node_id == peer_id)

        async def fake_get(url, timeout=None):
            return _make_response(503, {"detail": "down"})

        async def go():
            client = _client_with(fake_get)
            return await cm._pull_one_peer(client, peer)

        assert asyncio.run(go()) is False

    def test_skips_self_in_peer_payload(self):
        ident = cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
        )
        peer = next(m for m in cluster.members() if m.node_id == peer_id)
        body = {
            "members": [
                {
                    "node_id": ident.node_id,
                    "hostname": "i-am-the-loopback",
                    "address": "1.2.3.4",
                    "port": 9999,
                    "cluster_name": "c",
                    "forgather_version": "0.0.0",
                }
            ]
        }

        async def fake_get(url, timeout=None):
            return _make_response(200, body)

        async def go():
            client = _client_with(fake_get)
            await cm._pull_one_peer(client, peer)

        asyncio.run(go())
        self_m = next(
            m for m in cluster.members() if m.node_id == ident.node_id
        )
        assert self_m.last_source == "self"
        assert self_m.port == 8765

    def test_skips_foreign_cluster_in_payload(self):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
        )
        peer = next(m for m in cluster.members() if m.node_id == peer_id)
        alien = str(uuid.uuid4())
        body = {
            "members": [
                {
                    "node_id": alien,
                    "hostname": "alien",
                    "address": "10.0.0.99",
                    "port": 8765,
                    "cluster_name": "different",
                    "forgather_version": "1.1.0",
                }
            ]
        }

        async def fake_get(url, timeout=None):
            return _make_response(200, body)

        async def go():
            client = _client_with(fake_get)
            await cm._pull_one_peer(client, peer)

        asyncio.run(go())
        ids = {m.node_id for m in cluster.members()}
        assert alien not in ids


class TestTick:
    def test_successful_tick_keeps_peer_reachable(self):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
            now=0.0,
        )
        cluster._set_unreachable_after_for_tests(10.0)
        body = {
            "members": [
                {
                    "node_id": peer_id,
                    "hostname": "peer",
                    "address": "10.0.0.7",
                    "port": 8765,
                    "cluster_name": "c",
                    "forgather_version": "1.1.0",
                }
            ]
        }

        async def fake_get(url, timeout=None):
            return _make_response(200, body)

        async def go():
            client = _client_with(fake_get)
            await cm._tick(client)

        asyncio.run(go())
        peer = next(m for m in cluster.members() if m.node_id == peer_id)
        assert peer.reachable is True

    def test_failed_pulls_eventually_mark_unreachable(self):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
            now=0.0,
        )
        cluster._set_unreachable_after_for_tests(0.1)

        import time as _time

        original = _time.time
        try:
            _time.time = lambda: 9999.0  # type: ignore[assignment]

            async def fake_get(url, timeout=None):
                raise httpx.ConnectError("refused")

            async def go():
                client = _client_with(fake_get)
                await cm._tick(client)

            asyncio.run(go())
        finally:
            _time.time = original
        peer = next(m for m in cluster.members() if m.node_id == peer_id)
        assert peer.reachable is False

    def test_tick_with_no_peers_is_noop(self):
        cluster.activate("c", port=8765)

        async def fake_get(url, timeout=None):
            raise AssertionError("should not be called with no peers")

        async def go():
            client = _client_with(fake_get)
            await cm._tick(client)

        asyncio.run(go())  # must not raise


class TestLoopRobustness:
    def test_loop_survives_tick_exception(self, monkeypatch):
        cluster.activate("c", port=8765)
        ticks = []

        async def boom(client):
            ticks.append(1)
            if len(ticks) == 1:
                raise RuntimeError("first tick fails")

        monkeypatch.setattr(cm, "_tick", boom)

        async def go():
            task = asyncio.create_task(cm.membership_loop(tick_seconds=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(go())
        assert len(ticks) >= 2
