"""Tests for the DiLoCo-server cluster endpoints in routes/cluster.py.

Parallel to :mod:`test_routes_cluster_inference`, exercising the new
DiLoCo variants:

  - ``GET /api/cluster/diloco_servers_local`` (per-peer, tokens
    included, peer-mTLS carve-out);
  - ``GET /api/cluster/diloco_servers`` (master-aggregated, includes
    tokens — same shape as inference; see the model docstring);
  - ``POST /api/cluster/diloco_servers/refresh`` (wake hook).
"""

import forgather_server.auth as auth
import forgather_server.cluster as cluster
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from forgather_server import paths
from forgather_server.auth import AuthMiddleware
from forgather_server.routes import cluster as cluster_routes


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    cluster_dir = tmp_path / "cluster"
    cluster_dir.mkdir()
    journal_dir = cluster_dir / "journal"
    journal_dir.mkdir()
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    monkeypatch.setattr(paths, "cluster_state_dir", lambda: cluster_dir)
    monkeypatch.setattr(
        paths, "cluster_node_id_file", lambda: cluster_dir / "node_id"
    )
    monkeypatch.setattr(paths, "cluster_journal_dir", lambda: journal_dir)
    monkeypatch.setattr(paths, "server_state_dir", lambda: server_dir)
    monkeypatch.setattr(
        paths, "auth_token_file", lambda: server_dir / "auth_token"
    )
    monkeypatch.setattr(
        paths, "password_hash_file", lambda: server_dir / "password_hash"
    )
    from forgather_server import (
        cluster_dataset_inventory,
        cluster_diloco_inventory,
        cluster_inference_inventory,
        cluster_jobs,
        cluster_journal,
        cluster_membership,
    )

    cluster._reset_for_tests()
    cluster_jobs._reset_for_tests()
    cluster_journal._reset_for_tests()
    cluster_dataset_inventory._reset_master_state_for_tests()
    cluster_inference_inventory._reset_master_state_for_tests()
    cluster_diloco_inventory._reset_master_state_for_tests()
    cluster_membership._reset_role_listeners_for_tests()
    auth._reset_sessions_for_tests()
    yield


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(cluster_routes.router, prefix="/api")
    return app


class TestDiLoCoServersLocal:
    """``GET /diloco_servers_local`` returns the inventory module's
    view of this peer's DiLoCo servers; tokens included over the peer
    carve-out."""

    def _patch_inventory(self, monkeypatch, servers):
        from forgather_server import cluster_diloco_inventory

        monkeypatch.setattr(
            cluster_diloco_inventory, "local_servers", lambda: list(servers)
        )

    def test_returns_servers_with_tokens(self, monkeypatch):
        from forgather_server.cluster_diloco_inventory import LocalDiLoCo

        cluster.activate("c", port=8765)
        self._patch_inventory(
            monkeypatch,
            [
                LocalDiLoCo(
                    server_id="abc123",
                    base_url="http://node-a:8512",
                    auth_token="tok",
                    label="default:8512",
                    source="local",
                    peer_node_id=cluster.self_identity().node_id,
                    source_id="q1",
                    loopback=False,
                    verify_tls=True,
                )
            ],
        )
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/diloco_servers_local",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["self_node_id"] == cluster.self_identity().node_id
        assert len(body["servers"]) == 1
        s = body["servers"][0]
        assert s["base_url"] == "http://node-a:8512"
        # Token transit is the whole point of the endpoint.
        assert s["auth_token"] == "tok"
        assert s["source"] == "local"
        assert s["source_id"] == "q1"
        assert s["verify_tls"] is True
        assert (
            r.headers.get("x-forgather-node-id")
            == cluster.self_identity().node_id
        )

    def test_rejected_without_credentials(self, monkeypatch):
        cluster.activate("c", port=8765)
        self._patch_inventory(monkeypatch, [])
        client = TestClient(_make_app())
        # No bearer, no mTLS cert → 401 (the peer carve-out only opens
        # for mTLS-authenticated callers).
        r = client.get("/api/cluster/diloco_servers_local")
        assert r.status_code == 401


class TestDiLoCoServersAggregate:
    """``GET /diloco_servers`` returns the master's aggregated view.

    The model schema deliberately includes ``auth_token`` so the
    webui proxy can dial off-host upstreams without operator token
    handling. Pins that behavior so a future "strip tokens" change is
    a deliberate decision, not an accidental drift.
    """

    def _seed_master_inventory(self):
        from forgather_server import cluster_diloco_inventory as cdi

        cdi.master_inventory.set_master_state(True)
        cdi.master_inventory.merge_servers(
            {
                "srv-a": cdi.MasterServerEntry(
                    server_id="srv-a",
                    base_url="http://node-a:8512",
                    auth_token="tok-a",
                    label="default:8512",
                    source="local",
                    peer_node_id="peer-a",
                    source_id="q-a",
                    loopback=False,
                    healthy=True,
                ),
                "srv-b": cdi.MasterServerEntry(
                    server_id="srv-b",
                    base_url="https://gpu-box:8512",
                    auth_token="tok-b",
                    label="prod",
                    source="user",
                    peer_node_id="peer-b",
                    source_id="reg-1",
                    loopback=False,
                    verify_tls=False,
                    healthy=False,
                    last_health_error="conn refused",
                ),
            }
        )

    def test_master_returns_full_list_with_tokens(self):
        cluster.activate("c", port=8765)
        self._seed_master_inventory()
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/diloco_servers",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 2
        by_id = {s["server_id"]: s for s in body}
        # Deliberate token exposure — pinned by this test. If a future
        # change strips tokens server-side, update this assertion
        # together with the proxy's token-attach path so both halves
        # move together.
        assert by_id["srv-a"]["auth_token"] == "tok-a"
        assert by_id["srv-b"]["auth_token"] == "tok-b"
        # Per-source fields propagate.
        assert by_id["srv-a"]["source"] == "local"
        assert by_id["srv-b"]["source"] == "user"
        assert by_id["srv-b"]["verify_tls"] is False
        # Health surfaces as designed.
        assert by_id["srv-a"]["healthy"] is True
        assert by_id["srv-b"]["healthy"] is False
        assert by_id["srv-b"]["last_health_error"] == "conn refused"

    def test_non_master_with_no_reachable_master_returns_empty(self):
        """When this node isn't master and no master is reachable, the
        proxy returns an empty list rather than a 5xx — matches the
        dataset / inference variants' cold-start behavior."""
        cluster.activate("c", port=8765)
        # Don't set_master_state(True). No member entries, so
        # _master_member() returns None.
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/diloco_servers",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        assert r.json() == []


class TestDiLoCoServersRefresh:
    """``POST /diloco_servers/refresh`` wakes the inventory loops."""

    def test_wake_event_fires(self):
        from forgather_server import cluster_diloco_inventory as cdi

        cluster.activate("c", port=8765)

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            ev = cdi._register_wake_event()
            assert not ev.is_set()
            token = auth.load_token()
            client = TestClient(_make_app())
            r = client.post(
                "/api/cluster/diloco_servers/refresh",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert r.status_code == 204
            assert ev.is_set()
        finally:
            cdi._wake_events.clear()
            loop.close()
            asyncio.set_event_loop(None)

    def test_rejected_without_credentials(self):
        cluster.activate("c", port=8765)
        client = TestClient(_make_app())
        r = client.post("/api/cluster/diloco_servers/refresh")
        assert r.status_code == 401


class TestPeerAllowlistMembership:
    """The new DiLoCo endpoints must be in the mTLS peer allow-lists.

    Without these entries the master can't reach peers'
    ``diloco_servers_local`` (would 401 even with a valid client cert)
    and the ``/diloco_servers/refresh`` proxy-to-master would silently
    no-op.
    """

    def test_diloco_servers_local_is_peer_allowed_get(self):
        assert "/api/cluster/diloco_servers_local" in auth._PEER_ALLOWED_PATHS

    def test_diloco_servers_is_peer_allowed_get(self):
        assert "/api/cluster/diloco_servers" in auth._PEER_ALLOWED_PATHS

    def test_diloco_servers_refresh_is_peer_allowed_post(self):
        assert (
            "/api/cluster/diloco_servers/refresh"
            in auth._PEER_ALLOWED_MUTATIONS
        )
