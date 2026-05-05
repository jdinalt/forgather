"""Tests for routes/cluster.py and the peer-call auth carve-out.

Uses FastAPI's TestClient so the full middleware stack runs — that's
the only way to verify the carve-out is wired correctly.
"""

import uuid

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
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    monkeypatch.setattr(paths, "cluster_state_dir", lambda: cluster_dir)
    monkeypatch.setattr(
        paths, "cluster_node_id_file", lambda: cluster_dir / "node_id"
    )
    monkeypatch.setattr(paths, "server_state_dir", lambda: server_dir)
    monkeypatch.setattr(
        paths, "auth_token_file", lambda: server_dir / "auth_token"
    )
    monkeypatch.setattr(
        paths, "password_hash_file", lambda: server_dir / "password_hash"
    )
    cluster._reset_for_tests()
    auth._reset_sessions_for_tests()
    yield


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(cluster_routes.router, prefix="/api")
    return app


class TestEndpointShapes:
    def test_members_when_active(self):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.7",
            port=8765,
            cluster_name="c",
        )
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/members",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["cluster_name"] == "c"
        assert data["self_node_id"] == cluster.self_identity().node_id
        assert data["master_node_id"] is not None
        ids = {m["node_id"] for m in data["members"]}
        assert peer_id in ids

    def test_self_returns_null_when_inactive(self):
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json() is None

    def test_self_when_active(self):
        ident = cluster.activate("c", port=1234)
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["node_id"] == ident.node_id
        assert data["cluster_name"] == "c"
        assert data["port"] == 1234
        assert data["is_master"] is True  # only self in cluster

    def test_master_when_inactive(self):
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/master",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["cluster_active"] is False
        assert data["master_node_id"] is None
        assert data["is_self_master"] is False


class TestPeerCarveOut:
    def test_peer_get_allowed_without_token(self):
        # TestClient connects from 127.0.0.1 (varies by Starlette
        # version: testclient is the literal default). Add both
        # observed values as known peers so the carve-out fires.
        cluster.activate("c", port=8765)
        for addr in ("127.0.0.1", "testclient"):
            cluster.update_member(
                str(uuid.uuid4()),
                hostname=f"peer-{addr}",
                address=addr,
                port=8765,
                cluster_name="c",
            )
        client = TestClient(_make_app())
        r = client.get("/api/cluster/members")
        # No Authorization header. Must be accepted because we are a
        # "known peer" by source IP.
        assert r.status_code == 200, r.text

    def test_unknown_source_rejected(self):
        # Cluster active but no peer registered at 127.0.0.1 (or
        # whatever TestClient claims as source). Without a token, the
        # request must 401.
        cluster.activate("c", port=8765)
        # Wipe self's address out of the member table so nothing in
        # the table matches the testclient source.
        ident = cluster.self_identity()
        cluster._state._members[ident.node_id].address = "203.0.113.1"
        client = TestClient(_make_app())
        r = client.get("/api/cluster/members")
        assert r.status_code == 401

    def test_carve_out_inactive_when_cluster_off(self):
        # No activate() call. Even if the path is in the peer-allowed
        # list, the carve-out must not fire when cluster is not
        # active — otherwise these endpoints would be public on
        # standalone servers.
        client = TestClient(_make_app())
        r = client.get("/api/cluster/members")
        assert r.status_code == 401

    def test_post_not_carved_out(self):
        # The members endpoint is GET-only, but verify the principle
        # by adding a synthetic POST route under a peer-allowed path
        # (would only ever apply to a future mutating endpoint).
        cluster.activate("c", port=8765)
        cluster.update_member(
            str(uuid.uuid4()),
            hostname="peer",
            address="127.0.0.1",
            port=8765,
            cluster_name="c",
        )
        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.post("/api/cluster/members")
        async def fake_post():
            return {"ok": True}

        client = TestClient(app)
        r = client.post("/api/cluster/members")
        assert r.status_code == 401

    def test_normal_token_still_works(self):
        cluster.activate("c", port=8765)
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/members",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200


class TestClusterGpus:
    def test_inactive_returns_empty(self):
        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/gpus",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["nodes"] == []

    def test_self_short_circuits_no_network(self, monkeypatch):
        # Arrange a single-member (self) cluster and stub
        # ``gpu_monitor.snapshot`` so the aggregator can't accidentally
        # block on real NVML or hit the network for self.
        cluster.activate("c", port=8765)

        from forgather_server import gpu_monitor

        fake_gpu = gpu_monitor.GpuInfo(
            index=0,
            name="FakeGPU",
            total_mem_bytes=10_000_000_000,
            used_mem_bytes=1_000_000_000,
            util_pct=10,
            mem_util_pct=5,
            power_w=42.0,
            temp_c=50,
            fan_pct=20,
            processes=[],
            source="test",
            node="self",
            excluded=False,
            disabled=False,
            min_priority=0,
        )
        monkeypatch.setattr(gpu_monitor, "snapshot", lambda: [fake_gpu])

        token = auth.load_token()
        client = TestClient(_make_app())
        # Mount the gpus router so cluster_gpus's local fallback can
        # find _to_model — the import is already done at module load
        # but the test app does not need the gpus router mounted.
        r = client.get(
            "/api/cluster/gpus",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["nodes"]) == 1
        node = body["nodes"][0]
        assert node["reachable"] is True
        assert len(node["gpus"]) == 1
        assert node["gpus"][0]["name"] == "FakeGPU"

    def test_unreachable_peer_is_reported(self, monkeypatch):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.255.255.1",  # unroutable: forces a quick fail
            port=8765,
            cluster_name="c",
        )
        cluster.mark_unreachable(peer_id)

        # Stub local snapshot too so the test does not require NVML.
        from forgather_server import gpu_monitor

        monkeypatch.setattr(gpu_monitor, "snapshot", lambda: [])

        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.get(
            "/api/cluster/gpus",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        peers = [n for n in body["nodes"] if n["node_id"] == peer_id]
        assert len(peers) == 1
        assert peers[0]["reachable"] is False
        assert peers[0]["gpus"] == []
        assert peers[0]["error"]


class TestPathAllowsPeer:
    def test_known_peer_paths(self):
        assert auth.path_allows_peer("/api/cluster/members") is True
        assert auth.path_allows_peer("/api/cluster/self") is True
        assert auth.path_allows_peer("/api/cluster/master") is True

    def test_other_paths_not_carved_out(self):
        assert auth.path_allows_peer("/api/queue") is False
        assert auth.path_allows_peer("/api/gpus") is False
        assert auth.path_allows_peer("/api/cluster/anything-else") is False

    def test_gpus_local_carved_out(self):
        assert auth.path_allows_peer("/api/cluster/gpus_local") is True
