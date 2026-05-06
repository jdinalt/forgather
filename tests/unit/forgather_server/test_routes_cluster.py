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
    from forgather_server import cluster_jobs, cluster_journal

    cluster._reset_for_tests()
    cluster_jobs._reset_for_tests()
    cluster_journal._reset_for_tests()
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

    def test_post_on_get_only_path_not_carved_out(self):
        # /api/cluster/members is GET-only in the carve-out. POSTing
        # to it from a peer must still be rejected: the GET-allowed
        # set and the POST-allowed mutation set are disjoint.
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

    def test_post_on_mutation_path_carved_out(self):
        # /api/cluster/gpu_policy_local is in the explicit mutation
        # allow-list, so a POST from a known peer IP must pass the
        # gate without a token.
        cluster.activate("c", port=8765)
        # Register both possible TestClient source addresses (varies
        # by Starlette version) so the carve-out matches.
        for addr in ("127.0.0.1", "testclient"):
            cluster.update_member(
                str(uuid.uuid4()),
                hostname=f"peer-{addr}",
                address=addr,
                port=8765,
                cluster_name="c",
            )
        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.post("/api/cluster/gpu_policy_local")
        async def fake_post():
            return {"ok": True}

        client = TestClient(app)
        r = client.post("/api/cluster/gpu_policy_local")
        assert r.status_code == 200, r.text

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


class TestBandwidth:
    def test_bandwidth_local_streams_requested_size(self):
        # We don't actually need a cluster activated for the streaming
        # endpoint's correctness test — but the X-Forgather-Node-Id
        # header is only set when active.
        cluster.activate("c", port=8765)
        token = auth.load_token()
        client = TestClient(_make_app())
        size = 65536  # one chunk
        r = client.get(
            f"/api/cluster/bandwidth_local?bytes={size}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert len(r.content) == size
        # All bytes are the deterministic filler — checking one is
        # enough to catch a wiring mistake without hashing the body.
        assert set(r.content) == {ord("X")}
        assert r.headers.get("x-forgather-node-id")

    def test_bandwidth_local_clamps_huge_request(self):
        cluster.activate("c", port=8765)
        token = auth.load_token()
        client = TestClient(_make_app())
        # FastAPI's Query validation rejects out-of-range values with
        # 422; we don't want a malformed request to make the server
        # allocate gigabytes.
        r = client.get(
            "/api/cluster/bandwidth_local?bytes=99999999999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422

    def test_bandwidth_local_carved_out_for_peers(self):
        # The bandwidth target is in the peer-allowed GET list so the
        # master can hit it without holding a token. Verify the
        # carve-out fires for a known-peer source IP.
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
        r = client.get("/api/cluster/bandwidth_local?bytes=4096")
        assert r.status_code == 200, r.text


class TestClusterJobSubmit:
    """Cluster-coordinator submit path. The fanout step is monkey-
    patched away — we don't want to actually enqueue anything during
    unit tests — but the validation, rdzv-args computation, and
    bundle-record code paths run for real.
    """

    def _activate_with_two_members(
        self,
        version_a="1.1.0",
        version_b="1.1.0",
        *,
        with_interfaces: bool = True,
    ):
        ident = cluster.activate("c", port=8765)
        # Self gets a non-loopback address so the membership table is
        # realistic; updating self via update_member is normally done
        # by cluster_discovery.start, which the unit test bypasses.
        cluster.update_self_address("192.168.1.27")
        peer_id = str(uuid.uuid4())
        # Interface table mirrors the one a real probe would publish:
        # one entry per IPv4 interface, with the cluster's chosen
        # advertised address present so _derive_iface_from_member can
        # match it back to a name. Without that, auto-derive falls
        # through to the 422 branch.
        peer_ifaces = (
            [{"name": "enp4s0", "address": "192.168.1.162", "is_up": True}]
            if with_interfaces
            else []
        )
        self_ifaces = (
            [{"name": "enp212s0", "address": "192.168.1.27", "is_up": True}]
            if with_interfaces
            else []
        )
        cluster.update_member(
            peer_id,
            hostname="muthur",
            address="192.168.1.162",
            port=8765,
            cluster_name="c",
            forgather_version=version_b,
            probe={
                "versions": {
                    "forgather": version_b,
                    "torch": "2.10.0",
                    "nccl": "2.27.5",
                    "transformers": "5.7.0",
                },
                "interfaces": peer_ifaces,
            },
        )
        # Self's probe was set by activate() but with whatever was
        # importable at test time. Stamp a deterministic version dict
        # so the divergence check is honest about what's diverging.
        from forgather_server import cluster as _c

        _c._state._members[ident.node_id].probe = {
            "versions": {
                "forgather": version_a,
                "torch": "2.10.0",
                "nccl": "2.27.5",
                "transformers": "5.7.0",
            },
            "interfaces": self_ifaces,
        }
        return ident, peer_id

    def test_happy_path_submits_to_both_members(self, monkeypatch):
        ident, peer_id = self._activate_with_two_members()

        # Stub the fanout: capture payloads, return a synthetic queue id.
        import forgather_server.routes.cluster as routes

        captured: list = []

        async def fake_fanout(client, target, payload):
            captured.append((target.node_id, payload))
            return {
                "queue_id": f"q_{target.node_id[:8]}",
                "node_id": target.node_id,
            }

        monkeypatch.setattr(routes, "_fanout_training", fake_fanout)
        token = auth.load_token()
        client = TestClient(_make_app_with_full_router())
        r = client.post(
            "/api/cluster/jobs/submit",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "project_dir": "/proj",
                "config": "train.yaml",
                "members": [
                    {
                        "node_id": ident.node_id,
                        "nproc_per_node": 2,
                        "nccl_socket_ifname": "enp212s0",
                    },
                    {
                        "node_id": peer_id,
                        "nproc_per_node": 1,
                        "nccl_socket_ifname": "eth0",
                    },
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        cj = body["cluster_job"]
        assert len(cj["members"]) == 2
        # node_rank assignment follows the request order.
        ranks = {m["node_id"]: m["node_rank"] for m in cj["members"]}
        assert ranks[ident.node_id] == 0
        assert ranks[peer_id] == 1
        # rdzv host defaults to master (lowest UUID — could be either,
        # so just check it is one of them and the endpoint is well-
        # formed).
        assert cj["rdzv_endpoint"].endswith(":29400")
        # Both peers received a fanout call with the same rdzv_id and
        # endpoint, but different node_ranks.
        assert len(captured) == 2
        rdzv_ids = {c[1]["rdzv_args"]["rdzv_id"] for c in captured}
        assert len(rdzv_ids) == 1
        node_ranks = {
            c[0]: c[1]["rdzv_args"]["node_rank"] for c in captured
        }
        assert node_ranks == ranks
        # The operator-chosen interface name lands in extra_env for
        # all three socket-binding env vars: NCCL (CUDA collectives),
        # GLOO (CPU collectives), and TP (tensorpipe RPC). Pinning
        # only NCCL leaves Gloo's connectFullMesh resolving each
        # rank's address via socket.gethostname() — which on
        # Debian/Ubuntu returns 127.0.1.1 — and the trainer dies
        # before the first step.
        env_by_id = {c[0]: c[1]["extra_env"] for c in captured}
        for nid, expected_iface in (
            (ident.node_id, "enp212s0"),
            (peer_id, "eth0"),
        ):
            assert env_by_id[nid]["NCCL_SOCKET_IFNAME"] == expected_iface
            assert env_by_id[nid]["GLOO_SOCKET_IFNAME"] == expected_iface
            assert env_by_id[nid]["TP_SOCKET_IFNAME"] == expected_iface
        # Exactly one peer must be flagged is_host=True (the rdzv host),
        # all others is_host=False. Without this, c10d's broken
        # gethostname-based autodetection drops every node into client
        # mode and the rendezvous never binds.
        is_host_by_id = {c[0]: c[1]["rdzv_args"]["is_host"] for c in captured}
        rdzv_node_id = cj["rdzv_node_id"]
        assert is_host_by_id[rdzv_node_id] is True
        for nid, v in is_host_by_id.items():
            if nid != rdzv_node_id:
                assert v is False

    def test_version_mismatch_blocks_without_override(self, monkeypatch):
        ident, peer_id = self._activate_with_two_members(
            version_a="1.0.0", version_b="1.1.0"
        )
        import forgather_server.routes.cluster as routes

        async def fake_fanout(client, target, payload):
            raise AssertionError("should not fan out when blocked")

        monkeypatch.setattr(routes, "_fanout_training", fake_fanout)
        token = auth.load_token()
        client = TestClient(_make_app_with_full_router())
        r = client.post(
            "/api/cluster/jobs/submit",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "project_dir": "/proj",
                "config": "train.yaml",
                "members": [
                    {"node_id": ident.node_id, "nproc_per_node": 2},
                    {"node_id": peer_id, "nproc_per_node": 1},
                ],
            },
        )
        assert r.status_code == 409
        assert "version mismatch" in r.text.lower()

    def test_version_mismatch_allowed_with_override(self, monkeypatch):
        ident, peer_id = self._activate_with_two_members(
            version_a="1.0.0", version_b="1.1.0"
        )
        import forgather_server.routes.cluster as routes

        async def fake_fanout(client, target, payload):
            return {
                "queue_id": f"q_{target.node_id[:8]}",
                "node_id": target.node_id,
            }

        monkeypatch.setattr(routes, "_fanout_training", fake_fanout)
        token = auth.load_token()
        client = TestClient(_make_app_with_full_router())
        r = client.post(
            "/api/cluster/jobs/submit",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "project_dir": "/proj",
                "config": "train.yaml",
                "allow_version_mismatch": True,
                "members": [
                    {"node_id": ident.node_id, "nproc_per_node": 2},
                    {"node_id": peer_id, "nproc_per_node": 1},
                ],
            },
        )
        assert r.status_code == 200, r.text
        assert any("forgather" in w for w in r.json()["warnings"])

    def test_auto_derives_iface_when_operator_omits_it(self, monkeypatch):
        # When the modal's iface picker is left on "(auto)" — i.e.
        # nccl_socket_ifname is null — the server must match the
        # member's advertised address against its probe's interface
        # table and pin NCCL/Gloo/TP to that interface name. Without
        # this, Gloo's connectFullMesh publishes loopback addresses
        # because socket.gethostname() resolves to 127.0.1.1 on
        # Debian/Ubuntu, and the trainer dies before the first step.
        ident, peer_id = self._activate_with_two_members()
        import forgather_server.routes.cluster as routes

        captured: list = []

        async def fake_fanout(client, target, payload):
            captured.append((target.node_id, payload))
            return {"queue_id": f"q_{target.node_id[:8]}", "node_id": target.node_id}

        monkeypatch.setattr(routes, "_fanout_training", fake_fanout)
        token = auth.load_token()
        client = TestClient(_make_app_with_full_router())
        r = client.post(
            "/api/cluster/jobs/submit",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "project_dir": "/proj",
                "config": "train.yaml",
                "members": [
                    # No nccl_socket_ifname — auto-derive must fill in.
                    {"node_id": ident.node_id, "nproc_per_node": 1},
                    {"node_id": peer_id, "nproc_per_node": 1},
                ],
            },
        )
        assert r.status_code == 200, r.text
        env_by_id = {c[0]: c[1]["extra_env"] for c in captured}
        # Self's advertised address is 192.168.1.27 → enp212s0; peer's
        # is 192.168.1.162 → enp4s0. Auto-derive must pick those names.
        assert env_by_id[ident.node_id]["NCCL_SOCKET_IFNAME"] == "enp212s0"
        assert env_by_id[ident.node_id]["GLOO_SOCKET_IFNAME"] == "enp212s0"
        assert env_by_id[ident.node_id]["TP_SOCKET_IFNAME"] == "enp212s0"
        assert env_by_id[peer_id]["NCCL_SOCKET_IFNAME"] == "enp4s0"
        assert env_by_id[peer_id]["GLOO_SOCKET_IFNAME"] == "enp4s0"
        assert env_by_id[peer_id]["TP_SOCKET_IFNAME"] == "enp4s0"

    def test_rejects_when_iface_cannot_be_derived(self, monkeypatch):
        # Probe with no interfaces (e.g. a host that hasn't published
        # one yet, or a malformed payload) leaves auto-derive with
        # nothing to match. Must surface a 422 rather than silently
        # spawn a job that will deadlock in connectFullMesh.
        ident, peer_id = self._activate_with_two_members(
            with_interfaces=False
        )
        import forgather_server.routes.cluster as routes

        async def fake_fanout(client, target, payload):
            raise AssertionError("must not fan out when iface is unknown")

        monkeypatch.setattr(routes, "_fanout_training", fake_fanout)
        token = auth.load_token()
        client = TestClient(_make_app_with_full_router())
        r = client.post(
            "/api/cluster/jobs/submit",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "project_dir": "/proj",
                "config": "train.yaml",
                "members": [
                    {"node_id": ident.node_id, "nproc_per_node": 1},
                    {"node_id": peer_id, "nproc_per_node": 1},
                ],
            },
        )
        assert r.status_code == 422
        assert "interface" in r.text.lower()

    def test_unreachable_peer_rejected(self, monkeypatch):
        ident, peer_id = self._activate_with_two_members()
        cluster.mark_unreachable(peer_id)
        token = auth.load_token()
        client = TestClient(_make_app_with_full_router())
        r = client.post(
            "/api/cluster/jobs/submit",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "project_dir": "/proj",
                "config": "train.yaml",
                "members": [
                    {"node_id": ident.node_id, "nproc_per_node": 2},
                    {"node_id": peer_id, "nproc_per_node": 1},
                ],
            },
        )
        assert r.status_code == 400
        assert "unreachable" in r.text.lower()


def _make_app_with_full_router():
    """Like _make_app() but mounts the cluster router with prefix="/api"
    so /api/cluster/jobs/submit resolves correctly."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(cluster_routes.router, prefix="/api")
    return app


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

    def test_bandwidth_local_carved_out(self):
        assert auth.path_allows_peer("/api/cluster/bandwidth_local") is True

    def test_mutation_carve_out_disjoint_from_read_set(self):
        # The two carve-outs are different intentionally — guard
        # against accidentally widening the GET set to include the
        # mutation path or vice versa.
        assert auth.path_allows_peer_mutation("/api/cluster/gpu_policy_local")
        assert not auth.path_allows_peer_mutation("/api/cluster/members")
        assert not auth.path_allows_peer("/api/cluster/gpu_policy_local")
