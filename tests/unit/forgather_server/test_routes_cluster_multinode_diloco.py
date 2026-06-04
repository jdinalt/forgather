"""Tests for the DiLoCo composition path through ``/cluster/jobs/submit``.

PR-A wires the multi-node bundle endpoint so a single submit can carry
an optional ``diloco`` block. When set, every per-rank training job
joins one DiLoCo worker group (shared base ``worker_id``) and forwards
the param-server bearer through ``extra_env``.

The load-bearing invariants exercised here:

1. **One base ``worker_id`` across the bundle.** Resolved once on the
   master (auto-minted when the operator didn't supply one) and copied
   verbatim into every per-rank fanout payload. If this drifts the PP
   group fragments into N solo workers and parameter averaging
   silently degrades.

2. **Token forwarding via ``extra_env``.** A peer (non-master) cannot
   resolve the bearer for a server running on the master, so the
   master ships ``FORGATHER_DILOCO_SERVER_TOKEN`` to every peer.

3. **Persisted ``diloco`` on the bundle record + API model.** The CLI
   and webui can see the bundle as "one DiLoCo group" without having
   to peek into each peer's queue item.

The test stubs ``_fanout_training`` so payloads are captured in-process
without a real HTTP fanout to peers.
"""

import time
import uuid
from typing import Any, Dict, List

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
        cluster_jobs,
        cluster_journal,
        cluster_membership,
    )

    cluster._reset_for_tests()
    cluster_jobs._reset_for_tests()
    cluster_journal._reset_for_tests()
    cluster_dataset_inventory._reset_master_state_for_tests()
    cluster_membership._reset_role_listeners_for_tests()
    auth._reset_sessions_for_tests()
    yield


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(cluster_routes.router, prefix="/api")
    return app


def _activate_two_node_cluster(peer_iface: str = "eth0"):
    """Activate cluster mode and register one peer with a probe that
    advertises a matching interface so iface auto-derivation succeeds.

    Returns ``(self_node_id, peer_node_id)``.
    """
    cluster.activate("c", port=8765)
    self_id = cluster.self_identity().node_id
    # Make the self entry's address match an "interface" so the master
    # can auto-pick an iface for itself too.
    cluster.update_member(
        self_id,
        hostname="master-host",
        address="10.0.0.1",
        port=8765,
        cluster_name="c",
        source="local",
        probe={"interfaces": [{"name": "eth0", "address": "10.0.0.1"}]},
    )
    peer_id = str(uuid.uuid4())
    cluster.update_member(
        peer_id,
        hostname="peer-host",
        address="10.0.0.2",
        port=8765,
        cluster_name="c",
        source="peer_pull",
        probe={"interfaces": [{"name": peer_iface, "address": "10.0.0.2"}]},
    )
    return self_id, peer_id


def _capture_fanout(monkeypatch) -> List[Dict[str, Any]]:
    """Replace ``_fanout_training`` with a capture stub.

    Returns the list payloads are appended to. The stub fabricates a
    queue_id per call so the bundle assembly proceeds normally.
    """
    captured: List[Dict[str, Any]] = []

    async def _stub(client, target, payload):
        captured.append({"target_node_id": target.node_id, "payload": payload})
        return {
            "queue_id": f"q_{len(captured):03d}",
            "node_id": target.node_id,
        }

    monkeypatch.setattr(cluster_routes, "_fanout_training", _stub)
    return captured


def _submit_body(self_id: str, peer_id: str, *, diloco=None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "project_dir": "/tmp/project",
        "config": "test.yaml",
        "members": [
            {"node_id": self_id, "nproc_per_node": 2},
            {"node_id": peer_id, "nproc_per_node": 2},
        ],
        "allow_version_mismatch": True,
    }
    if diloco is not None:
        body["diloco"] = diloco
    return body


class TestMultiNodeDiLoCoComposition:
    def test_submit_without_diloco_unchanged(self, monkeypatch):
        """No regression for plain (non-DiLoCo) multi-node submits:
        no ``diloco`` block in per-peer payloads, none on the bundle."""
        self_id, peer_id = _activate_two_node_cluster()
        captured = _capture_fanout(monkeypatch)

        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.post(
            "/api/cluster/jobs/submit",
            json=_submit_body(self_id, peer_id),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        assert len(captured) == 2
        for entry in captured:
            assert "diloco" not in entry["payload"]
            assert "FORGATHER_DILOCO_SERVER_TOKEN" not in entry["payload"].get(
                "extra_env", {}
            )
        bundle = r.json()["cluster_job"]
        assert bundle.get("diloco") is None

    def test_explicit_worker_id_shared_across_payloads(self, monkeypatch):
        """An operator-supplied ``worker_id`` is the base id used by
        every per-rank payload — verbatim. This is the load-bearing
        invariant: same base → same DiLoCo group; drift → fragmented
        group + silent averaging degradation."""
        self_id, peer_id = _activate_two_node_cluster()
        captured = _capture_fanout(monkeypatch)

        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.post(
            "/api/cluster/jobs/submit",
            json=_submit_body(
                self_id,
                peer_id,
                diloco={
                    "server_addr": "http://10.0.0.1:8512",
                    "worker_id": "fixed_base",
                    "heartbeat_interval": 12.5,
                },
            ),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        assert len(captured) == 2
        worker_ids = {
            entry["payload"]["diloco"]["worker_id"] for entry in captured
        }
        assert worker_ids == {"fixed_base"}, (
            "every per-peer payload must share the operator's base worker_id"
        )
        # Other diloco fields also identical and verbatim.
        for entry in captured:
            d = entry["payload"]["diloco"]
            assert d["server_addr"] == "http://10.0.0.1:8512"
            assert d["heartbeat_interval"] == 12.5

    def test_missing_worker_id_auto_minted_and_shared(self, monkeypatch):
        """No operator-supplied ``worker_id`` → master mints exactly
        one memorable name and ships it to every peer (not one per
        peer, which would fragment the group)."""
        self_id, peer_id = _activate_two_node_cluster()
        captured = _capture_fanout(monkeypatch)

        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.post(
            "/api/cluster/jobs/submit",
            json=_submit_body(
                self_id,
                peer_id,
                diloco={"server_addr": "http://10.0.0.1:8512"},
            ),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        assert len(captured) == 2
        worker_ids = {
            entry["payload"]["diloco"]["worker_id"] for entry in captured
        }
        assert len(worker_ids) == 1, (
            "auto-minted base must be the same on every peer; got: "
            f"{worker_ids}"
        )
        only_id = next(iter(worker_ids))
        assert only_id and only_id.strip() == only_id
        # Bundle record carries the resolved base too.
        bundle = r.json()["cluster_job"]
        assert bundle["diloco"]["worker_id"] == only_id

    def test_token_forwarded_to_peers_when_local_server_known(
        self, monkeypatch
    ):
        """The master resolves the DiLoCo bearer once and ships it
        through ``extra_env``. Remote peers can't see the local
        diloco_server JobRecord, so without this they'd hit 401."""
        self_id, peer_id = _activate_two_node_cluster()
        captured = _capture_fanout(monkeypatch)

        # Stub the master's local token-resolution path so we don't
        # need a real diloco_server JobRecord.
        from forgather_server import scheduler

        monkeypatch.setattr(
            scheduler,
            "_diloco_token_for_server_addr",
            lambda addr: "test-bearer-token",
        )

        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.post(
            "/api/cluster/jobs/submit",
            json=_submit_body(
                self_id,
                peer_id,
                diloco={
                    "server_addr": "http://10.0.0.1:8512",
                    "worker_id": "base",
                },
            ),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        assert len(captured) == 2
        for entry in captured:
            env = entry["payload"]["extra_env"]
            assert env["FORGATHER_DILOCO_SERVER_TOKEN"] == "test-bearer-token"

    def test_no_token_when_server_is_external(self, monkeypatch):
        """When the master can't resolve a local JobRecord matching
        the server_addr (e.g. truly remote server), no token is
        forwarded — the worker will rely on its own configured
        credentials (env, dotenv, etc.)."""
        self_id, peer_id = _activate_two_node_cluster()
        captured = _capture_fanout(monkeypatch)

        from forgather_server import scheduler

        monkeypatch.setattr(
            scheduler, "_diloco_token_for_server_addr", lambda addr: None
        )

        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.post(
            "/api/cluster/jobs/submit",
            json=_submit_body(
                self_id,
                peer_id,
                diloco={"server_addr": "https://elsewhere.example:8512"},
            ),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        for entry in captured:
            env = entry["payload"]["extra_env"]
            assert "FORGATHER_DILOCO_SERVER_TOKEN" not in env

    def test_peer_handler_persists_diloco_on_job_params(self, monkeypatch):
        """The peer-side ``/api/cluster/training_local`` handler must
        spread the ``diloco`` block into the queue item's
        ``job_params`` (mirroring the existing ``rdzv_args`` /
        ``extra_env`` propagation). Without this, even if the master
        ships the block, the peer-side scheduler's
        ``_diloco_env_from_job_params`` reads an empty dict and the
        ranks register as solo workers."""
        # Activate the cluster so the peer-mTLS carve-out admits the
        # bearer-authed POST.
        cluster.activate("c", port=8765)
        from forgather_server import queue_store

        token = auth.load_token()
        client = TestClient(_make_app())
        payload = {
            "project_dir": "/tmp/project",
            "config": "test.yaml",
            "rdzv_args": {
                "nnodes": 2,
                "node_rank": 1,
                "rdzv_backend": "c10d",
                "rdzv_endpoint": "10.0.0.1:29400",
                "rdzv_id": "rd123",
                "nproc_per_node": 2,
                "is_host": False,
                "local_addr": "10.0.0.2",
            },
            "extra_env": {"FORGATHER_DILOCO_SERVER_TOKEN": "tok"},
            "cluster_job_id": "cj_test",
            "diloco": {
                "server_addr": "http://10.0.0.1:8512",
                "worker_id": "persisted_base",
                "heartbeat_interval": 5.0,
            },
        }
        r = client.post(
            "/api/cluster/training_local",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        queue_id = r.json()["queue_id"]
        item = queue_store.get_item(queue_id)
        assert item is not None
        assert item.job_params["diloco"] == payload["diloco"]
        # Sanity: existing pre-PR fields still present.
        assert item.job_params["rdzv_args"] == payload["rdzv_args"]
        assert item.job_params["extra_env"] == payload["extra_env"]
        assert item.job_params["cluster_job_id"] == "cj_test"

    def test_peer_handler_omits_diloco_when_absent(self, monkeypatch):
        """Plain (non-DiLoCo) multi-node enqueues land without a
        ``diloco`` key on ``job_params`` — pre-PR queue items are
        unaffected."""
        cluster.activate("c", port=8765)
        from forgather_server import queue_store

        token = auth.load_token()
        client = TestClient(_make_app())
        payload = {
            "project_dir": "/tmp/project",
            "config": "test.yaml",
            "rdzv_args": {
                "nnodes": 2,
                "node_rank": 0,
                "rdzv_backend": "c10d",
                "rdzv_endpoint": "10.0.0.1:29400",
                "rdzv_id": "rd123",
                "nproc_per_node": 2,
                "is_host": True,
                "local_addr": "10.0.0.1",
            },
            "extra_env": {},
            "cluster_job_id": "cj_plain",
        }
        r = client.post(
            "/api/cluster/training_local",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        item = queue_store.get_item(r.json()["queue_id"])
        assert item is not None
        assert "diloco" not in item.job_params

    def test_bundle_record_surfaces_diloco_on_submit_and_get(
        self, monkeypatch
    ):
        """The persisted bundle carries the resolved DiLoCo block; the
        submit response and the bundle-get response both expose it."""
        self_id, peer_id = _activate_two_node_cluster()
        _capture_fanout(monkeypatch)

        from forgather_server import scheduler

        monkeypatch.setattr(
            scheduler, "_diloco_token_for_server_addr", lambda addr: None
        )

        token = auth.load_token()
        client = TestClient(_make_app())
        r = client.post(
            "/api/cluster/jobs/submit",
            json=_submit_body(
                self_id,
                peer_id,
                diloco={
                    "server_addr": "http://10.0.0.1:8512",
                    "worker_id": "persisted",
                    "heartbeat_interval": 7.5,
                },
            ),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        bundle = r.json()["cluster_job"]
        cj_id = bundle["cluster_job_id"]
        assert bundle["diloco"] == {
            "server_addr": "http://10.0.0.1:8512",
            "worker_id": "persisted",
            "heartbeat_interval": 7.5,
        }

        r = client.get(
            f"/api/cluster/jobs/{cj_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["diloco"] == bundle["diloco"]
