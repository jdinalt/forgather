"""Tests for routes/diloco.py — server discovery + status proxy + registry."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from forgather_server import diloco_server_registry as reg
from forgather_server.routes import diloco as diloco_routes


def _patch_async_client(monkeypatch, handler):
    """Make ``httpx.AsyncClient(...)`` use a MockTransport in this module.

    The route uses ``async with httpx.AsyncClient(...) as client`` so we
    need a real AsyncClient (with a MockTransport) — not a MagicMock
    that breaks ``__aenter__``/``__aexit__``.
    """
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        # Drop kwargs the real AsyncClient understands so we can swap in
        # the MockTransport without colliding with verify / timeout.
        kwargs.pop("verify", None)
        kwargs.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(diloco_routes.httpx, "AsyncClient", factory)


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A FastAPI app with the diloco router mounted and a tmp registry."""
    target = tmp_path / "diloco_server_registry.json"
    monkeypatch.setattr(reg, "diloco_server_registry_file", lambda: target)
    a = FastAPI()
    a.include_router(diloco_routes.router, prefix="/api")
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def no_local_servers(monkeypatch):
    """Stub the JobRecord scan so tests don't see real local instances."""
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])


# ---------------------------------------------------------------------------
# Registry endpoints
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_list_empty(self, client, no_local_servers):
        r = client.get("/api/diloco/registry")
        assert r.status_code == 200
        assert r.json() == []

    def test_add_and_list(self, client, no_local_servers):
        r = client.post(
            "/api/diloco/registry",
            json={"label": "WAN", "base_url": "http://10.0.0.1:8512"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["label"] == "WAN"
        assert body["base_url"] == "http://10.0.0.1:8512"
        assert body["has_auth_token"] is False
        assert body["verify_tls"] is True
        assert len(body["id"]) == 8

        listed = client.get("/api/diloco/registry").json()
        assert len(listed) == 1
        assert listed[0]["id"] == body["id"]

    def test_add_rejects_missing_url(self, client):
        r = client.post("/api/diloco/registry", json={"label": "x", "base_url": ""})
        assert r.status_code == 400

    def test_add_rejects_bad_scheme(self, client):
        r = client.post(
            "/api/diloco/registry",
            json={"label": "x", "base_url": "ftp://x:1"},
        )
        assert r.status_code == 400

    def test_delete(self, client, no_local_servers):
        added = client.post(
            "/api/diloco/registry",
            json={"label": "x", "base_url": "http://x:1"},
        ).json()
        r = client.delete(f"/api/diloco/registry/{added['id']}")
        assert r.status_code == 200
        assert client.get("/api/diloco/registry").json() == []

    def test_delete_missing_returns_404(self, client):
        r = client.delete("/api/diloco/registry/nope")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Worker-name generation
# ---------------------------------------------------------------------------


class TestGenerateWorkerNames:
    def test_default_count_one(self, client):
        r = client.post("/api/diloco/generate-worker-names", json={})
        assert r.status_code == 200, r.text
        names = r.json()["names"]
        assert len(names) == 1
        assert "-" in names[0]

    def test_batch_is_unique(self, client):
        r = client.post("/api/diloco/generate-worker-names", json={"count": 50})
        assert r.status_code == 200, r.text
        names = r.json()["names"]
        assert len(names) == 50
        assert len(set(names)) == 50  # no duplicates within the batch

    def test_excludes_are_honored(self, client):
        # First batch, then a second batch excluding the first — the union
        # must stay collision-free (the resumable-pool use case).
        first = client.post(
            "/api/diloco/generate-worker-names", json={"count": 20}
        ).json()["names"]
        second = client.post(
            "/api/diloco/generate-worker-names",
            json={"count": 20, "exclude": first},
        ).json()["names"]
        assert len(second) == 20
        assert set(first).isdisjoint(set(second))

    def test_rejects_zero(self, client):
        r = client.post("/api/diloco/generate-worker-names", json={"count": 0})
        assert r.status_code == 400

    def test_rejects_too_many(self, client):
        r = client.post("/api/diloco/generate-worker-names", json={"count": 100000})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Unified servers list
# ---------------------------------------------------------------------------


class TestServersList:
    def test_lists_registered(self, client, no_local_servers):
        client.post(
            "/api/diloco/registry",
            json={"label": "WAN", "base_url": "http://10.0.0.1:8512"},
        )
        r = client.get("/api/diloco/servers")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["source"] == "registered"
        assert body[0]["base_url"] == "http://10.0.0.1:8512"
        assert body[0]["id"].startswith("registered:")

    def test_lists_local(self, client, monkeypatch):
        """Stub a fake JobRecord so the local-servers helper has work to do."""

        class FakeRec:
            queue_id = "q1"
            job_type = "diloco_server"
            config = "diloco:8512"
            status = "running"
            job_params = {"host": "127.0.0.1", "port": 8512}

        monkeypatch.setattr(
            diloco_routes.job_records, "list_records", lambda: [FakeRec()]
        )
        r = client.get("/api/diloco/servers")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["source"] == "local"
        assert body[0]["queue_id"] == "q1"
        assert body[0]["base_url"] == "http://127.0.0.1:8512"
        assert body[0]["alive"] is True

    def test_local_server_reports_has_auth_token(self, client, monkeypatch):
        """Locally-spawned servers default to bearer auth ON, so the
        lock indicator must reflect the JobRecord's token (post-#90
        finding: it was previously always None for local servers,
        inverting the lock vs registered remotes)."""

        class FakeRec:
            queue_id = "q1"
            job_type = "diloco_server"
            config = "diloco:8512"
            status = "running"
            auth_token = "secret-token"
            job_params = {"host": "127.0.0.1", "port": 8512}

        monkeypatch.setattr(
            diloco_routes.job_records, "list_records", lambda: [FakeRec()]
        )
        body = client.get("/api/diloco/servers").json()
        assert body[0]["has_auth_token"] is True

    def test_terminal_status_records_are_filtered(self, client, monkeypatch):
        """Dead servers (done / failed / aborted) shouldn't appear in the
        unified list — they can't be inspected or selected for training,
        and the Jobs view already shows them for diagnostics."""

        def fake(status):
            return type(
                "FakeRec",
                (),
                {
                    "queue_id": f"q-{status}",
                    "job_type": "diloco_server",
                    "config": "diloco:8512",
                    "status": status,
                    "job_params": {"host": "127.0.0.1", "port": 8512},
                },
            )()

        records = [fake("running"), fake("done"), fake("failed"), fake("aborted")]
        monkeypatch.setattr(diloco_routes.job_records, "list_records", lambda: records)
        body = client.get("/api/diloco/servers").json()
        assert len(body) == 1
        assert body[0]["queue_id"] == "q-running"

    def test_host_0000_is_remapped(self, client, monkeypatch):
        class FakeRec:
            queue_id = "q1"
            job_type = "diloco_server"
            config = "diloco:8512"
            status = "running"
            job_params = {"host": "0.0.0.0", "port": 8512}

        monkeypatch.setattr(
            diloco_routes.job_records, "list_records", lambda: [FakeRec()]
        )
        body = client.get("/api/diloco/servers").json()
        assert body[0]["base_url"] == "http://localhost:8512"

    def test_https_scheme_stamped_by_scheduler_is_respected(self, client, monkeypatch):
        """When the scheduler stamps ``scheme=https`` on the JobRecord
        (because TLS was provisioned), the base_url surfaced to the
        webui must use https://. Regression test for the bug where
        the Job card showed ``http://...`` for an actual TLS server,
        causing the proxy to hit ReadError speaking HTTP at a TLS
        socket."""

        class FakeRec:
            queue_id = "q1"
            job_type = "diloco_server"
            config = "diloco:8512"
            status = "running"
            job_params = {
                "host": "127.0.0.1",
                "port": 8512,
                "scheme": "https",
            }

        monkeypatch.setattr(
            diloco_routes.job_records, "list_records", lambda: [FakeRec()]
        )
        body = client.get("/api/diloco/servers").json()
        assert body[0]["base_url"] == "https://127.0.0.1:8512"

    def test_missing_scheme_falls_back_to_http(self, client, monkeypatch):
        """Records spawned before scheme-stamping landed (or services
        that never get the stamp) fall back to http — backwards
        compat for the pre-#90 wire shape."""

        class FakeRec:
            queue_id = "q1"
            job_type = "diloco_server"
            config = "diloco:8512"
            status = "running"
            job_params = {"host": "127.0.0.1", "port": 8512}

        monkeypatch.setattr(
            diloco_routes.job_records, "list_records", lambda: [FakeRec()]
        )
        body = client.get("/api/diloco/servers").json()
        assert body[0]["base_url"] == "http://127.0.0.1:8512"


# ---------------------------------------------------------------------------
# Status / info / control proxy
# ---------------------------------------------------------------------------


class TestProxy:
    def test_status_loopback_passes_through(
        self, client, no_local_servers, monkeypatch
    ):
        """Loopback is always allowed (no need to register)."""
        upstream_payload = {"status": "running", "mode": "sync"}

        def fake_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/status"
            return httpx.Response(200, json=upstream_payload)

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/server-status",
            params={"base": "http://127.0.0.1:8512"},
        )
        assert r.status_code == 200
        assert r.json() == upstream_payload

    def test_status_refused_for_unknown_remote(self, client, no_local_servers):
        r = client.get(
            "/api/diloco/server-status",
            params={"base": "http://10.0.0.99:8512"},
        )
        assert r.status_code == 403
        assert "registry" in r.json()["detail"].lower()

    def test_known_workers_passes_through(self, client, no_local_servers, monkeypatch):
        """The known-workers proxy forwards GET <base>/known_workers and
        returns the upstream roster verbatim (#103)."""
        upstream_payload = {
            "workers": [
                {
                    "worker_id": "w0",
                    "output_dir": "/runs/m_w0",
                    "last_registered": 1.0,
                    "running": False,
                }
            ]
        }

        def fake_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/known_workers"
            return httpx.Response(200, json=upstream_payload)

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/known-workers",
            params={"base": "http://127.0.0.1:8512"},
        )
        assert r.status_code == 200
        assert r.json() == upstream_payload

    def test_known_workers_refused_for_unknown_remote(self, client, no_local_servers):
        r = client.get(
            "/api/diloco/known-workers",
            params={"base": "http://10.0.0.99:8512"},
        )
        assert r.status_code == 403
        assert "registry" in r.json()["detail"].lower()

    def test_stats_history_passes_through(self, client, no_local_servers, monkeypatch):
        """The stats-history proxy forwards GET <base>/stats_history (with
        max_points) and returns the upstream history verbatim."""
        upstream_payload = {
            "records": [{"global_step": 10, "train_loss": 3.0}],
            "count": 1,
            "downsampled": False,
        }

        def fake_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/stats_history"
            assert request.url.params.get("max_points") == "500"
            return httpx.Response(200, json=upstream_payload)

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/stats-history",
            params={"base": "http://127.0.0.1:8512", "max_points": 500},
        )
        assert r.status_code == 200
        assert r.json() == upstream_payload

    def test_stats_history_refused_for_unknown_remote(self, client, no_local_servers):
        r = client.get(
            "/api/diloco/stats-history",
            params={"base": "http://10.0.0.99:8512"},
        )
        assert r.status_code == 403
        assert "registry" in r.json()["detail"].lower()

    def test_status_allowed_after_registering(
        self, client, no_local_servers, monkeypatch
    ):
        client.post(
            "/api/diloco/registry",
            json={"label": "WAN", "base_url": "http://10.0.0.99:8512"},
        )

        def fake_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "running"})

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/server-status",
            params={"base": "http://10.0.0.99:8512"},
        )
        assert r.status_code == 200

    def test_info_proxies(self, client, no_local_servers, monkeypatch):
        upstream = {
            "output_dir": "/tmp/m",
            "num_parameters": 1234,
            "expected_client_settings": {"sync_every": 500},
        }

        def fake_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/info"
            return httpx.Response(200, json=upstream)

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/server-info",
            params={"base": "http://127.0.0.1:8512"},
        )
        assert r.status_code == 200
        assert r.json() == upstream

    def test_work_queues_proxies(self, client, no_local_servers, monkeypatch):
        upstream = [
            {
                "dataset_id": "abc123",
                "shuffle_seed": 0,
                "total_units": 1024,
                "issued_count": 7,
                "completed_count": 3,
                "hint": {"length": 100000},
            }
        ]

        def fake_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/work/queues"
            return httpx.Response(200, json=upstream)

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/work-queues",
            params={"base": "http://127.0.0.1:8512"},
        )
        assert r.status_code == 200
        assert r.json() == upstream

    def test_work_queue_proxies_with_query_args(
        self, client, no_local_servers, monkeypatch
    ):
        upstream = {
            "dataset_id": "abc123",
            "shuffle_seed": 42,
            "total_units": 1024,
            "issued_count": 5,
            "completed_count": 2,
            "hint": {"length": 100000},
            "issued_bitmap_b64": "AwA=",
            "completed_bitmap_b64": "AQA=",
            "by_worker": {"alpha": {"units_issued": 3, "units_completed": 1}},
        }
        captured = {}

        def fake_handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["query"] = dict(request.url.params)
            return httpx.Response(200, json=upstream)

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/work-queue",
            params={
                "base": "http://127.0.0.1:8512",
                "dataset_id": "abc123",
                "shuffle_seed": 42,
            },
        )
        assert r.status_code == 200
        assert r.json() == upstream
        # Upstream got the path + the right query args.
        assert captured["path"] == "/work/queue"
        assert captured["query"] == {"dataset_id": "abc123", "shuffle_seed": "42"}

    def test_work_queue_refused_for_unregistered_base(self, client, no_local_servers):
        # SSRF: a non-loopback / non-registered base is refused.
        r = client.get(
            "/api/diloco/work-queue",
            params={
                "base": "http://wan-host:8512",
                "dataset_id": "abc",
                "shuffle_seed": 0,
            },
        )
        assert r.status_code == 403

    def test_control_rejects_unknown_action(self, client, no_local_servers):
        r = client.post(
            "/api/diloco/server-control/eat_my_homework",
            params={"base": "http://127.0.0.1:8512"},
            json={},
        )
        assert r.status_code == 400

    def test_control_proxies_known_action(self, client, no_local_servers, monkeypatch):
        captured = {}

        def fake_handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok"})

        _patch_async_client(monkeypatch, fake_handler)
        r = client.post(
            "/api/diloco/server-control/kick_worker",
            params={"base": "http://127.0.0.1:8512"},
            json={"worker_id": "w1"},
        )
        assert r.status_code == 200
        assert captured["path"] == "/control/kick_worker"
        assert captured["method"] == "POST"
        assert captured["body"] == {"worker_id": "w1"}

    def test_control_proxies_command_relay(self, client, no_local_servers, monkeypatch):
        # The webui's collective/per-worker controls + clean shutdown all
        # proxy through /control/command — it must be in the allowlist.
        captured = {}

        def fake_handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok", "workers": ["w0"]})

        _patch_async_client(monkeypatch, fake_handler)
        r = client.post(
            "/api/diloco/server-control/command",
            params={"base": "http://127.0.0.1:8512"},
            json={"command": "save_and_stop"},
        )
        assert r.status_code == 200
        assert captured["path"] == "/control/command"
        assert captured["body"] == {"command": "save_and_stop"}

    def test_upstream_unreachable_returns_502(
        self, client, no_local_servers, monkeypatch
    ):
        def fake_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/server-status",
            params={"base": "http://127.0.0.1:8512"},
        )
        assert r.status_code == 502

    def test_terminated_local_server_returns_502_not_403(self, client, monkeypatch):
        """A local DiLoCo server that has terminated must still pass the
        SSRF allowlist so the upstream attempt produces a 502 — not a 403
        that the webui's fetch wrapper used to misread as session-expired.
        Regression test for the bug where shutting down a DiLoCo server
        forced the user to re-enter their password.
        """

        class TerminatedRec:
            queue_id = "q-dead"
            job_type = "diloco_server"
            config = "diloco:8512"
            status = "done"  # terminal — won't appear in _local_servers()
            job_params = {"host": "192.168.9.43", "port": 8512}

        monkeypatch.setattr(
            diloco_routes.job_records, "list_records", lambda: [TerminatedRec()]
        )

        def fake_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/work-queue",
            params={
                "base": "http://192.168.9.43:8512",
                "dataset_id": "abc",
                "shuffle_seed": 0,
            },
        )
        assert r.status_code == 502, (
            "terminated-local DiLoCo URL should reach the upstream attempt "
            f"and surface its 502, but got {r.status_code}"
        )
