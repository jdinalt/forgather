"""Tests for the TensorBoard auth-gated reverse proxy.

The C3 fix defaults spawned TB instances to ``127.0.0.1`` so other
local users can't read training metadata directly. To keep the webui
working, ``/api/tb/{job_id}/...`` proxies through the auth-gated
forgather server. These tests cover the two pieces:

1. ``build_tensorboard_command`` defaults to ``--host 127.0.0.1`` and
   honours the ``path_prefix`` / ``bind_all`` opt-ins.
2. ``routes.tb_proxy`` looks up the job by ``queue_id``, refuses
   non-TB / unknown ids, and forwards requests to the upstream
   loopback port with the path-prefix preserved.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import httpx
import pytest
from forgather_server import job_records, tensorboard_ops
from forgather_server.routes import tb_proxy

# ---------------------------------------------------------------------------
# tensorboard_ops.build_tensorboard_command — default host + path_prefix
# ---------------------------------------------------------------------------


class TestBuildTensorboardCommand:
    def test_defaults_to_loopback_host(self):
        cmd = tensorboard_ops.build_tensorboard_command(logdir="/tmp/x", port=6006)
        # "--host 127.0.0.1" must appear so other local users can't reach
        # TB on its bound port.
        assert "--host" in cmd
        idx = cmd.index("--host")
        assert cmd[idx + 1] == "127.0.0.1"
        # No --bind_all when caller didn't ask for it.
        assert "--bind_all" not in cmd

    def test_bind_all_overrides_default(self):
        cmd = tensorboard_ops.build_tensorboard_command(
            logdir="/tmp/x", port=6006, bind_all=True
        )
        assert "--bind_all" in cmd
        # Default --host injection must not also fire when bind_all is set.
        assert "--host" not in cmd

    def test_explicit_host_overrides_default(self):
        cmd = tensorboard_ops.build_tensorboard_command(
            logdir="/tmp/x", port=6006, host="10.0.0.5"
        )
        assert "--host" in cmd
        idx = cmd.index("--host")
        assert cmd[idx + 1] == "10.0.0.5"

    def test_path_prefix_appended(self):
        cmd = tensorboard_ops.build_tensorboard_command(
            logdir="/tmp/x", port=6006, path_prefix="/api/tb/abc123"
        )
        assert "--path_prefix" in cmd
        idx = cmd.index("--path_prefix")
        assert cmd[idx + 1] == "/api/tb/abc123"

    def test_path_prefix_omitted_when_none(self):
        cmd = tensorboard_ops.build_tensorboard_command(logdir="/tmp/x", port=6006)
        assert "--path_prefix" not in cmd


# ---------------------------------------------------------------------------
# Helpers for the proxy tests
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_records(monkeypatch) -> Iterator[Dict[str, job_records.JobRecord]]:
    """Replace ``job_records.get_record`` with an in-memory map.

    The proxy looks up jobs by queue_id; we inject a dict keyed by
    queue_id and patch the route's import target.
    """
    store: Dict[str, job_records.JobRecord] = {}

    def fake_get_record(queue_id: str):
        return store.get(queue_id)

    monkeypatch.setattr(tb_proxy.job_records, "get_record", fake_get_record)
    yield store


@pytest.fixture
def isolated_auth(tmp_path, monkeypatch):
    """Point auth-token storage at a tmp dir and disable auth so the
    TestClient doesn't have to negotiate a bearer token for the proxy
    routes — we're testing proxy behaviour, not auth coverage."""
    monkeypatch.setenv("FORGATHER_HOME", str(tmp_path))
    from forgather_server import auth

    auth._reset_sessions_for_tests()
    auth._auth_disabled = True
    yield tmp_path
    auth._auth_disabled = False


def _make_tb_record(
    queue_id: str, port: int = 6006, **overrides
) -> job_records.JobRecord:
    rec = job_records.JobRecord(
        queue_id=queue_id,
        job_type="tensorboard",
        status="running",
        job_params={"port": port, "logdir": "/tmp/runs"},
        path_prefix=f"/api/tb/{queue_id}",
    )
    for k, v in overrides.items():
        setattr(rec, k, v)
    return rec


class _FakeAsyncClient:
    """Minimal stand-in for httpx.AsyncClient used by the proxy.

    Records every ``send`` and replies with a fixed canned response. The
    real client is replaced via ``monkeypatch.setattr`` on the route
    module so we don't have to spin up a real upstream HTTP server.
    """

    def __init__(self, captured: List[Dict[str, Any]], response):
        self._captured = captured
        self._response = response

    def build_request(self, method, url, *, content=None, headers=None, params=None):
        # Mirror httpx.Client.build_request enough that the proxy's
        # subsequent send() can introspect attrs if it needs to.
        req = httpx.Request(method, url, headers=headers or [], content=content)
        # Stash a reference for assertions; httpx normally consumes
        # ``params`` into the URL — track it explicitly here so tests
        # can compare against the raw query string the proxy passed.
        req.extensions["_test_params"] = params
        return req

    async def send(self, request: httpx.Request, *, stream: bool = False):
        self._captured.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "params": request.extensions.get("_test_params"),
                "content": request.content,
                "stream": stream,
            }
        )
        return self._response

    async def aclose(self):
        return None


def _client_factory(captured: List[Dict[str, Any]], response: httpx.Response):
    """Return a callable usable as ``httpx.AsyncClient`` replacement.

    The route does ``client = httpx.AsyncClient(...)`` so the
    replacement must be callable.
    """

    def _factory(*args, **kwargs):
        return _FakeAsyncClient(captured, response)

    return _factory


class _FakeResponse:
    """Tiny stand-in for httpx.Response that supports the proxy's calls.

    We avoid using ``httpx.Response`` directly because building one with
    ``content=`` materializes the body and disables ``aiter_raw``. The
    proxy only touches a small surface (status_code / headers / aread /
    aiter_raw / aclose) — easier to fake than to coerce httpx into
    streaming a pre-known body.
    """

    def __init__(
        self,
        status_code: int,
        body: bytes,
        headers: Dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._body = body
        self.headers = httpx.Headers(headers or {})

    async def aread(self) -> bytes:
        return self._body

    async def aiter_raw(self, chunk_size: int = 64 * 1024):
        # Single-chunk emission is enough for tests; the proxy doesn't
        # care about how the body is sliced.
        if self._body:
            yield self._body

    async def aclose(self):
        return None


def _make_response(
    status: int = 200,
    body: bytes = b"hello",
    headers: Dict[str, str] | None = None,
) -> _FakeResponse:
    return _FakeResponse(status, body, headers)


# ---------------------------------------------------------------------------
# Proxy lookups
# ---------------------------------------------------------------------------


class TestProxyLookups:
    def test_404_for_unknown_job(self, isolated_auth, stub_records):
        from fastapi.testclient import TestClient
        from forgather_server.app import create_app

        client = TestClient(create_app())
        resp = client.get("/api/tb/does-not-exist/")
        assert resp.status_code == 404

    def test_404_for_non_tb_job(self, isolated_auth, stub_records):
        from fastapi.testclient import TestClient
        from forgather_server.app import create_app

        # An eval job at the same queue_id slot must not be reachable
        # through the TB proxy.
        rec = job_records.JobRecord(
            queue_id="not-tb",
            job_type="eval",
            status="running",
            job_params={"port": 6006},
        )
        stub_records["not-tb"] = rec

        client = TestClient(create_app())
        resp = client.get("/api/tb/not-tb/")
        assert resp.status_code == 404

    def test_404_for_terminal_tb_job(self, isolated_auth, stub_records):
        from fastapi.testclient import TestClient
        from forgather_server.app import create_app

        rec = _make_tb_record("dead-tb")
        rec.status = "done"
        stub_records["dead-tb"] = rec

        client = TestClient(create_app())
        resp = client.get("/api/tb/dead-tb/")
        # 404 is fine here — the job is gone, the port is unbound; we
        # don't want the proxy attempting to dial it and 502'ing.
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Proxy forwarding
# ---------------------------------------------------------------------------


class TestProxyForwarding:
    def test_get_forwards_to_loopback_with_full_prefix(
        self, isolated_auth, stub_records, monkeypatch
    ):
        from fastapi.testclient import TestClient
        from forgather_server.app import create_app

        stub_records["abc"] = _make_tb_record("abc", port=6789)

        captured: List[Dict[str, Any]] = []
        response = _make_response(
            status=200,
            body=b"<html>tb</html>",
            headers={"content-type": "text/html"},
        )
        monkeypatch.setattr(
            tb_proxy.httpx,
            "AsyncClient",
            _client_factory(captured, response),
        )

        client = TestClient(create_app())
        resp = client.get("/api/tb/abc/data/runs?run=run1")

        assert resp.status_code == 200
        assert resp.content == b"<html>tb</html>"
        assert len(captured) == 1
        sent = captured[0]
        assert sent["method"] == "GET"
        # Upstream URL must keep the path_prefix intact and target the
        # loopback host on the recorded port.
        assert sent["url"].startswith("http://127.0.0.1:6789/api/tb/abc/data/runs")
        # Query string passes through verbatim.
        assert sent["params"] == "run=run1"

    def test_response_body_and_headers_passed_through(
        self, isolated_auth, stub_records, monkeypatch
    ):
        from fastapi.testclient import TestClient
        from forgather_server.app import create_app

        stub_records["abc"] = _make_tb_record("abc")

        captured: List[Dict[str, Any]] = []
        response = _make_response(
            status=200,
            body=b'{"ok": true}',
            headers={
                "content-type": "application/json",
                # x-tb-plugin is a fake custom header to confirm
                # non-hop-by-hop headers reach the client unchanged.
                "x-tb-plugin": "scalars",
            },
        )
        monkeypatch.setattr(
            tb_proxy.httpx,
            "AsyncClient",
            _client_factory(captured, response),
        )

        client = TestClient(create_app())
        resp = client.get("/api/tb/abc/data/scalars")

        assert resp.status_code == 200
        assert resp.content == b'{"ok": true}'
        # Custom upstream header round-trips.
        assert resp.headers.get("x-tb-plugin") == "scalars"
        # Content-Type makes it back too.
        assert resp.headers.get("content-type", "").startswith("application/json")

    def test_authorization_header_not_forwarded_upstream(
        self, isolated_auth, stub_records, monkeypatch
    ):
        from fastapi.testclient import TestClient
        from forgather_server.app import create_app

        stub_records["abc"] = _make_tb_record("abc")

        captured: List[Dict[str, Any]] = []
        response = _make_response()
        monkeypatch.setattr(
            tb_proxy.httpx,
            "AsyncClient",
            _client_factory(captured, response),
        )

        client = TestClient(create_app())
        # Even if the browser sent an Authorization header (unusual
        # because cookie auth is the norm), the proxy must not relay
        # it to the loopback TB process.
        resp = client.get(
            "/api/tb/abc/",
            headers={"authorization": "Bearer s3cret"},
        )
        assert resp.status_code == 200
        assert len(captured) == 1
        sent_headers = {k.lower(): v for k, v in captured[0]["headers"].items()}
        assert "authorization" not in sent_headers

    def test_upstream_502_on_connect_error(
        self, isolated_auth, stub_records, monkeypatch
    ):
        from fastapi.testclient import TestClient
        from forgather_server.app import create_app

        stub_records["abc"] = _make_tb_record("abc")

        # Force the AsyncClient.send to raise a connect-style error.
        class _FailingClient(_FakeAsyncClient):
            async def send(self, request, *, stream=False):  # type: ignore[override]
                raise httpx.ConnectError("connection refused")

        def _factory(*args, **kwargs):
            return _FailingClient([], _make_response())

        monkeypatch.setattr(tb_proxy.httpx, "AsyncClient", _factory)

        client = TestClient(create_app())
        resp = client.get("/api/tb/abc/")
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Path-prefix wiring at dispatch time
# ---------------------------------------------------------------------------


class TestSchedulerPathPrefixWiring:
    def test_launch_records_path_prefix_for_tb_jobs(self, monkeypatch, tmp_path):
        """``_launch`` should stamp ``path_prefix`` onto the JobRecord
        for tensorboard jobs so the proxy and the spawned TB CLI agree
        on the URL space."""
        from forgather_server import scheduler
        from forgather_server.queue_store import QueueItem

        # Capture add_record / update_record / remove_item without
        # actually writing anywhere on disk.
        recorded: List[job_records.JobRecord] = []

        def fake_add(rec):
            recorded.append(rec)
            return rec

        def fake_update(qid, **changes):
            return None

        def fake_remove(qid):
            return True

        monkeypatch.setattr(scheduler.job_records, "add_record", fake_add)
        monkeypatch.setattr(scheduler.job_records, "update_record", fake_update)
        monkeypatch.setattr(scheduler.queue_store, "remove_item", fake_remove)
        monkeypatch.setattr(
            scheduler.jobs_tty_dir, "__call__", lambda: tmp_path, raising=False
        )
        # jobs_tty_dir is a function — patch the attribute directly.
        monkeypatch.setattr(scheduler, "jobs_tty_dir", lambda: tmp_path)

        # Stub the launcher so we don't actually spawn tensorboard.
        from forgather_server.launcher import LaunchResult

        class _Stub:
            pid = 12345
            pgid = 12345

        def fake_spawn(*args, **kwargs):
            return LaunchResult(
                proc=_Stub(),  # type: ignore[arg-type]
                pid=12345,
                pgid=12345,
                cmd=["tensorboard"],
                tty_log_path=tmp_path / "x.tty",
            )

        monkeypatch.setattr(scheduler.launcher, "spawn_tensorboard_process", fake_spawn)

        # Also stub get_record for the builder's lookup.
        def fake_get_record(qid):
            for r in recorded:
                if r.queue_id == qid:
                    return r
            return None

        monkeypatch.setattr(scheduler.job_records, "get_record", fake_get_record)

        item = QueueItem(
            queue_id="tbjob1",
            project_dir="",
            config="",
            job_type="tensorboard",
            requested_gpus=0,
            job_params={"logdir": "/tmp/r", "port": 6006},
        )
        scheduler._launch(item, [])

        assert len(recorded) == 1
        assert recorded[0].path_prefix == "/api/tb/tbjob1"

    def test_launch_omits_path_prefix_for_non_tb_jobs(self, monkeypatch, tmp_path):
        from forgather_server import scheduler
        from forgather_server.queue_store import QueueItem

        recorded: List[job_records.JobRecord] = []

        monkeypatch.setattr(
            scheduler.job_records, "add_record", lambda r: (recorded.append(r), r)[1]
        )
        monkeypatch.setattr(
            scheduler.job_records, "update_record", lambda *a, **k: None
        )
        monkeypatch.setattr(scheduler.queue_store, "remove_item", lambda qid: True)
        monkeypatch.setattr(scheduler, "jobs_tty_dir", lambda: tmp_path)

        # Builder for unknown job_type falls back to training; stub it.
        from forgather_server.launcher import LaunchResult

        class _Stub:
            pass

        monkeypatch.setattr(
            scheduler.launcher,
            "spawn_training_process",
            lambda **kwargs: LaunchResult(
                proc=_Stub(),  # type: ignore[arg-type]
                pid=1,
                pgid=1,
                cmd=["torchrun"],
                tty_log_path=tmp_path / "x.tty",
            ),
        )

        item = QueueItem(
            queue_id="trainjob1",
            project_dir="/tmp/proj",
            config="cfg.yaml",
            job_type="training",
            requested_gpus=1,
        )
        scheduler._launch(item, [0])

        assert len(recorded) == 1
        assert recorded[0].path_prefix is None
