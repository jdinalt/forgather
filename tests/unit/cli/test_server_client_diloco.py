"""ServerClient DiLoCo methods: URL + request-body construction.

Constructs a real ServerClient (offline — no token file, http base so the
TLS branch is skipped) and swaps in a recording session, so we assert the
exact path/query/body each method sends without a live server.
"""

import asyncio

import pytest

from forgather.cli.server_client import ServerClient, ServerUnreachable


class _Resp:
    ok = True
    status_code = 200

    def __init__(self, payload=None, content=b""):
        self._payload = payload if payload is not None else {}
        self.content = content

    def json(self):
        return self._payload

    @property
    def text(self):
        return ""


class _RecordingSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None, **kwargs):
        self.calls.append(("GET", url, kwargs.get("params"), None))
        return _Resp()

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, None, json))
        return _Resp()

    def delete(self, url, timeout=None):
        self.calls.append(("DELETE", url, None, None))
        return _Resp()


def _client():
    c = ServerClient("http://127.0.0.1:8765")
    c.session = _RecordingSession()
    return c


def _last(c):
    return c.session.calls[-1]


def test_server_status_encodes_base():
    c = _client()
    c.diloco_server_status("http://192.168.9.43:8512")
    method, url, _, _ = _last(c)
    assert method == "GET"
    assert url == (
        "http://127.0.0.1:8765/api/diloco/server-status"
        "?base=http%3A%2F%2F192.168.9.43%3A8512"
    )


def test_work_queue_encodes_all_params():
    c = _client()
    c.diloco_work_queue("https://h:8512", "c4/en", 1234)
    _, url, _, _ = _last(c)
    assert "/api/diloco/work-queue?base=https%3A%2F%2Fh%3A8512" in url
    assert "dataset_id=c4%2Fen" in url
    assert "shuffle_seed=1234" in url


def test_stats_history_encodes_base_and_max_points():
    c = _client()
    c.diloco_stats_history("https://h:8512", max_points=500)
    method, url, _, _ = _last(c)
    assert method == "GET"
    assert "/api/diloco/stats-history?base=https%3A%2F%2Fh%3A8512" in url
    assert "max_points=500" in url


def test_generate_worker_names_body():
    c = _client()
    c.generate_diloco_worker_names(3, exclude=["a", "b"])
    method, url, _, body = _last(c)
    assert method == "POST"
    assert url.endswith("/api/diloco/generate-worker-names")
    assert body == {"count": 3, "exclude": ["a", "b"]}


def test_add_registry_body():
    c = _client()
    c.add_diloco_registry(
        base_url="https://h:8512", label="L", auth_token="t", verify_tls=False
    )
    _, url, _, body = _last(c)
    assert url.endswith("/api/diloco/registry")
    assert body == {
        "base_url": "https://h:8512",
        "verify_tls": False,
        "label": "L",
        "auth_token": "t",
    }


def test_server_control_body_and_query():
    c = _client()
    c.diloco_server_control(
        "command", "http://h:8512", command="save_and_stop", worker_id="w0"
    )
    method, url, _, body = _last(c)
    assert method == "POST"
    assert url == (
        "http://127.0.0.1:8765/api/diloco/server-control/command"
        "?base=http%3A%2F%2Fh%3A8512"
    )
    assert body == {"command": "save_and_stop", "worker_id": "w0"}


def test_enqueue_job_includes_dataset_source_only_when_set():
    c = _client()
    c.enqueue_job(
        project_dir="/p",
        config="cfg",
        job_type="training",
        job_params={"diloco": {"server_addr": "https://h:8512"}},
        dataset_source={"kind": "auto"},
    )
    _, _, _, body = _last(c)
    assert body["dataset_source"] == {"kind": "auto"}

    c2 = _client()
    c2.enqueue_job(project_dir="/p", config="cfg", job_type="training", job_params={})
    _, _, _, body2 = _last(c2)
    assert "dataset_source" not in body2


def test_job_tty_path_url():
    c = _client()
    c.job_tty_path("qZ")
    method, url, _, _ = _last(c)
    assert method == "GET" and url.endswith("/api/jobs/qZ/tty-path")


def test_ping_true_on_ok(monkeypatch):
    c = _client()
    assert c.ping() is True
    method, url, _, _ = _last(c)
    assert method == "GET" and url.endswith("/api/health")


# --- WebSocket TLS (the --follow / stream_tty fix) ---


def test_ws_ssl_context_no_hostname():
    c = _client()
    c._tls_bundle = None  # system trust; we only assert the hostname policy
    c._verify_hostname = False
    ctx = c._ws_ssl_context()
    assert ctx.check_hostname is False


def test_ws_ssl_context_verifies_hostname_by_default():
    c = _client()
    c._tls_bundle = None
    c._verify_hostname = True
    ctx = c._ws_ssl_context()
    assert ctx.check_hostname is True


def _drive_stream_tty(client):
    """Pull the first item from stream_tty so the websockets.connect call
    happens (then the fake connect aborts it)."""

    async def _run():
        agen = client.stream_tty("job1", follow=True)
        with pytest.raises(ServerUnreachable):
            await agen.__anext__()

    asyncio.run(_run())


def test_stream_tty_passes_ssl_for_wss(monkeypatch):
    """wss:// must get an SSL context built from the cluster trust material;
    without it the handshake hits the system store and rejects the
    self-signed cert (the --follow 'could not reach' bug)."""
    import websockets

    captured = {}

    async def fake_connect(url, **kw):
        captured["url"] = url
        captured["kw"] = kw
        raise OSError("stop after capture")

    monkeypatch.setattr(websockets, "connect", fake_connect)
    c = ServerClient("https://127.0.0.1:8765")  # wss
    c.session = _RecordingSession()
    _drive_stream_tty(c)
    assert captured["url"].startswith("wss://")
    assert "ssl" in captured["kw"] and captured["kw"]["ssl"] is not None


def test_stream_tty_no_ssl_for_ws(monkeypatch):
    import websockets

    captured = {}

    async def fake_connect(url, **kw):
        captured["url"] = url
        captured["kw"] = kw
        raise OSError("stop after capture")

    monkeypatch.setattr(websockets, "connect", fake_connect)
    c = ServerClient("http://127.0.0.1:8765")  # ws
    c.session = _RecordingSession()
    _drive_stream_tty(c)
    assert captured["url"].startswith("ws://")
    assert "ssl" not in captured["kw"]
