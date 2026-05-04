"""Auth-token tests for TrainerControlCallback.

Covers:

1. Token is generated on rank-0 ``on_train_begin`` and persisted at
   ``~/.forgather/jobs/{job_id}/auth_token`` with mode 0o600.
2. The aiohttp middleware accepts a valid bearer and rejects missing /
   wrong tokens with 401 + ``WWW-Authenticate: Bearer ...``.
3. ``disable_auth=True`` lets unauthenticated requests through and emits
   a WARNING.
4. ``HTTPTrainerControlClient`` automatically attaches the bearer header
   when ``auth_token`` is on disk for that job.

The middleware tests run the real aiohttp app via ``aiohttp.test_utils``
so we exercise the same code path the trainer uses in production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from forgather.ml.trainer.callbacks.control_callback import (  # noqa: E402
    TrainerControlCallback,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect ``Path.home()`` for the duration of the test."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _fake_state(rank_zero: bool = True):
    s = MagicMock()
    s.is_world_process_zero = rank_zero
    s.global_step = 0
    s.epoch = 0
    s.max_steps = 0
    return s


def _fake_args():
    a = MagicMock()
    a.output_dir = None
    a.logging_dir = None
    a.device = "cpu"
    a.load_best_model_at_end = False
    a.eval_strategy = "no"
    return a


# ----- 1. token persistence -------------------------------------------------


def test_auth_token_written_at_0600(isolated_home):
    cb = TrainerControlCallback(job_id="test_token_perms", enable_http=False)
    cb.auth_token = "deadbeef" * 8
    cb._write_auth_token()
    token_file = isolated_home / ".forgather" / "jobs" / cb.job_id / "auth_token"
    assert token_file.exists()
    assert _mode(token_file) == 0o600
    assert token_file.read_text() == cb.auth_token
    # control_dir itself should be 0o700.
    assert _mode(token_file.parent) == 0o700


def test_disabled_auth_does_not_write_token(isolated_home):
    cb = TrainerControlCallback(
        job_id="test_no_token", enable_http=False, disable_auth=True
    )
    cb.auth_token = None
    cb._write_auth_token()
    token_file = isolated_home / ".forgather" / "jobs" / cb.job_id / "auth_token"
    assert not token_file.exists()


def test_endpoint_json_records_bind_host(isolated_home):
    """endpoint.json["host"] reflects the actual bind address, not FQDN."""
    cb = TrainerControlCallback(
        job_id="test_endpoint_host", enable_http=False, host="127.0.0.1"
    )
    cb._write_endpoint_file(port=12345)
    ep_file = isolated_home / ".forgather" / "jobs" / cb.job_id / "endpoint.json"
    data = json.loads(ep_file.read_text())
    assert data["host"] == "127.0.0.1"
    assert data["port"] == 12345
    assert _mode(ep_file) == 0o600


# ----- 2. middleware behaviour ---------------------------------------------


async def _build_app_with_middleware(cb: TrainerControlCallback):
    """Replicate the routes/middleware setup from ``_run_http_server``
    without binding a TCP socket. The aiohttp test client owns the socket.
    """
    middlewares = []
    if not cb.disable_auth and cb.auth_token:
        middlewares.append(cb._make_auth_middleware())
    app = aiohttp.web.Application(middlewares=middlewares)

    async def status(_request):
        return aiohttp.web.json_response({"job_id": cb.job_id, "ok": True})

    app.router.add_get(f"/jobs/{cb.job_id}/status", status)
    return app


async def _exec_request(cb, *, headers=None):
    """Drive the aiohttp test client end-to-end and return (status, headers, body)."""
    app = await _build_app_with_middleware(cb)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get(f"/jobs/{cb.job_id}/status", headers=headers or {})
        try:
            body = await resp.json()
        except Exception:
            body = None
        return resp.status, dict(resp.headers), body


def test_valid_bearer_succeeds(isolated_home):
    cb = TrainerControlCallback(job_id="test_valid", enable_http=False)
    cb.auth_token = "a" * 64
    status, _, body = asyncio.run(
        _exec_request(cb, headers={"Authorization": f"Bearer {cb.auth_token}"})
    )
    assert status == 200
    assert body["ok"] is True


def test_missing_bearer_returns_401(isolated_home):
    cb = TrainerControlCallback(job_id="test_missing", enable_http=False)
    cb.auth_token = "b" * 64
    status, headers, body = asyncio.run(_exec_request(cb))
    assert status == 401
    assert headers.get("WWW-Authenticate") == 'Bearer realm="forgather-trainer"'
    assert body["detail"] == "authentication required"


def test_wrong_bearer_returns_401(isolated_home):
    cb = TrainerControlCallback(job_id="test_wrong", enable_http=False)
    cb.auth_token = "c" * 64
    status, _, _ = asyncio.run(
        _exec_request(cb, headers={"Authorization": "Bearer not-the-right-token"})
    )
    assert status == 401


def test_disable_auth_allows_anonymous(isolated_home):
    cb = TrainerControlCallback(
        job_id="test_disable", enable_http=False, disable_auth=True
    )
    # No middleware should be installed when auth is disabled.
    status, _, _ = asyncio.run(_exec_request(cb))
    assert status == 200


# ----- 3. on_train_begin generates token + warns ---------------------------


def test_on_train_begin_generates_token_and_persists(isolated_home, caplog):
    cb = TrainerControlCallback(job_id="test_otb", port=0, host="127.0.0.1")
    state = _fake_state(rank_zero=True)
    args = _fake_args()
    try:
        with caplog.at_level(logging.INFO):
            cb.on_train_begin(args, state, MagicMock())
        # Token persisted with 0o600.
        token_file = isolated_home / ".forgather" / "jobs" / cb.job_id / "auth_token"
        assert token_file.exists()
        assert _mode(token_file) == 0o600
        assert cb.auth_token and len(cb.auth_token) == 64
        assert token_file.read_text() == cb.auth_token
    finally:
        cb.on_train_end(args, state, MagicMock())


def test_on_train_begin_logs_warning_for_non_loopback_host(isolated_home, caplog):
    """0.0.0.0 (or other non-loopback) bind triggers an explicit warning."""
    cb = TrainerControlCallback(
        job_id="test_exposed", port=0, host="0.0.0.0", enable_http=False
    )
    # enable_http=False so we don't actually start the server thread; the
    # warning is emitted before that branch.
    cb.enable_http = True
    state = _fake_state(rank_zero=True)
    args = _fake_args()
    try:
        with caplog.at_level(logging.WARNING):
            cb.on_train_begin(args, state, MagicMock())
        assert any(
            "exposed beyond loopback" in rec.getMessage()
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        )
    finally:
        cb.on_train_end(args, state, MagicMock())


def test_on_train_begin_logs_warning_when_auth_disabled(isolated_home, caplog):
    cb = TrainerControlCallback(
        job_id="test_no_auth_warn", port=0, host="127.0.0.1", disable_auth=True
    )
    state = _fake_state(rank_zero=True)
    args = _fake_args()
    try:
        with caplog.at_level(logging.WARNING):
            cb.on_train_begin(args, state, MagicMock())
        assert any("auth DISABLED" in rec.getMessage() for rec in caplog.records)
        # And no token file should be on disk.
        token_file = isolated_home / ".forgather" / "jobs" / cb.job_id / "auth_token"
        assert not token_file.exists()
    finally:
        cb.on_train_end(args, state, MagicMock())


# ----- 4. trainer_control client picks up the token ------------------------


def test_http_client_loads_token_from_disk(isolated_home):
    from forgather.trainer_control import HTTPTrainerControlClient

    job_id = "test_client_picks_up"
    job_dir = isolated_home / ".forgather" / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "auth_token").write_text("xxxxx")
    assert HTTPTrainerControlClient._load_auth_token(job_id) == "xxxxx"
    assert HTTPTrainerControlClient._auth_headers(job_id) == {
        "Authorization": "Bearer xxxxx"
    }


def test_http_client_no_token_no_header(isolated_home):
    from forgather.trainer_control import HTTPTrainerControlClient

    assert HTTPTrainerControlClient._load_auth_token("nonexistent_job") is None
    assert HTTPTrainerControlClient._auth_headers("nonexistent_job") == {}


def test_http_client_send_command_attaches_bearer(isolated_home, monkeypatch):
    """Verify ``send_command`` actually sets the Authorization header."""
    from forgather.trainer_control import HTTPTrainerControlClient

    job_id = "test_send_attach"
    job_dir = isolated_home / ".forgather" / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "auth_token").write_text("topsecret")
    # endpoint.json is required for _get_job_info to succeed.
    (job_dir / "endpoint.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "host": "127.0.0.1",
                "port": 9999,
                "pid": 1,
                "started_at": 0.0,
            }
        )
    )

    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"message": "ok"}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return FakeResponse()

    client = HTTPTrainerControlClient()
    monkeypatch.setattr(client.session, "post", fake_post)
    resp = client.send_command(job_id, "graceful_stop")
    assert resp.success
    assert captured["headers"] == {"Authorization": "Bearer topsecret"}
    assert captured["url"].endswith(f"/jobs/{job_id}/control")
