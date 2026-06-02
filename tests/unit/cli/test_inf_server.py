"""Unit tests for `forgather inf server` scheduler-by-default routing.

`inf server` is a long-running service, so it submits to the scheduler by
default; --local-only runs it in the foreground, --local-fallback foregrounds
only when the server is down. ServerClient and the foreground runner are
monkeypatched.
"""

import argparse

import pytest

from forgather.cli import inference as inf
from forgather.cli import server_client


def test_pop_flag():
    toks = ["-m", "x", "--local-only", "--port", "8137"]
    assert inf._pop_flag(toks, "--local-only") is True
    assert toks == ["-m", "x", "--port", "8137"]
    assert inf._pop_flag(toks, "--local-only") is False


def test_strip_value_flags():
    toks = ["-m", "x", "--priority", "5", "--server", "http://h", "--port", "8137"]
    out = inf._strip_value_flags(toks, {"--priority", "--server"})
    assert out == ["-m", "x", "--port", "8137"]
    # equals form too
    assert inf._strip_value_flags(["--server=http://h", "-m", "x"], {"--server"}) == [
        "-m",
        "x",
    ]


class _FakeClient:
    def __init__(self, server=None, *, up=True):
        self.server = server
        self._up = up
        self.base = server or "http://127.0.0.1:8765"
        self.enqueued = None

    def ping(self):
        return self._up

    def enqueue_job(self, **kw):
        self.enqueued = kw
        return {"queue_id": "qi-1", "priority": kw["priority"]}


def _args(remainder):
    return argparse.Namespace(subcommand="server", project_dir=".", remainder=remainder)


def _fg_capture(captured):
    def _run(server_args):
        captured["args"] = server_args
        return 0

    return _run


def test_local_only_runs_foreground(monkeypatch):
    captured = {}
    monkeypatch.setattr(inf, "_run_server_foreground", _fg_capture(captured))
    rc = inf.server_cmd(_args(["--local-only", "-m", "/model", "--port", "8137"]))
    assert rc == 0
    assert captured["args"] == ["-m", "/model", "--port", "8137"]


def test_default_enqueues(monkeypatch, capsys):
    fake = _FakeClient(up=True)
    monkeypatch.setattr(server_client, "ServerClient", lambda server=None: fake)
    rc = inf.server_cmd(_args(["-m", "/model", "--port", "8137"]))
    assert rc == 0
    assert fake.enqueued["job_type"] == "inference"
    assert fake.enqueued["job_params"]["model_path"].endswith("/model")
    assert "queued: qi-1" in capsys.readouterr().out


def test_unreachable_no_fallback_errors(monkeypatch):
    monkeypatch.setattr(
        server_client, "ServerClient", lambda server=None: _FakeClient(up=False)
    )
    with pytest.raises(SystemExit) as exc:
        inf.server_cmd(_args(["-m", "/model"]))
    assert exc.value.code == 1


def test_unreachable_with_fallback_foregrounds(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        server_client, "ServerClient", lambda server=None: _FakeClient(up=False)
    )
    monkeypatch.setattr(inf, "_run_server_foreground", _fg_capture(captured))
    rc = inf.server_cmd(
        _args(["--local-fallback", "-m", "/model", "--priority", "2", "--port", "8137"])
    )
    assert rc == 0
    # forgather-only flags stripped before forwarding to the server script
    assert "--priority" not in captured["args"]
    assert captured["args"] == ["-m", "/model", "--port", "8137"]
