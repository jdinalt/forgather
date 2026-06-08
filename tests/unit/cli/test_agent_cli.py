"""`forgather agent` CLI client: arg parsing, SSE rendering, STATE logic."""

from __future__ import annotations

import json

from forgather.cli import agent
from forgather.cli.agent_args import create_agent_parser


# ---- arg parsing -----------------------------------------------------------


def test_parser_subcommands():
    p = create_agent_parser(None)
    ns = p.parse_args(["message", "hello", "--session", "s1"])
    assert ns.agent_subcommand == "message" and ns.text == "hello" and ns.session == "s1"
    assert p.parse_args(["approve", "a1"]).action_id == "a1"
    r = p.parse_args(["reject", "a2", "--reason", "use Y"])
    assert r.action_id == "a2" and r.reason == "use Y"
    assert p.parse_args(["use", "p1"]).profile_id == "p1"


# ---- SSE stream fakes ------------------------------------------------------


def _sse(ev):
    return "data: " + json.dumps(ev)


class _Resp:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status
        self.text = ""

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def json(self):
        return {"detail": "boom"}


class _Sess:
    def __init__(self, resp):
        self._resp = resp
        self.headers = {}

    def post(self, url, json=None, stream=None, timeout=None):
        return self._resp


class _Client:
    base = "http://x:8765"

    def __init__(self, resp):
        self.session = _Sess(resp)

    def _url(self, path):
        return self.base + "/api" + path


def test_stream_awaiting_approval(capsys):
    lines = [
        _sse({"type": "session", "session_id": "s1"}),
        _sse({"type": "text", "text": "I'll start it."}),
        _sse({"type": "tool_use", "id": "t1", "name": "start_inference_server", "input": {"port": 8137}}),
        _sse({"type": "action_card", "action_id": "a1", "risk": "confirm",
              "title": "Start service: inference:agent-default", "summary": "spawn",
              "extra": {"command": "forgather ... inf server", "job_type": "inference"}}),
        _sse({"type": "awaiting_approval", "session_id": "s1", "outstanding": ["t1"]}),
    ]
    st = agent._stream(_Client(_Resp(lines)), "/agent/message", {}, as_json=False)
    assert st["session_id"] == "s1"
    assert [c["action_id"] for c in st["pending"]] == ["a1"]
    out = capsys.readouterr().out
    assert "AWAITING_APPROVAL" in out and "a1" in out
    assert "forgather agent approve" in out  # next-step hint
    assert "start_inference_server" in out  # tool call shown


def test_stream_done_asks_question(capsys):
    lines = [
        _sse({"type": "session", "session_id": "s2"}),
        _sse({"type": "text", "text": "Which config should I use?"}),
        _sse({"type": "done", "session_id": "s2", "reason": "end_turn", "incomplete": False}),
    ]
    st = agent._stream(_Client(_Resp(lines)), "/agent/message", {}, as_json=False)
    assert st["done"]["reason"] == "end_turn" and not st["pending"]
    out = capsys.readouterr().out
    assert "DONE" in out and "session=s2" in out
    assert "forgather agent message --session s2" in out  # follow-up hint


def test_stream_incomplete_hints_continue(capsys):
    lines = [
        _sse({"type": "session", "session_id": "s3"}),
        _sse({"type": "done", "session_id": "s3", "reason": "max_tokens", "incomplete": True}),
    ]
    agent._stream(_Client(_Resp(lines)), "/agent/message", {}, as_json=False)
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out and "forgather agent continue --session s3" in out


def test_stream_json_mode_emits_raw_events(capsys):
    lines = [
        _sse({"type": "session", "session_id": "s4"}),
        _sse({"type": "done", "session_id": "s4", "reason": "end_turn", "incomplete": False}),
    ]
    agent._stream(_Client(_Resp(lines)), "/agent/message", {}, as_json=True)
    out = capsys.readouterr().out
    # Raw event JSONL is present (a line that parses back to the session event).
    assert any(json.loads(l).get("type") == "session"
               for l in out.splitlines() if l.startswith("{"))


def test_stream_http_error_returns_error(capsys):
    st = agent._stream(_Client(_Resp([], status=503)), "/agent/message", {}, as_json=False)
    assert st["error"] == "boom"


# ---- command dispatch (mocked client) --------------------------------------


class _GetClient:
    def __init__(self, payload):
        self._payload = payload

    def _get(self, path):
        class _R:
            def json(_self):
                return self._payload
        return _R()


def test_cmd_profiles_marks_active(capsys):
    client = _GetClient({
        "active_id": "p1",
        "profiles": [
            {"id": "p1", "provider": "anthropic", "model": "m", "base_url": "", "label": "Live"},
            {"id": "p2", "provider": "anthropic", "model": "q", "base_url": "http://k:8000", "label": "Local"},
        ],
    })
    rc = agent._cmd_profiles(client, None)
    out = capsys.readouterr().out
    assert rc == 0 and "p1" in out and "p2" in out and "active: p1" in out
    assert "* p1" in out  # active marker


def test_cmd_sessions_lists(capsys):
    import argparse
    client = _GetClient({"sessions": [
        {"session_id": "abc", "message_count": 4, "awaiting_approval": False, "updated_at": 0},
        {"session_id": "def", "message_count": 1, "awaiting_approval": True, "updated_at": 0},
    ]})
    rc = agent._cmd_sessions(client, argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert rc == 0 and "abc" in out and "def" in out and "session_id" in out


def test_cmd_sessions_empty(capsys):
    import argparse
    rc = agent._cmd_sessions(_GetClient({"sessions": []}), argparse.Namespace(json=False))
    assert rc == 0 and "no active sessions" in capsys.readouterr().out


def test_cmd_reject_sends_reason():
    import argparse
    captured = {}

    class _CapSess:
        headers = {}

        def post(self, url, json=None, stream=None, timeout=None):
            captured["json"] = json
            return _Resp([
                _sse({"type": "action_resolved", "action_id": "a1", "approved": False, "reason": "use Y"}),
                _sse({"type": "done", "session_id": "s", "reason": "end_turn", "incomplete": False}),
            ])

    class _CapClient:
        base = "http://x:8765"

        def __init__(self):
            self.session = _CapSess()

        def _url(self, p):
            return self.base + "/api" + p

    agent._cmd_reject(_CapClient(), argparse.Namespace(action_id="a1", reason="use Y", json=False))
    assert captured["json"] == {"action_id": "a1", "reason": "use Y"}


def test_cmd_forget_deletes():
    import argparse
    captured = {}

    class _Client:
        def _delete(self, path):
            captured["path"] = path

    rc = agent._cmd_forget(_Client(), argparse.Namespace(session_id="sess_x"))
    assert rc == 0 and captured["path"] == "/agent/sessions/sess_x"
