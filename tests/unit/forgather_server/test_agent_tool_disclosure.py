"""Hybrid tool disclosure: tiered serialization + meta tools + dispatch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from forgather_server.agent import tools_meta
from forgather_server.agent.loop import AgentLoop
from forgather_server.agent.providers.base import ToolCall
from forgather_server.agent.registry import (
    CONFIRM,
    EXTENDED,
    READ,
    Proposal,
    ToolRegistry,
    ToolSpec,
)
from forgather_server.agent.runtime import _resolve_disclosure_mode
from forgather_server.agent.session import Conversation, PendingTurn


# ---- fixtures --------------------------------------------------------------


def _echo_spec():
    return ToolSpec(
        name="echo",
        description="ECHO FULL: returns the text argument verbatim, at length.",
        summary="echo the text",
        json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda a: {"echoed": a["text"]},
        risk=READ,
        tier=EXTENDED,
    )


def _danger_spec():
    def _handler(a):
        return Proposal(title="dangerous", summary="would do a thing", commit=lambda: "did it")

    return ToolSpec(
        name="danger",
        description="DANGER FULL: a gated mutation.",
        summary="do a gated thing",
        json_schema={"type": "object", "properties": {}},
        handler=_handler,
        risk=CONFIRM,
        tier=EXTENDED,
    )


def _registry():
    reg = ToolRegistry()
    tools_meta.register_all(reg)
    reg.register(_echo_spec())
    reg.register(_danger_spec())
    return reg


class _FakeProvider:
    def format_tool_result(self, tool_use_id, content, *, is_error=False):
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }


async def _drive(loop, tc):
    conv = Conversation(session_id="t-disclosure")
    pt = PendingTurn(assistant_message={"role": "assistant", "content": []})
    events = [ev async for ev in loop._handle_tool_call(conv, tc, pt)]
    return conv, pt, events


# ---- serialization ---------------------------------------------------------


def test_inline_emits_all_with_summary_for_extended():
    reg = _registry()
    tools = {t["name"]: t for t in reg.anthropic_tools("inline")}
    # Extended tool present, shown by its SHORT summary (full prose absent).
    assert "echo" in tools
    assert tools["echo"]["description"] == "echo the text"
    assert "ECHO FULL" not in tools["echo"]["description"]
    # input_schema is always intact (direct callability preserved).
    assert tools["echo"]["input_schema"]["required"] == ["text"]
    # Meta helpers present; call_tool is NOT emitted inline.
    assert "list_tools" in tools and "tool_help" in tools
    assert "call_tool" not in tools


def test_deferred_hides_extended_and_emits_call_tool():
    reg = _registry()
    tools = {t["name"]: t for t in reg.anthropic_tools("deferred")}
    assert "echo" not in tools and "danger" not in tools
    assert "call_tool" in tools and "list_tools" in tools and "tool_help" in tools


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        _registry().anthropic_tools("bogus")


# ---- meta tools ------------------------------------------------------------


def test_tool_help_returns_full_description():
    reg = _registry()
    out = reg.get("tool_help").handler({"name": "echo"})
    assert "ECHO FULL" in out["description"]
    assert out["risk"] == READ and out["tier"] == EXTENDED
    assert out["input_schema"]["required"] == ["text"]


def test_tool_help_unknown_raises():
    reg = _registry()
    with pytest.raises(ValueError):
        reg.get("tool_help").handler({"name": "nope"})


def test_list_tools_lists_everything():
    reg = _registry()
    out = reg.get("list_tools").handler({})
    names = {t["name"] for t in out["tools"]}
    assert {"echo", "danger", "call_tool", "tool_help", "list_tools"} <= names


# ---- dispatch via the loop -------------------------------------------------


def test_call_tool_dispatches_read_inline():
    loop = AgentLoop(_FakeProvider(), _registry(), disclosure_mode="deferred")
    tc = ToolCall(id="tu1", name="call_tool", arguments={"name": "echo", "args": {"text": "hi"}})
    conv, pt, events = asyncio.run(_drive(loop, tc))
    results = [e for e in events if e["type"] == "tool_result"]
    assert results and not results[0]["is_error"]
    assert "hi" in results[0]["content"]
    assert "tu1" in pt.results and not pt.outstanding  # ran inline, no gate


def test_call_tool_preserves_confirm_risk():
    loop = AgentLoop(_FakeProvider(), _registry(), disclosure_mode="deferred")
    tc = ToolCall(id="tu2", name="call_tool", arguments={"name": "danger", "args": {}})
    conv, pt, events = asyncio.run(_drive(loop, tc))
    cards = [e for e in events if e["type"] == "action_card"]
    assert cards and cards[0]["risk"] == CONFIRM  # inner risk, not call_tool's
    assert "tu2" in pt.outstanding  # gated, awaiting approval


def test_call_tool_unknown_inner_errors():
    loop = AgentLoop(_FakeProvider(), _registry(), disclosure_mode="deferred")
    tc = ToolCall(id="tu3", name="call_tool", arguments={"name": "ghost", "args": {}})
    _, pt, events = asyncio.run(_drive(loop, tc))
    errs = [e for e in events if e["type"] == "tool_result" and e["is_error"]]
    assert errs and "ghost" in errs[0]["content"]


def test_call_tool_validates_inner_required_args():
    loop = AgentLoop(_FakeProvider(), _registry(), disclosure_mode="deferred")
    tc = ToolCall(id="tu4", name="call_tool", arguments={"name": "echo", "args": {}})
    _, pt, events = asyncio.run(_drive(loop, tc))
    errs = [e for e in events if e["type"] == "tool_result" and e["is_error"]]
    assert errs and "text" in errs[0]["content"]


# ---- mode resolution -------------------------------------------------------


@pytest.mark.parametrize(
    "base_url,pref,expected",
    [
        ("", "auto", "inline"),          # Claude -> inline
        ("https://kitt:8000", "auto", "deferred"),  # local/vLLM -> deferred
        ("https://kitt:8000", "inline", "inline"),  # explicit override wins
        ("", "deferred", "deferred"),    # explicit override wins
        ("", "", "inline"),              # blank == auto
    ],
)
def test_resolve_disclosure_mode(base_url, pref, expected):
    prof = SimpleNamespace(base_url=base_url, disclosure_mode=pref)
    assert _resolve_disclosure_mode(prof) == expected
