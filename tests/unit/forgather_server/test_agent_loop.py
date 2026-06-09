"""Unit tests for the agent loop + server-side approval gate.

Driven with a scripted fake ChatProvider so they need no network, no
``anthropic`` package, and no running server. Async generators are driven
directly with ``asyncio.run`` (no pytest-asyncio dependency).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from forgather_server.agent.loop import AgentLoop
from forgather_server.agent.providers.base import Done, TextDelta, ToolCall
from forgather_server.agent.registry import Proposal, ToolRegistry, ToolSpec, UiDirective
from forgather_server.agent.session import Conversation


class FakeProvider:
    """Emits a scripted list of events per ``stream_turn`` call."""

    def __init__(self, turns: List[List[Any]]):
        self.turns = list(turns)
        self.calls = 0

    async def stream_turn(self, messages, tools, *, system=None):
        turn = self.turns[self.calls]
        self.calls += 1
        for ev in turn:
            yield ev

    def format_tool_result(self, tool_use_id, content, *, is_error=False):
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }


def _collect(agen) -> List[Dict[str, Any]]:
    async def go():
        return [ev async for ev in agen]

    return asyncio.run(go())


def _types(events):
    return [e["type"] for e in events]


def _make_registry(state: Dict[str, int]):
    reg = ToolRegistry()

    def echo_read(args):
        state["read_calls"] += 1
        return f"READ_OK x={args.get('x')}"

    def make_change(args):
        # Preview only — must NOT have any side effect here.
        return Proposal(
            title=f"edit {args['path']}",
            path=args["path"],
            before="OLD",
            after="NEW",
            commit=lambda: _do_commit(state),
        )

    reg.register(
        ToolSpec(
            name="echo_read",
            description="echo",
            json_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            handler=echo_read,
            risk="read",
        )
    )
    reg.register(
        ToolSpec(
            name="make_change",
            description="propose a change",
            json_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            handler=make_change,
            risk="propose",
        )
    )
    return reg


def _do_commit(state):
    state["commit_calls"] += 1
    return "WROTE"


# --------------------------------------------------------------------------


def test_read_tool_runs_automatically():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="echo_read", arguments={"x": 7}), Done()],
            [TextDelta("all done"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s1")

    events = _collect(loop.run_user_message(conv, "hi"))

    assert state["read_calls"] == 1
    assert state["commit_calls"] == 0
    results = [e for e in events if e["type"] == "tool_result"]
    assert results and results[0]["content"] == "READ_OK x=7"
    assert results[0]["is_error"] is False
    assert events[-1]["type"] == "done"
    assert conv.pending_turn is None


def test_propose_pauses_without_side_effect():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [[ToolCall(id="t1", name="make_change", arguments={"path": "f.yaml"}), Done()]]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s2")

    events = _collect(loop.run_user_message(conv, "change f.yaml"))

    assert state["commit_calls"] == 0  # nothing applied yet
    cards = [e for e in events if e["type"] == "action_card"]
    assert len(cards) == 1
    assert cards[0]["after"] == "NEW"
    assert "awaiting_approval" in _types(events)
    assert conv.pending_turn is not None
    assert provider.calls == 1  # paused; provider not called again


def test_action_card_carries_verbatim_proposed_args():
    # Every approval card must surface the raw tool-call args the agent
    # proposed, independent of what the tool curated into ``extra`` — so the
    # user can always audit exactly what they're approving.
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    args = {"path": "f.yaml"}
    provider = FakeProvider(
        [[ToolCall(id="t1", name="make_change", arguments=args), Done()]]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s-args")

    events = _collect(loop.run_user_message(conv, "change f.yaml"))

    card = [e for e in events if e["type"] == "action_card"][0]
    assert card["proposed_args"] == {"path": "f.yaml"}


def test_approve_replays_commit_and_resumes():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="make_change", arguments={"path": "f.yaml"}), Done()],
            [TextDelta("applied"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s3")
    # NOTE: session module must know this conversation for apply_decision
    # to find it. run_user_message uses the conv we pass; register it.
    from forgather_server.agent import session as sess

    with sess._state._lock:
        sess._state.sessions[conv.session_id] = conv

    first = _collect(loop.run_user_message(conv, "change f.yaml"))
    action_id = [e for e in first if e["type"] == "action_card"][0]["action_id"]

    resumed = _collect(loop.apply_decision(action_id, approve=True))

    assert state["commit_calls"] == 1
    resolved = [e for e in resumed if e["type"] == "action_resolved"]
    assert resolved and resolved[0]["approved"] is True
    assert resumed[-1]["type"] == "done"
    assert conv.pending_turn is None
    assert provider.calls == 2


def test_reject_does_not_commit_but_resumes():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="make_change", arguments={"path": "f.yaml"}), Done()],
            [TextDelta("ok, skipping"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s4")
    from forgather_server.agent import session as sess

    with sess._state._lock:
        sess._state.sessions[conv.session_id] = conv

    first = _collect(loop.run_user_message(conv, "change f.yaml"))
    action_id = [e for e in first if e["type"] == "action_card"][0]["action_id"]

    resumed = _collect(loop.apply_decision(action_id, approve=False))

    assert state["commit_calls"] == 0
    resolved = [e for e in resumed if e["type"] == "action_resolved"]
    assert resolved and resolved[0]["approved"] is False
    assert resumed[-1]["type"] == "done"
    assert conv.pending_turn is None


def test_reject_with_reason_is_fed_to_the_model():
    import json

    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="make_change", arguments={"path": "f.yaml"}), Done()],
            [TextDelta("ok, using Y instead"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s4r")
    from forgather_server.agent import session as sess

    with sess._state._lock:
        sess._state.sessions[conv.session_id] = conv

    first = _collect(loop.run_user_message(conv, "change f.yaml"))
    action_id = [e for e in first if e["type"] == "action_card"][0]["action_id"]

    resumed = _collect(
        loop.apply_decision(action_id, approve=False, reason="use config Y instead")
    )
    resolved = [e for e in resumed if e["type"] == "action_resolved"][0]
    assert resolved["reason"] == "use config Y instead"
    assert state["commit_calls"] == 0
    # The reason reached the model as the rejected tool's result content.
    assert "use config Y instead" in json.dumps(conv.messages, default=str)


def test_missing_required_arg_is_error_not_crash():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="echo_read", arguments={}), Done()],
            [TextDelta("recovered"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s5")

    events = _collect(loop.run_user_message(conv, "go"))

    assert state["read_calls"] == 0  # never executed
    errs = [e for e in events if e["type"] == "tool_result" and e["is_error"]]
    assert errs and "missing required" in errs[0]["content"]
    assert events[-1]["type"] == "done"


def test_malformed_tool_json_is_error_not_crash():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="echo_read", arguments={}, parse_error="boom"), Done()],
            [TextDelta("recovered"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s6")

    events = _collect(loop.run_user_message(conv, "go"))

    assert state["read_calls"] == 0
    errs = [e for e in events if e["type"] == "tool_result" and e["is_error"]]
    assert errs and "could not parse" in errs[0]["content"]
    assert events[-1]["type"] == "done"


def test_iteration_cap_yields_incomplete_done():
    # A provider that keeps calling a read tool never finishes; the loop should
    # stop at the cap with an *incomplete* done (so the UI offers Continue),
    # not an error.
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)

    class LoopingProvider:
        calls = 0

        async def stream_turn(self, messages, tools, *, system=None):
            LoopingProvider.calls += 1
            yield ToolCall(id=f"t{LoopingProvider.calls}", name="echo_read", arguments={"x": 1})
            yield Done()

        def format_tool_result(self, tool_use_id, content, *, is_error=False):
            return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}

    loop = AgentLoop(LoopingProvider(), reg, max_iterations=2)
    conv = Conversation(session_id="cap")
    events = _collect(loop.run_user_message(conv, "go"))

    done = [e for e in events if e["type"] == "done"]
    assert done and done[-1]["incomplete"] is True
    assert done[-1]["reason"] == "max_iterations"


def test_continue_turn_resumes():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    # Conversation ended on an assistant message (e.g. truncated). Continue
    # should nudge with a user turn and produce more output.
    provider = FakeProvider([[TextDelta("more"), Done()]])
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="cont")
    conv.messages.append({"role": "assistant", "content": [{"type": "text", "text": "partial"}]})

    events = _collect(loop.continue_turn(conv))
    assert events[-1]["type"] == "done"
    # A user "continue" nudge was appended before the assistant's new turn.
    assert any(
        m["role"] == "user"
        and any(b.get("text") == "Please continue." for b in m["content"])
        for m in conv.messages
    )


def test_continue_turn_refuses_dangling_tool_use():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider([])  # must not be called
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="dangle")
    conv.messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    conv.messages.append(
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]}
    )
    events = _collect(loop.continue_turn(conv))
    assert events[-1]["type"] == "error"
    assert "mid-tool-call" in events[-1]["message"]
    assert conv.messages[-1]["role"] == "assistant"  # no user nudge appended
    assert provider.calls == 0


def test_incomplete_done_on_length_stop_reason():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider([[TextDelta("cut off"), Done(stop_reason="length")]])
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="len")
    events = _collect(loop.run_user_message(conv, "go"))
    done = [e for e in events if e["type"] == "done"]
    assert done and done[-1]["incomplete"] is True  # vLLM "length" == truncated


def test_action_resolved_echoes_reveal_for_create():
    # A create proposal carries reveal_kind; on approval the loop echoes
    # created_kind + created_path so the webui can refresh + reveal the item.
    state = {"read_calls": 0, "commit_calls": 0}
    reg = ToolRegistry()

    def make_project(args):
        return Proposal(
            title="New project: demo",
            path="/ws/demo",
            reveal_kind="project",
            commit=lambda: "created project at /ws/demo",
        )

    reg.register(
        ToolSpec(
            name="make_project",
            description="create a project",
            json_schema={"type": "object", "properties": {}},
            handler=make_project,
            risk="propose",
        )
    )
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="make_project", arguments={}), Done()],
            [TextDelta("done"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="rev")
    from forgather_server.agent import session as sess

    with sess._state._lock:
        sess._state.sessions[conv.session_id] = conv

    first = _collect(loop.run_user_message(conv, "make a project"))
    action_id = [e for e in first if e["type"] == "action_card"][0]["action_id"]
    resumed = _collect(loop.apply_decision(action_id, approve=True))

    resolved = [e for e in resumed if e["type"] == "action_resolved"]
    assert resolved
    assert resolved[0]["created_kind"] == "project"
    assert resolved[0]["created_path"] == "/ws/demo"


def test_action_resolved_reveal_none_for_edit():
    # A non-create proposal (make_change has no reveal_kind) reports
    # created_kind None so the webui knows not to refresh/reveal.
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="make_change", arguments={"path": "f.yaml"}), Done()],
            [TextDelta("applied"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="rev2")
    from forgather_server.agent import session as sess

    with sess._state._lock:
        sess._state.sessions[conv.session_id] = conv

    first = _collect(loop.run_user_message(conv, "change f.yaml"))
    action_id = [e for e in first if e["type"] == "action_card"][0]["action_id"]
    resumed = _collect(loop.apply_decision(action_id, approve=True))
    resolved = [e for e in resumed if e["type"] == "action_resolved"]
    assert resolved and resolved[0]["created_kind"] is None


def test_read_tool_ui_directive_is_emitted():
    # A read tool may return a UiDirective: the loop emits a ui_directive event
    # for the webui and feeds the directive's message back to the model.
    reg = ToolRegistry()

    def reveal(args):
        return UiDirective(
            action="reveal",
            payload={"path": "/p/x", "where": "projects"},
            message="revealed /p/x",
        )

    reg.register(
        ToolSpec(
            name="reveal",
            description="reveal",
            json_schema={"type": "object", "properties": {}},
            handler=reveal,
            risk="read",
        )
    )
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="reveal", arguments={}), Done()],
            [TextDelta("ok"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="ui")
    events = _collect(loop.run_user_message(conv, "show me"))

    directives = [e for e in events if e["type"] == "ui_directive"]
    assert directives and directives[0]["action"] == "reveal"
    assert directives[0]["payload"] == {"path": "/p/x", "where": "projects"}
    results = [e for e in events if e["type"] == "tool_result"]
    assert results and results[0]["content"] == "revealed /p/x"
    assert results[0]["is_error"] is False
    assert events[-1]["type"] == "done"


def test_unknown_tool_is_error_not_crash():
    state = {"read_calls": 0, "commit_calls": 0}
    reg = _make_registry(state)
    provider = FakeProvider(
        [
            [ToolCall(id="t1", name="does_not_exist", arguments={}), Done()],
            [TextDelta("recovered"), Done()],
        ]
    )
    loop = AgentLoop(provider, reg)
    conv = Conversation(session_id="s7")

    events = _collect(loop.run_user_message(conv, "go"))

    errs = [e for e in events if e["type"] == "tool_result" and e["is_error"]]
    assert errs and "unknown tool" in errs[0]["content"]
    assert events[-1]["type"] == "done"


def _bare_loop(mode):
    return AgentLoop(FakeProvider([]), ToolRegistry(), disclosure_mode=mode)


def test_clip_budget_is_provider_aware():
    from forgather_server.agent.loop import (
        _MAX_RESULT_CHARS_DEFERRED,
        _MAX_RESULT_CHARS_INLINE,
    )

    # deferred (small local model) clips far more aggressively than inline.
    assert _MAX_RESULT_CHARS_DEFERRED < _MAX_RESULT_CHARS_INLINE
    big = "z" * (_MAX_RESULT_CHARS_INLINE + 1000)

    inline = _bare_loop("inline")
    deferred = _bare_loop("deferred")

    # A blob that fits the inline budget passes untouched there...
    fits_inline = "z" * (_MAX_RESULT_CHARS_DEFERRED + 100)
    assert inline._clip(fits_inline) == fits_inline
    # ...but is truncated under the deferred budget.
    out = deferred._clip(fits_inline)
    assert "truncated" in out
    assert len(out) < len(fits_inline) + 200

    # The big blob is truncated even inline.
    assert "truncated" in inline._clip(big)


def test_clip_read_file_reports_resume_offset():
    from forgather_server.agent.loop import _MAX_RESULT_CHARS_DEFERRED

    deferred = _bare_loop("deferred")
    big = "y" * (_MAX_RESULT_CHARS_DEFERRED + 5000)

    # A clipped read_file result tells the model the offset to resume from.
    out = deferred._clip(big, tool_name="read_file", args={})
    assert f"offset={_MAX_RESULT_CHARS_DEFERRED}" in out
    assert "read_file again" in out

    # A continuation read (offset already set) advances the resume offset.
    out2 = deferred._clip(big, tool_name="read_file", args={"offset": 1000})
    assert f"offset={_MAX_RESULT_CHARS_DEFERRED + 1000}" in out2

    # A non-read tool gets the generic message (no offset hint).
    out3 = deferred._clip(big, tool_name="gpu_status", args={})
    assert "offset=" not in out3
    assert "truncated" in out3
