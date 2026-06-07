"""The Anthropic adapter must surface the full per-request token breakdown.

Each loop iteration re-sends the prefix, so reconciling with the billing
dashboard needs input + cache_read + cache_creation + output, not just
input/output. These tests drive ``stream_turn`` over a fake event stream and
pin that the emitted ``Usage`` carries the cache fields. Skipped if the
``anthropic`` package isn't installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from forgather_server.agent.providers.anthropic import (
    AnthropicProvider,
    _add_cache_control,
)
from forgather_server.agent.providers.base import Done, Usage


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e

        return gen()


def _drive(provider, events):
    import asyncio

    async def run():
        out = []
        async for ev in provider.stream_turn([{"role": "user", "content": []}], []):
            out.append(ev)
        return out

    # Fake client: messages.create returns the prepared stream.
    async def fake_create(**_kw):
        return _FakeStream(events)

    provider._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    return asyncio.run(run())


def test_usage_carries_cache_breakdown():
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=120,
                    cache_read_input_tokens=6800,
                    cache_creation_input_tokens=40,
                )
            ),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=55),
        ),
        SimpleNamespace(type="message_stop"),
    ]
    evs = _drive(AnthropicProvider(model="claude-x", api_key="k"), events)
    usages = [e for e in evs if isinstance(e, Usage)]
    assert len(usages) == 1
    u = usages[0]
    assert u.input_tokens == 120
    assert u.cache_read_input_tokens == 6800
    assert u.cache_creation_input_tokens == 40
    assert u.output_tokens == 55
    # Billed input reconciles as the sum of the three input components.
    assert u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens == 6960
    assert any(isinstance(e, Done) for e in evs)


def test_usage_defaults_when_cache_fields_absent():
    # A server that omits cache fields (e.g. vLLM) must not crash; they read 0.
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=200)),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=10),
        ),
        SimpleNamespace(type="message_stop"),
    ]
    evs = _drive(AnthropicProvider(model="qwen", base_url="http://localhost:8000"), events)
    u = [e for e in evs if isinstance(e, Usage)][0]
    assert u.input_tokens == 200
    assert u.cache_read_input_tokens == 0
    assert u.cache_creation_input_tokens == 0
    assert u.output_tokens == 10


# ---- prompt caching (cache_control injection) ------------------------------


def test_add_cache_control_anchors_system_and_last_message():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
    ]
    tools = [{"name": "a"}, {"name": "b"}]
    msgs, system, tools_out = _add_cache_control(messages, "SYS", tools)
    # system becomes a single cached text block (covers tools + system).
    assert system == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    # tools untouched when system anchors the head.
    assert tools_out == tools and "cache_control" not in tools_out[-1]
    # last message's last block gets the breakpoint.
    assert msgs[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # inputs not mutated.
    assert "cache_control" not in messages[-1]["content"][-1]


def test_add_cache_control_falls_back_to_last_tool_without_system():
    tools = [{"name": "a"}, {"name": "b"}]
    msgs, system, tools_out = _add_cache_control(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], None, tools
    )
    assert system is None
    # No system to anchor on -> cache the tools prefix via the last tool.
    assert tools_out[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tools[-1]  # original untouched


def test_add_cache_control_handles_empty_messages():
    msgs, system, tools_out = _add_cache_control([], None, [])
    assert msgs == [] and system is None and tools_out == []


def test_stream_turn_sends_cache_control_when_enabled():
    import asyncio

    captured = {}

    async def fake_create(**kw):
        captured.update(kw)
        return _FakeStream([SimpleNamespace(type="message_stop")])

    p = AnthropicProvider(model="claude-x", api_key="k", prompt_caching=True)
    p._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))

    async def run():
        async for _ in p.stream_turn(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            [{"name": "t"}],
            system="SYS",
        ):
            pass

    asyncio.run(run())
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }


def test_stream_turn_no_cache_control_when_disabled():
    import asyncio

    captured = {}

    async def fake_create(**kw):
        captured.update(kw)
        return _FakeStream([SimpleNamespace(type="message_stop")])

    p = AnthropicProvider(model="qwen", base_url="http://x", prompt_caching=False)
    p._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))

    async def run():
        async for _ in p.stream_turn(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            [{"name": "t"}],
            system="SYS",
        ):
            pass

    asyncio.run(run())
    assert captured["system"] == "SYS"  # untouched string
    assert "cache_control" not in captured["messages"][-1]["content"][-1]


def test_effective_max_tokens_counts_system_and_tools():
    # Regression: a 32K-context model with auto max_tokens (==32768) must not
    # grant a near-full-window output budget on the first request. The clamp
    # has to count system + tools, not just messages, or prompt + output
    # overflows the window (the reported 32224 + 545 > 32768 error).
    p = AnthropicProvider(
        model="qwen", base_url="http://localhost:8000",
        max_tokens=32768, max_model_len=32768,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    system = "s" * 6000
    tools = [
        {"name": f"t{i}", "description": "d" * 400,
         "input_schema": {"type": "object", "properties": {}}}
        for i in range(40)
    ]

    eff = p._effective_max_tokens(messages, system, tools)
    est = p._estimate_prompt_tokens(messages, system, tools)
    # The whole point: prompt estimate + output budget fits the window.
    assert eff + est <= p._max_model_len
    # And system + tools materially shrink the budget vs the old messages-only
    # estimate (which would have returned ~the full window).
    assert eff < p._effective_max_tokens(messages)


def test_effective_max_tokens_handles_cache_wrapped_system():
    # system can arrive as a list of cache_control-wrapped text blocks.
    p = AnthropicProvider(
        model="qwen", base_url="http://localhost:8000",
        max_tokens=8192, max_model_len=32768,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    sys_blocks = [{"type": "text", "text": "s" * 3000}]
    est = p._estimate_prompt_tokens(messages, sys_blocks, None)
    assert est >= 1000  # the 3000-char system block was counted
