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

from forgather_server.agent.providers.anthropic import AnthropicProvider
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
