"""Anthropic Messages API adapter.

Talks to Claude (``base_url=https://api.anthropic.com``) or to a local
vLLM model via vLLM's native Anthropic Messages API (point ``base_url``
at the vLLM server). The canonical conversation format (see
``providers/base.py``) is content-block style, which is already what the
Messages API expects, so translation is near-identity.

Robustness this adapter owns (so ``loop.py`` stays simple):

- **tool_use reassembly** — streaming delivers a tool call as a
  ``content_block_start`` then a run of ``input_json_delta`` fragments;
  we accumulate per block index and parse once at ``content_block_stop``.
- **malformed JSON tolerance** — local models emit partial/invalid
  tool-call JSON more often than Claude. A parse failure yields a
  ``ToolCall`` with ``parse_error`` set rather than raising; the loop
  feeds that back as an error tool_result so the model can retry.

The ``anthropic`` SDK is imported lazily so the rest of the agent package
(and its unit tests, which use a fake provider) import without the dep.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import Done, TextDelta, ToolCall, Usage

log = logging.getLogger("forgather_server.agent.anthropic")

# Conservative default; overridable via config. The Messages API requires
# max_tokens, and vLLM rejects values above the served model's context.
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    """ChatProvider backed by the Anthropic SDK.

    ``base_url=None`` uses the SDK default (Claude). Set it to a vLLM
    endpoint for local models. ``api_key`` is the Anthropic key for Claude
    or the vLLM bearer (or any non-empty placeholder when vLLM runs with
    ``--no-auth``) for local models.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:  # pragma: no cover - dep is declared
                raise RuntimeError(
                    "the 'anthropic' package is required for the agent; "
                    "install it (it is a declared dependency)"
                ) from e
            kwargs: Dict[str, Any] = {}
            # The SDK requires *some* api_key; vLLM --no-auth ignores it,
            # so pass a placeholder rather than failing construction.
            kwargs["api_key"] = self._api_key or "placeholder"
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def stream_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
    ) -> AsyncIterator[ProviderEvent]:  # type: ignore[name-defined]
        client = self._ensure_client()

        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "stream": True,
        }
        if tools:
            create_kwargs["tools"] = tools
        if system:
            create_kwargs["system"] = system

        # Per-index accumulators for tool_use blocks. index -> {id, name, json}
        tool_blocks: Dict[int, Dict[str, Any]] = {}

        stream = await client.messages.create(**create_kwargs)
        async for event in stream:
            etype = getattr(event, "type", None)

            if etype == "content_block_start":
                block = event.content_block
                if getattr(block, "type", None) == "tool_use":
                    tool_blocks[event.index] = {
                        "id": block.id,
                        "name": block.name,
                        "json": "",
                    }

            elif etype == "content_block_delta":
                delta = event.delta
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    yield TextDelta(text=delta.text)
                elif dtype == "input_json_delta":
                    acc = tool_blocks.get(event.index)
                    if acc is not None:
                        acc["json"] += delta.partial_json

            elif etype == "content_block_stop":
                acc = tool_blocks.pop(event.index, None)
                if acc is not None:
                    yield self._finalize_tool_call(acc)

            elif etype == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    yield Usage(
                        input_tokens=getattr(usage, "input_tokens", 0) or 0,
                        output_tokens=getattr(usage, "output_tokens", 0) or 0,
                    )

            elif etype == "message_stop":
                # stop_reason is carried on the message_delta in the raw
                # stream; surface a Done regardless so the loop terminates.
                yield Done(stop_reason=None)

    @staticmethod
    def _finalize_tool_call(acc: Dict[str, Any]) -> ToolCall:
        raw = acc["json"].strip()
        if not raw:
            # Anthropic emits no input_json_delta for a no-arg tool call;
            # an empty body is valid and means "{}".
            return ToolCall(id=acc["id"], name=acc["name"], arguments={})
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("tool input is not a JSON object")
            return ToolCall(id=acc["id"], name=acc["name"], arguments=parsed)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(
                "tool call %s (%s) had malformed JSON input: %s",
                acc["id"],
                acc["name"],
                e,
            )
            return ToolCall(
                id=acc["id"],
                name=acc["name"],
                arguments={},
                parse_error=f"{type(e).__name__}: {e}",
            )

    def format_tool_result(
        self, tool_use_id: str, content: str, *, is_error: bool = False
    ) -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        return block


# Re-export for the type annotation in stream_turn without a runtime import
# cycle (base imports nothing from here).
from .base import ProviderEvent  # noqa: E402  (placed last on purpose)
