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

# Per-request budgeting (only when max_model_len is known, i.e. vLLM):
# vLLM enforces prompt_tokens + max_tokens <= max_model_len, so the output
# budget must leave room for the (growing) prompt. We clamp max_tokens to
# the remaining context on every request using a cheap, deliberately
# *over*-estimated prompt size (bias toward a smaller, safe output budget
# rather than a hard "context length exceeded" error).
_MIN_OUTPUT_TOKENS = 512
_CONTEXT_SAFETY_MARGIN = 512
# Rough chars-per-token; intentionally low so the prompt estimate runs high.
_CHARS_PER_TOKEN = 3.0


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
        auth_token: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_model_len: Optional[int] = None,
        verify: Any = None,
    ) -> None:
        self.model = model
        # Upper bound on output tokens. When max_model_len is known, the
        # effective value is clamped per request to fit the remaining
        # context (see _effective_max_tokens).
        self.max_tokens = max_tokens
        self._max_model_len = max_model_len
        # api_key  -> sent as the ``x-api-key`` header (real Claude).
        # auth_token -> sent as ``Authorization: Bearer`` (what vLLM's
        # Anthropic Messages surface checks). They are mutually exclusive;
        # the runtime picks one based on whether base_url is a local server.
        self._api_key = api_key
        self._auth_token = auth_token
        self._base_url = base_url
        # httpx ``verify=`` value (False / ssl.SSLContext / path / True). Only
        # injected via a custom http_client when it's a non-default posture —
        # see agent_tls.build_verify. None/True keeps the SDK's own client.
        self._verify = verify
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
            # Prefer a bearer auth_token (vLLM); else x-api-key (Claude). The
            # SDK requires one credential even when the upstream ignores it
            # (vLLM --no-auth), so fall back to a placeholder api_key.
            if self._auth_token:
                kwargs["auth_token"] = self._auth_token
            elif self._api_key:
                kwargs["api_key"] = self._api_key
            else:
                kwargs["api_key"] = "placeholder"
            if self._base_url:
                kwargs["base_url"] = self._base_url
            # Custom TLS posture (self-signed / imported cert) → give the SDK
            # an httpx client built with that verify value. Skip for the
            # default (None/True) so the SDK keeps its own tuned client.
            if self._verify is not None and self._verify is not True:
                import httpx

                kwargs["http_client"] = httpx.AsyncClient(verify=self._verify)
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
            "max_tokens": self._effective_max_tokens(messages),
            "messages": messages,
            "stream": True,
        }
        if tools:
            create_kwargs["tools"] = tools
        if system:
            create_kwargs["system"] = system

        # Per-index accumulators for tool_use blocks. index -> {id, name, json}
        tool_blocks: Dict[int, Dict[str, Any]] = {}
        # Prompt token count arrives on message_start; output accrues on the
        # message_delta events. Carry input forward so each emitted Usage has
        # both (the UI shows context occupancy = input + output / window).
        input_tokens = 0
        # Why the turn ended (end_turn / max_tokens / tool_use); carried on the
        # final Done so the loop can flag a truncated turn for "Continue".
        stop_reason: Optional[str] = None

        stream = await client.messages.create(**create_kwargs)
        async for event in stream:
            etype = getattr(event, "type", None)

            if etype == "message_start":
                msg = getattr(event, "message", None)
                u = getattr(msg, "usage", None) if msg is not None else None
                if u is not None:
                    input_tokens = getattr(u, "input_tokens", 0) or 0

            elif etype == "content_block_start":
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
                delta = getattr(event, "delta", None)
                if delta is not None and getattr(delta, "stop_reason", None):
                    stop_reason = delta.stop_reason
                usage = getattr(event, "usage", None)
                if usage is not None:
                    # Some servers also echo input_tokens here; prefer the
                    # message_start value, fall back to the delta's.
                    if not input_tokens:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                    yield Usage(
                        input_tokens=input_tokens,
                        output_tokens=getattr(usage, "output_tokens", 0) or 0,
                        context_window=self._max_model_len,
                    )

            elif etype == "message_stop":
                # stop_reason was captured from the message_delta above.
                yield Done(stop_reason=stop_reason)

    def _effective_max_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Clamp the output budget so prompt + output fits the context.

        No-op when max_model_len is unknown (e.g. Claude) — the upstream
        enforces its own output limits there. For vLLM, keep
        ``estimated_prompt + max_tokens <= max_model_len``.
        """
        if not self._max_model_len:
            return self.max_tokens
        est_prompt = self._estimate_prompt_tokens(messages)
        avail = self._max_model_len - est_prompt - _CONTEXT_SAFETY_MARGIN
        if avail < _MIN_OUTPUT_TOKENS:
            # Conversation nearly fills the window; ask for the floor and let
            # the server return its diagnostic if it truly can't fit.
            return _MIN_OUTPUT_TOKENS
        return max(_MIN_OUTPUT_TOKENS, min(self.max_tokens, avail))

    @staticmethod
    def _estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
        """Cheap, deliberately-high estimate of the prompt's token count."""
        chars = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                chars += len(content)
                continue
            for block in content or []:
                if not isinstance(block, dict):
                    chars += len(str(block))
                    continue
                if block.get("text"):
                    chars += len(str(block["text"]))
                if block.get("content"):  # tool_result payload
                    chars += len(str(block["content"]))
                if block.get("input") is not None:  # tool_use args
                    chars += len(json.dumps(block["input"], default=str))
        return int(chars / _CHARS_PER_TOKEN) + 8 * len(messages)

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
