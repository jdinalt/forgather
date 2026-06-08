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
# rather than a hard "context length exceeded" error). The estimate counts
# system + tools + messages — on a small-context model the tool schemas and
# system prompt are most of the prompt; counting only messages grants an
# output budget that overflows the window on the first request.
_MIN_OUTPUT_TOKENS = 512
_CONTEXT_SAFETY_MARGIN = 512
# Rough chars-per-token; intentionally low so the prompt estimate runs high.
_CHARS_PER_TOKEN = 3.0


_EPHEMERAL = {"type": "ephemeral"}


def _add_cache_control(
    messages: List[Dict[str, Any]],
    system: Optional[str],
    tools: List[Dict[str, Any]],
):
    """Inject Anthropic ``cache_control`` breakpoints, without mutating inputs.

    Two breakpoints (well under the limit of 4) cover the whole re-sent prefix:

    - One on the stable head: the ``system`` block (which, per the cache order
      tools -> system -> messages, caches tools + system together). When there
      is no system prompt but there are tools, the breakpoint goes on the last
      tool instead so the tool schemas still cache.
    - One on the last message's last content block, so the conversation prefix
      caches and grows incrementally turn over turn.

    Copies only the containers it touches; the caller's stored conversation is
    left unchanged.
    """
    system_out: Any = system
    tools_out = tools
    if system:
        system_out = [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]
    elif tools:
        tools_out = list(tools)
        last_tool = dict(tools_out[-1])
        last_tool["cache_control"] = _EPHEMERAL
        tools_out[-1] = last_tool

    messages_out = messages
    if messages:
        messages_out = list(messages)
        last_msg = dict(messages_out[-1])
        content = last_msg.get("content")
        if isinstance(content, list) and content and isinstance(content[-1], dict):
            new_content = list(content)
            blk = dict(new_content[-1])
            blk["cache_control"] = _EPHEMERAL
            new_content[-1] = blk
            last_msg["content"] = new_content
            messages_out[-1] = last_msg
        elif isinstance(content, str) and content:
            last_msg["content"] = [
                {"type": "text", "text": content, "cache_control": _EPHEMERAL}
            ]
            messages_out[-1] = last_msg

    return messages_out, system_out, tools_out


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
        prompt_caching: bool = False,
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
        # Prompt caching (Anthropic ``cache_control`` breakpoints). Each loop
        # iteration re-sends system + tools + history; with caching that stable
        # prefix bills at ~0.1x instead of full rate. Off by default (the
        # runtime turns it on for Claude); vLLM does its own automatic prefix
        # caching and may reject cache_control, so it stays off there.
        self._prompt_caching = prompt_caching
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

        # Size the output budget BEFORE the cache_control transform, against the
        # full prompt — system + tools + messages. (The transform only adds
        # cache_control markers; it doesn't change the text we measure.)
        effective_max_tokens = self._effective_max_tokens(messages, system, tools)
        if self._prompt_caching:
            messages, system, tools = _add_cache_control(messages, system, tools)

        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "messages": messages,
            "stream": True,
        }
        if tools:
            create_kwargs["tools"] = tools
        if system:
            create_kwargs["system"] = system

        # Per-index accumulators for tool_use blocks. index -> {id, name, json}
        tool_blocks: Dict[int, Dict[str, Any]] = {}
        # Prompt token counts arrive on message_start; output accrues on the
        # message_delta events. Carry input + cache counts forward so each
        # emitted Usage carries the full per-request breakdown (the UI shows
        # context occupancy = input + output / window, and accumulates the
        # billed total = input + cache_read + cache_creation + output).
        input_tokens = 0
        cache_read = 0
        cache_creation = 0
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
                    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                    cache_creation = (
                        getattr(u, "cache_creation_input_tokens", 0) or 0
                    )

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
                    # Some servers also echo input/cache counts here; prefer the
                    # message_start values, fall back to the delta's.
                    if not input_tokens:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                    if not cache_read:
                        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                    if not cache_creation:
                        cache_creation = (
                            getattr(usage, "cache_creation_input_tokens", 0) or 0
                        )
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
                    # Per-request breakdown in the server log: this is what
                    # reconciles with the billing dashboard (each loop iteration
                    # re-sends the prefix). cache_read>0 confirms caching works.
                    log.info(
                        "agent request usage: input=%d cache_read=%d "
                        "cache_write=%d output=%d (billed_in=%d)",
                        input_tokens,
                        cache_read,
                        cache_creation,
                        output_tokens,
                        input_tokens + cache_read + cache_creation,
                    )
                    yield Usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_input_tokens=cache_read,
                        cache_creation_input_tokens=cache_creation,
                        context_window=self._max_model_len,
                    )

            elif etype == "message_stop":
                # stop_reason was captured from the message_delta above.
                yield Done(stop_reason=stop_reason)

    def _effective_max_tokens(
        self,
        messages: List[Dict[str, Any]],
        system: Any = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Clamp the output budget so prompt + output fits the context.

        No-op when max_model_len is unknown (e.g. Claude) — the upstream
        enforces its own output limits there. For vLLM, keep
        ``estimated_prompt + max_tokens <= max_model_len``, where the prompt
        is system + tools + messages — NOT just messages: the tool schemas and
        system prompt dominate the prompt on a small-context model, and
        omitting them grants a near-full-window output budget that overflows
        the very first request.
        """
        if not self._max_model_len:
            return self.max_tokens
        est_prompt = self._estimate_prompt_tokens(messages, system, tools)
        avail = self._max_model_len - est_prompt - _CONTEXT_SAFETY_MARGIN
        if avail < _MIN_OUTPUT_TOKENS:
            # Conversation nearly fills the window; ask for the floor and let
            # the server return its diagnostic if it truly can't fit.
            return _MIN_OUTPUT_TOKENS
        return max(_MIN_OUTPUT_TOKENS, min(self.max_tokens, avail))

    @staticmethod
    def _estimate_prompt_tokens(
        messages: List[Dict[str, Any]],
        system: Any = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Cheap, deliberately-high estimate of the prompt's token count.

        Counts the system prompt and the serialized tool schemas as well as
        the messages — all three are sent on every request and the latter two
        are often the bulk of the prompt.
        """
        chars = 0
        if isinstance(system, str):
            chars += len(system)
        elif system:  # cache_control-wrapped list of text blocks
            for b in system:
                chars += len(b.get("text", "") if isinstance(b, dict) else str(b))
        if tools:
            chars += len(json.dumps(tools, default=str))
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
