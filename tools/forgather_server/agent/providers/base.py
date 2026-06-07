"""Provider-neutral chat interface and the events the loop consumes.

The loop owns the *canonical* conversation format: a list of messages
whose ``content`` is a list of typed blocks. The shape is content-block
style (text / tool_use / tool_result) — conceptually provider-neutral;
each adapter translates it to and from its own wire format. The Anthropic
adapter passes it through nearly verbatim; a future OpenAI adapter would
flatten tool_use/tool_result into the OpenAI ``tool_calls`` shape.

Canonical message shapes (what ``loop.py`` builds and stores):

    {"role": "user", "content": [{"type": "text", "text": "..."}]}

    {"role": "assistant", "content": [
        {"type": "text", "text": "..."},
        {"type": "tool_use", "id": "toolu_x", "name": "...", "input": {...}},
    ]}

    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_x",
         "content": "...", "is_error": false},
    ]}

A provider yields a stream of the dataclass events below. Tool calls are
delivered *reassembled* (fragmented streaming deltas are joined by the
adapter) so the loop only ever sees a complete ``ToolCall``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Union


@dataclass
class TextDelta:
    """A chunk of assistant-visible text."""

    text: str


@dataclass
class ToolCall:
    """A fully-reassembled tool call from the model.

    ``arguments`` is the parsed JSON input. ``parse_error`` is set (and
    ``arguments`` left ``{}``) when the model emitted malformed/partial
    tool-call JSON — local vLLM models do this more often than Claude.
    The loop turns a ``parse_error`` into an error tool_result rather than
    crashing the turn.
    """

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    parse_error: Optional[str] = None


@dataclass
class Usage:
    """Token accounting for one turn (best-effort; may be partial).

    ``input_tokens`` is the prompt size of the request (the conversation's
    current context occupancy); ``context_window`` is the model's max context
    (``None`` when the provider doesn't report it, e.g. Claude).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    context_window: Optional[int] = None


@dataclass
class Done:
    """End of one assistant turn. ``stop_reason`` is provider-reported."""

    stop_reason: Optional[str] = None


ProviderEvent = Union[TextDelta, ToolCall, Usage, Done]


class ChatProvider(Protocol):
    """The seam. ``loop.py`` depends only on this, never on a vendor SDK."""

    def stream_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Stream one assistant turn.

        ``messages`` is the canonical conversation (see module docstring).
        ``tools`` is the registry's tool schema list in this provider's
        expected shape (the registry exposes a per-provider serializer).
        Yields ``TextDelta`` / ``ToolCall`` / ``Usage`` and finally exactly
        one ``Done``.
        """
        ...

    def format_tool_result(
        self, tool_use_id: str, content: str, *, is_error: bool = False
    ) -> Dict[str, Any]:
        """Build a canonical ``tool_result`` content block."""
        ...
