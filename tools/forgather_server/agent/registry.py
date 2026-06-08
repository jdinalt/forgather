"""In-process tool registry — the source of truth for agent tools.

Each tool wraps an existing ``*_ops.py`` function. A tool is classified
by ``risk``:

- ``read``    — runs automatically, no approval (inspection, search).
- ``propose`` — handler computes a **preview** and returns a ``Proposal``
  carrying the would-be change plus a ``commit`` closure. It performs no
  side effect. The loop pauses for user approval; approve runs ``commit``.
- ``confirm`` — like ``propose`` but for low-blast mutations with no rich
  diff (a simple approve-to-run); still gated.

The registry is provider-neutral. ``anthropic_tools()`` serializes the
specs into the Messages API tool shape; a future OpenAI adapter would add
its own serializer here. Keeping the registry the single source of truth
is also what lets a later ``forgather mcp`` server re-export the same
tools to external clients without duplicating definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

# A tool handler takes the parsed arguments dict and returns either a
# JSON-serializable result (read tools) or a Proposal (propose/confirm).
# It may be sync or async; the loop awaits coroutines.
Handler = Callable[[Dict[str, Any]], Union[Any, Awaitable[Any]]]

READ = "read"
PROPOSE = "propose"
CONFIRM = "confirm"
_RISK_LEVELS = frozenset({READ, PROPOSE, CONFIRM})

# Disclosure tier — controls how a tool is surfaced to the model:
#   core     — always in the tool array (full description).
#   extended — full description in ``inline`` mode; in ``deferred`` mode it is
#              dropped from the array and reached only via the ``call_tool``
#              dispatcher (discovered with list_tools / tool_help).
#   meta     — the disclosure helpers themselves; always present (except
#              ``call_tool``, which is ``deferred``-only — see ``dispatch``).
CORE = "core"
EXTENDED = "extended"
META = "meta"
_TIERS = frozenset({CORE, EXTENDED, META})

# Serialization modes for ``anthropic_tools``:
#   inline   — every tool in the array; extended tools carry a short summary.
#              The block is static across a session, so prompt caching holds.
#   deferred — only core + meta tools in the array; extended tools are hidden
#              and invoked through ``call_tool``. Keeps the array tiny for
#              limited-context local models.
INLINE = "inline"
DEFERRED = "deferred"
_MODES = frozenset({INLINE, DEFERRED})


@dataclass
class Proposal:
    """A previewed, not-yet-applied change returned by a propose/confirm tool.

    ``commit`` is the closure that performs the actual side effect when the
    user approves. The loop stores the Proposal verbatim at propose time
    and replays ``commit`` on approval — it never re-derives the change
    from model output, so the model cannot alter what gets written between
    preview and apply.
    """

    title: str
    summary: str = ""
    # Diff preview (optional): the file path and its before/after content.
    path: Optional[str] = None
    before: Optional[str] = None
    after: Optional[str] = None
    # Rendered preprocessor output of the candidate config (optional).
    pp_preview: Optional[str] = None
    # Arbitrary extra structured preview for non-file changes.
    extra: Dict[str, Any] = field(default_factory=dict)
    # When this proposal *creates* a navigable artifact, the kind of thing it
    # creates ("workspace" | "project" | "config"). On approval the loop
    # echoes this plus ``path`` so the webui can refresh the Projects tree and
    # reveal/expand to the new item (mirroring what a user-driven create does).
    # ``None`` for edits / non-creating changes.
    reveal_kind: Optional[str] = None
    # The actual side-effecting action, run only on approval. Returns a
    # result string fed back to the model as the tool_result.
    commit: Optional[Callable[[], Union[str, Awaitable[str]]]] = None

    def to_card(self, action_id: str, risk: str) -> Dict[str, Any]:
        """Serialize for the webui action card (no ``commit`` closure)."""
        return {
            "action_id": action_id,
            "risk": risk,
            "title": self.title,
            "summary": self.summary,
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "pp_preview": self.pp_preview,
            "extra": self.extra,
        }


@dataclass
class UiDirective:
    """A client-side action a ``read`` tool asks the webui to perform.

    Some read tools are useful precisely because they steer the UI — e.g.
    revealing a project the agent just located. The handler returns a
    ``UiDirective`` instead of plain data; the loop emits it as a
    ``ui_directive`` event for the webui and feeds ``message`` back to the
    model as the tool result (so the model knows it succeeded). It carries no
    side effect on the server and so needs no approval gate.
    """

    action: str  # e.g. "reveal"
    payload: Dict[str, Any] = field(default_factory=dict)
    message: str = "done"


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: Dict[str, Any]
    handler: Handler
    risk: str = READ
    # Disclosure tier (see CORE / EXTENDED / META above).
    tier: str = CORE
    # Short one-line description shown in ``inline`` mode for an ``extended``
    # tool (the full ``description`` is fetched on demand via ``tool_help``).
    # ``None`` => fall back to the full description.
    summary: Optional[str] = None
    # The ``call_tool`` dispatcher sets this. The loop intercepts a dispatch
    # tool: it resolves the inner tool from the call's ``name``/``args`` and
    # runs it under the *inner* tool's risk (so a confirm tool still gates).
    # A dispatch tool is only serialized in ``deferred`` mode.
    dispatch: bool = False

    def __post_init__(self) -> None:
        if self.risk not in _RISK_LEVELS:
            raise ValueError(f"unknown risk level: {self.risk!r}")
        if self.tier not in _TIERS:
            raise ValueError(f"unknown tier: {self.tier!r}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name!r}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def anthropic_tools(self, mode: str = INLINE) -> List[Dict[str, Any]]:
        """Serialize to the Anthropic Messages API tool shape for ``mode``.

        ``inline`` emits every tool (extended tools with their short summary);
        ``deferred`` emits only core + meta tools (plus the ``call_tool``
        dispatcher) and hides extended tools, which are then reached via
        ``call_tool``. ``input_schema`` is always sent in full so a tool
        stays directly callable with good argument hints.
        """
        if mode not in _MODES:
            raise ValueError(f"unknown disclosure mode: {mode!r}")
        out: List[Dict[str, Any]] = []
        for s in self._tools.values():
            if s.dispatch:
                if mode != DEFERRED:
                    continue  # call_tool is pointless when all tools are inline
            elif s.tier == EXTENDED and mode == DEFERRED:
                continue  # hidden from the array; reachable via call_tool
            use_summary = s.tier == EXTENDED and mode == INLINE and s.summary
            out.append(
                {
                    "name": s.name,
                    "description": s.summary if use_summary else s.description,
                    "input_schema": s.json_schema,
                }
            )
        return out

    def catalog(self) -> List[Dict[str, Any]]:
        """Compact listing of every registered tool (for list_tools)."""
        return [
            {"name": s.name, "tier": s.tier, "risk": s.risk,
             "summary": s.summary or s.description.split("\n", 1)[0]}
            for s in self._tools.values()
        ]
