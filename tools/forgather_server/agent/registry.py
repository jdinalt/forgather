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
class ToolSpec:
    name: str
    description: str
    json_schema: Dict[str, Any]
    handler: Handler
    risk: str = READ

    def __post_init__(self) -> None:
        if self.risk not in _RISK_LEVELS:
            raise ValueError(f"unknown risk level: {self.risk!r}")


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

    def anthropic_tools(self) -> List[Dict[str, Any]]:
        """Serialize to the Anthropic Messages API tool shape."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "input_schema": s.json_schema,
            }
            for s in self._tools.values()
        ]
