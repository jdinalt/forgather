"""Disclosure meta-tools: tool discovery + on-demand schema + dispatch.

These keep the tool array small without losing reach as the tool set grows
(the concern is context capacity for limited-window local models, not cost —
prompt caching already handles cost, and is off for vLLM anyway):

- ``list_tools`` — a compact catalog of every tool (name, tier, risk, one-line
  summary), so the model can discover ``extended`` tools that are not in the
  array under ``deferred`` mode.
- ``tool_help`` — the FULL description + input schema for one tool, fetched
  on demand (the detail trimmed from an ``extended`` tool's inline summary).
- ``call_tool`` — the dispatcher used in ``deferred`` mode to run an
  ``extended`` tool that isn't in the array. The agent loop intercepts a
  dispatch tool and runs the inner tool under the inner tool's *own* risk, so
  a confirm/propose tool still goes through the approval gate. This handler is
  only a safety net for a non-loop caller.

``register_all`` closes over the registry so the handlers can introspect it.
"""

from __future__ import annotations

from typing import Any, Dict

from .registry import META, READ, ToolRegistry, ToolSpec


def register_all(reg: ToolRegistry) -> None:
    def _list_tools(_args: Dict[str, Any]) -> Any:
        return {"tools": reg.catalog()}

    def _tool_help(args: Dict[str, Any]) -> Any:
        name = (args.get("name") or "").strip()
        spec = reg.get(name)
        if spec is None:
            raise ValueError(
                f"unknown tool {name!r} (use list_tools to see available tools)"
            )
        return {
            "name": spec.name,
            "risk": spec.risk,
            "tier": spec.tier,
            "description": spec.description,
            "input_schema": spec.json_schema,
        }

    def _call_tool(_args: Dict[str, Any]) -> Any:
        # The agent loop intercepts dispatch tools and never calls this; it
        # exists so a direct (non-loop) caller fails loudly instead of silently.
        raise RuntimeError(
            "call_tool is dispatched by the agent loop and must not be invoked "
            "directly"
        )

    reg.register(
        ToolSpec(
            name="list_tools",
            description=(
                "List every available tool with its tier (core/extended), risk, "
                "and a one-line summary. Use this to discover tools that are not "
                "shown directly — especially in deferred mode, where only core "
                "tools appear and extended tools are run via call_tool."
            ),
            json_schema={"type": "object", "properties": {}},
            handler=_list_tools,
            risk=READ,
            tier=META,
        )
    )
    reg.register(
        ToolSpec(
            name="tool_help",
            description=(
                "Get the full description and JSON input schema for one tool by "
                "name. Use before calling an unfamiliar tool (extended tools show "
                "only a short summary inline; their full guidance lives here)."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool name (see list_tools)."},
                },
                "required": ["name"],
            },
            handler=_tool_help,
            risk=READ,
            tier=META,
        )
    )
    reg.register(
        ToolSpec(
            name="call_tool",
            description=(
                "Run a tool that is not listed directly (an extended tool, in "
                "deferred mode). Pass the tool's name and its arguments object. "
                "Call tool_help(name) first to learn its schema. The tool runs "
                "with its own risk level — a confirm/propose tool still asks for "
                "your approval."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the tool to run (see list_tools)."},
                    "args": {"type": "object", "description": "Arguments object for that tool (see tool_help)."},
                },
                "required": ["name"],
            },
            handler=_call_tool,
            risk=READ,
            tier=META,
            dispatch=True,
        )
    )
