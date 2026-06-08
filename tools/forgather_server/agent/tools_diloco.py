"""Agent tools to monitor and control DiLoCo servers.

DiLoCo (distributed low-communication training) runs a parameter server
that workers sync against; watching workers/rounds and controlling the
server (checkpoint, relay a command to workers, shut down) is core to
operating a run. These wrap the ``_diloco`` helper (discovery +
``DiLoCoClient``). Starting/stopping the server itself is the generic
``start_diloco_server`` / ``stop_service``.

All extended-tier. ``diloco_control`` is CONFIRM-gated (it mutates a live
run); the read tools are auto.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from . import _diloco
from .registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry, ToolSpec


async def _list_diloco_servers(_args: Dict[str, Any]) -> Any:
    # Off the event loop: list_servers does blocking reachability probes.
    return {"servers": await asyncio.to_thread(_diloco.list_servers)}


async def _diloco_status(args: Dict[str, Any]) -> Any:
    # Off the event loop: status() makes blocking HTTP calls.
    return await asyncio.to_thread(_diloco.status, args.get("server_id") or None)


def _diloco_control(args: Dict[str, Any]) -> Proposal:
    server_id = args.get("server_id") or None
    action = (args.get("action") or "").strip()
    command = args.get("command") or None
    worker_id = args.get("worker_id") or None
    if action not in _diloco._CONTROL:
        raise ValueError(
            f"unknown action {action!r}; expected one of {sorted(_diloco._CONTROL)}"
        )
    if action == "relay" and command not in _diloco._RELAY_COMMANDS:
        raise ValueError(
            f"action 'relay' needs command in {sorted(_diloco._RELAY_COMMANDS)}"
        )

    detail = action if action != "relay" else f"relay {command}" + (
        f" to {worker_id}" if worker_id else " to all workers"
    )

    async def commit() -> str:
        # Off the event loop: control() makes a blocking HTTP call.
        out = await asyncio.to_thread(
            _diloco.control, server_id, action, command, worker_id
        )
        return f"diloco {detail} on {out['server']['id']}: {out['result']}"

    return Proposal(
        title=f"DiLoCo control: {detail}",
        summary=(
            "Run a control action on a live DiLoCo server: save_state "
            "(checkpoint the server), shutdown (stop it), or relay a "
            "save_checkpoint/save_and_stop/abort command to its workers."
        ),
        extra={
            "server_id": server_id,
            "action": action,
            "command": command,
            "worker_id": worker_id,
        },
        commit=commit,
    )


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="list_diloco_servers",
            description=(
                "List DiLoCo servers known to this node and the cluster, with "
                "base_url, source, reachability, and cached health. Use to pick "
                "a server_id for diloco_status / diloco_control."
            ),
            summary="List known DiLoCo servers (id/base_url/reachable).",
            json_schema={"type": "object", "properties": {}},
            handler=_list_diloco_servers,
            risk=READ,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="diloco_status",
            description=(
                "Get a DiLoCo server's live status (round/step, synced workers, "
                "etc.) and its worker roster (each worker_id + running flag). "
                "server_id is optional (defaults to the first reachable server; "
                "see list_diloco_servers)."
            ),
            summary="DiLoCo server live status + worker roster.",
            json_schema={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "From list_diloco_servers (default: first reachable)."},
                },
            },
            handler=_diloco_status,
            risk=READ,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="diloco_control",
            description=(
                "Control a live DiLoCo server (approval required). action: "
                "save_state (checkpoint the server now), shutdown (stop it), or "
                "relay (deliver a worker command on the next heartbeat — set "
                "command to save_checkpoint | save_and_stop | abort, optionally "
                "worker_id to target one worker instead of all). To start/stop "
                "the server process itself use start_diloco_server / stop_service."
            ),
            summary="Control a DiLoCo server: save_state | shutdown | relay (CONFIRM).",
            json_schema={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "From list_diloco_servers (default: first reachable)."},
                    "action": {
                        "type": "string",
                        "enum": ["save_state", "shutdown", "relay"],
                    },
                    "command": {
                        "type": "string",
                        "enum": ["save_checkpoint", "save_and_stop", "abort"],
                        "description": "Required when action=relay.",
                    },
                    "worker_id": {"type": "string", "description": "Target one worker (relay only); omit for all."},
                },
                "required": ["action"],
            },
            handler=_diloco_control,
            risk=CONFIRM,
            tier=EXTENDED,
        )
    )
