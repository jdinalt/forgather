"""Tier-3 (extended) agent tools: inference interaction, cluster status,
and cached config overrides.

- list_inference_servers / query_model: see running model servers and
  test-generate against one (the inference equivalent of the dataset/diloco
  helpers; tokens resolved server-side).
- cluster_status: node/master/member view for multi-node setups.
- get/set_config_overrides: the cached override values the submit UI uses.

All extended-tier. query_model and set_config_overrides are CONFIRM-gated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .. import overrides_store
from . import _inference
from .registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_advanced")


# ---- inference -------------------------------------------------------------


def _list_inference_servers(_args: Dict[str, Any]) -> Any:
    return {"servers": _inference.list_servers()}


def _query_model(args: Dict[str, Any]) -> Proposal:
    server_id = args.get("server_id") or None
    model = args.get("model") or None
    max_tokens = args.get("max_tokens")
    max_tokens = 256 if max_tokens in (None, "") else int(max_tokens)
    temperature = args.get("temperature")
    temperature = None if temperature in (None, "") else float(temperature)

    messages = args.get("messages")
    if not messages:
        prompt = args.get("prompt")
        if not prompt:
            raise ValueError("provide either messages (list) or prompt (string)")
        messages = [{"role": "user", "content": str(prompt)}]
    if not isinstance(messages, list):
        raise ValueError("messages must be a list of {role, content} objects")

    preview = messages[-1].get("content") if isinstance(messages[-1], dict) else str(messages[-1])

    def commit() -> str:
        out = _inference.chat(
            server_id, messages, model=model, max_tokens=max_tokens, temperature=temperature
        )
        content = (out.get("message") or {}).get("content")
        return (
            f"[{out['server']['id']} / {out.get('model')}] {content}\n"
            f"(usage: {out.get('usage')}, finish: {out.get('finish_reason')})"
        )

    return Proposal(
        title="Query inference model",
        summary="Send a chat completion to a running inference server (consumes "
        "compute on the server). Returns the model's reply.",
        extra={
            "server_id": server_id,
            "model": model,
            "max_tokens": max_tokens,
            "prompt_preview": (str(preview)[:200] if preview else None),
        },
        commit=commit,
    )


# ---- cluster ---------------------------------------------------------------


def _cluster_status(_args: Dict[str, Any]) -> Any:
    from .. import cluster

    try:
        if not cluster.is_active():
            return {"active": False, "note": "cluster mode is not active on this node"}
        self_id = cluster.self_identity()
        members = []
        for m in cluster.members():
            members.append(
                {
                    "node_id": m.node_id,
                    "hostname": m.hostname,
                    "address": m.address,
                    "port": m.port,
                    "reachable": m.reachable,
                    "last_source": m.last_source,
                    "tls": m.tls,
                }
            )
        return {
            "active": True,
            "cluster_name": getattr(self_id, "cluster_name", None),
            "self_node_id": getattr(self_id, "node_id", None),
            "master_node_id": cluster.master_node_id(),
            "is_self_master": cluster.is_self_master(),
            "members": members,
        }
    except Exception as e:  # never fail hard for a status read
        return {"active": False, "error": f"{type(e).__name__}: {e}"}


# ---- config overrides ------------------------------------------------------


def _get_config_overrides(args: Dict[str, Any]) -> Any:
    return overrides_store.get_overrides_payload(args["project_dir"], args["config"])


def _set_config_overrides(args: Dict[str, Any]) -> Proposal:
    project_dir = args["project_dir"]
    config = args["config"]
    values = args.get("values")
    if not isinstance(values, dict):
        raise ValueError("values must be an object of {dest: value} overrides")
    gpus = args.get("requested_gpus")
    requested_gpus = None if gpus in (None, "") else int(gpus)

    def commit() -> str:
        overrides_store.set_overrides(
            project_dir, config, values, requested_gpus=requested_gpus
        )
        return f"saved overrides for {config} ({len(values)} value(s))."

    return Proposal(
        title=f"Set overrides: {config}",
        summary="Persist the cached override values the submit UI uses for this "
        "config (affects future runs until cleared).",
        extra={"project_dir": project_dir, "config": config, "values": values,
               "requested_gpus": requested_gpus},
        commit=commit,
    )


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="list_inference_servers",
            description=(
                "List running inference (model) servers known to this node and "
                "the cluster, with base_url, served models, and reachability. "
                "Use to pick a server_id for query_model."
            ),
            summary="List running inference servers (for query_model).",
            json_schema={"type": "object", "properties": {}},
            handler=_list_inference_servers,
            risk=READ,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="query_model",
            description=(
                "Send a chat completion to a running inference server to "
                "test-generate (approval required; it consumes compute). Provide "
                "either prompt (a string) or messages (a list of {role, "
                "content}). server_id defaults to the first reachable server "
                "(see list_inference_servers); model defaults to the server's "
                "first served model. Optional max_tokens (default 256), "
                "temperature."
            ),
            summary="Test-generate against a running inference server (CONFIRM).",
            json_schema={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "From list_inference_servers (default: first reachable)."},
                    "prompt": {"type": "string", "description": "Single user prompt (alternative to messages)."},
                    "messages": {"type": "array", "description": "OpenAI-style [{role, content}] messages.", "items": {"type": "object"}},
                    "model": {"type": "string", "description": "Model id (default: server's first model)."},
                    "max_tokens": {"type": "integer", "description": "Max output tokens (default 256)."},
                    "temperature": {"type": "number"},
                },
            },
            handler=_query_model,
            risk=CONFIRM,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="cluster_status",
            description=(
                "Report cluster membership: this node's identity, the master, "
                "and the member table (node_id, hostname, address, reachable). "
                "Returns active:false on a single-node / non-cluster setup."
            ),
            summary="Cluster membership + master (multi-node).",
            json_schema={"type": "object", "properties": {}},
            handler=_cluster_status,
            risk=READ,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="get_config_overrides",
            description=(
                "Get the cached override values stored for a config (the values "
                "the submit/overrides UI last saved): {values, requested_gpus, "
                "multinode, dataset_source, updated_at}."
            ),
            summary="Read a config's cached override values.",
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config": {"type": "string"},
                },
                "required": ["project_dir", "config"],
            },
            handler=_get_config_overrides,
            risk=READ,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="set_config_overrides",
            description=(
                "Set the cached override values for a config (the dynamic-arg "
                "values the submit UI applies; from inspect_config's "
                "dynamic_args). Approval required. values is an object keyed by "
                "dest; optional requested_gpus. Affects future runs until "
                "cleared."
            ),
            summary="Persist a config's override values (CONFIRM).",
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config": {"type": "string"},
                    "values": {"type": "object", "description": "{dest: value} overrides (see inspect_config dynamic_args)."},
                    "requested_gpus": {"type": "integer"},
                },
                "required": ["project_dir", "config", "values"],
            },
            handler=_set_config_overrides,
            risk=CONFIRM,
            tier=EXTENDED,
        )
    )
