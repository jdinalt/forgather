"""Agent tools for long-running services (Sidebar -> Services).

One START tool per service type — each with an explicit, documented argument
schema — so the agent configures a service correctly instead of guessing at a
generic blob. ``list_services`` (read) and ``stop_service`` (confirm) stay
generic (stopping needs no per-type args). All wrap the same ``services`` +
``scheduler`` machinery the Services panel uses, so a service the agent starts
shows up there and survives a restart; signature dedup prevents double-spawn.

Start tools (all CONFIRM):
- ``start_dataset_server`` (core) — defaults are fine, so a no-arg call brings
  up a default dataset server (makes ``dataset_info`` usable).
- ``start_diloco_server`` (core) — DiLoCo param server (see docs/trainers/diloco.md).
- ``start_inference_server`` (core) — serve a model for query_model.
- ``start_tensorboard`` / ``start_mkdocs`` (extended) — nice-to-have.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

from .. import queue_store, scheduler, services
from .registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_services")


def _collect(args: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    """Service-arg dict from the provided fields (drop unset / empty)."""
    out: Dict[str, Any] = {}
    for k in fields:
        v = args.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    return out


def _list_services(_args: Dict[str, Any]) -> Any:
    rows = []
    for st in services.status_for_each(services.list_services()):
        s = st.service
        rows.append(
            {
                "type": s.type,
                "name": s.name,
                "enabled": s.enabled,
                "args": s.args,
                "running": st.running,
                "queue_id": st.queue_id,
                "status": st.status,
            }
        )
    return {"services": rows}


def _propose_start(svc_type: str, name: str, svc_args: Dict[str, Any]) -> Proposal:
    """Shared start logic: preview the resolved job, then on commit persist the
    service entry and enqueue it (unless a signature-identical instance is
    already active — dedup prevents double-spawn)."""
    candidate = services.Service(type=svc_type, name=name, enabled=True, args=svc_args)
    sig = candidate.signature()
    preview_item = services.build_queue_item(candidate)  # no side effect
    existing = services.active_signatures().get(sig)

    def commit() -> str:
        services.upsert_service(svc_type, name, True, svc_args)
        active = services.active_signatures().get(sig)
        if active:
            qid, st = active
            return (
                f"service {svc_type}:{name} already active as {qid} (status="
                f"{st}); persisted the entry, did not re-spawn."
            )
        item = services.build_queue_item(
            services.Service(type=svc_type, name=name, enabled=True, args=svc_args)
        )
        queue_store.add_item(item)
        return (
            f"started {svc_type} service {name!r} as job {item.queue_id}. "
            f"It takes a moment to come up; wait for it with "
            f"wait_for_job('{item.queue_id}', until='running') (NOT the default "
            "'terminal' — a healthy service never goes terminal), then verify "
            "it's reachable (list_dataset_servers / list_inference_servers / "
            "list_diloco_servers)."
        )

    return Proposal(
        title=f"Start service: {svc_type}:{name}",
        summary=(
            f"Persist and spawn a {svc_type} service. Long-running; it appears "
            "in Sidebar -> Services and survives a restart."
        ),
        extra={
            "type": svc_type,
            "name": name,
            "job_type": preview_item.job_type,
            "requested_gpus": preview_item.requested_gpus,
            "args": svc_args,
            "already_active": bool(existing),
        },
        commit=commit,
    )


def _name(args: Dict[str, Any]) -> str:
    return (args.get("name") or "agent-default").strip()


# ---- per-type start handlers ----------------------------------------------


def _start_dataset_server(args: Dict[str, Any]) -> Proposal:
    svc_args = _collect(
        args,
        ("port", "host", "no_hf", "allow_paths", "allow_downloads", "locals",
         "config_file", "no_auth"),
    )
    return _propose_start("dataset", _name(args), svc_args)


def _start_inference_server(args: Dict[str, Any]) -> Proposal:
    svc_args = _collect(
        args,
        ("model_path", "models", "port", "host", "requested_gpus", "dtype",
         "device", "from_checkpoint", "checkpoint_path", "compile",
         "disable_kv_cache", "keep_on_gpu", "attn_implementation",
         "chat_template", "cache_implementation", "no_auth"),
    )
    has_single = bool(svc_args.get("model_path"))
    has_multi = bool(svc_args.get("models"))
    if has_single == has_multi:  # neither or both
        raise ValueError(
            "provide exactly one of model_path (single model) or models "
            "(list of {name, path})"
        )
    if not svc_args.get("port"):
        raise ValueError("port is required for an inference server")
    return _propose_start("inference", _name(args), svc_args)


def _start_diloco_server(args: Dict[str, Any]) -> Proposal:
    svc_args = _collect(
        args,
        ("output_dir", "num_workers", "port", "host", "sync_every", "async_mode",
         "save_every", "save_total_limit", "num_fragments", "min_workers",
         "heartbeat_timeout", "outer_lr", "outer_momentum", "from_checkpoint",
         "run_name", "no_auth", "backend"),
    )
    advanced = args.get("advanced")
    if isinstance(advanced, dict):
        svc_args.update(advanced)  # passthrough for other `diloco server` flags
    if not svc_args.get("output_dir") or svc_args.get("num_workers") is None:
        raise ValueError("output_dir and num_workers are required for a diloco server")
    return _propose_start("diloco", _name(args), svc_args)


def _start_tensorboard(args: Dict[str, Any]) -> Proposal:
    svc_args = _collect(args, ("logdir", "port", "host", "bind_all"))
    if not svc_args.get("logdir") or not svc_args.get("port"):
        raise ValueError("logdir and port are required for tensorboard")
    return _propose_start("tensorboard", _name(args), svc_args)


def _start_mkdocs(args: Dict[str, Any]) -> Proposal:
    svc_args = _collect(
        args, ("config_file", "port", "host", "strict", "livereload", "dirty", "watch")
    )
    if not svc_args.get("config_file") or not svc_args.get("port"):
        raise ValueError("config_file and port are required for mkdocs")
    return _propose_start("mkdocs", _name(args), svc_args)


# ---- stop ------------------------------------------------------------------


def _stop_service(args: Dict[str, Any]) -> Proposal:
    svc_type = (args.get("type") or "").strip()
    name = (args.get("name") or "").strip()
    if svc_type not in services.SERVICE_TYPES:
        raise ValueError(
            f"unknown service type {svc_type!r}; expected one of "
            f"{sorted(services.SERVICE_TYPES)}"
        )
    if not name:
        raise ValueError("name is required (see list_services)")
    svc = services.get_service(svc_type, name)
    if svc is None:
        raise ValueError(f"no configured service {svc_type}:{name}")
    sig = svc.signature()
    existing = services.active_signatures().get(sig)

    def commit() -> str:
        services.set_enabled(svc_type, name, False)
        active = services.active_signatures().get(sig)
        if not active:
            return f"service {svc_type}:{name} disabled; no running instance to stop."
        qid = active[0]
        ok = scheduler.abort_or_cancel(qid)
        return (
            f"stopped {svc_type}:{name} (job {qid}): "
            f"{'aborted/cancelled' if ok else 'nothing to abort'}."
        )

    return Proposal(
        title=f"Stop service: {svc_type}:{name}",
        summary="Disable the service entry and abort its running instance.",
        extra={
            "type": svc_type,
            "name": name,
            "running": bool(existing),
            "queue_id": existing[0] if existing else None,
        },
        commit=commit,
    )


# A name field every start tool accepts.
_NAME_PROP = {"name": {"type": "string", "description": "Instance name (default 'agent-default'). Use distinct names to run several of one type."}}


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="list_services",
            description=(
                "List the configured long-running services (dataset / inference "
                "/ tensorboard / mkdocs / diloco) with their args and live "
                "status (running, queue_id). Use before a start_* / stop_service."
            ),
            json_schema={"type": "object", "properties": {}},
            handler=_list_services,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="start_dataset_server",
            description=(
                "Start a dataset server (Sidebar -> Services) — serves built/"
                "cached datasets for dataset_info and for training data routing. "
                "Defaults are fine: a no-arg call brings up a default server on "
                "port 8766. Use this when dataset_info reports none reachable. "
                "Approval required; persists to config and spawns. Then "
                "wait_for_job(queue_id, until='running')."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    **_NAME_PROP,
                    "port": {"type": "integer", "description": "Listen port (default 8766)."},
                    "host": {"type": "string", "description": "Bind host (default 127.0.0.1; 0.0.0.0 for LAN)."},
                    "no_hf": {"type": "boolean", "description": "Disable the HuggingFace cache backend."},
                    "allow_paths": {"type": "boolean", "description": "Allow loading datasets by filesystem path."},
                    "allow_downloads": {"type": "boolean", "description": "Allow the server to download datasets from HF."},
                    "locals": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": "Named local datasets: [[name, path], ...]."},
                    "config_file": {"type": "string", "description": "Optional dataset_server config file."},
                    "no_auth": {"type": "boolean", "description": "Disable the bearer-token gate (trusted host only)."},
                },
            },
            handler=_start_dataset_server,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="start_diloco_server",
            description=(
                "Start a DiLoCo parameter server (Sidebar -> Services) that "
                "workers sync against. output_dir (the model/checkpoint dir) and "
                "num_workers are required. DiLoCo has many tuning knobs — read "
                "docs/trainers/diloco.md before changing defaults; common ones "
                "are exposed here and the rest go in `advanced` (keys = "
                "`forgather diloco server` flags). Approval required; then "
                "wait_for_job(queue_id, until='running')."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    **_NAME_PROP,
                    "output_dir": {"type": "string", "description": "Model/checkpoint dir the server holds global params for."},
                    "num_workers": {"type": "integer", "description": "Expected worker count."},
                    "port": {"type": "integer", "description": "Listen port (default 8512)."},
                    "host": {"type": "string", "description": "Bind host (default 127.0.0.1)."},
                    "sync_every": {"type": "integer", "description": "Inner steps between syncs (default 500)."},
                    "async_mode": {"type": "boolean", "description": "Asynchronous (delayed-Nesterov) sync."},
                    "save_every": {"type": "integer", "description": "Save a checkpoint every N rounds (default 10)."},
                    "save_total_limit": {"type": "integer", "description": "Keep at most N checkpoints (default 3)."},
                    "num_fragments": {"type": "integer", "description": "Model fragments for staggered sync (default 1)."},
                    "min_workers": {"type": "integer", "description": "Minimum workers before syncing."},
                    "heartbeat_timeout": {"type": "number", "description": "Seconds before a silent worker is dropped."},
                    "outer_lr": {"type": "number", "description": "Outer (server) optimizer learning rate."},
                    "outer_momentum": {"type": "number", "description": "Outer optimizer momentum."},
                    "from_checkpoint": {"type": "string", "description": "Resume the server from this checkpoint."},
                    "run_name": {"type": "string", "description": "Run name for logging."},
                    "no_auth": {"type": "boolean", "description": "Disable the bearer-token gate."},
                    "backend": {"type": "string", "enum": ["http", "shared_memory", "collective"], "description": "Sync backend the worker group must use (default http). Advertised via /info; workers validate against it and fail loud on disagreement. Must match `submit --backend` for the workers."},
                    "advanced": {"type": "object", "description": "Other `forgather diloco server` flags (see docs/trainers/diloco.md), e.g. dylu, bf16_comm, upload_dtype, download_dtype, num_fragments tuning."},
                },
                "required": ["output_dir", "num_workers"],
            },
            handler=_start_diloco_server,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="start_inference_server",
            description=(
                "Start an inference (model) server (Sidebar -> Services) so the "
                "agent / webui can generate against it (see query_model). Provide "
                "EXACTLY ONE of model_path (single model) or models (a list of "
                "{name, path} for a multi-model server), plus a port. To serve a "
                "model you TRAINED (a Forgather output dir like output_models/"
                "<name>), set from_checkpoint=true so it loads the latest native "
                "checkpoint — a bare model_path expects an already-HF-format model "
                "and will fail to load a raw output dir. On unified-memory hardware "
                "(DGX Spark / Grace-Hopper) set keep_on_gpu=true. It reserves a "
                "GPU (requested_gpus, default 1); if none is free it QUEUES until "
                "one frees (check gpu_status). Approval required; then "
                "wait_for_job(queue_id, until='running')."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    **_NAME_PROP,
                    "model_path": {"type": "string", "description": "Path to a single model (output dir or HF model)."},
                    "models": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "path": {"type": "string"}}}, "description": "Multi-model server: [{name, path}, ...] (mutually exclusive with model_path)."},
                    "port": {"type": "integer", "description": "Listen port (required; e.g. 8137)."},
                    "host": {"type": "string", "description": "Bind host (default 127.0.0.1)."},
                    "requested_gpus": {"type": "integer", "description": "GPUs to reserve (default 1; 0 = CPU)."},
                    "dtype": {"type": "string", "description": "e.g. bfloat16 | float16 | float32."},
                    "device": {"type": "string", "description": "Explicit device override (e.g. 'cpu'); default auto."},
                    "from_checkpoint": {"type": "boolean", "description": "Load the latest native checkpoint instead of from_pretrained."},
                    "checkpoint_path": {"type": "string", "description": "Load a specific checkpoint path."},
                    "compile": {"type": "boolean", "description": "torch.compile the model."},
                    "disable_kv_cache": {"type": "boolean", "description": "Disable the KV cache."},
                    "keep_on_gpu": {"type": "boolean", "description": "Pin loaded models to GPU (required on unified-memory hosts)."},
                    "attn_implementation": {"type": "string", "description": "e.g. sdpa | eager | flash_attention_2."},
                    "chat_template": {"type": "string", "description": "Override the chat template."},
                    "cache_implementation": {"type": "string", "description": "KV cache implementation."},
                    "no_auth": {"type": "boolean", "description": "Disable the bearer-token gate."},
                },
                "required": ["port"],
            },
            handler=_start_inference_server,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="start_tensorboard",
            description=(
                "Start a TensorBoard server over a logdir (Sidebar -> Services). "
                "logdir and port are required. Approval required."
            ),
            summary="Start a TensorBoard server (logdir, port).",
            json_schema={
                "type": "object",
                "properties": {
                    **_NAME_PROP,
                    "logdir": {"type": "string", "description": "Directory of run logs to serve."},
                    "port": {"type": "integer", "description": "Listen port (e.g. 6006)."},
                    "host": {"type": "string", "description": "Bind host (default 127.0.0.1)."},
                    "bind_all": {"type": "boolean", "description": "Bind 0.0.0.0 and bypass the auth proxy (LAN)."},
                },
                "required": ["logdir", "port"],
            },
            handler=_start_tensorboard,
            risk=CONFIRM,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="start_mkdocs",
            description=(
                "Start an MkDocs live-docs server (Sidebar -> Services). "
                "config_file (mkdocs.yml) and port are required. Approval required."
            ),
            summary="Start an MkDocs server (config_file, port).",
            json_schema={
                "type": "object",
                "properties": {
                    **_NAME_PROP,
                    "config_file": {"type": "string", "description": "Path to mkdocs.yml."},
                    "port": {"type": "integer", "description": "Listen port (e.g. 9999)."},
                    "host": {"type": "string", "description": "Bind host (default localhost)."},
                    "strict": {"type": "boolean", "description": "Fail the build on warnings."},
                    "livereload": {"type": "boolean", "description": "Auto-reload on file changes (default true)."},
                    "dirty": {"type": "boolean", "description": "Dirty (incremental) reload."},
                    "watch": {"type": "array", "items": {"type": "string"}, "description": "Extra directories to watch."},
                },
                "required": ["config_file", "port"],
            },
            handler=_start_mkdocs,
            risk=CONFIRM,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="stop_service",
            description=(
                "Stop a configured service by type + name: disable its config "
                "entry and abort the running instance. Approval required. Use "
                "list_services to find the name."
            ),
            summary="Stop (disable + abort) a configured service by type+name.",
            json_schema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["dataset", "inference", "tensorboard", "mkdocs", "diloco"],
                    },
                    "name": {"type": "string"},
                },
                "required": ["type", "name"],
            },
            handler=_stop_service,
            risk=CONFIRM,
            tier=EXTENDED,
        )
    )
