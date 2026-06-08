"""Agent tools for long-running services (Sidebar -> Services).

Start / stop / list the spawned services the server manages: dataset,
inference, tensorboard, mkdocs, diloco. Wraps the same ``services`` +
``scheduler`` machinery the Services panel uses, so a service the agent
starts shows up there and survives a restart.

- ``list_services`` (READ): configured services + live status.
- ``start_service`` (CONFIRM): persist the entry (enabled) and spawn it,
  unless a signature-identical instance is already running/queued (dedup
  prevents double-spawn). With per-type defaults, ``start_service(type=
  "dataset")`` brings up a default dataset server — which is what makes
  ``dataset_info`` usable when none is reachable.
- ``stop_service`` (CONFIRM): disable the entry and abort its running
  instance.

All mutations are CONFIRM-gated. ``start_service`` validates the
type-required args (inference needs a model; diloco needs an output dir +
worker count) before committing, so the spawn doesn't fail obscurely.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .. import queue_store, scheduler, services
from .registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_services")


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


def _validate_service_args(svc_type: str, args: Dict[str, Any]) -> None:
    if svc_type == "inference" and not (args.get("model_path") or args.get("models")):
        raise ValueError(
            "inference service requires args.model_path (or args.models)"
        )
    if svc_type == "diloco" and not (
        args.get("output_dir") and args.get("num_workers") is not None
    ):
        raise ValueError(
            "diloco service requires args.output_dir and args.num_workers"
        )


def _start_service(args: Dict[str, Any]) -> Proposal:
    svc_type = (args.get("type") or "").strip()
    if svc_type not in services.SERVICE_TYPES:
        raise ValueError(
            f"unknown service type {svc_type!r}; expected one of "
            f"{sorted(services.SERVICE_TYPES)}"
        )
    name = (args.get("name") or "agent-default").strip()
    svc_args = dict(args.get("args") or {})
    _validate_service_args(svc_type, svc_args)

    candidate = services.Service(type=svc_type, name=name, enabled=True, args=svc_args)
    sig = candidate.signature()
    # Preview the resolved job (job_type / gpus / project_dir) — no side effect.
    preview_item = services.build_queue_item(candidate)
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
            "it's reachable (e.g. list_dataset_servers / list_inference_servers)."
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


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="list_services",
            description=(
                "List the configured long-running services (dataset / inference "
                "/ tensorboard / mkdocs / diloco) with their args and live "
                "status (running, queue_id). Use to see what's up before "
                "start_service / stop_service."
            ),
            json_schema={"type": "object", "properties": {}},
            handler=_list_services,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="start_service",
            description=(
                "Start a long-running service and persist it to the server "
                "config so it shows in Sidebar -> Services and restarts with the "
                "server. type is one of dataset/inference/tensorboard/mkdocs/"
                "diloco; name defaults to 'agent-default'; args are the service's "
                "settings (the same fields its modal submits). Sensible per-type "
                "defaults apply, so start_service(type='dataset') brings up a "
                "default dataset server (use this when dataset_info reports none "
                "reachable). inference needs args.model_path; diloco needs "
                "args.output_dir + args.num_workers. Signature dedup means a "
                "matching instance already running is NOT re-spawned. Approval "
                "required; returns a queue_id to watch with list_jobs."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["dataset", "inference", "tensorboard", "mkdocs", "diloco"],
                    },
                    "name": {"type": "string", "description": "Service instance name (default 'agent-default')."},
                    "args": {"type": "object", "description": "Service settings (e.g. {port, model_path, output_dir, num_workers, ...})."},
                },
                "required": ["type"],
            },
            handler=_start_service,
            risk=CONFIRM,
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
        )
    )
