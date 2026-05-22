"""HTTP API for configured services (auto-start entries).

GET /api/services              List every configured service with its
                               current running status.
POST /api/services             Upsert a single service (create or replace
                               by <type, name>).
DELETE /api/services/{type}/{name}
                               Remove a service entry from the config.
POST /api/services/{type}/{name}/enabled
                               Toggle the auto-start flag. When enabled
                               flips true the autostart pass runs so the
                               service is brought up immediately; when
                               flipped false the corresponding running
                               instance (if any) is aborted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import auth as auth_mod
from .. import scheduler, services

log = logging.getLogger("forgather_server.services_api")
router = APIRouter(tags=["services"])


class ServiceModel(BaseModel):
    type: str
    name: str
    enabled: bool
    args: Dict[str, Any]
    signature: str


class ServiceStatusModel(BaseModel):
    service: ServiceModel
    running: bool
    queue_id: Optional[str] = None
    status: Optional[str] = None


class UpsertRequest(BaseModel):
    type: str
    name: str
    enabled: bool = True
    args: Dict[str, Any] = {}


class EnabledRequest(BaseModel):
    enabled: bool


def _to_model(svc: services.Service) -> ServiceModel:
    return ServiceModel(
        type=svc.type,
        name=svc.name,
        enabled=svc.enabled,
        # ``args`` is rendered verbatim in ServicesPanel.tsx via
        # formatArgValue(); a service for an inference / dataset
        # server can carry an ``auth_token`` (or ``--auth-token``)
        # value the operator typed in. Redact in demo mode so it
        # doesn't leak through to the webui.
        args=auth_mod.redact_sensitive_in_demo(svc.args),
        signature=svc.signature(),
    )


def _status_to_model(s: services.ServiceStatus) -> ServiceStatusModel:
    return ServiceStatusModel(
        service=_to_model(s.service),
        running=s.running,
        queue_id=s.queue_id,
        status=s.status,
    )


@router.get("/services", response_model=List[ServiceStatusModel])
def list_services():
    return [
        _status_to_model(s) for s in services.status_for_each(services.list_services())
    ]


@router.post("/services", response_model=ServiceStatusModel)
def upsert_service(req: UpsertRequest):
    try:
        svc = services.upsert_service(req.type, req.name, req.enabled, req.args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # If the operator created (or re-enabled) an entry, run the autostart
    # pass so the service comes up immediately rather than waiting for
    # the next server restart.
    if svc.enabled:
        try:
            services.autostart()
        except Exception:
            log.exception("autostart after upsert failed")
    statuses = services.status_for_each([svc])
    return _status_to_model(statuses[0])


@router.delete("/services/{svc_type}/{name}", response_model=Dict[str, Any])
def delete_service(svc_type: str, name: str):
    # Best-effort: abort any running instance whose signature matches
    # this service before erasing it. Without this, the queue / Jobs
    # rows linger after the entry vanishes.
    existing = services.get_service(svc_type, name)
    if existing is not None:
        active = services.active_signatures().get(existing.signature())
        if active is not None:
            qid, _ = active
            try:
                scheduler.abort_or_cancel(qid)
            except Exception:
                log.exception("abort during service delete failed")
    if not services.delete_service(svc_type, name):
        raise HTTPException(
            status_code=404,
            detail=f"service not found: {svc_type}:{name}",
        )
    return {"deleted": f"{svc_type}:{name}"}


@router.post(
    "/services/{svc_type}/{name}/enabled",
    response_model=ServiceStatusModel,
)
def set_enabled(svc_type: str, name: str, req: EnabledRequest):
    svc = services.set_enabled(svc_type, name, req.enabled)
    if svc is None:
        raise HTTPException(
            status_code=404,
            detail=f"service not found: {svc_type}:{name}",
        )
    if req.enabled:
        try:
            services.autostart()
        except Exception:
            log.exception("autostart after enable failed")
    else:
        # Disabling => stop the corresponding running instance. Look up
        # by signature so a manually-spawned matching job is also caught.
        active = services.active_signatures().get(svc.signature())
        if active is not None:
            qid, _ = active
            try:
                scheduler.abort_or_cancel(qid)
            except Exception:
                log.exception("abort during disable failed")
    statuses = services.status_for_each([svc])
    return _status_to_model(statuses[0])
