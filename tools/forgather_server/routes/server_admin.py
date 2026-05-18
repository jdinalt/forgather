"""Server-level admin endpoints.

  POST /api/server/restart
      In-place process restart. The handler returns immediately
      ({"restart": "scheduled"}), then a background task waits long
      enough for the response body to flush and calls ``os.execv`` on
      the current Python interpreter with the original ``sys.argv``.
      The process image is replaced — PID, controlling TTY, and
      stdio FDs are preserved; the listening socket closes via
      ``SOCK_CLOEXEC`` (Python's default since 3.4) so the rebooted
      server can rebind. Config-file changes take effect on the new
      boot.

  POST /api/server/shutdown
      Graceful process exit. Optional body ``{"stop_jobs": true}``
      first SIGTERMs every non-terminal JobRecord's process group
      (training, inference, dataset_server, …) so those subprocesses
      go down with the server instead of being left detached on the
      host.

Spawned subprocesses (training, inference, dataset_server, …) keep
running across the exec — they belong to their own process groups,
and the new server reattaches via the existing
``_reattach_or_cleanup_on_startup`` path on boot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from fastapi import APIRouter
from pydantic import BaseModel

from .. import job_records, scheduler

log = logging.getLogger("forgather_server.admin")
router = APIRouter(tags=["admin"])


@router.post("/server/restart")
async def restart_server():
    """Schedule an in-place ``execv`` restart.

    Returns immediately so the HTTP response can flush before the
    exec replaces the process image; without the short sleep the
    browser would occasionally see a truncated body.
    """

    async def _exec_after_response():
        # Half a second is plenty for the response to flush. The
        # event-loop tick after the handler returns lets uvicorn
        # write the body, then we hop into the new process image.
        await asyncio.sleep(0.5)
        log.info(
            "in-place restart: execv(%s, %s)",
            sys.executable,
            sys.argv,
        )
        # Stdout/stderr buffers don't survive execv — flush so the
        # banner / log lines hit the terminal before the new
        # interpreter takes over.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except OSError:
            # execv only returns on failure. If we get here the new
            # interpreter never started — log and exit hard so the
            # supervisor (if any) sees a failure rather than a
            # half-dead server.
            log.exception("execv failed; exiting")
            os._exit(1)

    asyncio.create_task(_exec_after_response())
    return {"restart": "scheduled"}


class ShutdownRequest(BaseModel):
    stop_jobs: bool = False


@router.post("/server/shutdown")
async def shutdown_server(req: ShutdownRequest):
    """Schedule a graceful exit.

    When ``stop_jobs`` is true, SIGTERM every non-terminal JobRecord's
    process group first so spawned trainers / inference servers / etc.
    are torn down with the server. Otherwise the subprocesses are left
    running and the next ``forgather server`` boot will reattach them.
    """
    stop_jobs = bool(req.stop_jobs)
    killed: list[str] = []
    if stop_jobs:
        for rec in job_records.list_records():
            if rec.status in job_records.TERMINAL_STATUSES:
                continue
            try:
                if scheduler.abort_record(rec.queue_id):
                    killed.append(rec.queue_id)
            except Exception:
                log.exception("failed to abort %s during shutdown", rec.queue_id)

    async def _exit_after_response():
        await asyncio.sleep(0.5)
        log.info("shutdown requested; exiting (stop_jobs=%s)", stop_jobs)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # SIGTERM to ourselves so uvicorn's signal handler unwinds the
        # ASGI app cleanly. If anything in the handler chain hangs, the
        # supervisor (or operator) can still SIGKILL from outside.
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            log.exception("self-SIGTERM failed; exiting hard")
            os._exit(0)

    asyncio.create_task(_exit_after_response())
    return {"shutdown": "scheduled", "stopped_jobs": killed}
