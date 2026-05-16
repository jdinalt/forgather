"""Server-level admin endpoints.

Currently exposes a single endpoint:

  POST /api/server/restart
      Schedule an in-place process restart. The handler flips a
      module-level flag, then sends ``SIGTERM`` to its own PID so
      uvicorn unwinds cleanly. The main entry point checks the flag
      after ``uvicorn.run`` returns and ``os.execv``s a fresh
      interpreter with the same argv — config-file changes take
      effect on the new boot.

Spawned subprocesses (training, inference, dataset_server, …) keep
running across the exec — they belong to their own process groups,
and the new server reattaches via the existing
``_reattach_or_cleanup_on_startup`` path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from fastapi import APIRouter

log = logging.getLogger("forgather_server.admin")
router = APIRouter(tags=["admin"])


# Set by the restart endpoint, read by ``server.main`` after uvicorn
# returns. Module-level so it survives the FastAPI lifespan teardown.
_RESTART_REQUESTED: bool = False


def is_restart_requested() -> bool:
    return _RESTART_REQUESTED


@router.post("/server/restart")
async def restart_server():
    """Schedule an in-place restart.

    Returns immediately so the HTTP response can flush before uvicorn
    starts unwinding. The signal is sent from a background task with
    a short delay; that way the response body reaches the client even
    if the OS schedules the signal before the response writer flushes.
    """
    global _RESTART_REQUESTED
    _RESTART_REQUESTED = True

    async def _shutdown_after_response():
        # Half a second is plenty for the response to flush; the
        # browser doesn't get a half-finished body if we let the loop
        # spin once after the handler returns.
        await asyncio.sleep(0.5)
        log.info("sending SIGTERM to self for restart")
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_shutdown_after_response())
    return {"restart": "scheduled"}
