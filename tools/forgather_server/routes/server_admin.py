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

Spawned subprocesses (training, inference, dataset_server, …) keep
running across the exec — they belong to their own process groups,
and the new server reattaches via the existing
``_reattach_or_cleanup_on_startup`` path on boot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from fastapi import APIRouter

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
