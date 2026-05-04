"""FastAPI app factory for the Forgather server."""

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from . import scheduler, search_roots
from .auth import AuthMiddleware
from .routes import auth as auth_routes
from .routes import configs as configs_routes
from .routes import docs as docs_routes
from .routes import fs as fs_routes
from .routes import generation_configs as generation_configs_routes
from .routes import gpus as gpus_routes
from .routes import inference_proxy as inference_proxy_routes
from .routes import jobs as jobs_routes
from .routes import models as models_routes
from .routes import projects as projects_routes
from .routes import queue as queue_routes
from .routes import search_roots as search_roots_routes
from .routes import tb_proxy as tb_proxy_routes

log = logging.getLogger("forgather_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run the dispatcher loop alongside the HTTP server.

    The dispatcher starts ``enabled=True`` so a freshly-restarted
    server resumes dispatch immediately — operators were forgetting
    to flip the switch after a restart and finding their queues
    silently stalled. Pause anytime via the ⏸ button in the sidebar
    header (``POST /api/queue/scheduler {enabled: false}``).
    """
    task = asyncio.create_task(scheduler.dispatcher_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    # Mount the OpenAPI schema and Swagger / Redoc UIs under ``/api/`` so
    # AuthMiddleware gates them. The defaults (``/openapi.json``, ``/docs``,
    # ``/redoc``) bypass the gate and leak the full route map — including
    # parameter shapes — to any local user, which violates the
    # other-local-users-on-host threat model.
    app = FastAPI(
        title="Forgather Server",
        version="0.1.0",
        description="Web frontend for Forgather project and job management (prototype).",
        lifespan=lifespan,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        swagger_ui_oauth2_redirect_url="/api/docs/oauth2-redirect",
    )

    # Auth middleware is added FIRST so CORS ends up outermost — that
    # way preflight OPTIONS requests are answered by CORS without ever
    # reaching the auth gate (browsers don't send credentials on
    # preflight, so an auth check here would break every cross-origin
    # request).
    app.add_middleware(AuthMiddleware)

    # CORS for the Vite dev server, which serves the SPA on its own port
    # and proxies /api. ``allow_credentials`` must be true so the
    # session cookie can flow back to the browser; that in turn forbids
    # ``allow_origins=['*']`` per the CORS spec, so we list the dev
    # server origins explicitly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/server-identity")
    async def server_identity():
        """Stable per-server identity for namespacing client-side
        persisted state.

        ``localStorage`` in the browser is keyed by *origin*
        (scheme + host + port), not by which Forgather server is
        currently reachable through that origin. Two different
        servers SSH-forwarded onto the same loopback port share
        localStorage and cross-contaminate persisted defaults
        (e.g. an MkDocs configFile that points at
        ``/path/from/server-A/mkdocs.yml`` ends up filling the
        modal connected to server B). The frontend mixes this
        identity into every persisted key so each server gets its
        own bucket. ``repo_root`` is included for diagnostics —
        the ``identity`` hash is what the namespace actually uses.
        """
        repo = search_roots.forgather_repo_root()
        return {
            "repo_root": repo,
            # 12 chars of hex is plenty for collision-avoidance
            # within one user's set of Forgather installs while
            # keeping the localStorage keys readable in devtools.
            "identity": hashlib.sha256(repo.encode("utf-8")).hexdigest()[:12],
        }

    app.include_router(auth_routes.router, prefix="/api")
    app.include_router(search_roots_routes.router, prefix="/api")
    app.include_router(projects_routes.router, prefix="/api")
    app.include_router(configs_routes.router, prefix="/api")
    app.include_router(docs_routes.router, prefix="/api")
    app.include_router(fs_routes.router, prefix="/api")
    app.include_router(generation_configs_routes.router, prefix="/api")
    app.include_router(gpus_routes.router, prefix="/api")
    app.include_router(inference_proxy_routes.router, prefix="/api")
    app.include_router(jobs_routes.router, prefix="/api")
    app.include_router(models_routes.router, prefix="/api")
    app.include_router(queue_routes.router, prefix="/api")
    # Auth-gated reverse proxy to spawned TensorBoard instances. Defaults
    # in tensorboard_ops bind TB to loopback so other local users can't
    # reach it directly; this proxy mounts it under /api/tb/{job_id}/...
    # so the webui (gated by AuthMiddleware above) can still serve it.
    app.include_router(tb_proxy_routes.router, prefix="/api")

    # Serve the built webui if it's present.
    webui_dist = Path(__file__).resolve().parent / "webui" / "dist"
    if webui_dist.is_dir():
        app.mount(
            "/",
            CachingStaticFiles(directory=str(webui_dist), html=True),
            name="webui",
        )

    return app


class CachingStaticFiles(StaticFiles):
    """StaticFiles with SPA-correct Cache-Control headers.

    Vite emits content-hashed filenames under ``/assets/`` and a single
    ``index.html`` at the root that references them. The bundles are
    safe to cache forever (their names change when contents change),
    but ``index.html`` must always be revalidated — otherwise the
    browser keeps loading old hashed bundles after a redeploy. By
    default Starlette emits no ``Cache-Control`` for either, leaving
    browsers to fall back to heuristic freshness, which on
    ``index.html`` means a freshly-rebuilt webui can stay invisible
    until the user issues a hard reload (Ctrl+Shift+R).
    """

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            if path.startswith("assets/"):
                response.headers["cache-control"] = (
                    "public, max-age=31536000, immutable"
                )
            else:
                # index.html and any other top-level file: always
                # revalidate. ``no-cache`` still uses the conditional
                # ETag/Last-Modified round-trip, so unchanged files
                # answer with a 304 — cheap, and correct.
                response.headers["cache-control"] = "no-cache"
        return response
