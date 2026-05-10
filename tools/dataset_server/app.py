"""
FastAPI application factory for the forgather dataset server.

The factory takes a :class:`ServerState` and an optional bearer token
and returns a configured :class:`fastapi.FastAPI`. Tests use this to
build the app in-process and drive it with a test client; the
``server.py`` entry point uses it to mount the app under uvicorn.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException

from .routes import API_VERSION, build_router, error_handler
from .state import ServerState


def create_app(
    state: ServerState,
    auth_token: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    state
        Mutable server state (handle cache, local mappings, policy).
    auth_token
        If non-empty, every gated endpoint requires
        ``Authorization: Bearer <token>``. ``None`` (or empty)
        disables auth — matches ``--no-auth``.
    """
    app = FastAPI(
        title="forgather dataset server",
        version=API_VERSION,
    )
    # ``state.auth_required`` is reflected in /v1/health and
    # /v1/auth/status — keep it in sync with the actual gating.
    state.auth_required = bool(auth_token)
    app.include_router(build_router(state, auth_token))

    # Translate raised HTTPExceptions into our {"error": "..."} body
    # shape, matching the previous PoC. FastAPI's default body uses
    # {"detail": "..."}; tests and existing curl examples expect
    # "error".
    app.add_exception_handler(HTTPException, error_handler)
    return app
