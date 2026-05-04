"""R1: regression tests for /api auth coverage.

These tests guard against three classes of regression:

1. A route is mounted on the FastAPI app outside of ``/api/`` (and not
   in ``auth._OPEN_PATHS``) — silently bypassing ``AuthMiddleware``.
2. A path listed in ``_OPEN_PATHS`` does not actually correspond to a
   registered route (typo / stale entry — misconfiguration, not a
   security bug, but worth catching).
3. ``AuthMiddleware`` does not enforce its policy on a representative
   sample of routes (open-listed paths pass through; protected paths
   return the middleware's 401 body; static mount serves SPA).
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute, Mount
from fastapi.testclient import TestClient
from starlette.routing import Route as StarletteRoute
from starlette.routing import WebSocketRoute

# Static mount points are the only intentional non-/api/ routes; they
# serve the built SPA and must be reachable before the user logs in.
# Starlette stores a mount at "/" with ``Mount.path == ""`` (the trailing
# slash is stripped), so accept both spellings.
ALLOWED_STATIC_MOUNT_PATHS = frozenset({"/", ""})


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point FORGATHER_HOME at a tmp dir; reset auth module state."""
    monkeypatch.setenv("FORGATHER_HOME", str(tmp_path))
    from forgather_server import auth

    auth._reset_sessions_for_tests()
    auth._auth_disabled = False
    yield tmp_path


@pytest.fixture
def app(isolated_home):
    from forgather_server.app import create_app

    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


def _walk_routes(routes):
    """Yield (path, kind, route_obj) for every leaf route on the app.

    ``kind`` is one of "http", "websocket", or "mount". Mounts are
    yielded as a single record (we don't recurse into StaticFiles —
    the mount path itself is what matters for auth gating).
    """
    for r in routes:
        if isinstance(r, Mount):
            path = getattr(r, "path", "")
            yield (path, "mount", r)
            # Mounts can wrap a router with nested routes (not the case
            # for StaticFiles, but be safe). Recurse only for non-Static
            # mounts that expose a routes attribute.
            sub = getattr(r.app, "routes", None)
            if sub and not _is_static_files(r.app):
                for inner in _walk_routes(sub):
                    # Inner paths are relative to the mount; prefix.
                    inner_path, inner_kind, inner_obj = inner
                    yield (path.rstrip("/") + inner_path, inner_kind, inner_obj)
        elif isinstance(r, (APIWebSocketRoute, WebSocketRoute)):
            yield (r.path, "websocket", r)
        elif isinstance(r, (APIRoute, StarletteRoute)):
            yield (r.path, "http", r)
        else:
            # Unknown route type — surface it so the test author can
            # decide how to classify it rather than silently skipping.
            yield (getattr(r, "path", repr(r)), "unknown", r)


def _is_static_files(app_obj) -> bool:
    """True if the mounted ASGI app is a StaticFiles instance."""
    from fastapi.staticfiles import StaticFiles

    return isinstance(app_obj, StaticFiles)


def _is_allowed(path: str, kind: str, route_obj, open_paths) -> bool:
    if kind == "mount":
        return path in ALLOWED_STATIC_MOUNT_PATHS and _is_static_files(route_obj.app)
    if path in open_paths:
        return True
    if path.startswith("/api/"):
        return True
    return False


class TestRouteAuthCoverage:
    def test_every_route_is_under_api_or_open_listed(self, app):
        """Every HTTP/WS route must be /api/-prefixed or in _OPEN_PATHS.

        Static mounts at ``/`` (StaticFiles serving the SPA) are the
        only sanctioned exception. Anything else mounted on the app
        bypasses ``AuthMiddleware`` and is a security regression.
        """
        from forgather_server import auth

        open_paths = set(auth._OPEN_PATHS)
        offenders: list[tuple[str, str, str]] = []
        for path, kind, route_obj in _walk_routes(app.routes):
            if not _is_allowed(path, kind, route_obj, open_paths):
                offenders.append((path, kind, type(route_obj).__name__))

        assert not offenders, (
            "Routes mounted outside /api/ and not in _OPEN_PATHS bypass "
            "AuthMiddleware:\n"
            + "\n".join(f"  {kind:9} {cls:24} {path}" for path, kind, cls in offenders)
        )

    def test_open_paths_correspond_to_registered_routes(self, app):
        """Every entry in _OPEN_PATHS must match a real registered route.

        A typo here (e.g. ``/api/health/`` vs ``/api/health``) silently
        leaves the open-listed path requiring auth because it isn't
        actually a route. Not a security bug — but a misconfiguration.
        """
        from forgather_server import auth

        registered_paths = {
            path
            for path, kind, _ in _walk_routes(app.routes)
            if kind in ("http", "websocket")
        }
        missing = [p for p in auth._OPEN_PATHS if p not in registered_paths]
        assert not missing, f"Paths in _OPEN_PATHS without a matching route: {missing}"


class TestAuthMiddlewareEnforcement:
    """End-to-end checks against the live AuthMiddleware via TestClient."""

    def test_health_passes_through(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        # Body comes from the route, not the middleware.
        assert r.json() == {"status": "ok"}

    def test_auth_status_passes_through(self, client):
        r = client.get("/api/auth/status")
        assert r.status_code == 200
        body = r.json()
        assert "authenticated" in body

    def test_login_route_handles_invalid_creds_not_middleware(self, client):
        """POST /api/auth/login with bad creds: middleware must NOT
        intercept; the route handler returns its own 401 (different
        body than the middleware's ``authentication required``)."""
        r = client.post("/api/auth/login", json={"token": "definitely-wrong"})
        # Whatever the route's verdict, the body is NOT the middleware's
        # canonical 401 payload.
        assert r.content != b'{"detail":"authentication required"}'

    def test_protected_route_blocked_by_middleware(self, client):
        r = client.get("/api/queue")
        assert r.status_code == 401
        assert r.content == b'{"detail":"authentication required"}'
        assert r.headers.get("www-authenticate", "").lower().startswith("bearer")

    def test_static_mount_not_intercepted_by_middleware(self, client):
        """Hitting a path served by the StaticFiles mount must not
        produce the middleware's 401. Either StaticFiles serves the
        SPA (200) or replies 404 — but never 401."""
        r = client.get("/")
        assert r.status_code != 401
        assert r.content != b'{"detail":"authentication required"}'
