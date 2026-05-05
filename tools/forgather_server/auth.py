"""Token + password authentication for the Forgather server.

Threat model: a single-user server bound to ``127.0.0.1`` is reachable by
any other local user on the same host (loopback ports are not isolated by
uid). This module raises the bar for cross-user access by gating every
``/api/`` request on a bearer token, an authenticated session cookie, or
a query-string token (used for browser bootstrap and WebSocket auth).

The token persists across server restarts in ``~/.forgather/server/auth_token``
(mode 0600) so CLI clients can read it without user interaction. The
optional password lives in ``~/.forgather/server/password_hash`` (also
mode 0600) and is used only for browser logins after the initial
token-bootstrap.

The module is small and dependency-free on purpose: the FastAPI integration
is a thin ASGI middleware so the same gate applies to plain HTTP routes
and WebSockets without per-route boilerplate.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Mapping, Optional
from urllib.parse import parse_qs

from . import paths
from ._atomic import atomic_write_text

log = logging.getLogger("forgather_server.auth")

# 32 bytes -> 64 hex characters. Plenty of entropy and still pasteable.
TOKEN_LENGTH_BYTES = 32

# Session ids are url-safe base64 of 32 random bytes (~43 chars).
SESSION_LENGTH_BYTES = 32

# Sessions are kept in-memory only; a 30-day TTL matches the typical
# browser tab lifetime for an interactive workstation.
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30

SESSION_COOKIE_NAME = "forgather_session"

# OWASP 2023 minimum for PBKDF2-HMAC-SHA256. Tuned for ~100 ms on a
# modern laptop; password verification is rare so the cost is fine.
PBKDF2_ITERATIONS = 600_000
PBKDF2_SALT_BYTES = 16
PBKDF2_KEY_BYTES = 32

_OPEN_PATHS = frozenset(
    {
        "/api/health",
        "/api/auth/status",
        "/api/auth/login",
    }
)

# Cluster API endpoints that may be called by a peer node without the
# bearer token. The carve-out is gated on the source IP belonging to a
# known cluster member — see ``_request_is_from_peer`` below — so it is
# not equivalent to making these paths fully public.
#
# Limited to read-only GETs in v1. Anything that mutates state still
# requires the regular auth credential, even from a peer.
_PEER_ALLOWED_PATHS = frozenset(
    {
        "/api/cluster/members",
        "/api/cluster/self",
        "/api/cluster/master",
        # Read-only local GPU snapshot used by the master's
        # cluster-wide aggregator. Going through a cluster-scoped
        # alias (rather than carving out the existing /api/gpus
        # path) keeps the trusted-peer surface explicitly inside
        # the cluster namespace.
        "/api/cluster/gpus_local",
    }
)

# Module-level state. Sessions intentionally do not survive process
# restart — both the bearer token and the password still work, so a
# restart only forces a re-login for already-open browser tabs.
_sessions: dict[str, float] = {}
_auth_disabled: bool = False


# ---------------------------------------------------------------------------
# Configuration toggles
# ---------------------------------------------------------------------------


def disable_auth() -> None:
    """Disable all auth checks. Used by ``--no-auth`` for legacy/dev use."""
    global _auth_disabled
    _auth_disabled = True
    log.warning("authentication is DISABLED")


def auth_disabled() -> bool:
    return _auth_disabled


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


def load_token() -> str:
    """Return the persisted bearer token, generating one if missing."""
    path = paths.auth_token_file()
    if path.exists():
        text = path.read_text().strip()
        if text:
            return text
    return generate_and_save_token()


def generate_and_save_token() -> str:
    token = secrets.token_hex(TOKEN_LENGTH_BYTES)
    path = paths.auth_token_file()
    atomic_write_text(path, token + "\n", mode=0o600)
    # Belt-and-suspenders: if the target already existed with looser
    # perms, the rename preserves them on some filesystems.
    try:
        os.chmod(path, 0o600)
    except OSError as e:
        log.warning("could not chmod %s to 0600: %s", path, e)
    return token


def regenerate_token() -> str:
    """Rotate the bearer token. Existing sessions remain valid."""
    return generate_and_save_token()


def verify_token(presented: Optional[str]) -> bool:
    if not presented:
        return False
    actual = load_token()
    return hmac.compare_digest(presented.strip(), actual)


# ---------------------------------------------------------------------------
# Password persistence
# ---------------------------------------------------------------------------


def has_password() -> bool:
    p = paths.password_hash_file()
    try:
        return p.stat().st_size > 0
    except FileNotFoundError:
        return False


def set_password(password: str) -> None:
    if not password:
        raise ValueError("password may not be empty")
    salt = secrets.token_bytes(PBKDF2_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS, PBKDF2_KEY_BYTES
    )
    line = f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}\n"
    path = paths.password_hash_file()
    atomic_write_text(path, line, mode=0o600)
    # Belt-and-suspenders: rename can preserve a pre-existing destination's
    # looser mode bits on some filesystems.
    try:
        os.chmod(path, 0o600)
    except OSError as e:
        log.warning("could not chmod %s to 0600: %s", path, e)


def verify_password(password: Optional[str]) -> bool:
    if not password:
        return False
    p = paths.password_hash_file()
    try:
        line = p.read_text().strip()
    except FileNotFoundError:
        return False
    parts = line.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        iters = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected = bytes.fromhex(parts[3])
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iters, len(expected)
    )
    return hmac.compare_digest(candidate, expected)


def clear_password() -> None:
    p = paths.password_hash_file()
    try:
        p.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def create_session() -> str:
    sid = secrets.token_urlsafe(SESSION_LENGTH_BYTES)
    _sessions[sid] = time.time()
    return sid


def revoke_session(sid: Optional[str]) -> None:
    if sid:
        _sessions.pop(sid, None)


def session_valid(sid: Optional[str]) -> bool:
    if not sid:
        return False
    created = _sessions.get(sid)
    if created is None:
        return False
    if time.time() - created > SESSION_TTL_SECONDS:
        _sessions.pop(sid, None)
        return False
    return True


def _reset_sessions_for_tests() -> None:
    """Test helper: drop all in-memory sessions."""
    _sessions.clear()


# ---------------------------------------------------------------------------
# Request authentication
# ---------------------------------------------------------------------------


def credential_kind(
    headers: Mapping[str, str],
    query: Mapping[str, str],
    cookies: Mapping[str, str],
) -> Optional[str]:
    """Identify which credential channel authenticated the request.

    Returns one of "disabled", "token", "cookie", "query_token", or None.
    Used by privileged endpoints (e.g. set-password) that distinguish a
    cookie session from a fresh bearer token.
    """
    if _auth_disabled:
        return "disabled"
    auth_header = headers.get("authorization") or headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        if verify_token(auth_header[7:]):
            return "token"
    if session_valid(cookies.get(SESSION_COOKIE_NAME)):
        return "cookie"
    qtok = query.get("token")
    if qtok and verify_token(qtok):
        return "query_token"
    return None


def authenticate(
    headers: Mapping[str, str],
    query: Mapping[str, str],
    cookies: Mapping[str, str],
) -> Optional[str]:
    """Return the credential kind if authenticated, else None.

    Truthy/falsy semantics are preserved for existing callers.
    """
    return credential_kind(headers, query, cookies)


def path_requires_auth(path: str) -> bool:
    """All ``/api/`` paths require auth except the explicit open list.

    Static webui assets and the SPA bundle are intentionally open: the
    login UI itself has to load before the user can authenticate.
    """
    if path in _OPEN_PATHS:
        return False
    if not path.startswith("/api/"):
        return False
    return True


def path_allows_peer(path: str) -> bool:
    """True if a known cluster peer may call ``path`` without auth.

    See ``_PEER_ALLOWED_PATHS`` for the rationale.
    """
    return path in _PEER_ALLOWED_PATHS


def _request_is_from_peer(scope) -> bool:
    """True if the request's source IP belongs to a known cluster peer.

    The cluster module is imported lazily so this auth module remains
    importable in environments where multi-node mode is not active
    (and therefore zeroconf is not loaded).
    """
    client = scope.get("client")
    if not client:
        return False
    address = client[0] if isinstance(client, (tuple, list)) else None
    if not address:
        return False
    try:
        from . import cluster
    except Exception:
        return False
    if not cluster.is_active():
        return False
    return cluster.is_peer_address(address)


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------


class AuthMiddleware:
    """Pure-ASGI gate so HTTP and WebSocket scopes share one code path."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path_requires_auth(path):
            await self.app(scope, receive, send)
            return

        # Headers come through as a list of (bytes, bytes) tuples; lowercase
        # the names so callers can use either Authorization or authorization.
        headers: dict[str, str] = {}
        for k, v in scope.get("headers", []):
            try:
                headers[k.decode("latin-1").lower()] = v.decode("latin-1")
            except UnicodeDecodeError:
                continue

        query_raw = scope.get("query_string", b"")
        query_flat: dict[str, str] = {}
        if query_raw:
            try:
                parsed = parse_qs(query_raw.decode("latin-1"))
                query_flat = {k: v[0] for k, v in parsed.items() if v}
            except Exception:
                query_flat = {}

        cookies = _parse_cookie_header(headers.get("cookie", ""))

        if authenticate(headers, query_flat, cookies):
            await self.app(scope, receive, send)
            return

        # Cluster peer-call carve-out: a GET on a peer-allowed path,
        # originating from a node we already know about, is treated as
        # an inter-node call. Limited to GET so any mutating endpoint
        # still requires a regular credential even from a peer.
        if (
            scope_type == "http"
            and scope.get("method", "").upper() == "GET"
            and path_allows_peer(path)
            and _request_is_from_peer(scope)
        ):
            await self.app(scope, receive, send)
            return

        if scope_type == "websocket":
            # Accept then close with a policy-violation code so the
            # client gets a clean handshake completion before rejection;
            # some WS clients raise opaque errors on a bare 403.
            await send({"type": "websocket.close", "code": 4401})
            return

        body = b'{"detail":"authentication required"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b'Bearer realm="forgather"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _parse_cookie_header(raw: str) -> dict[str, str]:
    """Tiny cookie-header parser. Avoids depending on ``http.cookies``,
    which is more lenient than we need and slower on hot paths."""
    out: dict[str, str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        value = value.strip()
        if name:
            out[name] = value
    return out
