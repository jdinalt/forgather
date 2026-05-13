"""Token + password authentication for the Forgather server.

Threat model: a single-user server bound to ``127.0.0.1`` is reachable by
any other local user on the same host (loopback ports are not isolated by
uid). This module raises the bar for cross-user access by gating every
``/api/`` request on a bearer token, an authenticated session cookie, or
a query-string token (used for browser bootstrap and WebSocket auth).

The token persists across server restarts in ``~/.config/forgather/server/auth_token``
(mode 0600) so CLI clients can read it without user interaction. The
optional password lives in ``~/.config/forgather/server/password_hash`` (also
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

# Cluster API endpoints that another forgather node may call without
# a bearer token. The caller authenticates by presenting a CA-signed
# TLS client cert (mTLS — see ``_request_has_client_cert`` below); the
# auth gate then checks the path against this allow-list. These paths
# are the inter-node surface, intentionally narrow and explicit.
#
# Limited to read-only GETs; mutations have their own (smaller) list
# in ``_PEER_ALLOWED_MUTATIONS`` because granting writes to peers is
# a deliberate decision per endpoint.
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
        # Bandwidth self-test target — peer GETs this to time the
        # transfer. Returns a deterministic in-memory blob; never
        # touches state.
        "/api/cluster/bandwidth_local",
        # Per-rank job-status lookup. The master rolls up cluster-job
        # status by GETting this on each peer with the queue_id of
        # the assignment. Read-only — exposes a small status snapshot
        # of one local queue item, nothing else.
        "/api/cluster/training_status_local",
        # Cluster jobs list — read-only view of the bundle records.
        # Non-master nodes proxy to master via this path so every
        # cluster-mode webui shows the same job list. Returning the
        # bundle catalogue across the LAN is consistent with the
        # trusted-peer security contract.
        "/api/cluster/jobs",
        # Dataset-server inventory: each peer's local list of
        # dataset_servers (JobRecord-spawned + user-registered),
        # including the bearer token. The master aggregator polls
        # this every ~10s to build the cluster-wide routing index.
        # Tokens leaving this surface stay within the cluster bearer
        # trust boundary — see cluster_dataset_inventory.py.
        "/api/cluster/dataset_servers_local",
        # Master-aggregated dataset inventory + router. Non-master
        # nodes proxy these GETs to master so every webui and every
        # training client sees the same cluster-wide view.
        "/api/cluster/dataset_inventory",
        "/api/cluster/dataset_servers",
        "/api/cluster/dataset_router/resolve",
    }
)

# Peer-allowed mutating endpoints. POST is permitted from an mTLS-
# authenticated peer on these paths only — narrower than the GET
# allow-list above. Each entry here represents a deliberate decision
# that "another node may change my state with only cluster-CA-cert
# proof of identity"; the list should stay small.
_PEER_ALLOWED_MUTATIONS = frozenset(
    {
        # GPU enable/disable + priority gate. Lets the cluster Nodes
        # view route the click-to-toggle action to the owning node.
        "/api/cluster/gpu_policy_local",
        # Cluster-coordinator submit (Phase 3). The master generates
        # rdzv args and POSTs one of these to each participating
        # peer to enqueue the per-rank training job. Narrower than
        # carving out the entire /api/queue surface — the handler
        # only constructs training items with caller-supplied rdzv
        # args, never the other job_types.
        "/api/cluster/training_local",
        # Cluster-coordinator cancel: master DELETEs through this
        # path on each peer to abort the local queue item. Modeled
        # as POST so the carve-out (which only allows GET / POST)
        # applies cleanly without widening it to DELETE.
        "/api/cluster/training_cancel_local",
        # On-demand wake for the master's dataset-server collect
        # loop. Non-master nodes proxy here so an add/delete on a
        # peer's user-registry surfaces in the cluster inventory
        # within ~1 s instead of one collect tick. Read-only-ish:
        # the handler just sets an asyncio.Event.
        "/api/cluster/dataset_servers/refresh",
    }
)

# Peer-allowed path *prefixes* for endpoints whose final segments are
# templated by server_id, queue_id, etc. Exact-match doesn't fit
# templated routes; matching by prefix keeps the gate narrow as long
# as the prefix itself is unambiguous (every entry here must be a
# string no other API endpoint can start with).
_PEER_ALLOWED_PATH_PREFIXES = frozenset(
    {
        # Master-side cluster dataset_server proxy. Non-master nodes
        # forward webui ``/api/cluster/dataset_server_proxy/{id}/...``
        # GETs (status / datasets / cache / local / length / iter) to
        # the master over mTLS. The op set is validated against
        # ``_ALLOWED_PROXY_OPS`` in routes/cluster.py before any
        # forwarding happens.
        "/api/cluster/dataset_server_proxy/",
    }
)

_PEER_ALLOWED_MUTATION_PREFIXES = frozenset(
    {
        # Same family as above; the ``load`` op is POST. Anything
        # else under this prefix is rejected by _ALLOWED_PROXY_OPS in
        # routes/cluster.py.
        "/api/cluster/dataset_server_proxy/",
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
    """True if a known cluster peer may GET ``path`` without auth.

    See ``_PEER_ALLOWED_PATHS`` for the rationale.
    """
    if path in _PEER_ALLOWED_PATHS:
        return True
    return any(path.startswith(p) for p in _PEER_ALLOWED_PATH_PREFIXES)


def path_allows_peer_mutation(path: str) -> bool:
    """True if a known cluster peer may POST ``path`` without auth."""
    if path in _PEER_ALLOWED_MUTATIONS:
        return True
    return any(path.startswith(p) for p in _PEER_ALLOWED_MUTATION_PREFIXES)


def _request_has_client_cert(scope) -> bool:
    """True if the TLS handshake presented a CA-validated client cert.

    The custom uvicorn protocol (``ForgatherProtocol``) sets
    ``scope["extensions"]["tls"]["client_cert_verified"]`` whenever the
    peer presents a cert that passed validation against the cluster CA
    (``ssl_cert_reqs=CERT_OPTIONAL`` + ``ssl_ca_certs=<bundle>`` on the
    listener). Presence is therefore proof of cluster membership — a
    cert signed by our CA is by definition a legitimate peer.

    Returns False for plain-HTTP listeners, for TLS listeners without
    the custom protocol, and for connections where the peer did not
    present a cert.
    """
    extensions = scope.get("extensions") or {}
    tls_ext = extensions.get("tls") or {}
    return bool(tls_ext.get("client_cert_verified"))


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

        # Cluster inter-node call: an mTLS-authenticated peer (proven
        # by presenting a CA-signed client cert in the TLS handshake)
        # may GET a peer-allowed path or POST one of the explicitly
        # mutation-allowed cluster endpoints without a bearer token.
        # The path allow-lists encode what an inter-node call is
        # allowed to do; cert presence proves who is making it.
        if scope_type == "http" and _request_has_client_cert(scope):
            method = scope.get("method", "").upper()
            if (method == "GET" and path_allows_peer(path)) or (
                method == "POST" and path_allows_peer_mutation(path)
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
