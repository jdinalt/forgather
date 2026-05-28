"""Per-port bearer-token auth for the DiLoCo parameter server.

Mirrors :mod:`tools.dataset_server.auth` but is stdlib-only — the
DiLoCo server uses :mod:`http.server`, not FastAPI, so this module
provides a request-handler-style verifier instead of a FastAPI
dependency.

Token discovery layout:

* ``<forgather_config_dir>/diloco_server/<port>.token`` (mode 0600,
  parent dir 0700). On Linux this is
  ``~/.config/forgather/diloco_server/<port>.token``.

The DiLoCo CLI's ``forgather diloco server`` subcommand and the
:class:`~forgather.ml.diloco.client.DiLoCoClient` both look here when
no explicit token / env var is supplied. The token is mirrored to the
forgather_server's ``JobRecord.auth_token`` field by the scheduler
when DiLoCo is spawned as a managed job, so the webui proxy can also
authenticate to the upstream server (see
``tools/forgather_server/routes/diloco.py``).
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import secrets
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from forgather.preprocess import forgather_config_dir

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}

#: Realm value used in ``WWW-Authenticate`` headers and log lines.
SERVICE_REALM = "forgather-diloco"


# ---------------------------------------------------------------------------
# Per-port token file
# ---------------------------------------------------------------------------


def diloco_tokens_dir() -> Path:
    """Directory holding per-port token files (mode 0700).

    Tightens both the parent ``<forgather_config_dir>`` and the
    ``diloco_server`` subdirectory to mode 0700 to keep token existence
    from leaking through ``ls`` (the contents themselves are always
    0600, but enumeration is an information leak in its own right).
    """
    home = Path(forgather_config_dir())
    try:
        os.chmod(home, 0o700)
    except OSError:
        pass
    d = home / "diloco_server"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def standalone_token_file(port: int) -> Path:
    """Path to the per-port token file for ``port``."""
    return diloco_tokens_dir() / f"{int(port)}.token"


def write_standalone_token(port: int, token: str) -> Path:
    """Atomically write ``token`` to the per-port token file (mode 0600).

    Uses ``os.open`` with explicit mode + ``os.fchmod`` to defeat
    permissive umasks, writes via a tmp suffix, then ``os.replace``
    for atomicity. Cleans up the tmp file on partial write.
    """
    path = standalone_token_file(port)
    tmp = path.with_suffix(".token.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w") as f:
            f.write(token)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


# ---------------------------------------------------------------------------
# Client-side token discovery (loopback URL → per-port file)
# ---------------------------------------------------------------------------


def url_is_local(url: str) -> bool:
    """True if ``url``'s hostname is a loopback alias."""
    try:
        host = urlparse(url).hostname
    except (TypeError, ValueError):
        return False
    return host in _LOCAL_HOSTS


def url_port(url: str) -> Optional[int]:
    """Port from ``url``, or ``None`` if absent / malformed."""
    try:
        port = urlparse(url).port
    except (TypeError, ValueError):
        return None
    return port


def read_standalone_token(url: str) -> Optional[str]:
    """Auto-discover a bearer token for a loopback ``url``.

    Returns the trimmed token string if ``url`` is loopback and a
    persisted token file exists for that port; ``None`` otherwise.
    Empty files return ``None`` (treated as "missing").
    """
    if not url_is_local(url):
        return None
    port = url_port(url)
    if port is None:
        return None
    try:
        token = standalone_token_file(port).read_text().strip()
    except OSError:
        return None
    return token or None


# ---------------------------------------------------------------------------
# CLI arg shape + token resolution (mirrors dataset_server precedence)
# ---------------------------------------------------------------------------


def add_auth_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard auth CLI flags to a server subparser.

    Shared with :mod:`forgather.cli.diloco_args` so the operator-facing
    surface matches dataset_server's. ``--auth-token`` /
    ``--auth-token-file`` / ``--no-auth`` form a mutex group; the
    others stand alone.
    """
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument(
        "--auth-token",
        default=None,
        help=(
            "Bearer token clients must present in 'Authorization: Bearer "
            "<token>'. Auto-generated and persisted per-port if neither "
            "this nor --auth-token-file is given."
        ),
    )
    auth_group.add_argument(
        "--auth-token-file",
        default=None,
        type=os.path.expanduser,
        help=(
            "Read the bearer token from this file (mode 0600 expected). "
            "Avoids exposing the token via argv."
        ),
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help=(
            "Disable bearer-token auth. Any host able to reach the bind "
            "port becomes able to control the server — only set this on "
            "a trusted LAN."
        ),
    )
    parser.add_argument(
        "--regen-token",
        action="store_true",
        help=(
            "Generate a fresh auth token at startup, overwriting the "
            "persisted per-port token file. Existing peer connections "
            "will start getting 401 until they pick up the new token."
        ),
    )
    parser.add_argument(
        "--quiet-tokens",
        action="store_true",
        help=(
            "Don't print the bearer token (or curl example) to stderr "
            "at launch. Token is still written to its per-port file "
            "when auto-generated."
        ),
    )


def resolve_auth_token(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Tuple[Optional[str], str]:
    """Resolve the effective bearer token from CLI args.

    Returns ``(token, source)``. ``token`` is ``None`` for
    ``--no-auth``; ``source`` is one of:

    - ``"cli"`` — explicit ``--auth-token`` (or ``--no-auth``).
    - ``"file"`` — explicit ``--auth-token-file``.
    - ``"persisted"`` — loaded from the per-port file.
    - ``"generated"`` — freshly minted; caller should persist it.
    - ``"regenerated"`` — freshly minted because ``--regen-token``
      was passed; caller should persist it and emit a loud banner.
    """
    if getattr(args, "no_auth", False):
        return None, "cli"
    if getattr(args, "auth_token", None):
        return args.auth_token.strip(), "cli"
    if getattr(args, "auth_token_file", None):
        try:
            text = Path(args.auth_token_file).read_text().strip()
        except OSError as exc:
            parser.error(f"could not read --auth-token-file: {exc}")
        if not text:
            parser.error(f"auth-token-file is empty: {args.auth_token_file}")
        return text, "file"
    if getattr(args, "regen_token", False):
        return secrets.token_hex(32), "regenerated"
    token_path = standalone_token_file(args.port)
    if token_path.is_file():
        try:
            text = token_path.read_text().strip()
        except OSError as exc:
            parser.error(f"could not read persisted token at {token_path}: {exc}")
        if text:
            return text, "persisted"
        # Empty file: fall through to mint.
    return secrets.token_hex(32), "generated"


def format_auth_mode(args: argparse.Namespace, token_source: Optional[str]) -> str:
    """Human-readable auth-mode line for the startup banner.

    Renders the *actual* outcome (cli / file / persisted / generated /
    regenerated) rather than re-deriving it from argv precedence,
    matching dataset_server's behavior for first-run vs reused-token
    diagnostics.
    """
    if getattr(args, "no_auth", False) or token_source is None:
        return "disabled (--no-auth)"
    if token_source == "cli":
        return "token via --auth-token"
    if token_source == "file":
        return f"token from file: {args.auth_token_file}"
    if token_source == "persisted":
        return "persisted per-port (reused existing token file)"
    if token_source == "generated":
        return "generated (minted + persisted to per-port file)"
    if token_source == "regenerated":
        return "regenerated (--regen-token; per-port file overwritten)"
    return token_source


# ---------------------------------------------------------------------------
# Request-time bearer verification
# ---------------------------------------------------------------------------


def _send_401(
    handler: BaseHTTPRequestHandler, message: str = "authentication required"
) -> None:
    """Send a 401 + WWW-Authenticate body and close the response.

    The body matches the ``{"error": ...}`` shape used by every other
    DiLoCo endpoint so existing clients can parse it uniformly.
    """
    import json

    body = json.dumps({"error": message, "realm": SERVICE_REALM}).encode("utf-8")
    handler.send_response(401)
    handler.send_header("WWW-Authenticate", f'Bearer realm="{SERVICE_REALM}"')
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def verify_bearer(
    handler: BaseHTTPRequestHandler, expected_token: Optional[str]
) -> bool:
    """Check the request's ``Authorization`` header against ``expected_token``.

    Returns ``True`` and lets the caller proceed when:

    * ``expected_token`` is falsy (auth disabled), OR
    * the header carries a ``Bearer`` token that matches
      ``expected_token`` via constant-time compare.

    Otherwise sends 401 + ``WWW-Authenticate: Bearer realm="…"`` and
    returns ``False`` so the caller can early-return.

    The constant-time compare prevents a partial-match timing leak
    that would let an attacker fingerprint the token byte-by-byte.
    """
    if not expected_token:
        return True
    auth_header = handler.headers.get("Authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        _send_401(handler)
        return False
    presented = auth_header.split(" ", 1)[1].strip()
    if not hmac.compare_digest(presented, expected_token):
        _send_401(handler)
        return False
    return True
