"""
Per-port bearer-token auth for the dataset server.

Mirrors ``tools/inference_server/auth_paths.py`` and
``tools/inference_server/routes.py:_make_verify_bearer``: when the
server starts without ``--auth-token`` / ``--auth-token-file`` /
``--no-auth`` it auto-generates a 64-hex token and writes it to a
per-port file under
``<forgather_config_dir>/dataset_server/<port>.token`` (on Linux,
``~/.config/forgather/dataset_server/<port>.token``). Local clients
(CLI diagnostics, the loader-side `RemoteBackend`) discover the
token by reading that file when their URL is loopback.

Token files are mode 0600 in a directory mode 0700, and removed when
the server exits.
"""

from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import Header, HTTPException

from forgather.preprocess import forgather_config_dir

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}

#: Used in WWW-Authenticate / log lines.
SERVICE_REALM = "forgather-dataset"


def dataset_server_tokens_dir() -> Path:
    """Directory holding per-port token files (mode 0700).

    Also tightens the parent ``<forgather_config_dir>`` to 0o700. On a
    fresh install where the user has only ever run the standalone
    dataset_server (never the forgather_server itself, which does its
    own tighten), the parent could otherwise stay at the umask default
    — exposing the existence of ``dataset_server/`` to other local
    users via plain ``ls``. Token contents remain protected (0600 in
    0700 dir) but enumeration is itself a small information leak.
    """
    home = Path(forgather_config_dir())
    try:
        os.chmod(home, 0o700)
    except OSError:
        pass
    d = home / "dataset_server"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def standalone_token_file(port: int) -> Path:
    return dataset_server_tokens_dir() / f"{int(port)}.token"


def write_standalone_token(port: int, token: str) -> Path:
    """Atomically write ``token`` to the per-port token file (0600).

    Belt-and-suspenders on the mode: ``os.open(..., mode)`` is subject
    to the process umask (a permissive umask can clear bits), so an
    extra ``os.fchmod`` enforces the exact mode before any data lands
    on disk. Cleanup on partial write removes the tmp file rather
    than letting it linger.
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


def url_is_local(url: str) -> bool:
    try:
        host = urlparse(url).hostname
    except (TypeError, ValueError):
        return False
    return host in _LOCAL_HOSTS


def url_port(url: str) -> Optional[int]:
    try:
        port = urlparse(url).port
    except (TypeError, ValueError):
        return None
    return port


def read_standalone_token(url: str) -> Optional[str]:
    """If ``url`` is local and a token file exists for its port, return it."""
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


def make_verify_bearer(auth_token: str):
    """Build a FastAPI dependency that enforces ``Authorization: Bearer <token>``.

    Constant-time compare via ``hmac.compare_digest`` so a partial-match
    timing leak can't fingerprint the token. When ``auth_token`` is
    empty/None the route should not register this dep at all.
    """
    expected = auth_token

    async def verify_bearer(
        authorization: Optional[str] = Header(default=None),
    ):
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="authentication required",
                headers={"WWW-Authenticate": f'Bearer realm="{SERVICE_REALM}"'},
            )
        token = authorization.split(" ", 1)[1].strip()
        if not hmac.compare_digest(token, expected):
            raise HTTPException(
                status_code=401,
                detail="authentication required",
                headers={"WWW-Authenticate": f'Bearer realm="{SERVICE_REALM}"'},
            )
        return None

    return verify_bearer
