"""Shared per-port bearer-token file location for the standalone inference
server and its bundled CLI client.

When the server is started without ``--auth-token`` / ``--auth-token-file``
/ ``--no-auth`` it auto-generates a random token and writes it here so that
the bundled client running on the same host can pick it up without the user
having to copy/paste the value out of the server's stderr. Files are mode
0600, in a directory chmod 0700, and removed when the server exits.

The lookup is intentionally local-only: the client only consults the cache
when its ``--url`` resolves to ``127.0.0.1`` / ``::1`` / ``localhost``, so a
captured token cannot leak via a remote URL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _forgather_home() -> Path:
    env = os.environ.get("FORGATHER_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".forgather"


def inference_tokens_dir() -> Path:
    """Directory holding per-port standalone-server token files (0700)."""
    d = _forgather_home() / "inference"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def standalone_token_file(port: int) -> Path:
    """Path to the shared token file for a server bound to ``port``."""
    return inference_tokens_dir() / f"{int(port)}.token"


def write_standalone_token(port: int, token: str) -> Path:
    """Atomically write ``token`` to the standalone-server token file."""
    path = standalone_token_file(port)
    tmp = path.with_suffix(".token.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
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
    """Return True if ``url`` resolves to a loopback host."""
    try:
        host = urlparse(url).hostname
    except (TypeError, ValueError):
        return False
    return host in _LOCAL_HOSTS


def url_port(url: str) -> Optional[int]:
    """Return the explicit port from ``url`` (no protocol-default fallback)."""
    try:
        port = urlparse(url).port
    except (TypeError, ValueError):
        return None
    return port


def read_standalone_token(url: str) -> Optional[str]:
    """If ``url`` is local and a standalone server token file exists for its
    port, return the token. Otherwise None."""
    if not url_is_local(url):
        return None
    port = url_port(url)
    if port is None:
        return None
    path = standalone_token_file(port)
    try:
        token = path.read_text().strip()
    except OSError:
        return None
    return token or None
