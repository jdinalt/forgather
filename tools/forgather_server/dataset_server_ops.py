"""Server-side wrappers for the Forgather dataset server.

Mirrors the argv produced by ``forgather dataset-server start`` so the
scheduler can spawn the dataset server as just another job type. The
dataset server is CPU-only (no GPUs reserved), long-lived (runs until
killed), and supports its own bearer-token auth via ``--auth-token-file``.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

from .search_roots import forgather_repo_root


def build_dataset_server_command(
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    log_level: str = "INFO",
    no_hf: bool = False,
    allow_paths: bool = False,
    allow_downloads: bool = False,
    locals_: Optional[Sequence[Tuple[str, str]]] = None,
    config_file: Optional[str] = None,
    auth_token_file: Optional[str] = None,
    no_auth: bool = False,
    quiet_tokens: bool = False,
) -> List[str]:
    """Build the argv for ``tools/dataset_server/server.py``.

    ``locals_`` is a sequence of ``(name, path)`` pairs, each turned into
    ``--local NAME=PATH``. The auth token is always passed via a 0600
    file (when present) so it never lands in argv where local users could
    read it via ``ps``.
    """
    forgather_dir = forgather_repo_root()
    server_script = os.path.join(forgather_dir, "tools", "dataset_server", "server.py")
    cmd: List[str] = [
        sys.executable,
        server_script,
        "-H",
        host,
        "-p",
        str(port),
        "-l",
        log_level,
    ]
    if no_hf:
        cmd.append("--no-hf")
    if allow_paths:
        cmd.append("--allow-paths")
    if allow_downloads:
        cmd.append("--allow-downloads")
    if locals_:
        for name, path in locals_:
            cmd.extend(["--local", f"{name}={path}"])
    if config_file:
        cmd.extend(["--config", config_file])
    if no_auth:
        cmd.append("--no-auth")
    elif auth_token_file:
        cmd.extend(["--auth-token-file", auth_token_file])
    if quiet_tokens:
        cmd.append("--quiet-tokens")
    return cmd
