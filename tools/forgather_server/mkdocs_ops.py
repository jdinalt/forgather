"""Server-side wrappers for launching ``mkdocs serve``.

MkDocs is a pure documentation server — no GPUs, no forgather imports.
The scheduler treats it as a job type so it queues alongside training /
eval / inference / tensorboard and shows up in the unified Jobs view
with TTY + kill controls; ``requested_gpus`` is always zero.

The server shells out to the stock ``mkdocs`` CLI rather than importing
it as a library, mirroring how ``tensorboard_ops`` shells out to the TB
CLI. That keeps the dependency graph clean (mkdocs is dev-only) and
lets users pick which mkdocs install to use just by adjusting PATH on
the server process.
"""

from __future__ import annotations

from typing import List, Optional


def build_mkdocs_command(
    *,
    config_file: str,
    host: str,
    port: int,
    strict: bool = False,
    livereload: bool = True,
    dirty: bool = False,
    watch: Optional[List[str]] = None,
) -> List[str]:
    """Build the argv for ``mkdocs serve``.

    ``config_file`` is required and should be an absolute path to the
    project's ``mkdocs.yml`` so the spawn doesn't depend on the server's
    cwd. ``host`` + ``port`` get folded into a single ``--dev-addr`` flag
    (the only form mkdocs accepts).
    """
    cmd: List[str] = [
        "mkdocs",
        "serve",
        "-f",
        config_file,
        "-a",
        f"{host}:{port}",
    ]
    if strict:
        cmd.append("--strict")
    if not livereload:
        cmd.append("--no-livereload")
    if dirty:
        cmd.append("--dirty")
    for w in watch or []:
        if w:
            cmd.extend(["--watch", w])
    return cmd
