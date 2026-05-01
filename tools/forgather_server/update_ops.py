"""Server-side wrappers for ``forgather update``.

Mirrors the argv that ``forgather update`` would build by calling
``tools/update_model/update.py`` directly. Same pattern as
:mod:`convert_ops` and :mod:`finalize_ops` — the server reuses the same
script the CLI does, just spawned with stdout/stderr piped into the
job's TTY log.

Update is fire-and-forget: no trainer-control endpoint. Like convert /
finalize, the only supported lifecycle actions are kill / force-kill.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from .search_roots import forgather_repo_root


def build_update_command(
    *,
    src_model_path: str,
    dst_model_path: str,
    arch: Optional[str] = None,
    from_version: Optional[int] = None,
    to_version: Optional[int] = None,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    no_strict: bool = False,
    safetensors: bool = False,
    converter_paths: Optional[List[str]] = None,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> List[str]:
    """Build the argv for ``tools/update_model/update.py``.

    Mirrors the CLI flag set documented by ``forgather update --help``.
    Source and destination are positional and required. Other flags are
    optional and inherit their script-level defaults when omitted (the
    update script reads ``forgather_arch`` / ``forgather_arch_version``
    from the source ``config.json`` when ``arch`` / ``from_version`` are
    not supplied; ``to_version`` defaults to the converter's current
    ``arch_version``).
    """
    forgather_dir = forgather_repo_root()
    script = os.path.join(forgather_dir, "tools", "update_model", "update.py")

    cmd: List[str] = [sys.executable, script]

    if arch:
        cmd.extend(["--arch", arch])
    if from_version is not None:
        cmd.extend(["--from-version", str(from_version)])
    if to_version is not None:
        cmd.extend(["--to-version", str(to_version)])
    if checkpoint:
        cmd.extend(["-c", checkpoint])
    if device:
        cmd.extend(["--device", device])
    if dtype:
        cmd.extend(["--dtype", dtype])
    if no_strict:
        cmd.append("--no-strict")
    if safetensors:
        cmd.append("--safetensors")
    for cp in converter_paths or []:
        if cp:
            cmd.extend(["--converter-path", cp])
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["--log-level", log_level])

    # Positionals last for parser clarity.
    cmd.extend([src_model_path, dst_model_path])
    return cmd
