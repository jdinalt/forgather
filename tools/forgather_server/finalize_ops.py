"""Server-side wrappers for ``forgather finalize``.

Mirrors the argv that ``forgather finalize`` would build by calling
``tools/finalize_model/finalize_model.py`` directly. Same pattern as
:mod:`convert_ops` and :mod:`eval_ops`.

Finalize is fire-and-forget: no trainer-control endpoint. The only
supported lifecycle actions are kill / force-kill.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from .search_roots import forgather_repo_root


def build_finalize_command(
    *,
    source: str,
    dest: str,
    checkpoint: Optional[str] = None,
    add_tokens: Optional[str] = None,
    skip_default_tokens: bool = False,
    chat_template_path: Optional[str] = None,
    no_auto_stop_tokens: bool = False,
    stop_tokens: Optional[str] = None,
    generation_config: Optional[str] = None,
    keep_optimizer: bool = False,
    root_copy: bool = False,
    safetensors: bool = False,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> List[str]:
    """Build the argv for ``tools/finalize_model/finalize_model.py``.

    Mirrors the CLI flag set documented by ``forgather finalize --help``.
    Source and destination are positional and required.
    """
    forgather_dir = forgather_repo_root()
    script = os.path.join(forgather_dir, "tools", "finalize_model", "finalize_model.py")

    cmd: List[str] = [sys.executable, script]

    if checkpoint:
        cmd.extend(["-c", checkpoint])
    if add_tokens:
        cmd.extend(["--add-tokens", add_tokens])
    if skip_default_tokens:
        cmd.append("--skip-default-tokens")
    if chat_template_path:
        cmd.extend(["-t", chat_template_path])
    if no_auto_stop_tokens:
        cmd.append("--no-auto-stop-tokens")
    if stop_tokens:
        cmd.extend(["--stop-tokens", stop_tokens])
    if generation_config:
        cmd.extend(["--generation-config", generation_config])
    if keep_optimizer:
        cmd.append("--keep-optimizer")
    if root_copy:
        cmd.append("--root-copy")
    if safetensors:
        cmd.append("--safetensors")
    if dtype:
        cmd.extend(["--dtype", dtype])
    if device:
        cmd.extend(["--device", device])
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["--log-level", log_level])

    # Positionals last for parser clarity.
    cmd.extend([source, dest])
    return cmd
