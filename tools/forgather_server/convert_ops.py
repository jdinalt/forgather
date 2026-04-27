"""Server-side wrappers for ``forgather convert``.

Mirrors the argv that ``forgather convert`` would build by calling
``tools/convert_model/convert.py`` directly. Same pattern as
:mod:`inference_ops` and :mod:`eval_ops` — the server reuses the same
script the CLI does, just spawned with stdout/stderr piped into the
job's TTY log.

Convert is fire-and-forget: no trainer-control endpoint. Like eval, the
only supported lifecycle actions are kill / force-kill.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from .search_roots import forgather_repo_root


def build_convert_command(
    *,
    src_model_path: str,
    dst_model_path: str,
    reverse: bool = False,
    model_type: Optional[str] = None,
    dtype: Optional[str] = None,
    max_length: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    generation_test: bool = False,
    dry_run: bool = False,
    prompt: Optional[str] = None,
    compare_text_file: Optional[str] = None,
    debug_params: bool = False,
    chat_template_path: Optional[str] = None,
    add_tokens: Optional[str] = None,
    skip_default_tokens: bool = False,
    converter_paths: Optional[List[str]] = None,
    log_level: str = "INFO",
) -> List[str]:
    """Build the argv for ``tools/convert_model/convert.py``.

    Mirrors the CLI flag set documented by ``forgather convert --help``.
    Source and destination are positional and required. Direction is
    auto-detected by the script unless ``reverse=True`` is passed.
    """
    forgather_dir = forgather_repo_root()
    script = os.path.join(forgather_dir, "tools", "convert_model", "convert.py")

    cmd: List[str] = [sys.executable, script]

    if reverse:
        cmd.append("--reverse")
    if model_type:
        cmd.extend(["--model-type", model_type])
    if dtype:
        cmd.extend(["--dtype", dtype])
    if max_length is not None:
        cmd.extend(["--max-length", str(max_length)])
    if checkpoint_path:
        cmd.extend(["-c", checkpoint_path])
    if device:
        cmd.extend(["--device", device])
    if generation_test:
        cmd.append("-g")
    if dry_run:
        cmd.append("--dry-run")
    if prompt:
        cmd.extend(["--prompt", prompt])
    if compare_text_file:
        cmd.extend(["--compare-text-file", compare_text_file])
    if debug_params:
        cmd.append("--debug-params")
    if chat_template_path:
        cmd.extend(["-t", chat_template_path])
    if add_tokens:
        cmd.extend(["--add-tokens", add_tokens])
    if skip_default_tokens:
        cmd.append("--skip-default-tokens")
    for cp in converter_paths or []:
        if cp:
            cmd.extend(["--converter-path", cp])
    cmd.extend(["--log-level", log_level])

    # Positional args go last so the parser is unambiguous about which
    # tokens are flags vs the src/dst paths.
    cmd.extend([src_model_path, dst_model_path])
    return cmd
