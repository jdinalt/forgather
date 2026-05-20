"""Server-side wrappers for the Forgather inference server.

Mirrors the argv produced by ``forgather inf server`` (actually the direct
``tools/inference_server/server.py`` invocation it wraps) so the scheduler
can spawn inference as just another job type. Unlike eval, inference jobs
are long-lived: they run until killed. The server has no trainer-control
protocol, so kill / force-kill are the only control actions.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .search_roots import forgather_repo_root


def build_inference_command(
    *,
    # Single-model legacy path: pass ``model_path``.
    # Multi-model path: pass ``models`` (list of {"name": str, "path": str}).
    # Exactly one of the two must be set.
    model_path: Optional[str] = None,
    models: Optional[Sequence[Dict[str, Any]]] = None,
    port: int,
    host: str = "127.0.0.1",
    # Inside the subprocess ``CUDA_VISIBLE_DEVICES`` is restricted by the
    # launcher to the reserved GPU(s), so the device is always cuda:0 from
    # the process's point of view regardless of the outside index.
    device: str = "cuda:0",
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    # Three checkpoint modes:
    #   - ``checkpoint_path`` is a specific path → ``-c <path>``
    #   - ``from_checkpoint`` is True and no path → ``-c`` (latest)
    #   - neither → load via Transformers ``from_pretrained`` (no ``-c``)
    checkpoint_path: Optional[str] = None,
    from_checkpoint: bool = False,
    compile: bool = False,
    disable_kv_cache: bool = False,
    ignore_eos: bool = False,
    keep_on_gpu: bool = False,
    chat_template: Optional[str] = None,
    cache_implementation: Optional[str] = None,
    compile_args: Optional[str] = None,
    log_level: str = "INFO",
    # Pass the token via a 0600 file (NOT --auth-token) so it never lands
    # in argv where any local user can read it via /proc or `ps`.
    auth_token_file: Optional[str] = None,
    no_auth: bool = False,
) -> List[str]:
    """Build the argv for ``tools/inference_server/server.py``.

    Callers (the launcher) pass through. The -c flag is tri-valued:
    present with a path pins the load to that checkpoint; present alone
    loads the latest Forgather checkpoint; absent uses
    ``AutoModelForCausalLM.from_pretrained`` against ``model_path``.
    """
    if (model_path is None) == (models is None):
        raise ValueError(
            "build_inference_command: exactly one of 'model_path' or 'models' "
            "must be provided"
        )
    if models is not None and len(models) > 1 and checkpoint_path:
        raise ValueError(
            "checkpoint_path is not supported with multiple models; pass "
            "from_checkpoint=True to load each model's latest checkpoint"
        )

    forgather_dir = forgather_repo_root()
    server_script = os.path.join(
        forgather_dir, "tools", "inference_server", "server.py"
    )
    cmd: List[str] = [
        sys.executable,
        server_script,
        "-H",
        host,
        "-p",
        str(port),
        "-d",
        device,
        "-l",
        log_level,
    ]
    if models is not None:
        for entry in models:
            name = entry["name"]
            path = entry["path"]
            cmd.extend(["-m", f"{name}={path}"])
    else:
        cmd.extend(["-m", model_path])
    if dtype:
        cmd.extend(["-T", dtype])
    if attn_implementation:
        cmd.extend(["-a", attn_implementation])
    if checkpoint_path:
        cmd.extend(["-c", checkpoint_path])
    elif from_checkpoint:
        cmd.append("-c")
    if compile:
        cmd.append("--compile")
    if disable_kv_cache:
        cmd.append("--disable-kv-cache")
    if ignore_eos:
        cmd.append("--ignore-eos")
    if keep_on_gpu:
        cmd.append("--keep-on-gpu")
    if chat_template:
        cmd.extend(["-t", chat_template])
    if cache_implementation:
        cmd.extend(["--cache-implementation", cache_implementation])
    if compile_args:
        cmd.extend(["--compile-args", compile_args])
    if no_auth:
        cmd.append("--no-auth")
    elif auth_token_file:
        cmd.extend(["--auth-token-file", auth_token_file])
    return cmd
