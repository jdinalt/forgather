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
from typing import List, Optional

from .search_roots import forgather_repo_root


def build_inference_command(
    *,
    model_path: str,
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
    chat_template: Optional[str] = None,
    cache_implementation: Optional[str] = None,
    compile_args: Optional[str] = None,
    log_level: str = "INFO",
) -> List[str]:
    """Build the argv for ``tools/inference_server/server.py``.

    Callers (the launcher) pass through. The -c flag is tri-valued:
    present with a path pins the load to that checkpoint; present alone
    loads the latest Forgather checkpoint; absent uses
    ``AutoModelForCausalLM.from_pretrained`` against ``model_path``.
    """
    forgather_dir = forgather_repo_root()
    server_script = os.path.join(
        forgather_dir, "tools", "inference_server", "server.py"
    )
    cmd: List[str] = [
        sys.executable,
        server_script,
        "-m",
        model_path,
        "-H",
        host,
        "-p",
        str(port),
        "-d",
        device,
        "-l",
        log_level,
    ]
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
    if chat_template:
        cmd.extend(["-t", chat_template])
    if cache_implementation:
        cmd.extend(["--cache-implementation", cache_implementation])
    if compile_args:
        cmd.extend(["--compile-args", compile_args])
    return cmd
