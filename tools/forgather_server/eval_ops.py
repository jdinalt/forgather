"""Server-side wrappers around the Forgather eval subsystem.

Uses the library's own discovery helpers (``forgather.eval_config``) so the
CLI (``forgather eval list`` / ``show`` / ``test``) and the web server share
a single implementation. The server never shells out to ``forgather eval``;
it constructs the same argv that ``cli/eval.test_cmd`` produces and hands
it to the launcher, identical to how training jobs are spawned.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from forgather.cli.eval_args import forward_eval_script_args_from_params
from forgather.eval_config import iter_eval_configs
from forgather.user_config import eval_search_paths

from .search_roots import forgather_repo_root

# ---------------------------------------------------------------------------
# Server-side override for the eval search path.
#
# Populated by ``configure_eval_search_paths`` (called from server.py once
# CLI args are parsed). Mirrors the ``meta_templates.configure_roots`` shape:
# user-provided dirs come first in scan order so leaves there can shadow the
# bundled defaults on name collision; ``disable_default`` drops the bundled
# ``examples/evaluation`` directory entirely. Both pieces also remain
# configurable via ``~/.config/forgather/config.yaml``'s ``eval.*`` keys,
# which the library reads for plain ``forgather eval`` CLI invocations
# without the server in the loop.

_extra_eval_dirs: List[str] = []
_disable_default_eval: bool = False


def configure_eval_search_paths(
    extra_dirs: List[str], *, disable_default: bool = False
) -> None:
    """Set the active eval-project search path additions.

    ``extra_dirs`` come first in scan order. Non-existent paths are
    quietly dropped (a typo shouldn't disable discovery; the operator
    sees the empty contribution but the loop keeps going).
    """
    global _extra_eval_dirs, _disable_default_eval
    cleaned: List[str] = []
    for d in extra_dirs:
        ap = os.path.abspath(d)
        if os.path.isdir(ap) and ap not in cleaned:
            cleaned.append(ap)
    _extra_eval_dirs = cleaned
    _disable_default_eval = bool(disable_default)


def _resolve_search_paths() -> List[str]:
    """Combine the CLI overrides with the user-config-driven defaults.

    Order: server CLI extras first, then library's ``eval_search_paths``
    (which itself sources the bundled default + user-config extras).
    Drop the bundled default when ``--no-default-eval`` was passed.
    """
    base = eval_search_paths(forgather_repo_root())
    if _disable_default_eval:
        bundled = os.path.abspath(
            os.path.join(forgather_repo_root(), "examples", "evaluation")
        )
        base = [p for p in base if os.path.abspath(p) != bundled]
    # De-dup while preserving order.
    seen: set = set()
    out: List[str] = []
    for p in [*_extra_eval_dirs, *base]:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(ap)
    return out


@dataclass
class EvalConfigEntry:
    """One row for the eval-config picker.

    Shape mirrors what the user sees in ``forgather eval show``. The
    server returns these to the web UI so it can render a dropdown
    without re-implementing discovery logic.
    """

    name: str
    project_dir: str
    template: str
    description: str
    default_batch_size: int
    default_max_length: int
    default_stride: int


def list_eval_configs() -> List[EvalConfigEntry]:
    """Return every discoverable eval config.

    Search paths combine three sources, in this order: any directories
    passed via the server's ``--eval-dir`` flag, then the library's
    default discovery (bundled ``examples/evaluation`` plus the user's
    ``~/.config/forgather/config.yaml`` ``eval.search_paths``). The
    bundled default is dropped when ``--no-default-eval`` is set.
    """
    paths = _resolve_search_paths()
    out: List[EvalConfigEntry] = []
    for name, project_dir, template, data in iter_eval_configs(paths):
        out.append(
            EvalConfigEntry(
                name=name,
                project_dir=project_dir,
                template=template,
                description=data.description,
                default_batch_size=data.default_batch_size,
                default_max_length=data.default_max_length,
                default_stride=data.default_stride,
            )
        )
    out.sort(key=lambda e: e.name.lower())
    return out


def build_eval_command(
    *,
    eval_project: str,
    eval_template: str,
    model_path: str,
    gpu_indices: Optional[List[int]] = None,
    nproc_override: Optional[str] = None,
    **passthrough,
) -> List[str]:
    """Build the subprocess argv for an eval run.

    Mirror of ``forgather.cli.eval.test_cmd``. The caller resolves the eval
    config's ``project_dir`` / ``template`` first via
    :func:`list_eval_configs` and passes them in, so the server doesn't
    repeat the config walk for every enqueue.

    ``trainer="simple"`` bypasses torchrun (direct python); ``ddp`` /
    ``pipeline`` use ``torchrun --standalone --nproc-per-node gpu``. The
    scheduler gates which GPUs are visible via ``CUDA_VISIBLE_DEVICES`` on
    the subprocess env, same as training jobs.

    All other args (``--trainer``, ``--checkpoint``, ``--fused-loss``, etc.)
    are declared once in ``forgather.cli.eval_args._EVAL_SCRIPT_ARGS`` and
    forwarded by ``forward_eval_script_args_from_params``. To add a new
    passthrough flag, append a single entry to that spec — no signature
    change here.
    """
    forgather_dir = forgather_repo_root()
    eval_script = os.path.join(forgather_dir, "scripts", "eval_script.py")

    trainer = passthrough.get("trainer", "ddp")
    if trainer == "simple":
        cmd: List[str] = [sys.executable, eval_script]
    else:
        # Explicit caller override (EvalModal nproc field, or another
        # caller bypassing the default) wins. Otherwise: zero-GPU
        # dispatch falls back from "gpu" to "1" so torchrun's "gpu"
        # sentinel doesn't crash with "invalid literal for int() with
        # base 10: 'gpu'" when CUDA_VISIBLE_DEVICES="" hides all
        # devices (mirrors src/forgather/cli/eval.py). When
        # gpu_indices is None the caller didn't tell us, so preserve
        # the legacy "gpu" default (the scheduler always passes a
        # list, so this branch is only hit by direct callers).
        if nproc_override is not None:
            nproc = nproc_override
        elif gpu_indices is not None and not gpu_indices:
            nproc = "1"
        else:
            nproc = "gpu"
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc-per-node",
            nproc,
            eval_script,
        ]

    cmd.extend(
        [
            "--eval-project",
            eval_project,
            "--eval-config",
            eval_template,
            "--model",
            model_path,
        ]
    )
    cmd.extend(forward_eval_script_args_from_params(passthrough))
    return cmd
