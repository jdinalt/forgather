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

    Search paths are resolved the same way the CLI resolves them — the
    repo's ``examples/evaluation`` directory by default, extensible via
    ``~/.config/forgather/config.yaml`` ``eval.search_paths``.
    """
    paths = eval_search_paths(forgather_repo_root())
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
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc-per-node",
            "gpu",
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
