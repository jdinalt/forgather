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
    ``~/.forgather/config.yaml`` ``eval.search_paths``.
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
    checkpoint_path: Optional[str] = None,
    no_checkpoint: bool = False,
    trainer: str = "ddp",
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    max_steps: int = -1,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    compile: bool = False,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Build the subprocess argv for an eval run.

    Mirror of ``forgather.cli.eval.test_cmd`` (lines ~155–201). The caller
    resolves the eval config's ``project_dir`` / ``template`` first via
    :func:`list_eval_configs` and passes them in, so the server doesn't
    repeat the config walk for every enqueue.

    ``trainer="simple"`` bypasses torchrun (direct python); ``ddp`` /
    ``pipeline`` use ``torchrun --standalone --nproc-per-node gpu``. The
    scheduler gates which GPUs are visible via ``CUDA_VISIBLE_DEVICES`` on
    the subprocess env, same as training jobs.
    """
    forgather_dir = forgather_repo_root()
    eval_script = os.path.join(forgather_dir, "scripts", "eval_script.py")

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
            "--trainer",
            trainer,
            "--dtype",
            dtype,
            "--attn-implementation",
            attn_implementation,
            "--max-steps",
            str(max_steps),
        ]
    )
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if max_length is not None:
        cmd.extend(["--max-length", str(max_length)])
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    if no_checkpoint:
        cmd.append("--no-checkpoint")
    if compile:
        cmd.append("--compile")
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    return cmd
