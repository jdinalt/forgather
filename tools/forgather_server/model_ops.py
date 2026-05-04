"""Server-side wrappers for ``forgather model``.

Mirrors the argv that ``forgather model construct/test`` would build so
the scheduler can dispatch model construct/test runs as just another job
type. Same pattern as :mod:`eval_ops` and :mod:`finalize_ops`.

Model jobs are fire-and-forget: no trainer-control endpoint, no save /
stop. Lifecycle actions are kill / force-kill, like eval. They may
allocate one or more GPUs (for ``--device cuda`` or AMP) but the
scheduler also allows zero (``--device cpu`` / ``meta``) so the user
can spot-check a model definition without burning a GPU.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from . import config_ops

# Subcommand names accepted by ``forgather model``. The scheduler
# validates against this set before dispatch.
MODEL_SUBCOMMANDS = ("construct", "test")


def _dynamic_args_to_argv(
    project_dir: str,
    config_name: str,
    dynamic_args: Dict[str, Any],
) -> List[str]:
    """Render ``{dest: value}`` to ``--cli-name value`` pairs.

    Unlike ``scripts/train_script.py``, the model / dataset CLI subcommands
    don't accept a single ``--dynamic-args`` JSON blob — each dynamic arg
    is added to argparse as its own flag. So we look up the schema, find
    the ``cli_name`` for each dest, and emit one flag per entry.
    Booleans become bare flags when truthy and are dropped otherwise; any
    dest the schema doesn't know about is silently skipped (matches the
    overrides cache behavior).
    """
    if not dynamic_args:
        return []
    try:
        schema = config_ops.load_dynamic_args(project_dir, config_name)
    except Exception:
        # If the schema can't load (parse error, etc.) the caller will
        # also fail downstream; emit nothing rather than hand-roll
        # arguments we can't validate.
        return []
    by_dest = {a.dest: a for a in schema}
    out: List[str] = []
    for dest, value in dynamic_args.items():
        spec = by_dest.get(dest)
        if spec is None or value is None:
            continue
        if spec.type == "bool":
            if bool(value):
                out.append(spec.cli_name)
            continue
        out.extend([spec.cli_name, str(value)])
    return out


def build_model_command(
    *,
    project_dir: str,
    config_name: str,
    subcommand: str,
    dynamic_args: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    no_init_weights: bool = False,
    load_from_checkpoint: Optional[str] = None,
    gradient_checkpointing: bool = False,
    fuse_optim_with_backward: bool = False,
    refresh_model: bool = False,
    save_checkpoint: bool = False,
    safetensors: bool = False,
    # ``test`` subcommand only — ignored when subcommand == "construct".
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    steps: Optional[int] = None,
    lr: Optional[float] = None,
    dataset_project: Optional[str] = None,
    dataset_config: Optional[str] = None,
    packed: bool = False,
    compile: bool = False,
    compile_backend: Optional[str] = None,
    compile_mode: Optional[str] = None,
    compile_dynamic: Optional[bool] = None,
    compile_fullgraph: bool = False,
    amp: Optional[str] = None,
) -> List[str]:
    """Build argv for ``forgather model construct/test``.

    Invokes the CLI as a Python module so the same code paths run as
    interactive use. Global ``-p`` / ``-t`` come before the subcommand
    name so the global parser sees them; everything past ``model`` is
    parsed by the model-subcommand parser.
    """
    if subcommand not in MODEL_SUBCOMMANDS:
        raise ValueError(
            f"unknown model subcommand: {subcommand!r}; "
            f"expected one of {MODEL_SUBCOMMANDS}"
        )

    cmd: List[str] = [
        sys.executable,
        "-m",
        "forgather.cli",
        "-p",
        project_dir,
        "-t",
        config_name,
        "model",
    ]
    if device:
        cmd.extend(["--device", device])
    if dtype:
        cmd.extend(["--dtype", dtype])
    if no_init_weights:
        cmd.append("--no-init-weights")
    if load_from_checkpoint:
        cmd.extend(["--load-from-checkpoint", load_from_checkpoint])
    if gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if fuse_optim_with_backward:
        cmd.append("--fuse-optim-with-backward")
    if refresh_model:
        cmd.append("--refresh-model")
    if save_checkpoint:
        cmd.append("--save-checkpoint")
    if safetensors:
        cmd.append("--safetensors")

    # Dynamic args are added by ``parse_dynamic_args`` to the parent
    # ``model`` parser, not the subcommand subparser, so they belong
    # before the subcommand name.
    cmd.extend(_dynamic_args_to_argv(project_dir, config_name, dynamic_args or {}))

    cmd.append(subcommand)

    if subcommand == "test":
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
        if sequence_length is not None:
            cmd.extend(["--sequence-length", str(sequence_length)])
        if steps is not None:
            cmd.extend(["--steps", str(steps)])
        if lr is not None:
            cmd.extend(["--lr", str(lr)])
        if dataset_project:
            cmd.extend(["--dataset-project", dataset_project])
        if dataset_config:
            cmd.extend(["--dataset-config", dataset_config])
        if packed:
            cmd.append("--packed")
        if compile:
            cmd.append("--compile")
        if compile_backend:
            cmd.extend(["--compile-backend", compile_backend])
        if compile_mode:
            cmd.extend(["--compile-mode", compile_mode])
        if compile_dynamic is False:
            cmd.append("--no-compile-dynamic")
        if compile_fullgraph:
            cmd.append("--compile-fullgraph")
        if amp:
            cmd.extend(["--amp", amp])

    return cmd
