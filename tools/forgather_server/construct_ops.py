"""Server-side wrappers for ``forgather construct``.

Mirrors the argv that ``forgather construct`` would build so the scheduler
can dispatch construct runs as just another job type. Same pattern as
:mod:`model_ops` — every dynamic arg is rendered as its own ``--cli-name
value`` pair (the construct CLI registers each one with argparse rather
than accepting a single ``--dynamic-args`` JSON blob).

Construct jobs are fire-and-forget: no trainer-control endpoint, no
save / stop. Lifecycle actions are kill / force-kill. GPUs are optional
— most targets materialize on the meta device or CPU, but the user may
opt into a GPU through the modal so targets that allocate real tensors
(or run a tokenizer trainer on CUDA) have one available.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from . import config_ops


def _dynamic_args_to_argv(
    project_dir: str,
    config_name: str,
    dynamic_args: Dict[str, Any],
) -> List[str]:
    """Render ``{dest: value}`` to ``--cli-name value`` pairs.

    The ``forgather construct`` CLI registers each dynamic arg as its own
    argparse flag (via ``parse_dynamic_args``), so we look up the schema
    to recover each entry's ``cli_name`` and emit one flag per dest.
    Booleans become bare flags when truthy; unknown dests are silently
    skipped (the overrides cache may hold stale keys after a schema
    edit).
    """
    if not dynamic_args:
        return []
    try:
        schema = config_ops.load_dynamic_args(project_dir, config_name)
    except Exception:
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


def build_construct_command(
    *,
    project_dir: str,
    config_name: str,
    target: str = "main",
    dynamic_args: Optional[Dict[str, Any]] = None,
    call: bool = False,
) -> List[str]:
    """Build argv for ``forgather construct``.

    Invokes the CLI as a Python module so the same code paths run as on
    the command line. Global ``-p`` / ``-t`` come before the subcommand
    name; ``--target`` / ``--call`` are parsed by the construct
    subparser.
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "forgather.cli",
        "-p",
        project_dir,
        "-t",
        config_name,
    ]
    # Dynamic args are added by ``parse_dynamic_args`` to the *global*
    # parser, so they sit before the subcommand name.
    cmd.extend(_dynamic_args_to_argv(project_dir, config_name, dynamic_args or {}))
    cmd.append("construct")
    if target:
        cmd.extend(["--target", target])
    if call:
        cmd.append("--call")
    return cmd
