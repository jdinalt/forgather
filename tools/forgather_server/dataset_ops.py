"""Server-side wrappers for ``forgather dataset``.

Mirrors the argv that ``forgather dataset`` would build so the
scheduler can dispatch dataset preprocessing / inspection runs as just
another job type. Same pattern as :mod:`model_ops`.

Dataset jobs are CPU-only (no GPUs reserved), fire-and-forget. Lifecycle
actions are kill / force-kill, like eval and model.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from .model_ops import _dynamic_args_to_argv


def build_dataset_command(
    *,
    project_dir: str,
    config_name: str,
    dynamic_args: Optional[Dict[str, Any]] = None,
    tokenizer_path: Optional[str] = None,
    pp: bool = False,
    histogram: bool = False,
    target: Optional[str] = None,
    histogram_samples: Optional[int] = None,
    examples: Optional[int] = None,
    features: Optional[List[str]] = None,
    tokenized: bool = False,
    num_shards: Optional[int] = None,
    shard_index: Optional[int] = None,
    select_range: Optional[str] = None,
    seed: Optional[int] = None,
    example_stride: Optional[int] = None,
    truncate: Optional[int] = None,
) -> List[str]:
    """Build argv for ``forgather dataset``.

    Invokes the CLI as a Python module so the same code paths run as
    interactive use. Global ``-p`` / ``-t`` come before the subcommand
    name; dataset-specific flags follow.
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "forgather.cli",
        "-p",
        project_dir,
        "-t",
        config_name,
        "dataset",
    ]
    if tokenizer_path:
        cmd.extend(["--tokenizer-path", tokenizer_path])
    if pp:
        cmd.append("--pp")
    if histogram:
        cmd.append("--histogram")
    if target:
        cmd.extend(["--target", target])
    if histogram_samples is not None:
        cmd.extend(["--histogram-samples", str(histogram_samples)])
    if examples is not None:
        cmd.extend(["--examples", str(examples)])
    if features:
        # `--features` is nargs='*'; argparse stops consuming when it hits
        # a flag-shaped token. Refuse feature names beginning with `-` so a
        # caller can't inject argparse flags through this channel.
        for f in features:
            if not isinstance(f, str) or not f or f.startswith("-"):
                raise ValueError(
                    f"invalid feature name {f!r}: must be a non-empty string "
                    "that does not start with '-'"
                )
        cmd.append("--features")
        cmd.extend(features)
    if tokenized:
        cmd.append("--tokenized")
    if num_shards is not None:
        cmd.extend(["--num-shards", str(num_shards)])
    if shard_index is not None:
        cmd.extend(["--shard-index", str(shard_index)])
    if select_range:
        cmd.extend(["--select-range", select_range])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if example_stride is not None:
        cmd.extend(["--example-stride", str(example_stride)])
    if truncate is not None:
        cmd.extend(["--truncate", str(truncate)])

    cmd.extend(_dynamic_args_to_argv(project_dir, config_name, dynamic_args or {}))
    return cmd
