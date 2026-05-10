"""
forgather dataset-server — wrapper around tools/dataset_server.

Forwards all args to ``python -m tools.dataset_server`` so options like
``--port``, ``--host``, ``--allow-load``, and ``--log-level`` reach
the underlying script unchanged. Default port is 8766 (8765 is the
forgather orchestration server).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def dataset_server_cmd(args) -> int:
    """Launch the forgather dataset server.

    Args:
        args: Parsed arguments with ``remainder`` containing forwarded args.
    """
    repo_root = _repo_root()
    cmd_args = [sys.executable, "-m", "tools.dataset_server"]
    if hasattr(args, "remainder") and args.remainder:
        cmd_args.extend(args.remainder)

    print(f"Running: {' '.join(cmd_args)}")
    print()

    # Run from the repo root so `tools.dataset_server` resolves on
    # the import path even when the user invokes from a project subdir.
    result = subprocess.run(cmd_args, cwd=str(repo_root))
    return result.returncode


def _repo_root() -> Path:
    """Resolve the repo root from this file's location.

    This file lives at src/forgather/cli/dataset_server.py — walk four
    levels up to reach the repo root that contains ``tools/``.
    """
    return Path(__file__).resolve().parents[3]
