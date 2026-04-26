"""
Finalize command for Forgather CLI.

Forwards arguments to ``tools/finalize_model/finalize_model.py``.
"""

import subprocess
import sys
from pathlib import Path


def finalize_cmd(args):
    """Launch the finalize_model script.

    Args:
        args: Parsed command-line arguments with ``dummy`` and ``remainder``
            attributes (set in main.py for REMAINDER-style passthrough).
    """
    script_path = _get_script_path()

    cmd_args = [sys.executable, str(script_path)]

    if hasattr(args, "dummy") and args.dummy:
        cmd_args.append(args.dummy)

    if hasattr(args, "remainder") and args.remainder:
        cmd_args.extend(args.remainder)

    print(f"Running: {' '.join(cmd_args)}")
    print()

    result = subprocess.run(cmd_args)
    return result.returncode


def _get_script_path():
    """Resolve the absolute path of the finalize_model script."""
    current_file = Path(__file__).resolve()
    forgather_root = current_file.parent.parent.parent.parent
    script_path = forgather_root / "tools" / "finalize_model" / "finalize_model.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find finalize script at {script_path}. "
            f"Expected it relative to the forgather installation."
        )

    return script_path
