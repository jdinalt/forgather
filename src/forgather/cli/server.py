"""
Forgather server command — launches the web frontend backend.
"""

import subprocess
import sys
from pathlib import Path


def server_cmd(args):
    """Launch the forgather web server.

    Args:
        args: Parsed arguments with ``remainder`` containing forwarded args.
    """
    script_path = _get_server_script_path()

    cmd_args = [sys.executable, str(script_path)]
    if hasattr(args, "remainder") and args.remainder:
        cmd_args.extend(args.remainder)

    print(f"Running: {' '.join(cmd_args)}")
    print()

    result = subprocess.run(cmd_args)
    return result.returncode


def _get_server_script_path():
    """Resolve tools/forgather_server/server.py relative to this file.

    This file lives at src/forgather/cli/server.py; the server entry point is
    at tools/forgather_server/server.py. Walk four levels up to find the repo
    root, then descend.
    """
    current_file = Path(__file__).resolve()
    forgather_root = current_file.parent.parent.parent.parent
    script_path = forgather_root / "tools" / "forgather_server" / "server.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find forgather server script at {script_path}."
        )
    return script_path
