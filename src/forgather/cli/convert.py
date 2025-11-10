"""
Model conversion command for Forgather CLI.
"""

import os
import subprocess
import sys
from pathlib import Path


def convert_cmd(args):
    """
    Launch the model conversion script.

    Args:
        args: Parsed command-line arguments with dummy and remainder attributes
    """
    # Get path to convert_llama.py script
    script_path = _get_script_path("convert_llama.py")

    # Build command: python convert_llama.py <forwarded_args>
    cmd_args = [sys.executable, str(script_path)]

    # Add dummy argument if it's not empty (it captures the first positional)
    if hasattr(args, "dummy") and args.dummy:
        cmd_args.append(args.dummy)

    # Forward all remaining arguments
    if hasattr(args, "remainder") and args.remainder:
        cmd_args.extend(args.remainder)

    # Print command for transparency
    print(f"Running: {' '.join(cmd_args)}")
    print()

    # Run conversion script in foreground (blocking)
    result = subprocess.run(cmd_args)
    return result.returncode


def _get_script_path(script_name):
    """
    Get the absolute path to the conversion script.

    Args:
        script_name: Name of the script (e.g., 'convert_llama.py')

    Returns:
        Path object pointing to the script

    Raises:
        FileNotFoundError: If the script cannot be found
    """
    # Try to find the script relative to this file's location
    # This file is at: src/forgather/cli/convert.py
    # Script is at: scripts/convert_llama.py

    # Get the forgather root directory (3 levels up from this file)
    current_file = Path(__file__).resolve()
    forgather_root = current_file.parent.parent.parent.parent

    script_path = forgather_root / "scripts" / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find conversion script at {script_path}. "
            f"Expected to find it relative to forgather installation."
        )

    return script_path
