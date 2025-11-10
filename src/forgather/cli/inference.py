"""
Inference server and client commands for Forgather CLI.
"""

import os
import subprocess
import sys
from pathlib import Path


def inf_cmd(args):
    """
    Dispatch to inference server or client command.

    Args:
        args: Parsed command-line arguments with subcommand and remainder attributes
    """
    if args.subcommand == "server":
        return server_cmd(args)
    elif args.subcommand == "client":
        return client_cmd(args)
    else:
        # This should never happen due to argparse choices validation
        print(f"Error: Unknown subcommand '{args.subcommand}'", file=sys.stderr)
        return 1


def server_cmd(args):
    """
    Launch the inference server script.

    Args:
        args: Parsed arguments with remainder containing forwarded args
    """
    # Get path to server.py script
    script_path = _get_script_path("server.py")

    # Build command: python server.py <forwarded_args>
    cmd_args = [sys.executable, str(script_path)]

    # Forward all remaining arguments
    if hasattr(args, "remainder") and args.remainder:
        cmd_args.extend(args.remainder)

    # Print command for transparency
    print(f"Running: {' '.join(cmd_args)}")
    print()

    # Run server in foreground (blocking)
    result = subprocess.run(cmd_args)
    return result.returncode


def client_cmd(args):
    """
    Launch the inference client script.

    Args:
        args: Parsed arguments with remainder containing forwarded args
    """
    # Get path to client.py script
    script_path = _get_script_path("client.py")

    # Build command: python client.py <forwarded_args>
    cmd_args = [sys.executable, str(script_path)]

    # Forward all remaining arguments
    if hasattr(args, "remainder") and args.remainder:
        cmd_args.extend(args.remainder)

    # Print command for transparency
    print(f"Running: {' '.join(cmd_args)}")
    print()

    # Run client
    result = subprocess.run(cmd_args)
    return result.returncode


def _get_script_path(script_name):
    """
    Get the absolute path to an inference server script.

    Args:
        script_name: Name of the script (e.g., 'server.py' or 'client.py')

    Returns:
        Path object pointing to the script

    Raises:
        FileNotFoundError: If the script cannot be found
    """
    # Try to find the script relative to this file's location
    # This file is at: src/forgather/cli/inference.py
    # Scripts are at: tools/inference_server/<script_name>

    # Get the forgather root directory (3 levels up from this file)
    current_file = Path(__file__).resolve()
    forgather_root = current_file.parent.parent.parent.parent

    script_path = forgather_root / "tools" / "inference_server" / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find inference script at {script_path}. "
            f"Expected to find it relative to forgather installation."
        )

    return script_path
