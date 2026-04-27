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
    if args.remainder and any(
        t == "--enqueue" or t.startswith("--enqueue=") for t in args.remainder
    ):
        return _enqueue_inference(args)

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


def _enqueue_inference(args):
    import argparse

    p = argparse.ArgumentParser(prog="forgather inf server --enqueue", add_help=True)
    p.add_argument("--enqueue", action="store_true", required=True)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-p", "--port", type=int, default=8137)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--from-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument("--disable-kv-cache", action="store_true")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--cache-implementation", default=None)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--chat-template", default=None)
    p.add_argument("--compile-args", default=None)
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--server", default=None)
    sub = p.parse_args(args.remainder)

    job_params = {
        "model_path": os.path.abspath(sub.model),
        "port": sub.port,
        "host": sub.host,
        "dtype": sub.dtype,
        "from_checkpoint": bool(sub.from_checkpoint),
        "compile": bool(sub.compile),
        "disable_kv_cache": bool(sub.disable_kv_cache),
    }
    if sub.attn_implementation:
        job_params["attn_implementation"] = sub.attn_implementation
    if sub.cache_implementation:
        job_params["cache_implementation"] = sub.cache_implementation
    if sub.checkpoint_path:
        job_params["checkpoint_path"] = sub.checkpoint_path
    if sub.chat_template:
        job_params["chat_template"] = sub.chat_template
    if sub.compile_args:
        job_params["compile_args"] = sub.compile_args

    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient(sub.server)
    try:
        item = client.enqueue_job(
            project_dir=os.path.abspath(args.project_dir),
            config=f"inference:{sub.port}",
            job_type="inference",
            job_params=job_params,
            requested_gpus=1,
            priority=sub.priority,
        )
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"queued: {item['queue_id']} (inference:{sub.port}, priority={item['priority']})"
    )
    return 0
