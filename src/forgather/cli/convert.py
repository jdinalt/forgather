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
    if args.remainder and any(
        t == "--enqueue" or t.startswith("--enqueue=") for t in args.remainder
    ):
        return _enqueue_convert(args)

    # Get path to convert_llama.py script
    script_path = _get_script_path()

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


def _get_script_path():
    """
    Get the absolute path to the conversion script.

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

    script_path = forgather_root / "tools" / "convert_model" / "convert.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find conversion script at {script_path}. "
            f"Expected to find it relative to forgather installation."
        )

    return script_path


def _enqueue_convert(args):
    import argparse

    p = argparse.ArgumentParser(prog="forgather convert --enqueue", add_help=True)
    p.add_argument("--enqueue", action="store_true", required=True)
    p.add_argument("--src", "--src-model-path", dest="src", required=True)
    p.add_argument("--dst", "--dst-model-path", dest="dst", required=True)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--generation-test", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-default-tokens", action="store_true")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--dtype", default=None)
    p.add_argument("--model-type", default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--chat-template-path", default=None)
    p.add_argument("--add-tokens", default=None)
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--server", default=None)
    sub = p.parse_args(args.remainder)

    dst = os.path.abspath(sub.dst)
    job_params = {
        "src_model_path": os.path.abspath(sub.src),
        "dst_model_path": dst,
        "reverse": bool(sub.reverse),
        "generation_test": bool(sub.generation_test),
        "dry_run": bool(sub.dry_run),
        "skip_default_tokens": bool(sub.skip_default_tokens),
        "log_level": sub.log_level,
    }
    if sub.dtype:
        job_params["dtype"] = sub.dtype
    if sub.model_type:
        job_params["model_type"] = sub.model_type
    if sub.max_length is not None:
        job_params["max_length"] = sub.max_length
    if sub.checkpoint_path:
        job_params["checkpoint_path"] = sub.checkpoint_path
    if sub.device:
        job_params["device"] = sub.device
    if sub.prompt:
        job_params["prompt"] = sub.prompt
    if sub.chat_template_path:
        job_params["chat_template_path"] = sub.chat_template_path
    if sub.add_tokens:
        job_params["add_tokens"] = sub.add_tokens

    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient(sub.server)
    try:
        item = client.enqueue_job(
            project_dir=os.path.abspath(args.project_dir),
            config=f"convert:{os.path.basename(dst)}",
            job_type="convert",
            job_params=job_params,
            requested_gpus=0,
            priority=sub.priority,
        )
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"queued: {item['queue_id']} (convert:{os.path.basename(dst)}, priority={item['priority']})"
    )
    return 0
