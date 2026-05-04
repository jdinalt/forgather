"""
Finalize command for Forgather CLI.

Forwards arguments to ``tools/finalize_model/finalize_model.py``.
"""

import os
import subprocess
import sys
from pathlib import Path


def finalize_cmd(args):
    """Launch the finalize_model script.

    Args:
        args: Parsed command-line arguments with ``dummy`` and ``remainder``
            attributes (set in main.py for REMAINDER-style passthrough).
    """
    if args.remainder and any(
        t == "--enqueue" or t.startswith("--enqueue=") for t in args.remainder
    ):
        return _enqueue_finalize(args)

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


def _enqueue_finalize(args):
    import argparse

    p = argparse.ArgumentParser(prog="forgather finalize --enqueue", add_help=True)
    p.add_argument("--enqueue", action="store_true", required=True)
    p.add_argument("--source", "--src", dest="source", required=True)
    p.add_argument("--dest", "--dst", dest="dest", required=True)
    p.add_argument("--skip-default-tokens", action="store_true")
    p.add_argument("--no-auto-stop-tokens", action="store_true")
    p.add_argument("--keep-optimizer", action="store_true")
    p.add_argument("--root-copy", action="store_true")
    p.add_argument("--safetensors", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--add-tokens", default=None)
    p.add_argument("--chat-template-path", default=None)
    p.add_argument("--stop-tokens", default=None)
    p.add_argument("--generation-config", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--server", default=None)
    sub = p.parse_args(args.remainder)

    dest = os.path.abspath(sub.dest)
    job_params = {
        "source": os.path.abspath(sub.source),
        "dest": dest,
        "skip_default_tokens": bool(sub.skip_default_tokens),
        "no_auto_stop_tokens": bool(sub.no_auto_stop_tokens),
        "keep_optimizer": bool(sub.keep_optimizer),
        "root_copy": bool(sub.root_copy),
        "safetensors": bool(sub.safetensors),
        "dry_run": bool(sub.dry_run),
        "log_level": sub.log_level,
        "device": sub.device,
    }
    if sub.checkpoint:
        job_params["checkpoint"] = sub.checkpoint
    if sub.add_tokens:
        job_params["add_tokens"] = sub.add_tokens
    if sub.chat_template_path:
        job_params["chat_template_path"] = sub.chat_template_path
    if sub.stop_tokens:
        job_params["stop_tokens"] = sub.stop_tokens
    if sub.generation_config:
        job_params["generation_config"] = sub.generation_config
    if sub.dtype:
        job_params["dtype"] = sub.dtype

    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient(sub.server)
    try:
        item = client.enqueue_job(
            project_dir=os.path.abspath(args.project_dir),
            config=f"finalize:{os.path.basename(dest)}",
            job_type="finalize",
            job_params=job_params,
            requested_gpus=0,
            priority=sub.priority,
        )
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"queued: {item['queue_id']} (finalize:{os.path.basename(dest)}, priority={item['priority']})"
    )
    return 0
