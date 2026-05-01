"""
Update command for Forgather CLI.

Forwards arguments to ``tools/update_model/update.py``.
"""

import os
import subprocess
import sys
from pathlib import Path


def update_cmd(args):
    """Launch the update_model script.

    Args:
        args: Parsed command-line arguments with ``dummy`` and ``remainder``
            attributes (set in main.py for REMAINDER-style passthrough).
    """
    if args.remainder and any(
        t == "--enqueue" or t.startswith("--enqueue=") for t in args.remainder
    ):
        return _enqueue_update(args)

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
    """Resolve the absolute path of the update_model script."""
    current_file = Path(__file__).resolve()
    forgather_root = current_file.parent.parent.parent.parent
    script_path = forgather_root / "tools" / "update_model" / "update.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find update script at {script_path}. "
            f"Expected it relative to the forgather installation."
        )

    return script_path


def _enqueue_update(args):
    import argparse

    p = argparse.ArgumentParser(prog="forgather update --enqueue", add_help=True)
    p.add_argument("--enqueue", action="store_true", required=True)
    p.add_argument("--src", "--src-model-path", dest="src", required=True)
    p.add_argument("--dst", "--dst-model-path", dest="dst", required=True)
    p.add_argument("--arch", default=None)
    p.add_argument("--from-version", type=int, default=None)
    p.add_argument("--to-version", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default=None)
    p.add_argument("--no-strict", action="store_true")
    p.add_argument("--safetensors", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--server", default=None)
    sub = p.parse_args(args.remainder)

    dst = os.path.abspath(sub.dst)
    job_params = {
        "src_model_path": os.path.abspath(sub.src),
        "dst_model_path": dst,
        "device": sub.device,
        "no_strict": bool(sub.no_strict),
        "safetensors": bool(sub.safetensors),
        "dry_run": bool(sub.dry_run),
        "log_level": sub.log_level,
    }
    if sub.arch:
        job_params["arch"] = sub.arch
    if sub.from_version is not None:
        job_params["from_version"] = sub.from_version
    if sub.to_version is not None:
        job_params["to_version"] = sub.to_version
    if sub.checkpoint:
        job_params["checkpoint"] = sub.checkpoint
    if sub.dtype:
        job_params["dtype"] = sub.dtype

    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient(sub.server)
    try:
        item = client.enqueue_job(
            project_dir=os.path.abspath(args.project_dir),
            config=f"update:{os.path.basename(dst)}",
            job_type="update",
            job_params=job_params,
            requested_gpus=0,
            priority=sub.priority,
        )
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"queued: {item['queue_id']} (update:{os.path.basename(dst)}, priority={item['priority']})"
    )
    return 0
