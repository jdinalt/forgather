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


def _pop_flag(tokens, name):
    """Remove every occurrence of a value-less flag (``--x`` or ``--x=...``)
    from *tokens* in place; return True if any were present."""
    found = False
    out = []
    for t in tokens:
        if t == name or t.startswith(name + "="):
            found = True
            continue
        out.append(t)
    tokens[:] = out
    return found


def _strip_value_flags(tokens, names):
    """Return a copy of *tokens* with each flag in *names* and its value
    removed (handles ``--flag value`` and ``--flag=value``)."""
    out = []
    skip = False
    for t in tokens:
        if skip:
            skip = False
            continue
        if t in names:
            skip = True  # also drop the following value
            continue
        if any(t.startswith(n + "=") for n in names):
            continue
        out.append(t)
    return out


def _run_server_foreground(server_args):
    """Run the inference server script in the foreground (blocking)."""
    script_path = _get_script_path("server.py")
    cmd_args = [sys.executable, str(script_path), *server_args]
    print(f"Running: {' '.join(cmd_args)}")
    print()
    return subprocess.run(cmd_args).returncode


def server_cmd(args):
    """
    Launch the inference server.

    The inference server is a long-running service, so it is submitted to the
    forgather-server scheduler (background) by default — like dataset-server and
    the DiLoCo server. ``--local-only`` runs it in the foreground instead;
    ``--local-fallback`` foregrounds only when the server is unreachable. The
    old ``--enqueue`` flag is now the default and is accepted as a deprecated
    no-op.

    Args:
        args: Parsed arguments with remainder containing forwarded args
    """
    remainder = list(getattr(args, "remainder", None) or [])
    local_only = _pop_flag(remainder, "--local-only")
    local_fallback = _pop_flag(remainder, "--local-fallback")
    if _pop_flag(remainder, "--enqueue"):
        print(
            "note: --enqueue is deprecated; `inf server` now submits to the "
            "scheduler by default (use --local-only to run in the foreground).",
            file=sys.stderr,
        )

    if local_only:
        # Strip the scheduler-only flags so the foreground server script
        # (which doesn't know them) doesn't choke — same as the
        # --local-fallback foreground path.
        server_args = _strip_value_flags(remainder, {"--priority", "--server"})
        return _run_server_foreground(server_args)
    return _enqueue_inference(args, remainder, local_fallback=local_fallback)


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


def _enqueue_inference(args, remainder, local_fallback=False):
    import argparse

    p = argparse.ArgumentParser(
        prog="forgather inf server",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run an OpenAI-compatible inference server for one or more models.\n"
            "\n"
            "Disposition: the inference server is a long-running service, so by\n"
            "default it is SUBMITTED TO THE SCHEDULER and runs in the background\n"
            "(like dataset-server / the DiLoCo server) — manage it with\n"
            "`forgather job` (list/logs/stop). Control where it runs with the\n"
            "locality flags below."
        ),
        epilog=(
            "examples:\n"
            "  forgather inf server -m ./out/model            # background (scheduler)\n"
            "  forgather inf server -m ./out/model --local-only   # foreground here\n"
            "  forgather inf server -m a=./m1 -m b=./m2        # multi-model\n"
        ),
    )

    loc = p.add_argument_group("disposition (where it runs)")
    loc.add_argument(
        "--local-only",
        action="store_true",
        help="Run the server in the foreground here instead of scheduling it.",
    )
    loc.add_argument(
        "--local-fallback",
        action="store_true",
        help="Foreground only if the forgather server is unreachable.",
    )
    loc.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Scheduler priority when submitted as a job (default: 0).",
    )
    loc.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help="forgather-server URL to submit to (default: env / 127.0.0.1:8765).",
    )

    model = p.add_argument_group("model + serving")
    model.add_argument(
        "-m",
        "--model",
        action="append",
        required=True,
        metavar="PATH | NAME=PATH",
        help=(
            "Model PATH or NAME=PATH; pass multiple times for multi-model "
            "inference. Requests dispatch by OpenAI 'model' field."
        ),
    )
    model.add_argument(
        "-p", "--port", type=int, default=8137, help="Bind port (default: 8137)."
    )
    model.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)."
    )
    model.add_argument(
        "--dtype", default="bfloat16", help="Model dtype (default: bfloat16)."
    )
    model.add_argument(
        "--from-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load from a training checkpoint dir rather than a finalized model.",
    )
    model.add_argument(
        "--checkpoint-path",
        default=None,
        metavar="PATH",
        help="Explicit checkpoint path (single-model; implies --from-checkpoint).",
    )
    model.add_argument(
        "--compile", action="store_true", help="torch.compile the model(s)."
    )
    model.add_argument(
        "--compile-args",
        default=None,
        metavar="JSON",
        help="JSON of kwargs passed to torch.compile.",
    )
    model.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="Disable the KV cache (debugging / memory-constrained).",
    )
    model.add_argument(
        "--keep-on-gpu",
        action="store_true",
        help="Multi-model: keep all models GPU-resident (no CPU swap).",
    )
    model.add_argument(
        "--attn-implementation",
        default=None,
        metavar="IMPL",
        help="Attention impl (e.g. flash_attention_2, sdpa, eager).",
    )
    model.add_argument(
        "--cache-implementation",
        default=None,
        metavar="IMPL",
        help="KV-cache implementation (e.g. static, dynamic).",
    )
    model.add_argument(
        "--chat-template",
        default=None,
        metavar="PATH|NAME",
        help="Override the chat template used to format requests.",
    )
    sub = p.parse_args(remainder)

    # Parse -m args into name/path specs. ``NAME=PATH`` or bare ``PATH``.
    models_list = []
    for raw in sub.model:
        if "=" in raw:
            name, _, path = raw.partition("=")
            name = name.strip()
            path = path.strip()
            if not name or not path:
                print(f"error: --model {raw!r}: empty name or path", file=sys.stderr)
                raise SystemExit(2)
        else:
            path = raw.strip()
            name = os.path.basename(os.path.normpath(path))
        models_list.append({"name": name, "path": os.path.abspath(path)})

    if len(models_list) > 1 and sub.checkpoint_path:
        print(
            "error: --checkpoint-path is not supported with multiple --model "
            "(use --from-checkpoint without an explicit path)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    job_params = {
        "port": sub.port,
        "host": sub.host,
        "dtype": sub.dtype,
        "from_checkpoint": bool(sub.from_checkpoint),
        "compile": bool(sub.compile),
        "disable_kv_cache": bool(sub.disable_kv_cache),
        "keep_on_gpu": bool(sub.keep_on_gpu),
    }
    if len(models_list) == 1:
        # Single-model: keep the existing job_params shape so older
        # scheduler/inference_ops code paths continue to work unchanged.
        job_params["model_path"] = models_list[0]["path"]
    else:
        job_params["models"] = models_list
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

    from . import submit_orch
    from .server_client import ServerUnreachable

    # Reuse the shared locality decision: server-required by default,
    # --local-fallback drops to a foreground server when it's down. (--local-only
    # was handled in server_cmd before we got here.)
    locality = argparse.Namespace(
        via_server=sub.server, local_only=False, local_fallback=local_fallback
    )
    try:
        client = submit_orch.use_orchestrator(locality)
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    if client is None:
        # --local-fallback and the server is down → foreground server.
        server_args = _strip_value_flags(remainder, {"--priority", "--server"})
        return _run_server_foreground(server_args)
    try:
        item = submit_orch.submit_single(
            client,
            project_dir=os.path.abspath(args.project_dir),
            config=f"inference:{sub.port}",
            job_type="inference",
            job_params=job_params,
            requested_gpus=1,
            priority=sub.priority,
            dynamic_args=None,
        )
    except (ServerUnreachable, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"queued: {item['queue_id']} (inference:{sub.port}, priority={item['priority']})"
    )
    return 0
