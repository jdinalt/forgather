"""DiLoCo CLI commands - server, status, and worker."""

import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _load_model_state_dict(model_path: str, from_checkpoint: bool = False):
    """Load a model state_dict from a path."""
    import torch

    if from_checkpoint:
        # Load from Forgather checkpoint format - look for model.safetensors or model state files
        from safetensors.torch import load_file

        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            return load_file(safetensors_path)

        # Try pytorch format
        pt_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location="cpu", weights_only=True)

        raise FileNotFoundError(
            f"No model weights found in checkpoint at {model_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )
    else:
        # Load using AutoModelForCausalLM
        from transformers import AutoModelForCausalLM

        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="cpu"
        )
        state_dict = model.state_dict()
        del model
        return state_dict


def _server_cmd(args):
    """Start DiLoCo parameter server."""
    import torch

    from forgather.ml.diloco.server import DiLoCoServer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DiLoCo Server] %(levelname)s: %(message)s",
    )

    # Load model
    state_dict = _load_model_state_dict(args.model_path, args.from_checkpoint)

    num_params = sum(p.numel() for p in state_dict.values())
    param_bytes = sum(p.numel() * p.element_size() for p in state_dict.values())
    print(f"Model loaded: {num_params:,} parameters ({param_bytes / 1e6:.1f} MB)")

    # Build outer optimizer factory
    nesterov = not args.no_nesterov
    outer_lr = args.outer_lr
    outer_momentum = args.outer_momentum

    def outer_optimizer_factory(params):
        return torch.optim.SGD(
            params, lr=outer_lr, momentum=outer_momentum, nesterov=nesterov
        )

    print(
        f"Outer optimizer: SGD(lr={outer_lr}, momentum={outer_momentum}, nesterov={nesterov})"
    )

    # Async mode settings
    async_mode = getattr(args, "async_mode", False)
    dn_buffer_size = getattr(args, "dn_buffer_size", 0)
    dylu = getattr(args, "dylu", False)
    dylu_base = getattr(args, "dylu_base_sync_every", 500)

    if async_mode:
        mode_str = "async"
        if dn_buffer_size > 0:
            mode_str += f", DN(buffer={dn_buffer_size})"
        if dylu:
            mode_str += f", DyLU(base={dylu_base})"
        print(f"Mode: {mode_str}")
    else:
        print("Mode: sync")

    # Fault tolerance settings
    heartbeat_timeout = getattr(args, "heartbeat_timeout", 120.0)
    min_workers = getattr(args, "min_workers", 1)

    if heartbeat_timeout > 0:
        print(f"Health monitoring: timeout={heartbeat_timeout}s, min_workers={min_workers}")
    else:
        print("Health monitoring: disabled")

    # Dashboard
    dashboard_enabled = not getattr(args, "no_dashboard", False)

    # Create server
    server = DiLoCoServer(
        model_state_dict=state_dict,
        num_workers=args.num_workers,
        port=args.port,
        outer_optimizer_factory=outer_optimizer_factory,
        host=args.host,
        save_dir=args.save_dir,
        save_every_n_rounds=args.save_every,
        async_mode=async_mode,
        dn_buffer_size=dn_buffer_size,
        dylu_enabled=dylu,
        dylu_base_sync_every=dylu_base,
        heartbeat_timeout=heartbeat_timeout,
        min_workers=min_workers,
        dashboard_enabled=dashboard_enabled,
    )

    # Resume from saved state if requested
    if args.resume:
        print(f"Resuming from {args.resume}")
        server.load_state(args.resume)

    print(f"Starting DiLoCo server on {args.host}:{args.port}")
    if dashboard_enabled:
        print(f"Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"Waiting for {args.num_workers} worker(s)...")

    server.run()


def _status_cmd(args):
    """Get DiLoCo server status."""
    from forgather.ml.diloco.client import DiLoCoClient

    client = DiLoCoClient(args.server, timeout=10)

    try:
        status = client.get_status()
    except Exception as e:
        print(f"Error connecting to server at {args.server}: {e}")
        return 1

    print("DiLoCo Server Status")
    print("=" * 50)
    print(f"  Status:        {status.get('status', 'unknown')}")
    print(f"  Mode:          {status.get('mode', 'sync')}")
    print(f"  Sync round:    {status.get('sync_round', 0)}")
    print(f"  Workers:       {status.get('num_registered', 0)}/{status.get('num_workers', '?')}")

    if status.get("uptime_seconds"):
        uptime = status["uptime_seconds"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        print(f"  Uptime:        {hours}h {minutes}m")

    # Async-specific fields
    if status.get("mode") == "async":
        print(f"  Submissions:   {status.get('total_submissions', 0)}")
        dn_buf = status.get("dn_buffer_size", 0)
        if dn_buf > 0:
            print(f"  DN buffer:     {status.get('dn_buffered', 0)}/{dn_buf}")
        if status.get("dylu_enabled"):
            print(f"  DyLU base H:   {status.get('dylu_base_sync_every', '?')}")

    # Fault tolerance
    deaths = status.get("total_worker_deaths", 0)
    if deaths > 0:
        print(f"  Worker deaths: {deaths}")
    hb_timeout = status.get("heartbeat_timeout", 0)
    if hb_timeout > 0:
        print(f"  HB timeout:    {hb_timeout}s")

    pending = status.get("pending_submissions", [])
    if pending:
        print(f"  Pending sync:  {', '.join(pending)}")

    workers = status.get("workers", {})
    if workers:
        print()
        print("Workers:")
        print(f"  {'ID':<30} {'Host':<15} {'Round':<8} {'Steps/s':<10} {'Last HB'}")
        print("  " + "-" * 75)

        import datetime

        for wid, winfo in workers.items():
            last_hb = datetime.datetime.fromtimestamp(
                winfo.get("last_heartbeat", 0)
            ).strftime("%H:%M:%S")
            print(
                f"  {wid:<30} "
                f"{winfo.get('hostname', '?'):<15} "
                f"{winfo.get('sync_round', 0):<8} "
                f"{winfo.get('steps_per_second', 0):<10.2f} "
                f"{last_hb}"
            )

    return 0


def _worker_cmd(args):
    """
    Launch training as a DiLoCo worker.

    This wraps the standard training command, injecting DiLoCo configuration
    via environment variables that the training script picks up.
    """
    # Set DiLoCo environment variables for the training script
    env = os.environ.copy()
    env["DILOCO_SERVER"] = args.server
    env["DILOCO_SYNC_EVERY"] = str(args.sync_every)
    env["DILOCO_BF16_COMM"] = "0" if args.no_bf16 else "1"
    env["DILOCO_DYLU"] = "1" if getattr(args, "dylu", False) else "0"
    env["DILOCO_HEARTBEAT_INTERVAL"] = str(getattr(args, "heartbeat_interval", 30.0))
    env["DILOCO_NUM_FRAGMENTS"] = str(getattr(args, "num_fragments", 1))

    if args.worker_id:
        env["DILOCO_WORKER_ID"] = args.worker_id

    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    # Build the forgather train command from remaining args
    cmd_args = [sys.executable, "-m", "forgather"]

    # Pass through project dir and config template from global args
    if hasattr(args, "project_dir") and args.project_dir != ".":
        cmd_args.extend(["-p", args.project_dir])

    if hasattr(args, "config_template") and args.config_template:
        cmd_args.extend(["-t", args.config_template])

    cmd_args.append("train")

    # Forward remaining arguments
    remainder = args.remainder
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    cmd_args.extend(remainder)

    cmd_str = " ".join(cmd_args)
    diloco_info = (
        f"DiLoCo: server={args.server}, sync_every={args.sync_every}, "
        f"bf16={'yes' if not args.no_bf16 else 'no'}"
    )
    num_frags = getattr(args, "num_fragments", 1)
    if num_frags > 1:
        diloco_info += f", fragments={num_frags}"
    if getattr(args, "dylu", False):
        diloco_info += ", dylu=yes"
    if args.worker_id:
        diloco_info += f", worker_id={args.worker_id}"

    print(diloco_info)
    print(f"Command: {cmd_str}")

    if not args.dry_run:
        subprocess.run(cmd_args, env=env)


def diloco_cmd(args):
    """Handle diloco subcommands."""
    subcmd = getattr(args, "diloco_subcommand", None)

    if subcmd == "server":
        return _server_cmd(args)
    elif subcmd == "status":
        return _status_cmd(args)
    elif subcmd == "worker":
        return _worker_cmd(args)
    else:
        print("Usage: forgather diloco {server|status|worker}")
        print("Run 'forgather diloco --help' for details.")
        return 1
