"""DiLoCo CLI commands - server, status, and worker."""

import argparse
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _server_cmd(args):
    """Start DiLoCo parameter server."""
    import torch

    from forgather.ml.diloco.auth import (
        format_auth_mode,
        resolve_auth_token,
        standalone_token_file,
        write_standalone_token,
    )
    from forgather.ml.diloco.server import DiLoCoServer
    from forgather.tls import enforce_non_loopback_policy
    from forgather.tls.runtime import is_tls_active, stdlib_ssl_context

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DiLoCo Server] %(levelname)s: %(message)s",
    )

    # Echo the resolved configuration up-front so the TTY log contains
    # exactly what we were asked to do — useful for diagnosing webui /
    # autostart issues where the launching command isn't otherwise
    # visible. argv first (verbatim from the caller), then the parsed
    # namespace (post-defaults).
    print(f"argv: {sys.argv}")
    print("parsed args:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v!r}")

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
        print(
            f"Health monitoring: timeout={heartbeat_timeout}s, min_workers={min_workers}"
        )
    else:
        print("Health monitoring: disabled")

    save_total_limit = getattr(args, "save_total_limit", 3)
    default_work_units = getattr(args, "default_work_units", 1024)

    # Resolve an ephemeral ``--port 0`` to the concrete port the server
    # will actually bind *now*, before anything keys off the port. The
    # per-port token file, the startup banner, and the DiLoCoServer
    # constructor all read ``args.port`` downstream; if we left it at 0
    # the token would be written to ``0.token`` while the listener bound
    # a different port, and loopback token auto-discovery would 401.
    # DiLoCoServer applies the same ``port or _find_available_port()``
    # rule, so passing the resolved port keeps both ends in sync.
    if not args.port:
        args.port = DiLoCoServer._find_available_port()
        print(f"Resolved ephemeral --port 0 to {args.port}")

    # ------------------------------------------------------------------
    # Security (issue #90): resolve bearer token + TLS context. Both
    # default to "on" via the persisted per-port file / shared TLS
    # provisioning, matching dataset_server. ``--no-auth`` / ``--no-tls``
    # opt out for the trusted-LAN case.
    # ------------------------------------------------------------------
    # Need an ArgumentParser handle for resolve_auth_token's parser.error.
    # We don't have one in scope here; build a thin wrapper.
    _auth_parser = argparse.ArgumentParser(prog="forgather diloco server")
    auth_token, token_source = resolve_auth_token(_auth_parser, args)
    if token_source in ("generated", "regenerated") and auth_token is not None:
        write_standalone_token(args.port, auth_token)
    elif token_source == "persisted":
        # Persisted file already exists — no write needed.
        pass

    if not getattr(args, "quiet_tokens", False):
        print(f"Auth: {format_auth_mode(args, token_source)}")
        if auth_token is not None and token_source in (
            "generated",
            "regenerated",
            "persisted",
        ):
            print(f"  token file: {standalone_token_file(args.port)}")

    # TLS: build a stdlib SSLContext (or None for cleartext). Refuse
    # non-loopback binds without TLS unless --insecure was passed.
    ssl_context = stdlib_ssl_context(args)
    tls_on = ssl_context is not None
    enforce_non_loopback_policy(
        args.host,
        tls_enabled=tls_on,
        insecure=getattr(args, "insecure", False),
        service="diloco-server",
    )
    if tls_on:
        print(f"TLS: enabled ({'mTLS+bearer' if auth_token else 'mTLS-or-cleartext'})")
    else:
        print("TLS: disabled (cleartext)")

    # Two-port bulk plane (issue #90). Defaults when --bulk-port is set:
    # cleartext, no auth — matching torch.distributed's posture on a
    # trusted LAN. Explicit --bulk-tls / --bulk-auth flip those bits.
    bulk_port = getattr(args, "bulk_port", None)
    bulk_ssl_context = None
    bulk_auth_enabled = True  # ignored when bulk_port is None
    if bulk_port is not None:
        bulk_tls = getattr(args, "bulk_tls", None)
        if bulk_tls is True:
            # Use the same SSL context for the bulk listener — same
            # cluster identity, same CA bundle. Distinct contexts add
            # no security here.
            bulk_ssl_context = ssl_context
            if bulk_ssl_context is None:
                _auth_parser.error(
                    "--bulk-tls requires the control plane to also be on "
                    "TLS (pass --tls or provision the cluster)."
                )
        # else: bulk_tls is False (explicit --no-bulk-tls) or None
        # (default → cleartext); both leave bulk_ssl_context=None.

        bulk_auth = getattr(args, "bulk_auth", None)
        # Default when --bulk-port is set: bulk auth OFF (opt-out for
        # throughput). Explicit --bulk-auth turns it on.
        bulk_auth_enabled = bool(bulk_auth) if bulk_auth is not None else False
        # Requiring the bearer on a *cleartext* bulk listener would make
        # every worker POST the control-plane token in plaintext (the
        # bulk and control listeners share one secret). A LAN sniffer
        # would then capture full control-plane authority — exactly the
        # "host takeover" boundary the two-port split is meant to hold.
        # Refuse the combination: either secure the bulk port with
        # --bulk-tls, or run it --no-bulk-auth.
        if bulk_auth_enabled and auth_token and bulk_ssl_context is None:
            _auth_parser.error(
                "--bulk-auth requires the bulk listener to be on TLS "
                "(pass --bulk-tls). Sending the bearer token over a "
                "cleartext bulk port would leak the control-plane "
                "credential to anyone on the network. Use --no-bulk-auth "
                "for an unauthenticated cleartext bulk plane."
            )
        print(
            f"Bulk listener: port={bulk_port} "
            f"({'TLS' if bulk_ssl_context else 'cleartext'}, "
            f"{'auth' if bulk_auth_enabled and auth_token else 'no-auth'})"
        )

    # Create server
    server = DiLoCoServer(
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        from_checkpoint=args.from_checkpoint,
        port=args.port,
        outer_optimizer_factory=outer_optimizer_factory,
        host=args.host,
        save_every_n_rounds=args.save_every,
        save_total_limit=save_total_limit,
        async_mode=async_mode,
        dn_buffer_size=dn_buffer_size,
        dylu_enabled=dylu,
        dylu_base_sync_every=dylu_base,
        heartbeat_timeout=heartbeat_timeout,
        min_workers=min_workers,
        default_work_units=default_work_units,
        auth_token=auth_token,
        ssl_context=ssl_context,
        bulk_port=bulk_port,
        bulk_ssl_context=bulk_ssl_context,
        bulk_auth_enabled=bulk_auth_enabled,
    )

    print(f"Starting DiLoCo server on {args.host}:{args.port}")
    print(f"Waiting for {args.num_workers} worker(s)...")
    print()
    print("To stop the server:")
    print(
        "  Ctrl-C              Stop server"
        + (" (saves state automatically)" if args.output_dir else "")
    )
    print(f"  curl -X POST        http://localhost:{args.port}/control/shutdown")
    print(f"  forgather webui     DiLoCo view → Control card → Shutdown server")
    print()

    server.run()


def _status_cmd(args):
    """Get DiLoCo server status."""
    from forgather.ml.diloco.client import DiLoCoClient

    # Token + verify_tls are picked up from explicit args / env /
    # loopback per-port file by DiLoCoClient automatically.
    client = DiLoCoClient(
        args.server,
        timeout=10,
        token=getattr(args, "auth_token", None),
        verify_tls=not getattr(args, "no_verify_tls", False),
    )

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
    print(
        f"  Workers:       {status.get('num_registered', 0)}/{status.get('num_workers', '?')}"
    )

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
    # Set DiLoCo environment variables for the training script. Only
    # client-local knobs are forwarded; sync_every / bf16_comm / dylu /
    # num_fragments are server-authoritative and resolved from /info by
    # the worker at startup (no client override).
    env = os.environ.copy()
    env["DILOCO_SERVER"] = args.server
    env["DILOCO_HEARTBEAT_INTERVAL"] = str(getattr(args, "heartbeat_interval", 30.0))

    if args.worker_id:
        env["DILOCO_WORKER_ID"] = args.worker_id

    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    # Build the forgather train command from remaining args
    import shutil

    forgather_bin = shutil.which("forgather")
    if forgather_bin is None:
        # Fallback: use the entry point module directly
        forgather_bin = [sys.executable, "-c", "from forgather.cli import main; main()"]
    else:
        forgather_bin = [forgather_bin]
    cmd_args = list(forgather_bin)

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
    # Strip leading "train" if user passed it (we already add it above)
    if remainder and remainder[0] == "train":
        remainder = remainder[1:]
    cmd_args.extend(remainder)

    cmd_str = " ".join(cmd_args)
    # sync_every / bf16 / dylu / num_fragments come from the server's /info
    # at startup, so they aren't known here — the worker logs them once it
    # negotiates with the server.
    diloco_info = f"DiLoCo: server={args.server}"
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
