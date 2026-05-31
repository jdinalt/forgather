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
    from forgather.tls.discovery import primary_routable_ip
    from forgather.tls.runtime import is_tls_active, stdlib_ssl_context

    _WILDCARD_HOSTS = ("0.0.0.0", "::", "")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DiLoCo Server] %(levelname)s: %(message)s",
    )

    # Echo the resolved configuration up-front so the TTY log contains
    # exactly what we were asked to do — useful for diagnosing webui /
    # autostart issues where the launching command isn't otherwise
    # visible. argv first (from the caller), then the parsed namespace
    # (post-defaults). A bearer passed inline (``--auth-token <secret>``)
    # is redacted from both so it can't leak into a captured/public TTY;
    # the webui/demo spawn path uses ``--auth-token-file`` (a path, not the
    # secret) so it's unaffected.
    def _redact_argv(argv):
        out, skip = [], False
        for tok in argv:
            if skip:
                out.append("<redacted>")
                skip = False
            elif tok == "--auth-token":
                out.append(tok)
                skip = True
            elif tok.startswith("--auth-token="):
                out.append("--auth-token=<redacted>")
            else:
                out.append(tok)
        return out

    print(f"argv: {_redact_argv(sys.argv)}")
    print("parsed args:")
    for k, v in sorted(vars(args).items()):
        shown = "<redacted>" if (k == "auth_token" and v) else v
        print(f"  {k} = {shown!r}")

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
    # The human-facing auth banner (token value + curl) is printed below,
    # once TLS state and the display host are resolved, so the example URL
    # carries the right scheme and a routable address.

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

    # Cleartext bulk plane (issue #90). A single toggle: the bulk
    # endpoints move to a separate cleartext, unauthenticated listener on
    # a server-picked ephemeral port whose sole purpose is to bypass TLS
    # for throughput on a trusted LAN. No port/TLS/auth knobs — a TLS bulk
    # plane gains nothing over the control port, and a bearer over a
    # sniffable socket is theater. The actual port is logged at bind and
    # delivered to workers over the TLS control plane.
    bulk_cleartext = getattr(args, "bulk_cleartext", False)
    if bulk_cleartext:
        print(
            "Bulk listener: cleartext, no-auth, server-assigned "
            "(ephemeral) port — TLS bypassed for throughput"
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
        sync_every=args.sync_every,
        bf16_comm=args.bf16_comm,
        num_fragments=args.num_fragments,
        heartbeat_timeout=heartbeat_timeout,
        min_workers=min_workers,
        default_work_units=default_work_units,
        auth_token=auth_token,
        ssl_context=ssl_context,
        bulk_cleartext=bulk_cleartext,
    )

    # Resolve the display host + scheme for the startup banner. A wildcard
    # bind (``0.0.0.0``) is not something a worker can dial, so show the
    # primary interface's IP instead — the operator copies this straight
    # onto worker ``--server`` lines. Mirrors the dataset server's banner.
    scheme = "https" if tls_on else "http"
    display_host = args.host
    if args.host in _WILDCARD_HOSTS:
        routable = primary_routable_ip()
        if routable:
            display_host = routable

    # Auth banner — like the dataset server, print the bearer token (and a
    # ready-to-run curl) so setting up workers from the CLI is copy-paste.
    # ``--quiet-tokens`` (set by the webui in --demo mode) suppresses the
    # value; the per-port token file still carries it for legitimate peers.
    quiet_tokens = bool(getattr(args, "quiet_tokens", False))
    if auth_token is None:
        print(
            "!! DiLoCo server is running with --no-auth — any host that can "
            "reach the bind port has full control (shutdown, optimizer, sync)"
        )
    else:
        if token_source == "regenerated":
            print(
                "!! --regen-token: replacing the persisted per-port token. "
                "Existing workers will need to re-pull."
            )
        print(f"Auth: {format_auth_mode(args, token_source)}")
        if quiet_tokens:
            # Demo/public-TTY mode: reveal nothing sensitive — not the
            # token value and not the on-disk path. Legitimate peers
            # still resolve the token via the per-port file.
            print("  bearer-token enabled (value suppressed by --quiet-tokens)")
        else:
            print(f"  auth token: {auth_token}")
            print("  workers must send 'Authorization: Bearer <token>'")
            print(
                f'  curl -H "Authorization: Bearer {auth_token}" '
                f"{scheme}://{display_host}:{args.port}/status"
            )
            if token_source in ("generated", "regenerated", "persisted"):
                print(f"  token file: {standalone_token_file(args.port)}")

    print()
    print(f"Starting DiLoCo server on {scheme}://{display_host}:{args.port}")
    if display_host != args.host:
        print(
            f"  (bound to {args.host}; showing primary interface "
            f"{display_host} so workers can reach it)"
        )
    print(f"Waiting for {args.num_workers} worker(s)...")
    print()
    print("To stop the server:")
    print(
        "  Ctrl-C              Stop server"
        + (" (saves state automatically)" if args.output_dir else "")
    )
    print(
        f"  curl -X POST        {scheme}://{display_host}:{args.port}/control/shutdown"
        + (' -H "Authorization: Bearer <token>"' if auth_token else "")
    )
    print(f"  forgather webui     DiLoCo view → Control card → Shutdown server")
    print()

    # Flush stdout before the blocking serve loop. When the TTY is a pipe
    # (the webui scheduler captures it), stdout is block-buffered, so the
    # whole banner above — and the argv/parsed-args diagnostic printed at
    # entry — would otherwise sit in the buffer until the process exits,
    # making the bearer token appear only *after* the server is stopped.
    # The server's own logging goes to stderr (flushed per line), so
    # without this the two streams land badly out of order.
    sys.stdout.flush()
    server.run()


def _status_cmd(args):
    """Get DiLoCo server status — rich, orchestrator-first with direct fallback.

    The status pulls the live ``/status`` snapshot plus the known-worker
    roster and (with ``--queues``) the work-unit queues. When the forgather
    server is reachable AND knows this target, the read goes through its
    proxy (which resolves the upstream token + TLS for us); otherwise we
    talk to the parameter server directly. ``--direct`` forces the latter.
    """
    from . import diloco_orch as orch

    want_queues = getattr(args, "queues", False)
    as_json = getattr(args, "json", False)

    client, base = orch.resolve_orchestrator_base(args)
    if base is not None:
        merged = orch.assemble_status(
            get_status=lambda: client.diloco_server_status(base),
            get_info=lambda: client.diloco_server_info(base),
            get_known_workers=lambda: client.diloco_known_workers(base),
            get_work_queues=(
                (lambda: client.diloco_work_queues(base)) if want_queues else None
            ),
        )
        source = {"via": "orchestrator", "base": base}
    else:
        from forgather.ml.diloco.client import DiLoCoClient

        # Token + verify_tls are picked up from explicit args / env /
        # loopback per-port file by DiLoCoClient automatically.
        c = DiLoCoClient(
            args.server,
            timeout=10,
            max_retries=0,  # status is a probe — fail fast, no backoff storm
            token=getattr(args, "auth_token", None),
            verify_tls=not getattr(args, "no_verify_tls", False),
        )
        try:
            first = c.get_status()
        except Exception as e:
            if as_json:
                print(json.dumps({"error": str(e), "server": args.server}))
            else:
                print(f"Error connecting to server at {args.server}: {e}")
            return 1
        merged = orch.assemble_status(
            get_status=lambda: first,
            get_info=c.get_info,
            get_known_workers=c.get_known_workers,
            get_work_queues=c.get_work_queues if want_queues else None,
        )
        source = {"via": "direct", "server": args.server}

    if as_json:
        print(json.dumps({"source": source, **merged}, default=str, indent=2))
        return 0
    return orch.render_status(merged, want_queues=want_queues)


def _make_client(args, timeout=10):
    """Build a DiLoCoClient from the shared control-plane connection args."""
    from forgather.ml.diloco.client import DiLoCoClient

    return DiLoCoClient(
        args.server,
        timeout=timeout,
        token=getattr(args, "auth_token", None),
        verify_tls=not getattr(args, "no_verify_tls", False),
    )


# CLI action name -> trainer-control command the server relays.
_CONTROL_ACTION_MAP = {
    "save": "save_checkpoint",
    "save-stop": "save_and_stop",
    "abort": "abort",
}


def _control_cmd(args):
    """Relay a trainer-control command to one or all workers."""
    command = _CONTROL_ACTION_MAP[args.action]
    client = _make_client(args)
    try:
        resp = client.relay_command(command, worker_id=args.worker_id)
    except Exception as e:
        print(f"Error contacting server at {args.server}: {e}")
        return 1
    workers = resp.get("workers", [])
    if not workers:
        print("No workers registered — nothing to do.")
        return 0
    print(f"Queued '{command}' for {len(workers)} worker(s): {', '.join(workers)}")
    print("Each worker applies it on its next heartbeat.")
    return 0


def _shutdown_cmd(args):
    """Stop the DiLoCo server — cleanly (default) or immediately (--force)."""
    import time

    client = _make_client(args)

    if args.force:
        print("Force shutdown: stopping server now (workers will lose sync).")
        try:
            client.shutdown()
        except Exception as e:
            # The server closes the socket as it exits; tolerate that.
            print(f"  (server stop signalled; {type(e).__name__})")
        print("Server stop signalled.")
        return 0

    # Clean shutdown: save-stop all workers, wait for them to exit, then
    # checkpoint + stop the server.
    try:
        resp = client.relay_command("save_and_stop")
    except Exception as e:
        print(f"Error contacting server at {args.server}: {e}")
        return 1
    targets = list(resp.get("workers", []))
    if targets:
        print(f"Save & stop queued for {len(targets)} worker(s): {', '.join(targets)}")
        print(f"Waiting up to {int(args.timeout)}s for workers to stop…")
        deadline = time.time() + args.timeout
        remaining = set(targets)
        while remaining and time.time() < deadline:
            time.sleep(2.0)
            try:
                status = client.get_status()
            except Exception as e:
                print(f"  (status poll failed: {e}; retrying)")
                continue
            live = set(status.get("workers", {}).keys())
            stopped = remaining - live
            for wid in stopped:
                print(f"  stopped: {wid}")
            remaining = remaining & live
            print(f"  {len(targets) - len(remaining)}/{len(targets)} stopped")
        if remaining:
            print(
                f"Timed out: {len(remaining)} worker(s) still running "
                f"({', '.join(sorted(remaining))}). Server left running so you "
                f"can troubleshoot; re-run, or use --force."
            )
            return 1
        print("All workers stopped.")
    else:
        print("No workers registered — skipping worker stop.")

    print("Saving server checkpoint…")
    try:
        client.save_state()
        print("  server checkpoint saved.")
    except Exception as e:
        print(f"  server checkpoint failed: {e} (continuing to stop)")

    print("Stopping server…")
    try:
        client.shutdown()
    except Exception as e:
        print(f"  (server stop signalled; {type(e).__name__})")
    print("Done.")
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
    elif subcmd == "control":
        return _control_cmd(args)
    elif subcmd == "shutdown":
        return _shutdown_cmd(args)
    elif subcmd == "worker":
        return _worker_cmd(args)
    elif subcmd == "servers":
        from .diloco_orch import servers_cmd

        return servers_cmd(args)
    elif subcmd == "logs":
        from .diloco_orch import logs_cmd

        return logs_cmd(args)
    else:
        print(
            "Usage: forgather diloco "
            "{server|worker|status|servers|logs|control|shutdown}"
        )
        print("Run 'forgather diloco --help' for details.")
        return 1
