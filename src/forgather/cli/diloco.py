"""DiLoCo CLI commands - server, status, and worker."""

import argparse
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _server_cmd(args):
    """Start DiLoCo parameter server.

    By default this enqueues a scheduled ``diloco_server`` job through the
    forgather server (background, GPU-scheduled, TTY-captured), and errors
    if the server isn't reachable. ``--local-fallback`` runs it in the
    foreground when the server is down; ``--local-only`` always runs it in
    the foreground (this is also how the scheduler spawns the actual
    server, so it never re-enqueues itself).
    """
    from . import diloco_orch as orch
    from .server_client import ServerUnreachable

    try:
        client = orch.use_orchestrator(args)
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1
    if client is not None:
        return orch.launch_server(args)

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
    from forgather.tls.runtime import (
        is_tls_active,
        server_tls_files,
        stdlib_ssl_context,
    )

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
    # The same cert/key/CA the control-plane SSLContext uses, as file paths, so
    # the gRPC bulk listener (issue #154) can build matching TLS credentials
    # (gRPC needs PEM material, not a Python SSLContext). (None,None,None) off.
    grpc_cert, grpc_key, grpc_ca = server_tls_files(args)
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
        # Wire precision (issue #130). The argparse defaults are
        # ``upload_dtype=None`` / ``bf16_comm=None``, which the
        # DiLoCoServer constructor reconciles into "neither passed →
        # use the legacy bf16 default". Passing both raises ValueError
        # — desirable to surface operator scripts that mix old and
        # new flags.
        upload_dtype=args.upload_dtype,
        upload_sr=args.upload_sr,
        download_dtype=args.download_dtype,
        download_sr=args.download_sr,
        wire_format=args.wire_format,
        backend=getattr(args, "backend", "http"),
        grpc_enabled=getattr(args, "grpc_enabled", False),
        bf16_comm=args.bf16_comm,
        num_fragments=args.num_fragments,
        heartbeat_timeout=heartbeat_timeout,
        min_workers=min_workers,
        default_work_units=default_work_units,
        auth_token=auth_token,
        ssl_context=ssl_context,
        tls_cert_file=grpc_cert,
        tls_key_file=grpc_key,
        tls_ca_file=grpc_ca,
        bulk_cleartext=bulk_cleartext,
        run_name=getattr(args, "run_name", None),
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
    talk to the parameter server directly. ``--local-only`` forces the
    direct path; ``--local-fallback`` uses it only when the server is down;
    the default errors if the server is unreachable.
    """
    from . import diloco_orch as orch
    from .server_client import ServerUnreachable

    want_queues = getattr(args, "queues", False)
    as_json = getattr(args, "json", False)

    if as_json and getattr(args, "watch", False):
        print(
            "error: --watch and --json are mutually exclusive "
            "(watch is an interactive refreshing view).",
            file=sys.stderr,
        )
        return 1

    try:
        client, base = orch.resolve_orchestrator_base(args)
    except ServerUnreachable as e:
        if as_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(str(e), file=sys.stderr)
        return 1
    if base is not None:
        get_status = lambda: client.diloco_server_status(base)  # noqa: E731
        get_info = lambda: client.diloco_server_info(base)  # noqa: E731
        get_known = lambda: client.diloco_known_workers(base)  # noqa: E731
        get_queues = (lambda: client.diloco_work_queues(base)) if want_queues else None
        get_queue_detail = (
            (lambda ds, seed: client.diloco_work_queue(base, ds, seed))  # noqa: E731
            if want_queues
            else None
        )
        source = {"via": "orchestrator", "base": base}
        target = base
    else:
        from forgather.ml.diloco.client import DiLoCoClient

        # Direct path: no discovery possible, so fall back to the loopback
        # default when --server is omitted. Token + verify_tls are picked up
        # from explicit args / env / loopback per-port file automatically.
        direct_server = (
            getattr(args, "diloco_server", None) or orch.DEFAULT_DIRECT_SERVER
        )
        c = DiLoCoClient(
            direct_server,
            timeout=10,
            max_retries=0,  # status is a probe — fail fast, no backoff storm
            token=getattr(args, "auth_token", None),
            verify_tls=not getattr(args, "no_verify_tls", False),
        )
        get_status = c.get_status
        get_info = c.get_info
        get_known = c.get_known_workers
        get_queues = c.get_work_queues if want_queues else None
        get_queue_detail = c.get_work_queue if want_queues else None
        source = {"via": "direct", "server": direct_server}
        target = direct_server

    def _render_once():
        # The core /status read is REQUIRED — a failure here means the
        # server is down / unreachable (directly or through the proxy), so
        # we report it and return non-zero rather than printing a
        # healthy-looking "unknown" snapshot. The remaining reads (info /
        # workers / queues) stay best-effort via assemble_status so a
        # partial server still renders.
        try:
            status = get_status()
        except Exception as e:
            if as_json:
                print(json.dumps({"error": str(e), "source": source}))
            else:
                print(f"Error reading DiLoCo status for {target}: {e}")
            return 1
        merged = orch.assemble_status(
            get_status=lambda: status,
            get_info=get_info,
            get_known_workers=get_known,
            get_work_queues=get_queues,
            get_work_queue_detail=get_queue_detail,
        )
        if as_json:
            print(json.dumps({"source": source, **merged}, default=str, indent=2))
            return 0
        return orch.render_status(merged, want_queues=want_queues)

    if not getattr(args, "watch", False):
        return _render_once()

    # Watch mode: poll in-process, reusing the same client/connection across
    # ticks (no per-tick subprocess, unlike `watch -n N forgather …`). Ctrl-C
    # exits cleanly — in the interactive CLI it returns to the prompt.
    import time

    interval = max(0.1, float(getattr(args, "interval", 2.0)))
    use_clear = sys.stdout.isatty()
    try:
        while True:
            if use_clear:
                # Clear screen + home cursor (ANSI); falls back to a rule
                # for non-TTY sinks.
                print("\033[2J\033[H", end="")
            else:
                print("\n" + "=" * 50)
            print(
                f"forgather diloco status — {target} — "
                f"{time.strftime('%H:%M:%S')} "
                f"(every {interval:g}s, Ctrl-C to stop)\n"
            )
            _render_once()
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
        return 0


# CLI action name -> trainer-control command the server relays.
_CONTROL_ACTION_MAP = {
    "save": "save_checkpoint",
    "save-stop": "save_and_stop",
    "abort": "abort",
}


def _control_cmd(args):
    """Relay a trainer-control command to one or all workers."""
    from . import diloco_orch as orch
    from .server_client import ServerUnreachable

    command = _CONTROL_ACTION_MAP[args.action]
    try:
        ops, label = orch.make_control_ops(args)
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1
    try:
        resp = ops.relay(command, worker_id=args.worker_id)
    except Exception as e:
        print(f"Error contacting server at {label}: {e}")
        return 1
    workers = resp.get("workers", [])
    if not workers:
        print("No workers registered — nothing to do.")
        return 0
    print(f"Queued '{command}' for {len(workers)} worker(s): {', '.join(workers)}")
    print("Each worker applies it on its next heartbeat.")
    return 0


def _shutdown_cmd(args):
    """Stop the DiLoCo server — cleanly (default) or immediately (--force).

    Routes the control actions through the forgather server by default (see
    make_control_ops); ``--local-only`` goes straight to the parameter
    server, ``--local-fallback`` does so only when the server is down, and
    the default errors if the server is unreachable.
    """
    import time

    from . import diloco_orch as orch
    from .server_client import ServerUnreachable

    try:
        ops, label = orch.make_control_ops(args)
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.force:
        print("Force shutdown: stopping server now (workers will lose sync).")
        try:
            ops.shutdown()
        except Exception as e:
            # The server closes the socket as it exits; tolerate that.
            print(f"  (server stop signalled; {type(e).__name__})")
        print("Server stop signalled.")
        return 0

    # Clean shutdown: save-stop all workers, wait for them to exit, then
    # checkpoint + stop the server.
    try:
        resp = ops.relay("save_and_stop")
    except Exception as e:
        print(f"Error contacting server at {label}: {e}")
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
                status = ops.get_status()
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
        ops.save_state()
        print("  server checkpoint saved.")
    except Exception as e:
        print(f"  server checkpoint failed: {e} (continuing to stop)")

    print("Stopping server…")
    try:
        ops.shutdown()
    except Exception as e:
        print(f"  (server stop signalled; {type(e).__name__})")
    print("Done.")
    return 0


def _load_dynamic_schema(project_dir, config_template):
    """Return the config's ``dynamic_args`` schema (list of entry dicts), or
    ``[]`` on any failure.

    ``config_template=None`` resolves the project's DEFAULT config (via
    ``Project(config_name=None)``), matching the parser side
    (``diloco_args.parse_dynamic_args``) and ``forgather train`` — so the
    worker collects + forwards dynamic args even without an explicit ``-t``.
    """
    try:
        from forgather import MetaConfig, Project

        pdir = MetaConfig.find_project_dir(project_dir)
        proj = Project(config_name=config_template, project_dir=pdir)
        if "dynamic_args" in proj.config:
            return proj("dynamic_args") or []
    except Exception:
        pass
    return []


def _schema_dests(schema):
    """argparse dests (``max_steps``) for each entry in a dynamic_args schema."""
    dests = []
    for entry in schema or []:
        if not isinstance(entry, dict):
            continue
        names = entry.get("names")
        if isinstance(names, str):
            names = [names]
        if not names:
            continue
        long = next((n for n in names if str(n).startswith("--")), names[0])
        dests.append(long.lstrip("-").replace("-", "_"))
    return dests


def _worker_dynamic_args(args, schema):
    """Collect the config's dynamic args, scoped to the schema's dests.

    Two callers reach the worker path with the dynamic args in different
    places: ``forgather diloco worker`` keeps them as namespace attributes (the
    diloco parser deliberately doesn't propagate ``_dynamic_arg_names`` — see
    diloco_args.create_diloco_parser), while ``forgather submit`` declares them
    on its own subparser, so main.py partitions them into ``args._dynamic_args``
    and strips the attributes. Support both: take the partitioned dict, then
    overlay any namespace-attribute values. ``None`` is filtered so unset valued
    args fall back to template defaults.
    """
    out = {}
    partitioned = getattr(args, "_dynamic_args", None) or {}
    for dest in _schema_dests(schema):
        if partitioned.get(dest) is not None:
            out[dest] = partitioned[dest]
        v = getattr(args, dest, None)
        if v is not None:
            out[dest] = v
    return out


def _dynamic_cli_from_schema(schema, dynamic_args):
    """Pure schema→CLI-tokens reconstruction (testable without a Project).

    ``schema`` is the config's ``dynamic_args`` list of entry dicts
    (``{names, type, action, …}``). For each parsed ``(dest, value)``:
    store_true/store_false args emit a bare flag only when the value differs
    from the action's implied default; everything else emits ``--flag value``.
    Args absent from the schema fall back to ``--flag value``.
    """
    by_dest = {}
    for entry in schema or []:
        if not isinstance(entry, dict):
            continue
        names = entry.get("names")
        if isinstance(names, str):
            names = [names]
        if not names:
            continue
        long = next((n for n in names if str(n).startswith("--")), names[0])
        dest = long.lstrip("-").replace("-", "_")
        by_dest[dest] = (long, entry)

    toks = []
    for dest, val in dynamic_args.items():
        meta = by_dest.get(dest)
        flag = meta[0] if meta else "--" + dest.replace("_", "-")
        action = meta[1].get("action") if meta else None
        if action in ("store_true", "store_false"):
            implied_default = action == "store_false"
            if bool(val) != implied_default:
                toks.append(flag)
        else:
            toks.extend([flag, str(val)])
    return toks


def _worker_cmd(args):
    """
    Launch training as a DiLoCo worker.

    The worker-launch implementation behind ``forgather submit --diloco``
    (submit_cmd calls this directly). Don't rename or change its signature
    without updating ``submit.submit_cmd``.

    By default the worker(s) are enqueued as scheduled training jobs through
    the forgather server — the path that supports ``--count N`` (auto-named),
    ``--dataset auto|server:<id>``, and central auth. ``--local-only`` (or
    ``--local-fallback`` when the server is down) instead runs a single
    worker in the foreground by wrapping ``forgather train`` with DiLoCo env
    vars; the default errors if the server is unreachable.

    Dynamic/template args are accepted the standard way (built from the
    config's ``dynamic_args`` metadata, like ``forgather train``) and are
    forwarded — as ``EnqueueRequest.dynamic_args`` on the orchestrator path,
    or as ``--dynamic-args <json>`` to the spawned trainer on the direct path.
    """
    from . import diloco_orch as orch
    from . import submit_orch
    from .server_client import ServerUnreachable

    schema = _load_dynamic_schema(
        getattr(args, "project_dir", "."), getattr(args, "config_template", None)
    )
    dynamic_args = _worker_dynamic_args(args, schema)
    # Enforce the config's required/bounded dynamic args up front (same check
    # `forgather submit`/`train` do), so a missing required arg fails here
    # rather than deep inside the spawned worker job.
    submit_orch.validate_dynamic_args(
        getattr(args, "project_dir", "."),
        getattr(args, "config_template", None),
        dynamic_args,
    )
    count = getattr(args, "count", 1) or 1
    resume = getattr(args, "resume_workers", False)

    # --resume-workers is its own mode: re-launch the stopped workers the
    # server already knows. It doesn't select/create new workers, so it can't
    # be combined with --worker-id / --count.
    if resume and (getattr(args, "worker_id", None) or count > 1):
        print(
            "error: --resume-workers restarts the stopped workers the server "
            "already knows; it can't be combined with --worker-id or --count.",
            file=sys.stderr,
        )
        return 1

    try:
        client = orch.use_orchestrator(args)
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1

    if resume:
        if client is None:
            print(
                "error: --resume-workers requires the forgather server (it "
                "provides the known-worker roster); not available with "
                "--local-only.",
                file=sys.stderr,
            )
            return 1
        return orch.launch_resume(args, dynamic_args)

    if client is not None:
        return orch.launch_workers(args, dynamic_args)

    # --- Direct / foreground path (forgather server not reachable) ---
    if count > 1:
        print(
            "error: --count > 1 launches multiple background jobs and "
            "requires the forgather server (start it with 'forgather "
            "server', or run one worker at a time directly).",
            file=sys.stderr,
        )
        return 1

    # Set DiLoCo environment variables for the training script. Only
    # client-local knobs are forwarded; sync_every / bf16_comm / dylu /
    # num_fragments are server-authoritative and resolved from /info by
    # the worker at startup (no client override).
    env = os.environ.copy()
    # Direct/foreground: no discovery here, so default to loopback when
    # --server is omitted.
    env["DILOCO_SERVER"] = (
        getattr(args, "diloco_server", None) or orch.DEFAULT_DIRECT_SERVER
    )
    env["DILOCO_HEARTBEAT_INTERVAL"] = str(getattr(args, "heartbeat_interval", 30.0))

    # Always set DILOCO_WORKER_ID — when the operator didn't supply one,
    # mint a memorable two-word name so the worker doesn't surface as
    # the worker.py auto-generated ``worker_<hostname>_<8hex>`` fallback.
    # Same naming convention the orchestrator path uses via
    # ``client.generate_diloco_worker_names`` and the queue route's
    # auto-fill for blank submissions.
    if args.worker_id:
        env["DILOCO_WORKER_ID"] = args.worker_id
    else:
        from forgather.utils import generate_name

        env["DILOCO_WORKER_ID"] = generate_name()

    # `forgather submit --diloco` doesn't carry --devices (it's a scheduler
    # submit command); the direct/foreground worker inherits the parent env's
    # CUDA_VISIBLE_DEVICES.
    devices = getattr(args, "devices", None)
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = devices

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

    # Forward dynamic/template args to the spawned `forgather train` as its
    # own first-class flags (`--max-steps 500`). `forgather train` does NOT
    # accept `--dynamic-args` (that's a flag on the generated training
    # *script*, which train re-derives from first-class flags), so we
    # reconstruct the flags from the parsed dict using the config's
    # dynamic_args schema to emit the right form (store_true flag vs valued
    # option). This is what lets `forgather -t cfg diloco worker --max-steps
    # 500` reach the training job. The orchestrator path forwards the dict
    # directly as EnqueueRequest.dynamic_args instead.
    if dynamic_args:
        cmd_args.extend(_dynamic_cli_from_schema(schema, dynamic_args))

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
    # Report DILOCO_WORKER_ID unconditionally so the operator sees the
    # auto-minted memorable name (when no --worker-id was passed) in the
    # banner — without this they only learn the name from the running
    # worker's log line.
    diloco_info = (
        f"DiLoCo: server={env['DILOCO_SERVER']}, "
        f"worker_id={env['DILOCO_WORKER_ID']}"
    )

    print(diloco_info)
    print(f"Command: {cmd_str}")

    if args.dry_run:
        return 0
    # Propagate the trainer's exit code so a failed worker surfaces as a
    # non-zero `forgather diloco worker` exit (scriptability).
    return subprocess.run(cmd_args, env=env).returncode


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
    elif subcmd == "servers":
        from .diloco_orch import servers_cmd

        return servers_cmd(args)
    elif subcmd == "logs":
        from .diloco_orch import logs_cmd

        return logs_cmd(args)
    elif subcmd == "register":
        from .diloco_orch import register_cmd

        return register_cmd(args)
    elif subcmd == "unregister":
        from .diloco_orch import unregister_cmd

        return unregister_cmd(args)
    else:
        print(
            "Usage: forgather diloco {server|worker|status|servers|logs|"
            "register|unregister|control|shutdown}"
        )
        print("Run 'forgather diloco --help' for details.")
        return 1
