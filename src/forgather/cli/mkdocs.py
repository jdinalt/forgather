"""Run `mkdocs serve` locally or enqueue it on the forgather-server."""

import os
import subprocess
import sys


def mkdocs_cmd(args):
    """Run the docs server.

    The docs server is long-running, so it is submitted to the
    forgather-server scheduler (background) by default — like inf server and
    dataset-server; check its output with `forgather job logs <id>`.
    `--local-only` runs `mkdocs serve` in the foreground instead;
    `--local-fallback` foregrounds only when the server is unreachable.
    """
    from . import submit_orch

    config_file = os.path.abspath(args.config_file)
    if getattr(args, "enqueue", False):
        print(
            "note: --enqueue is deprecated; mkdocs now submits to the scheduler "
            "by default (use --local-only to run the foreground server).",
            file=sys.stderr,
        )

    if not getattr(args, "local_only", False):
        from .server_client import ServerUnreachable

        try:
            client = submit_orch.use_orchestrator(args)
        except ServerUnreachable as e:
            print(str(e), file=sys.stderr)
            raise SystemExit(1)
        if client is not None:
            return _enqueue(args, config_file, client)
        # client is None → --local-fallback and server down → foreground.

    cmd = [
        "mkdocs",
        "serve",
        "-f",
        config_file,
        "--dev-addr",
        f"{args.host}:{args.port}",
    ]
    if args.strict:
        cmd.append("--strict")
    if not args.livereload:
        cmd.append("--no-livereload")
    if args.dirty:
        cmd.append("--dirty")
    for w in args.watch:
        cmd.extend(["--watch", w])

    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd).returncode


def _enqueue(args, config_file, client):
    job_params = {
        "config_file": config_file,
        "host": args.host,
        "port": args.port,
        "strict": bool(args.strict),
        "livereload": bool(args.livereload),
        "dirty": bool(args.dirty),
    }
    if args.watch:
        job_params["watch"] = list(args.watch)

    from . import submit_orch
    from .server_client import ServerUnreachable

    try:
        item = submit_orch.submit_single(
            client,
            project_dir=os.path.abspath(args.project_dir),
            config=f"mkdocs:{args.port}",
            job_type="mkdocs",
            job_params=job_params,
            requested_gpus=0,
            priority=args.priority,
            dynamic_args=None,
        )
    except (ServerUnreachable, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"queued: {item['queue_id']} (mkdocs:{args.port}, priority={item['priority']})"
    )
    return 0
