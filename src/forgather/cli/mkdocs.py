"""Run `mkdocs serve` locally or enqueue it on the forgather-server."""

import os
import subprocess
import sys


def mkdocs_cmd(args):
    config_file = os.path.abspath(args.config_file)
    if args.enqueue:
        return _enqueue(args, config_file)

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


def _enqueue(args, config_file):
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

    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient.from_args(args)
    try:
        item = client.enqueue_job(
            project_dir=os.path.abspath(args.project_dir),
            config=f"mkdocs:{args.port}",
            job_type="mkdocs",
            job_params=job_params,
            requested_gpus=0,
            priority=args.priority,
        )
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"queued: {item['queue_id']} (mkdocs:{args.port}, priority={item['priority']})"
    )
    return 0
