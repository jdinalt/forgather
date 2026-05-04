"""Per-job control and log subcommands for the forgather-server CLI."""

import asyncio
import sys


def _do_control(client, job_id, action):
    result = client.job_control(job_id, action)
    if result.get("success"):
        print(f"OK: {result.get('message', action)}")
    else:
        print(f"FAIL: {result.get('message', action)}", file=sys.stderr)
        sys.exit(1)


def _do_tail(client, job_id):
    async def _run():
        try:
            async for kind, data in client.stream_tty(job_id, follow=True):
                if kind == "bytes":
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                else:
                    print(f"error: {data}", file=sys.stderr)
        except KeyboardInterrupt:
            pass

    asyncio.run(_run())


def job_cmd(args):
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient.from_args(args)

    sub = getattr(args, "job_subcommand", None)
    if sub is None:
        print(
            "error: specify a subcommand (status, save, stop, save-stop, abort, kill, force-kill, tail, dump, logs)",
            file=sys.stderr,
        )
        sys.exit(1)

    job_id = args.job_id

    try:
        if sub == "status":
            r = client.job_status(job_id)
            if r.status_code == 409:
                print("still starting; trainer endpoint not yet registered")
                return
            if not r.ok:
                try:
                    detail = r.json().get("detail", r.text)
                except Exception:
                    detail = r.text
                print(f"server: {detail}", file=sys.stderr)
                sys.exit(1)
            data = r.json()
            for key, value in data.items():
                print(f"{key}: {value}")

        elif sub in ("save", "stop", "save-stop", "abort", "kill"):
            _do_control(client, job_id, sub)

        elif sub == "force-kill":
            if not getattr(args, "yes", False):
                print(
                    "force-kill requires --yes (SIGKILL is destructive)",
                    file=sys.stderr,
                )
                sys.exit(1)
            _do_control(client, job_id, "force-kill")

        elif sub in ("tail",):
            _do_tail(client, job_id)

        elif sub in ("dump", "logs"):
            data = client.job_dump(job_id)
            sys.stdout.buffer.write(data)

        else:
            print(f"error: unknown subcommand: {sub}", file=sys.stderr)
            sys.exit(1)

    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(1)
