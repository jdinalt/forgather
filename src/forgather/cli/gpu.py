"""GPU status and policy subcommands for the forgather-server CLI."""

import sys


def _fmt_mem(b):
    if b is None:
        return "?"
    return f"{b / 1024**3:.1f}"


def _fmt_pct(v):
    if v is None:
        return "?"
    return f"{v:.0f}"


def _fmt_val(v, fmt=None):
    if v is None:
        return "?"
    if fmt:
        return fmt.format(v)
    return str(v)


def _gpu_status(client):
    gpus = client.list_gpus()

    col = {
        "idx": 4,
        "name": 24,
        "util": 5,
        "mem": 14,
        "temp": 6,
        "pwr": 7,
        "fan": 5,
        "dis": 10,
        "pri": 6,
        "pids": 6,
    }

    header = (
        f"{'Idx':>{col['idx']}}  "
        f"{'Name':<{col['name']}}  "
        f"{'Util%':>{col['util']}}  "
        f"{'Mem GB used/total':>{col['mem']}}  "
        f"{'Temp C':>{col['temp']}}  "
        f"{'Pwr W':>{col['pwr']}}  "
        f"{'Fan%':>{col['fan']}}  "
        f"{'Disabled':<{col['dis']}}  "
        f"{'MinPri':>{col['pri']}}  "
        f"{'PIDs':>{col['pids']}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for g in gpus:
        idx = g.get("index", "?")
        name = (g.get("name") or "?")[: col["name"]]
        util = _fmt_pct(g.get("util_pct"))
        used = _fmt_mem(g.get("used_mem_bytes"))
        total = _fmt_mem(g.get("total_mem_bytes"))
        mem = f"{used}/{total}"
        temp = _fmt_val(g.get("temp_c"), "{:.0f}")
        pwr = _fmt_val(g.get("power_w"), "{:.0f}")
        fan = _fmt_pct(g.get("fan_pct"))
        disabled = g.get("disabled", False)
        excluded = g.get("excluded", False)
        dis_str = "(excluded)" if excluded else str(disabled)
        min_pri = _fmt_val(g.get("min_priority"))
        procs = g.get("processes") or []
        pid_count = len(procs)
        print(
            f"{str(idx):>{col['idx']}}  "
            f"{name:<{col['name']}}  "
            f"{util:>{col['util']}}  "
            f"{mem:>{col['mem']}}  "
            f"{temp:>{col['temp']}}  "
            f"{pwr:>{col['pwr']}}  "
            f"{fan:>{col['fan']}}  "
            f"{dis_str:<{col['dis']}}  "
            f"{min_pri:>{col['pri']}}  "
            f"{str(pid_count):>{col['pids']}}"
        )


def gpu_cmd(args):
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient.from_args(args)

    sub = getattr(args, "gpu_subcommand", None)
    if sub is None:
        print(
            "error: specify a subcommand (status, disable, enable, priority, kill)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        if sub == "status":
            _gpu_status(client)

        elif sub == "disable":
            result = client.set_gpu_policy(args.idx, disabled=True)
            print(
                f"gpu {args.idx}: disabled={result.get('disabled')} min_priority={result.get('min_priority')}"
            )

        elif sub == "enable":
            result = client.set_gpu_policy(args.idx, disabled=False)
            print(
                f"gpu {args.idx}: disabled={result.get('disabled')} min_priority={result.get('min_priority')}"
            )

        elif sub == "priority":
            result = client.set_gpu_policy(args.idx, min_priority=args.level)
            print(
                f"gpu {args.idx}: disabled={result.get('disabled')} min_priority={result.get('min_priority')}"
            )

        elif sub == "kill":
            if not args.yes:
                print(
                    "gpu kill requires --yes (this kills ALL compute processes on the card, not just server jobs)",
                    file=sys.stderr,
                )
                sys.exit(1)
            result = client.kill_gpu_processes(args.idx)
            killed = result.get("killed", [])
            failed = result.get("failed", [])
            print(f"gpu {args.idx}: killed={killed}, failed={failed}")

        else:
            print(f"error: unknown subcommand: {sub}", file=sys.stderr)
            sys.exit(1)

    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
