"""
forgather dataset-server — multi-action wrapper.

Subcommands:

- ``start [...]`` runs ``tools/dataset_server/server.py`` as a
  subprocess (REMAINDER passthrough so server flags reach the script
  unchanged).
- ``status`` / ``list`` / ``cache`` / ``local`` are pure HTTP client
  calls against a running server.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ._dataset_server_client import (
    AuthRequired,
    DatasetServerClient,
    ServerError,
)


def dataset_server_cmd(args) -> int:
    sub = getattr(args, "ds_subcommand", None) or "start"
    if sub == "start":
        return _start_cmd(args)
    if sub == "status":
        return _status_cmd(args)
    if sub == "list":
        return _list_cmd(args)
    if sub == "cache":
        return _cache_cmd(args)
    if sub == "local":
        return _local_cmd(args)
    print(f"Error: Unknown dataset-server action: {sub}", file=sys.stderr)
    return 2


# ----- start -----


def _parse_local_maps(raw):
    """``["NAME=PATH", ...]`` → ``[[name, path], ...]``; raises ValueError on
    a malformed entry. Empty/None → ``[]``."""
    pairs = []
    for entry in raw or []:
        name, sep, path = str(entry).partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--local expects NAME=PATH, got {entry!r}")
        pairs.append([name, path])
    return pairs


def _start_job_params(args, local_pairs) -> dict:
    """Build the ``dataset_server`` job_params the scheduler consumes.

    Identical shape to the webui's "Start dataset server" modal (buildArgs):
    the scheduler provisions the per-port auth token + TLS itself, so no
    token/cert is sent here.
    """
    params = {
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "no_auth": bool(args.no_auth),
        # regen_token is only meaningful when auth is on.
        "regen_token": bool(args.regen_token) and not bool(args.no_auth),
        "no_hf": bool(args.no_hf),
        "allow_paths": bool(args.allow_paths),
        "allow_downloads": bool(args.allow_downloads),
    }
    if args.config:
        params["config_file"] = args.config
    if local_pairs:
        params["locals"] = local_pairs
    return params


def _start_via_server(client, args, local_pairs) -> int:
    """Enqueue a scheduled ``dataset_server`` job through the forgather server.

    The scheduler starts it in the background (CPU-only), captures its TTY,
    provisions auth/TLS, and the cluster dataset inventory picks it up — so
    it shows up as a job and is known to the cluster, exactly like a
    webui-launched server.
    """
    from .server_client import AuthRequired, ServerUnreachable

    # Extra/unknown server flags can't be honored by the scheduler (it owns
    # the managed flag surface); they only apply to a foreground launch.
    extra = getattr(args, "extra", None) or []
    if extra:
        print(
            "error: these flags are only supported with --local-only: "
            f"{' '.join(extra)}",
            file=sys.stderr,
        )
        return 1

    try:
        item = client.enqueue_job(
            # No project context for this tool; the launcher ignores
            # project_dir for dataset_server jobs (matches the webui).
            project_dir="/",
            config=f"dataset:{args.port}",
            job_type="dataset_server",
            job_params=_start_job_params(args, local_pairs),
            requested_gpus=0,
            priority=getattr(args, "priority", 0),
        )
    except (ServerUnreachable, AuthRequired, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(item, indent=2))
        return 0
    qid = item.get("queue_id")
    print(f"Enqueued dataset server job {qid} (the scheduler will start it).")
    print("  status:  forgather dataset-server status")
    print(f"  logs:    forgather job tail {qid} -f")
    print(f"  stop:    forgather job stop {qid}")
    return 0


def _start_local(args, local_pairs) -> int:
    """Run the dataset server in the foreground (the pre-scheduler path)."""
    repo_root = _repo_root()
    cmd_args = [
        sys.executable,
        "-m",
        "tools.dataset_server",
        "-H",
        args.host,
        "-p",
        str(args.port),
        "-l",
        args.log_level,
    ]
    if args.no_hf:
        cmd_args.append("--no-hf")
    if args.allow_paths:
        cmd_args.append("--allow-paths")
    if args.allow_downloads:
        cmd_args.append("--allow-downloads")
    for name, path in local_pairs:
        cmd_args.extend(["--local", f"{name}={path}"])
    if args.no_auth:
        cmd_args.append("--no-auth")
    if args.regen_token:
        cmd_args.append("--regen-token")
    if args.config:
        cmd_args.extend(["--config", args.config])
    # Forward any extra/unknown flags verbatim (e.g. TLS options).
    cmd_args.extend(getattr(args, "extra", None) or [])
    print(f"Running: {' '.join(cmd_args)}")
    print()
    return subprocess.run(cmd_args, cwd=str(repo_root)).returncode


def _start_cmd(args) -> int:
    # Orchestrator-first (matches `forgather diloco server`): enqueue a
    # scheduled job by default; --local-only forces foreground; --local-
    # fallback uses foreground only when the server is unreachable; the
    # default errors if the server is down (no silent local degrade).
    from . import diloco_orch as orch  # generic orchestrator-locality helper
    from .server_client import ServerUnreachable

    try:
        local_pairs = _parse_local_maps(getattr(args, "local_maps", None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        client = orch.use_orchestrator(args)
    except ServerUnreachable as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if client is not None:
        return _start_via_server(client, args, local_pairs)
    return _start_local(args, local_pairs)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ----- diagnostic actions -----


def _status_cmd(args) -> int:
    client = DatasetServerClient.from_args(args)
    return _emit(args, client.health, _fmt_health)


def _list_cmd(args) -> int:
    client = DatasetServerClient.from_args(args)
    return _emit(args, client.list_datasets, _fmt_handles)


def _cache_cmd(args) -> int:
    client = DatasetServerClient.from_args(args)
    return _emit(args, client.list_hf_cache, _fmt_hf_cache)


def _local_cmd(args) -> int:
    client = DatasetServerClient.from_args(args)
    return _emit(args, client.list_local, _fmt_local)


# ----- output helpers -----


def _emit(args, fetch, format_human) -> int:
    try:
        payload = fetch()
    except (AuthRequired, ServerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    format_human(payload)
    return 0


def _fmt_health(payload: dict) -> None:
    print(
        f"service: {payload.get('service', '?')}  version: {payload.get('version', '?')}"
    )
    print(f"status:  {payload.get('status', '?')}")
    pol = payload.get("policy", {}) or {}
    print("policy:")
    print(f"  auth_required:    {pol.get('auth_required', '?')}")
    print(f"  hf_cache_enabled: {pol.get('hf_cache_enabled', '?')}")
    print(f"  allow_paths:      {pol.get('allow_paths', '?')}")
    print(f"  allow_downloads:  {pol.get('allow_downloads', '?')}")
    print(f"  local_count:      {pol.get('local_count', '?')}")


def _fmt_handles(payload: dict) -> None:
    handles = payload.get("handles", []) or []
    if not handles:
        print("(no handles loaded)")
        return
    width = max((len(h.get("handle", "")) for h in handles), default=8)
    print(f"{'HANDLE':<{width}}  {'LENGTH':>10}  SOURCE  ARGS")
    for h in handles:
        args_str = json.dumps(h.get("load_args", {}), sort_keys=True)
        print(
            f"{h.get('handle', ''):<{width}}  "
            f"{h.get('length', '?'):>10}  "
            f"{h.get('source', '?'):<6}  "
            f"{args_str}"
        )


def _fmt_local(payload: dict) -> None:
    items = payload.get("local", []) or []
    if not items:
        print("(no local mappings configured)")
        return
    width = (
        max((len(it.get("name", "")) for it in items), default=4) + 6
    )  # 'local/' prefix
    print(f"{'NAME':<{width}}  PATH")
    for it in items:
        name = f"local/{it.get('name', '?')}"
        print(f"{name:<{width}}  {it.get('path', '?')}")


def _fmt_hf_cache(payload: dict) -> None:
    root = payload.get("cache_root", "?")
    datasets = payload.get("datasets", []) or []
    print(f"cache_root: {root}")
    print(f"datasets:   {len(datasets)}")
    print()
    if not datasets:
        print("(cache is empty or unreadable)")
        return
    for d in datasets:
        print(f"- {d.get('repo', '?')}  ({_human_bytes(d.get('size_bytes', 0))})")
        for cfg in d.get("configs", []) or []:
            cfg_name = cfg.get("config", "?")
            ver = cfg.get("version") or "?"
            splits = cfg.get("splits") or []
            split_str = (
                ", ".join(_split_summary(s) for s in splits)
                if splits
                else "(no splits info)"
            )
            print(f"    {cfg_name} @ {ver}  -- {split_str}")


def _split_summary(s: dict) -> str:
    name = s.get("name", "?")
    n = s.get("num_examples")
    if n is None:
        return name
    return f"{name}={n:,}"


def _human_bytes(n: int) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"
