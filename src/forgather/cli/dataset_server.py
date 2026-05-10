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


def _start_cmd(args) -> int:
    repo_root = _repo_root()
    cmd_args = [sys.executable, "-m", "tools.dataset_server"]
    remainder = getattr(args, "remainder", None) or []
    cmd_args.extend(remainder)
    print(f"Running: {' '.join(cmd_args)}")
    print()
    return subprocess.run(cmd_args, cwd=str(repo_root)).returncode


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
