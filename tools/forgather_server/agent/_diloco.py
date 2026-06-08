"""Discover and talk to DiLoCo servers on behalf of the agent's DiLoCo tools.

Mirrors ``_dataset_servers.py``: discovery reuses
``cluster_diloco_inventory`` (JobRecord-spawned + user registry for the
local node, plus the master inventory for the cluster); queries/control go
through the stdlib ``DiLoCoClient``. Tokens are resolved here and kept
internal — ``list_servers`` projects them out.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from forgather.ml.diloco.client import DiLoCoClient

log = logging.getLogger("forgather_server.agent.diloco")

_PROBE_TIMEOUT = 4.0
_QUERY_TIMEOUT = 30.0


def _discover() -> List[Dict[str, Any]]:
    """Merge local-node + cluster DiLoCo servers into internal entries
    (id, label, base_url, source, token, verify_tls, healthy). Deduped by
    id; local entries win over cluster ones."""
    from .. import cluster_diloco_inventory as cdi

    entries: Dict[str, Dict[str, Any]] = {}
    try:
        for s in cdi.local_servers():
            entries.setdefault(
                s.server_id,
                {
                    "id": s.server_id,
                    "label": s.label,
                    "base_url": s.base_url,
                    "source": s.source,  # "local" | "user"
                    "token": s.auth_token or None,
                    "verify_tls": s.verify_tls,
                    "healthy": None,
                },
            )
    except Exception:
        log.debug("local diloco-server discovery failed", exc_info=True)
    try:
        for s in cdi.master_inventory.servers_snapshot():
            entries.setdefault(
                s.server_id,
                {
                    "id": s.server_id,
                    "label": s.label,
                    "base_url": s.base_url,
                    "source": "cluster",
                    "token": s.auth_token or None,
                    "verify_tls": s.verify_tls,
                    "healthy": getattr(s, "healthy", None),
                },
            )
    except Exception:
        log.debug("cluster diloco-server discovery failed", exc_info=True)
    return list(entries.values())


def _client(entry: Dict[str, Any], *, timeout: float, max_retries: int = 3) -> DiLoCoClient:
    return DiLoCoClient(
        server_addr=entry["base_url"],
        token=entry.get("token") or None,
        verify_tls=entry.get("verify_tls", True),
        timeout=timeout,
        max_retries=max_retries,
    )


def _reachable(entry: Dict[str, Any]) -> bool:
    try:
        _client(entry, timeout=_PROBE_TIMEOUT, max_retries=1).get_status()
        return True
    except Exception:
        return False


def _probe_all(entries: List[Dict[str, Any]]) -> List[bool]:
    """Reachability for every entry, probed concurrently (bounded pool)."""
    if not entries:
        return []
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(entries))) as ex:
        return list(ex.map(_reachable, entries))


def list_servers() -> List[Dict[str, Any]]:
    """Public list for the agent: id/label/base_url/source/reachable/healthy
    (no token)."""
    entries = _discover()
    reach = _probe_all(entries)  # concurrent; callers run this off the loop
    return [
        {
            "id": e["id"],
            "label": e["label"],
            "base_url": e["base_url"],
            "source": e["source"],
            "reachable": ok,
            "healthy": e.get("healthy"),
        }
        for e, ok in zip(entries, reach)
    ]


def _pick(server_id: Optional[str]) -> Dict[str, Any]:
    entries = _discover()
    if not entries:
        raise ValueError(
            "No DiLoCo server is known. Start one with "
            "start_diloco_server(output_dir=..., num_workers=...) and "
            "retry."
        )
    if server_id:
        for e in entries:
            if e["id"] == server_id:
                return e
        raise ValueError(
            f"no DiLoCo server with id {server_id!r} "
            f"(use list_diloco_servers to see available ids)"
        )
    for e in entries:  # first reachable; local entries are listed first
        if _reachable(e):
            return e
    raise ValueError(
        "No DiLoCo server is reachable. Start one with "
        "start_diloco_server(...) and retry."
    )


def status(server_id: Optional[str] = None) -> Dict[str, Any]:
    """Live status + worker roster for a DiLoCo server."""
    chosen = _pick(server_id)
    client = _client(chosen, timeout=_QUERY_TIMEOUT)
    st = client.get_status()
    workers: Any
    try:
        roster = client.get_known_workers().get("workers", []) or []
        workers = [
            {"worker_id": w.get("worker_id"), "running": w.get("running")}
            for w in roster
        ]
    except Exception:
        workers = None
    return {
        "server": {"id": chosen["id"], "base_url": chosen["base_url"]},
        "status": st,
        "workers": workers,
    }


# action -> (client method name, needs a `command`?)
_CONTROL = {
    "save_state": ("save_state", False),
    "shutdown": ("shutdown", False),
    "relay": ("relay_command", True),  # command in {save_checkpoint,save_and_stop,abort}
}
_RELAY_COMMANDS = {"save_checkpoint", "save_and_stop", "abort"}


def control(
    server_id: Optional[str],
    action: str,
    command: Optional[str] = None,
    worker_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a control action against a DiLoCo server (the side-effecting half;
    callers gate it behind approval)."""
    if action not in _CONTROL:
        raise ValueError(f"unknown action {action!r}; expected one of {sorted(_CONTROL)}")
    method, needs_command = _CONTROL[action]
    if needs_command and command not in _RELAY_COMMANDS:
        raise ValueError(
            f"action 'relay' needs command in {sorted(_RELAY_COMMANDS)}"
        )
    chosen = _pick(server_id)
    client = _client(chosen, timeout=_QUERY_TIMEOUT)
    if action == "relay":
        resp = client.relay_command(command, worker_id=worker_id)
    else:
        resp = getattr(client, method)()
    return {"server": {"id": chosen["id"], "base_url": chosen["base_url"]}, "result": resp}
