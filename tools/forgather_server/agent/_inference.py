"""Discover and query inference servers on behalf of the agent's inference
tools.

Discovery reuses ``cluster_inference_inventory`` (JobRecord-spawned servers
+ cluster snapshot); calls go to the OpenAI-compatible ``/v1/*`` endpoints
with the server's bearer (resolved here, kept internal). TLS posture: plain
http is a no-op; loopback https is accepted unverified (a loopback peer
can't be MITM'd); a routable https host uses system trust.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("forgather_server.agent.inference")

_PROBE_TIMEOUT = httpx.Timeout(connect=4.0, read=4.0, write=4.0, pool=4.0)
_QUERY_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


def _discover() -> List[Dict[str, Any]]:
    from .. import cluster_inference_inventory as cii

    entries: Dict[str, Dict[str, Any]] = {}
    try:
        for s in cii.local_servers():
            entries.setdefault(
                s.server_id,
                {
                    "id": s.server_id,
                    "label": s.label,
                    "base_url": s.base_url,
                    "source": "local",
                    "token": s.auth_token or None,
                    "loopback": getattr(s, "loopback", False),
                    "models": list(getattr(s, "models", []) or []),
                },
            )
    except Exception:
        log.debug("local inference discovery failed", exc_info=True)
    try:
        for s in cii.master_inventory.servers_snapshot():
            entries.setdefault(
                s.server_id,
                {
                    "id": s.server_id,
                    "label": s.label,
                    "base_url": s.base_url,
                    "source": "cluster",
                    "token": s.auth_token or None,
                    "loopback": getattr(s, "loopback", False),
                    "models": list(getattr(s, "models", []) or []),
                },
            )
    except Exception:
        log.debug("cluster inference discovery failed", exc_info=True)
    return list(entries.values())


def _verify(entry: Dict[str, Any]) -> Any:
    base = (entry.get("base_url") or "").lower()
    if not base.startswith("https"):
        return True  # http: ignored by httpx
    if entry.get("loopback"):
        return False  # loopback https: a loopback peer can't be MITM'd
    return True  # routable https: system trust


def _headers(entry: Dict[str, Any]) -> Dict[str, str]:
    tok = entry.get("token")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _reachable(entry: Dict[str, Any]) -> bool:
    try:
        with httpx.Client(timeout=_PROBE_TIMEOUT, verify=_verify(entry)) as c:
            r = c.get(entry["base_url"].rstrip("/") + "/health", headers=_headers(entry))
        return r.status_code < 500
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
    entries = _discover()
    # Probe reachability concurrently — each probe blocks up to _PROBE_TIMEOUT,
    # so serial probing would be O(N * timeout). (Callers should still run this
    # off the event loop, e.g. via asyncio.to_thread.)
    reach = _probe_all(entries)
    return [
        {
            "id": e["id"],
            "label": e["label"],
            "base_url": e["base_url"],
            "source": e["source"],
            "models": e["models"],
            "reachable": ok,
        }
        for e, ok in zip(entries, reach)
    ]


def _pick(server_id: Optional[str]) -> Dict[str, Any]:
    entries = _discover()
    if not entries:
        raise ValueError(
            "No inference server is known. Start one with "
            "start_inference_server(model_path=..., port=...) and retry."
        )
    if server_id:
        for e in entries:
            if e["id"] == server_id:
                return e
        raise ValueError(
            f"no inference server with id {server_id!r} "
            f"(use list_inference_servers to see available ids)"
        )
    for e in entries:
        if _reachable(e):
            return e
    raise ValueError("No inference server is reachable. Start one and retry.")


def chat(
    server_id: Optional[str],
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_tokens: int = 256,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Send a chat completion to a running inference server and return the
    assistant message + usage (compact projection of the OpenAI response)."""
    chosen = _pick(server_id)
    use_model = model or (chosen["models"][0] if chosen["models"] else None)
    body: Dict[str, Any] = {"messages": messages, "max_tokens": int(max_tokens)}
    if use_model:
        body["model"] = use_model
    if temperature is not None:
        body["temperature"] = float(temperature)
    url = chosen["base_url"].rstrip("/") + "/v1/chat/completions"
    with httpx.Client(timeout=_QUERY_TIMEOUT, verify=_verify(chosen)) as c:
        r = c.post(url, json=body, headers=_headers(chosen))
        r.raise_for_status()
        data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return {
        "server": {"id": chosen["id"], "base_url": chosen["base_url"]},
        "model": data.get("model") or use_model,
        "message": {"role": msg.get("role"), "content": msg.get("content")},
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
    }
