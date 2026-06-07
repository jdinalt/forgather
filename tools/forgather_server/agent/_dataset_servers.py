"""Discover and query dataset servers on behalf of the agent's dataset
metadata tools.

Discovery reuses ``cluster_dataset_inventory`` (which already merges
JobRecord-spawned servers + the user registry for the local node, and the
master inventory for the cluster), and queries go through the stdlib
``DatasetServerClient``. We deliberately keep the bearer tokens internal —
``list_dataset_servers`` projects them out before returning to the model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from forgather.cli._dataset_server_client import DatasetServerClient

log = logging.getLogger("forgather_server.agent.dataset_servers")

_PROBE_TIMEOUT = 4.0  # cheap reachability check
_QUERY_TIMEOUT = 30.0


def _discover() -> List[Dict[str, Any]]:
    """Merge local-node + cluster dataset servers into internal entries
    carrying the token (id, label, base_url, source, token, verify_tls).
    Deduped by server id; local entries win over cluster ones."""
    from .. import cluster_dataset_inventory as cdi

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
                },
            )
    except Exception:  # cluster identity / registry unavailable
        log.debug("local dataset-server discovery failed", exc_info=True)
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
                },
            )
    except Exception:
        log.debug("cluster dataset-server discovery failed", exc_info=True)
    return list(entries.values())


def _client(entry: Dict[str, Any], *, timeout: float) -> DatasetServerClient:
    return DatasetServerClient(
        url=entry["base_url"],
        token=entry.get("token") or None,
        insecure=not entry.get("verify_tls", True),
        timeout=timeout,
    )


def _reachable(entry: Dict[str, Any]) -> bool:
    try:
        _client(entry, timeout=_PROBE_TIMEOUT).health()
        return True
    except Exception:
        return False


def list_servers() -> List[Dict[str, Any]]:
    """Public list for the agent: id/label/base_url/source/reachable (no token)."""
    out = []
    for e in _discover():
        out.append(
            {
                "id": e["id"],
                "label": e["label"],
                "base_url": e["base_url"],
                "source": e["source"],
                "reachable": _reachable(e),
            }
        )
    return out


def _matches(candidate: Optional[str], dataset: str) -> bool:
    if not candidate:
        return False
    return candidate == dataset or candidate == f"local/{dataset}" or (
        f"local/{candidate}" == dataset
    )


def _pick(entries: List[Dict[str, Any]], server_id: Optional[str]) -> Dict[str, Any]:
    if not entries:
        raise ValueError(
            "No dataset server is reachable. Start one (e.g. from the webui's "
            "Datasets view, or `forgather dataset-server start`) and retry."
        )
    if server_id:
        for e in entries:
            if e["id"] == server_id:
                if not _reachable(e):
                    raise ValueError(f"dataset server {server_id!r} is not reachable")
                return e
        raise ValueError(
            f"no dataset server with id {server_id!r} "
            f"(use list_dataset_servers to see available ids)"
        )
    for e in entries:  # first reachable; local entries are listed first
        if _reachable(e):
            return e
    raise ValueError(
        "No dataset server is reachable. Start one (e.g. from the webui's "
        "Datasets view, or `forgather dataset-server start`) and retry."
    )


def info(
    dataset: str, server_id: Optional[str] = None, split: Optional[str] = None
) -> Dict[str, Any]:
    """Return splits / #examples / features for ``dataset`` from a dataset
    server. Tries the HF-cache inventory and the local-dataset listing; if
    features are still unknown, falls back to POST /v1/load (which loads the
    dataset on the server)."""
    chosen = _pick(_discover(), server_id)
    client = _client(chosen, timeout=_QUERY_TIMEOUT)

    splits: Dict[str, Any] = {}
    features: List[str] = []
    source: Optional[str] = None

    def _add_features(fs):
        for f in fs or []:
            if f not in features:
                features.append(f)

    try:
        hf = client.list_hf_cache()
    except Exception:
        hf = {}
    for repo in hf.get("datasets", []) or []:
        if not _matches(repo.get("repo"), dataset):
            continue
        for cfg in repo.get("configs", []) or []:
            for sp in cfg.get("splits", []) or []:
                splits.setdefault(sp.get("name"), sp.get("num_examples"))
            _add_features(cfg.get("features"))
        source = "hf_cache"

    try:
        loc = client.list_local()
    except Exception:
        loc = {}
    for entry in loc.get("local", []) or []:
        if not _matches(entry.get("name"), dataset):
            continue
        for sp in entry.get("splits", []) or []:
            splits.setdefault(sp.get("name"), sp.get("num_examples"))
        _add_features(entry.get("features"))
        source = source or "local"

    # Features fallback: load the dataset to read its column names. Only when
    # we have nothing else — this triggers a real load on the server.
    if splits and not features:
        target_split = split or next(iter(splits), None)
        try:
            r = client.load({"path": dataset, "split": target_split})
            _add_features(r.get("column_names"))
        except Exception:
            log.debug("dataset_info load fallback failed", exc_info=True)

    if not splits and not features:
        raise ValueError(
            f"dataset {dataset!r} was not found on dataset server "
            f"{chosen['id']!r} ({chosen['base_url']}). If you just built it, "
            f"give the server a moment; otherwise check the dataset name."
        )

    return {
        "server": {"id": chosen["id"], "base_url": chosen["base_url"]},
        "source": source,
        "splits": [{"name": n, "num_examples": splits[n]} for n in splits],
        "features": features,
    }
