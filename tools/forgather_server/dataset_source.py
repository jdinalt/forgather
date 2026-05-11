"""Resolve a webui-supplied ``dataset_source`` choice to env vars.

The submit modal's dataset-source selector picks where a training job
should fetch examples from. Two shapes the webui sends:

- ``{"kind": "local"}`` — use the in-process loader. No env vars; the
  training script's default ``fast_load_iterable_dataset`` path runs.
- ``{"kind": "server", "server_id": "..."}`` — route through a named
  dataset_server. ``server_id`` is one of:

  * ``local:<queue_id>`` — a dataset_server the forgather_server itself
    spawned. URL + token come from the matching JobRecord.
  * ``user:<entry_id>`` — a URL the operator registered via the
    Datasets → Servers tab. URL + token come from the registry.

This module owns the resolution. ``resolve_to_env`` returns a dict of
env vars to merge into the spawn's ``extra_env``; ``None`` means "no
server, run locally". On a stale id (registry deleted, dataset_server
exited) it raises ``DatasetSourceError`` so the enqueue path can
surface a helpful 400 to the caller instead of silently falling back.

Env var shape matches what the loader already reads:

- ``FORGATHER_DATASET_SERVER`` — base URL.
- ``FORGATHER_DATASET_SERVER_TOKEN`` — bearer token (omitted when the
  server runs ``--no-auth`` and the JobRecord has no token).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import dataset_server_registry, job_records


class DatasetSourceError(ValueError):
    """Resolution failed (unknown id, server not running, etc.)."""


def resolve_to_env(source: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Return the env vars to add to ``extra_env``, or ``None`` for local.

    ``source`` shape is webui-owned and validated here. Anything not
    matching the documented shape returns ``None`` (treated as local)
    so an older webui that doesn't send the field still works; only
    *known* shapes with stale ids raise.
    """
    if not source or not isinstance(source, dict):
        return None
    kind = source.get("kind")
    if kind in (None, "local"):
        return None
    if kind != "server":
        # Unknown kind — treat as local rather than refusing to submit.
        # A future webui adding a new kind without a backend bump
        # shouldn't break training submits.
        return None

    server_id = source.get("server_id")
    if not isinstance(server_id, str) or ":" not in server_id:
        raise DatasetSourceError(f"invalid server_id: {server_id!r}")
    src_kind, _, src_value = server_id.partition(":")

    if src_kind == "local":
        # JobRecord lookup. The webui only allows picking *alive*
        # servers, but a job may have exited between the modal opening
        # and the submit click — raise so the user sees a clear
        # message and re-opens the modal.
        for r in job_records.list_records():
            if r.job_type != "dataset_server" or r.queue_id != src_value:
                continue
            if r.status not in {"starting", "running"}:
                raise DatasetSourceError(
                    f"dataset_server {src_value} is not running "
                    f"(status={r.status}); re-open the submit modal "
                    "to pick a different source"
                )
            params = r.job_params or {}
            port = params.get("port")
            if port is None:
                raise DatasetSourceError(
                    f"dataset_server {src_value} has no port in job_params"
                )
            host = params.get("host") or "127.0.0.1"
            # Server-spawned dataset_servers bind to loopback by default
            # (or to 0.0.0.0 when the operator picked that). Either way
            # the training subprocess on this host reaches it via
            # localhost, since it runs in the same machine as the
            # forgather_server. Translate 0.0.0.0 → localhost; leave
            # other binds (rare, e.g. a real LAN address) alone.
            client_host = "localhost" if host == "0.0.0.0" else host
            env: Dict[str, str] = {
                "FORGATHER_DATASET_SERVER": f"http://{client_host}:{int(port)}",
            }
            if r.auth_token:
                env["FORGATHER_DATASET_SERVER_TOKEN"] = r.auth_token
            return env
        raise DatasetSourceError(
            f"no local dataset_server with queue_id={src_value!r} "
            "(it may have exited since the modal was opened)"
        )

    if src_kind == "user":
        for entry in dataset_server_registry.list_entries():
            if entry.id != src_value:
                continue
            env = {"FORGATHER_DATASET_SERVER": entry.base_url}
            if entry.auth_token:
                env["FORGATHER_DATASET_SERVER_TOKEN"] = entry.auth_token
            return env
        raise DatasetSourceError(
            f"no user-registered dataset_server with id={src_value!r} "
            "(it may have been deleted)"
        )

    raise DatasetSourceError(
        f"unknown server_id prefix: {src_kind!r} (expected 'local' or 'user')"
    )
