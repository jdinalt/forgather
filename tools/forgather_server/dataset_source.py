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


def resolve_to_env(
    source: Optional[Dict[str, Any]],
    *,
    remote_host_override: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Return the env vars to add to ``extra_env``, or ``None`` for local.

    ``source`` shape is webui-owned and validated here. Two forward-
    compat cases short-circuit to "local" (None env vars):

    - ``source`` is None / not a dict — older webui omitting the field.
    - ``source.kind`` is None or ``"local"`` — explicit local choice.

    Any other ``kind`` value (or a stale id under ``"server"``) raises
    ``DatasetSourceError`` so the operator sees a clear 400 rather
    than a job that silently fell back. Earlier drafts of this code
    treated unknown kinds as local for forward-compat, but that hid
    operator typos and out-of-sync webui/server versions — the rest
    of the resolver raises in equivalent "can't act on the choice"
    cases, so this matches.

    ``remote_host_override`` is for the cluster-dispatch path: when
    the resolved env will be shipped to a *peer* training process,
    ``127.0.0.1`` is wrong (that's the peer's loopback, not the
    master's dataset server). Set this to the master's cluster-
    routable address; the resolver substitutes it in place of the
    loopback-fallback. Single-node submits leave this as ``None``
    and get the loopback URL as before.
    """
    if not source or not isinstance(source, dict):
        return None
    kind = source.get("kind")
    if kind in (None, "local"):
        return None
    if kind == "auto":
        # Cluster auto-routing: the resilient client consults
        # /api/cluster/dataset_router/resolve on the local
        # forgather_server every time it needs to (re)connect. No
        # token is shipped via env — the client uses
        # $FORGATHER_SERVER_TOKEN (or the bearer file) to talk to its
        # local forgather_server, which holds the upstream token in
        # the master inventory.
        return {"FORGATHER_DATASET_SERVER": "auto"}
    if kind != "server":
        raise DatasetSourceError(
            f"unknown dataset_source kind: {kind!r} "
            "(expected 'local', 'server', or 'auto')"
        )

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
            # the training subprocess on this host reaches it via the
            # loopback alias, since it runs on the same machine as the
            # forgather_server. Translate 0.0.0.0 → 127.0.0.1 rather
            # than "localhost" — on IPv6-first hosts (some container
            # setups, some /etc/hosts orderings) "localhost" can resolve
            # to ::1 first, which fails against an IPv4-only wildcard
            # bind. 127.0.0.1 always matches the IPv4 wildcard.
            # Pick the host the *trainer process* will actually dial.
            #
            #   * remote_host_override (set by the cluster submit path):
            #     the master's cluster-routable IP. Used when the
            #     trainer is on a peer node — that peer's loopback is
            #     not the master's. Required for multi-node training
            #     against a master-spawned dataset server.
            #   * No override + bind host 0.0.0.0 → translate to
            #     127.0.0.1 (single-node case, trainer on same host
            #     as the dataset server).
            #   * No override + explicit bind host → trust it as-is.
            #
            # A loopback-only bind (host == "127.0.0.1" or "localhost")
            # combined with a remote_host_override is an operator error
            # — the dataset server isn't reachable from peers at all.
            # Surface that with a clear message instead of letting it
            # fail at connect time on the peer.
            if remote_host_override:
                if host in ("127.0.0.1", "::1", "localhost"):
                    raise DatasetSourceError(
                        f"dataset_server {src_value} is bound to loopback "
                        f"only ({host}); peers in the cluster cannot reach "
                        "it. Restart the dataset server with --host 0.0.0.0 "
                        "(or a routable LAN IP) to share it across the cluster."
                    )
                client_host = remote_host_override
            else:
                client_host = "127.0.0.1" if host == "0.0.0.0" else host
            from forgather.tls import client_scheme

            scheme = client_scheme(client_host)
            env: Dict[str, str] = {
                "FORGATHER_DATASET_SERVER": f"{scheme}://{client_host}:{int(port)}",
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
