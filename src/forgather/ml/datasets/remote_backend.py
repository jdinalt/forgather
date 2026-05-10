"""
RemoteBackend — `IterableDatasetBackend` implementation that
talks to a `tools.dataset_server` over HTTP.

The client holds purely local state — ``(server_url, handle, seed,
position)``. ``shuffle(seed)`` and ``seek(position)`` are local-only
(they mint a new instance with updated state). ``__iter__`` opens a
streaming HTTP request that carries the current state, so the server
is effectively stateless wrt clients and multiple clients can share a
handle without interfering with each other's iteration cursors.

Wire format: newline-delimited JSON. See ``tools/dataset_server`` for
the matching server side.

Auth: when the server is configured for bearer-token auth, the
client must send ``Authorization: Bearer <token>``. Tokens are
either passed explicitly via the ``token`` constructor argument, or
auto-discovered for localhost URLs by reading the per-port file the
server publishes under ``$FORGATHER_HOME/dataset_server/<port>.token``.
The ``$FORGATHER_DATASET_SERVER_TOKEN`` environment variable wins
over both.

This is intentionally minimal: no retries, no compression, no
connection pooling. Wrap it in a ``ComposableIterableDataset`` for
map / filter / shard / state_dict semantics.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


#: Env var that overrides any auto-discovered or passed-in token.
TOKEN_ENV_VAR = "FORGATHER_DATASET_SERVER_TOKEN"

#: Hostnames that count as "this machine" for token auto-discovery.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _read_localhost_token_file(url: str) -> Optional[str]:
    """Read the per-port token file the dataset server publishes for
    localhost clients.

    Mirrors `tools.dataset_server.auth.read_standalone_token` but is
    inlined so the loader doesn't depend on the ``tools/`` package
    being importable from arbitrary client processes (e.g. inside a
    training script that imports forgather but never adds ``tools/``
    to ``sys.path``).

    Returns ``None`` if the URL isn't loopback, has no explicit port,
    or the file doesn't exist / is empty.
    """
    try:
        parsed = urlparse(url)
    except (TypeError, ValueError):
        return None
    if parsed.hostname not in _LOOPBACK_HOSTS:
        return None
    port = parsed.port
    if port is None:
        return None
    home_env = os.environ.get("FORGATHER_HOME")
    if home_env:
        home = Path(home_env).expanduser()
    else:
        home = Path.home() / ".forgather"
    token_path = home / "dataset_server" / f"{int(port)}.token"
    try:
        text = token_path.read_text().strip()
    except OSError:
        return None
    return text or None


def _from_jsonable(value):
    """Decode the ``__bytes_b64__`` tagged dicts the server may emit."""
    import base64

    if isinstance(value, dict):
        if "__bytes_b64__" in value and len(value) == 1:
            return base64.b64decode(value["__bytes_b64__"])
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


def resolve_auth_token(url: str, explicit: Optional[str]) -> Optional[str]:
    """Discover the bearer token for ``url``.

    Order: explicit > ``$FORGATHER_DATASET_SERVER_TOKEN`` > localhost
    per-port file. Returns ``None`` if no token is found — the server
    might be running with ``--no-auth``, in which case requests are
    accepted unauthenticated.

    Note: the localhost lookup uses an inlined reader rather than
    importing from ``tools.dataset_server.auth`` because the loader
    runs from arbitrary client processes that don't have the
    ``tools/`` directory on ``sys.path``.
    """
    if explicit:
        return explicit
    env = os.environ.get(TOKEN_ENV_VAR)
    if env:
        return env.strip() or None
    return _read_localhost_token_file(url)


class RemoteBackend(IterableDatasetBackend):
    """
    Network-proxy backend.

    Parameters
    ----------
    url : str
        Base URL of the dataset server, e.g. ``"http://host:8766"``.
    handle : str
        Server-side identifier for the registered backend to consume.
    seed : int, optional
        Shuffle seed; ``None`` means no shuffle requested.
    position : int, optional
        Initial flat example index. Default ``0``.
    timeout : float, optional
        Per-request HTTP timeout (seconds). Default ``60``.
    token : str, optional
        Explicit bearer token. If omitted, the constructor consults
        ``$FORGATHER_DATASET_SERVER_TOKEN`` and (for localhost URLs)
        ``$FORGATHER_HOME/dataset_server/<port>.token``.
    """

    def __init__(
        self,
        url: str,
        handle: str,
        seed: Optional[int] = None,
        position: int = 0,
        timeout: float = 60.0,
        token: Optional[str] = None,
        column_names: Optional[list[str]] = None,
    ):
        if position < 0:
            raise ValueError(f"position must be non-negative, got {position}")
        self._url = url.rstrip("/")
        self._handle = handle
        self._seed = seed
        self._position = position
        self._timeout = timeout
        # Resolved once at construction; if you change tokens, build a
        # new client. Most callers won't notice.
        self._token = resolve_auth_token(self._url, token)
        self._cached_len: Optional[int] = None
        # Schema cache. The loader passes column_names from the
        # /v1/load response so the client can answer column-aware
        # APIs (e.g. preprocess_dataset's remove_columns) without an
        # extra round trip; if not supplied here, we fetch it lazily
        # from /v1/datasets/{handle} on first access.
        self._column_names: Optional[list[str]] = (
            list(column_names) if column_names is not None else None
        )

    # ----- Backend interface -----

    def __iter__(self) -> Iterator[dict]:
        """
        Open a streaming /iter request from the current position and
        yield decoded examples. Updates ``self._position`` as each
        example arrives so callers can capture progress mid-stream.
        """
        params: dict[str, str] = {"position": str(self._position)}
        if self._seed is not None:
            params["seed"] = str(self._seed)
        url = f"{self._url}/v1/datasets/{self._handle}/iter?{urlencode(params)}"
        req = Request(url, method="GET", headers=self._headers())
        with urlopen(req, timeout=self._timeout) as resp:
            for raw in resp:
                line = raw.rstrip(b"\n")
                if not line:
                    continue
                example = _from_jsonable(json.loads(line.decode("utf-8")))
                self._position += 1
                yield example

    def __len__(self) -> int:
        if self._cached_len is None:
            url = f"{self._url}/v1/datasets/{self._handle}/length"
            req = Request(url, method="GET", headers=self._headers())
            with urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            self._cached_len = int(payload["length"])
        return self._cached_len

    def shuffle(self, seed: Optional[int] = None) -> "RemoteBackend":
        """
        Return a new client with the new seed; position resets to 0.

        No RPC is issued — the seed travels with the next ``/iter``
        request. The cached length is preserved (shuffling doesn't
        change the underlying example count).
        """
        new = RemoteBackend(
            self._url,
            self._handle,
            seed=seed,
            position=0,
            timeout=self._timeout,
            token=self._token,
        )
        new._cached_len = self._cached_len
        return new

    def seek(self, position: int) -> "RemoteBackend":
        """
        Return a new client positioned at the given flat example index.

        No RPC is issued — the position travels with the next
        ``/iter`` request.
        """
        if position < 0:
            raise ValueError(f"position must be non-negative, got {position}")
        new = RemoteBackend(
            self._url,
            self._handle,
            seed=self._seed,
            position=position,
            timeout=self._timeout,
            token=self._token,
        )
        new._cached_len = self._cached_len
        return new

    def position(self) -> int:
        return self._position

    # ----- Optional metadata -----

    @property
    def n_shards(self) -> int:
        # The remote layer doesn't expose physical sharding info; the
        # server may have any number of files behind the handle.
        return 1

    @property
    def column_names(self) -> Optional[list[str]]:
        """Column names of the underlying dataset.

        Populated either from the `/v1/load` response (most common —
        the loader passes them through) or by a lazy GET to
        `/v1/datasets/{handle}` on first access. Returns ``None`` if
        the server can't determine them.
        """
        if self._column_names is not None:
            return self._column_names
        url = f"{self._url}/v1/datasets/{self._handle}"
        req = Request(url, method="GET", headers=self._headers())
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug("column_names lookup failed: %s", exc)
            return None
        cols = payload.get("column_names")
        if cols is not None:
            self._column_names = list(cols)
        return self._column_names

    # ----- helpers -----

    def _headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def __repr__(self) -> str:
        return (
            f"RemoteBackend(url={self._url!r}, "
            f"handle={self._handle!r}, seed={self._seed}, "
            f"position={self._position})"
        )
