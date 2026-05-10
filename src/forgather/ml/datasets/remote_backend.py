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
from typing import Iterator, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


#: Env var that overrides any auto-discovered or passed-in token.
TOKEN_ENV_VAR = "FORGATHER_DATASET_SERVER_TOKEN"


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
    """
    if explicit:
        return explicit
    env = os.environ.get(TOKEN_ENV_VAR)
    if env:
        return env.strip() or None
    # Lazy import to avoid pulling FastAPI/uvicorn into the loader hot path.
    try:
        from tools.dataset_server.auth import read_standalone_token
    except ImportError:
        # tools/ not on sys.path — try the directly-installed name.
        try:
            from dataset_server.auth import read_standalone_token  # type: ignore
        except ImportError:
            return None
    try:
        return read_standalone_token(url)
    except Exception as exc:
        logger.debug("auth-token auto-discovery failed: %s", exc)
        return None


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
