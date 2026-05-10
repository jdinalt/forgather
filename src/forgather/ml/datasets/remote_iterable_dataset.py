"""
RemoteIterableDataset — `IterableDatasetBackend` implementation that
talks to a `tools.dataset_server` over HTTP.

The client holds purely local state — ``(server_url, handle, seed,
position)``. ``shuffle(seed)`` and ``seek(position)`` are local-only
(they mint a new instance with updated state). ``__iter__`` opens a
streaming HTTP request that carries the current state, so the server
is effectively stateless wrt clients and multiple clients can share a
handle without interfering with each other's iteration cursors.

Wire format: newline-delimited JSON. See ``tools/dataset_server`` for
the matching server side.

This is a proof-of-concept implementation intended to validate that
the backend interface is sufficient for remote consumption. It is
deliberately minimal: no retries, no compression, no auth, no
connection pooling. Wrap it in a ``ComposableIterableDataset`` for
map / filter / shard / state_dict semantics.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


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


class RemoteIterableDataset(IterableDatasetBackend):
    """
    Network-proxy backend.

    Parameters
    ----------
    url : str
        Base URL of the dataset server, e.g. ``"http://host:8765"``.
    handle : str
        Server-side identifier for the registered backend to consume.
    seed : int, optional
        Shuffle seed; ``None`` means no shuffle requested.
    position : int, optional
        Initial flat example index. Default ``0``.
    timeout : float, optional
        Per-request HTTP timeout (seconds). Default ``60``.
    """

    def __init__(
        self,
        url: str,
        handle: str,
        seed: Optional[int] = None,
        position: int = 0,
        timeout: float = 60.0,
    ):
        if position < 0:
            raise ValueError(f"position must be non-negative, got {position}")
        self._url = url.rstrip("/")
        self._handle = handle
        self._seed = seed
        self._position = position
        self._timeout = timeout
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
        req = Request(url, method="GET")
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
            req = Request(url, method="GET")
            with urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            self._cached_len = int(payload["length"])
        return self._cached_len

    def shuffle(self, seed: Optional[int] = None) -> "RemoteIterableDataset":
        """
        Return a new client with the new seed; position resets to 0.

        No RPC is issued — the seed travels with the next ``/iter``
        request. The cached length is preserved (shuffling doesn't
        change the underlying example count).
        """
        new = RemoteIterableDataset(
            self._url, self._handle, seed=seed, position=0, timeout=self._timeout
        )
        new._cached_len = self._cached_len
        return new

    def seek(self, position: int) -> "RemoteIterableDataset":
        """
        Return a new client positioned at the given flat example index.

        No RPC is issued — the position travels with the next
        ``/iter`` request.
        """
        if position < 0:
            raise ValueError(f"position must be non-negative, got {position}")
        new = RemoteIterableDataset(
            self._url,
            self._handle,
            seed=self._seed,
            position=position,
            timeout=self._timeout,
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

    def __repr__(self) -> str:
        return (
            f"RemoteIterableDataset(url={self._url!r}, "
            f"handle={self._handle!r}, seed={self._seed}, "
            f"position={self._position})"
        )
