"""
ResilientRemoteBackend — wraps :class:`RemoteBackend` with retry +
re-establish semantics so a transient network blip or server restart
doesn't abort a long-running training job.

The wrapper holds the original dataset descriptor (``path``, ``name``,
``split``, ``data_files``, ``revision``) and the current ``(seed,
position)``. When the inner backend raises
:class:`DatasetServerUnreachable`, the wrapper applies exponential
backoff (1s → 30s cap), re-issues `POST /v1/load` against the same
``base_url`` (or — Phase 4 — a freshly-resolved one from the cluster
router), and builds a new inner ``RemoteBackend`` at the captured
``position``. Iteration resumes at the next example.

Resolver hook
-------------
``resolver`` is a callback returning ``(base_url, token)`` for the
next attempt. The default (``None``) is **reconnect-only** mode: the
wrapper keeps hitting the same URL with the same token. Cluster
``auto`` mode passes a resolver that consults the local
forgather_server's ``/api/cluster/dataset_router/resolve``.

Retry budget
------------
By default the wrapper retries forever; only ``KeyboardInterrupt`` /
``SIGTERM`` abort. The
``FORGATHER_DATASET_CLIENT_MAX_RETRY_SECONDS`` env var sets an
optional ceiling on cumulative backoff sleep.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Iterator, Optional
from urllib.request import Request, urlopen

from .iterable_backend import IterableDatasetBackend
from .remote_backend import (
    DatasetServerUnreachable,
    RemoteBackend,
    _make_ssl_context,
    _translate_request_error,
    resolve_auth_token,
)

logger = logging.getLogger(__name__)

#: Env var setting an optional ceiling on cumulative retry backoff
#: seconds. Unset = retry forever (only operator termination aborts).
MAX_RETRY_SECONDS_ENV_VAR = "FORGATHER_DATASET_CLIENT_MAX_RETRY_SECONDS"


Resolver = Callable[[dict], tuple[str, Optional[str]]]


def _do_load_once(
    base_url: str,
    token: Optional[str],
    load_args: dict,
    timeout: float = 300.0,
) -> dict:
    """Issue ``POST /v1/load`` once and return the parsed response.

    Raises :class:`DatasetServerUnreachable` on transient network /
    5xx failures; other ``HTTPError`` cases propagate unchanged (so
    auth or bad-request errors surface as logic failures, not retry
    loops).
    """
    url = base_url.rstrip("/") + "/v1/load"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps({k: v for k, v in load_args.items() if v is not None}).encode(
        "utf-8"
    )
    req = Request(url, data=body, method="POST", headers=headers)
    ssl_context = _make_ssl_context(base_url)
    try:
        with urlopen(req, timeout=timeout, context=ssl_context) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise _translate_request_error(exc) from exc


class ResilientRemoteBackend(IterableDatasetBackend):
    """
    Retry-aware wrapper around :class:`RemoteBackend`.

    Parameters
    ----------
    base_url, token : str, Optional[str]
        Initial dataset_server URL and bearer token. Used until a
        ``resolver`` is supplied and the inner backend fails.
    load_args : dict
        Descriptor passed to ``POST /v1/load`` (``path``, ``name``,
        ``split``, ``data_files``, ``revision``) — the same fields
        :func:`forgather.ml.datasets.fast_hf_loader._remote_load_iterable_dataset`
        already builds.
    handle, length, column_names : optional
        Result of the *initial* ``/v1/load`` call. If supplied, the
        wrapper uses them directly for the first connection so the
        caller's eager-load cost isn't paid twice.
    resolver : callable, optional
        Callback ``(load_args) -> (base_url, token)`` invoked on
        failure to obtain a fresh URL + token. Default: reconnect to
        the same URL.
    seed, position : int, optional
        Initial iteration state. ``position`` defaults to 0.
    max_retry_seconds : float, optional
        Cumulative cap on backoff sleep. ``None`` reads
        ``$FORGATHER_DATASET_CLIENT_MAX_RETRY_SECONDS`` or defaults to
        no cap.
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str],
        load_args: dict,
        *,
        handle: Optional[str] = None,
        length: Optional[int] = None,
        column_names: Optional[list[str]] = None,
        resolver: Optional[Resolver] = None,
        seed: Optional[int] = None,
        position: int = 0,
        max_retry_seconds: Optional[float] = None,
    ):
        if position < 0:
            raise ValueError(f"position must be non-negative, got {position}")
        self._base_url = base_url
        self._token = token
        self._load_args = dict(load_args)
        self._resolver = resolver
        self._seed = seed
        self._position = position
        self._cached_len: Optional[int] = length
        self._column_names: Optional[list[str]] = (
            list(column_names) if column_names is not None else None
        )
        self._inner: Optional[RemoteBackend] = None
        if handle is not None:
            self._inner = RemoteBackend(
                base_url,
                handle,
                seed=seed,
                position=position,
                token=token,
                column_names=self._column_names,
            )
            if length is not None:
                self._inner._cached_len = length  # avoid an extra /length RPC
        if max_retry_seconds is None:
            env_cap = os.environ.get(MAX_RETRY_SECONDS_ENV_VAR)
            if env_cap:
                try:
                    max_retry_seconds = float(env_cap)
                except ValueError:
                    logger.warning(
                        "ignoring non-numeric %s=%r",
                        MAX_RETRY_SECONDS_ENV_VAR,
                        env_cap,
                    )
                    max_retry_seconds = None
        self._max_retry_seconds = max_retry_seconds

    # ----- IterableDatasetBackend interface -----

    def __iter__(self) -> Iterator[dict]:
        elapsed = 0.0
        attempt = 0
        while True:
            try:
                self._ensure_inner()
                # Inner backend may raise DatasetServerUnreachable
                # mid-stream; its `position` advances as examples are
                # yielded, so on retry we resume from the very next
                # unread example.
                for example in self._inner:  # type: ignore[union-attr]
                    self._position = self._inner.position()  # type: ignore[union-attr]
                    attempt = 0
                    elapsed = 0.0
                    yield example
                return
            except DatasetServerUnreachable as exc:
                if self._inner is not None:
                    self._position = self._inner.position()
                attempt, elapsed = self._sleep_or_raise(exc, attempt, elapsed)
                self._inner = None  # force fresh load on next attempt

    def __len__(self) -> int:
        if self._cached_len is not None:
            return self._cached_len
        attempt = 0
        elapsed = 0.0
        while True:
            try:
                self._ensure_inner()
                self._cached_len = len(self._inner)  # type: ignore[arg-type]
                return self._cached_len
            except DatasetServerUnreachable as exc:
                attempt, elapsed = self._sleep_or_raise(exc, attempt, elapsed)
                self._inner = None

    def shuffle(self, seed: Optional[int] = None) -> "ResilientRemoteBackend":
        """Return a fresh wrapper at position 0 with the new seed."""
        return ResilientRemoteBackend(
            self._base_url,
            self._token,
            self._load_args,
            handle=self._inner._handle if self._inner is not None else None,
            length=self._cached_len,
            column_names=self._column_names,
            resolver=self._resolver,
            seed=seed,
            position=0,
            max_retry_seconds=self._max_retry_seconds,
        )

    def seek(self, position: int) -> "ResilientRemoteBackend":
        if position < 0:
            raise ValueError(f"position must be non-negative, got {position}")
        return ResilientRemoteBackend(
            self._base_url,
            self._token,
            self._load_args,
            handle=self._inner._handle if self._inner is not None else None,
            length=self._cached_len,
            column_names=self._column_names,
            resolver=self._resolver,
            seed=self._seed,
            position=position,
            max_retry_seconds=self._max_retry_seconds,
        )

    def position(self) -> int:
        return self._position

    # ----- Optional metadata -----

    @property
    def n_shards(self) -> int:
        return 1

    @property
    def column_names(self) -> Optional[list[str]]:
        if self._column_names is not None:
            return self._column_names
        try:
            self._ensure_inner()
        except DatasetServerUnreachable:
            return None
        if self._inner is not None:
            cols = self._inner.column_names
            if cols is not None:
                self._column_names = list(cols)
                return self._column_names
        return None

    # ----- helpers -----

    def _ensure_inner(self) -> None:
        if self._inner is not None:
            return
        if self._resolver is not None:
            self._base_url, self._token = self._resolver(self._load_args)
        elif self._token is None:
            # Re-resolve from env / token file in case the file was
            # written after the first attempt.
            self._token = resolve_auth_token(self._base_url, explicit=None)
        result = _do_load_once(self._base_url, self._token, self._load_args)
        handle = result["handle"]
        if "length" in result:
            try:
                self._cached_len = int(result["length"])
            except (TypeError, ValueError):
                pass
        cols = result.get("column_names")
        if cols is not None:
            self._column_names = list(cols)
        self._inner = RemoteBackend(
            self._base_url,
            handle,
            seed=self._seed,
            position=self._position,
            token=self._token,
            column_names=self._column_names,
        )
        if self._cached_len is not None:
            self._inner._cached_len = self._cached_len

    def _sleep_or_raise(
        self, exc: DatasetServerUnreachable, attempt: int, elapsed: float
    ) -> tuple[int, float]:
        delay = min(30.0, 2.0**attempt)
        if (
            self._max_retry_seconds is not None
            and elapsed + delay > self._max_retry_seconds
        ):
            logger.error(
                "dataset_server unreachable after %.0fs (cap %.0fs); aborting: %s",
                elapsed,
                self._max_retry_seconds,
                exc,
            )
            raise exc
        logger.warning(
            "dataset_server unreachable (%s); retrying in %.1fs (attempt %d, "
            "elapsed %.0fs)",
            exc,
            delay,
            attempt + 1,
            elapsed,
        )
        time.sleep(delay)
        return attempt + 1, elapsed + delay

    def __repr__(self) -> str:
        return (
            f"ResilientRemoteBackend(base_url={self._base_url!r}, "
            f"seed={self._seed}, position={self._position}, "
            f"resolver={'auto' if self._resolver else None})"
        )
