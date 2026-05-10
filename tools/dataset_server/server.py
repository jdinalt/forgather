"""
HTTP dataset server (proof of concept).

Endpoints
---------

- ``GET /v1/datasets`` — list registered handles.
- ``GET /v1/datasets/{handle}/length`` — JSON ``{"length": int}``.
- ``GET /v1/datasets/{handle}/iter?seed=<int>&position=<int>&limit=<int>``
  — newline-delimited JSON stream of examples. ``seed`` and ``limit``
  are optional; ``position`` defaults to 0.
- ``POST /v1/load`` — body ``{"path", "name", "split", "data_files",
  "revision"}`` matching the `fast_load_iterable_dataset` signature.
  Server lazily loads the dataset locally (caching by hash of the
  args) and returns ``{"handle": str, "length": int}`` so the client
  can use ``handle`` for subsequent ``/iter`` and ``/length`` calls.
  Disabled by default; pass ``allow_load=True`` to ``DatasetServer``
  (or ``--allow-load`` on the CLI) to enable.

The server is intentionally stateless with respect to clients — every
``/iter`` call carries the seed and position to start from. This means
multiple clients can hit the same handle concurrently without
trampling each other's iteration state.

Backend handling
----------------

The handle resolves to an `IterableDatasetBackend` registered via
``DatasetServer.register`` or auto-registered through ``/v1/load``.
For each ``/iter`` request the server applies ``backend.shuffle(seed)``
(if seed given) and ``backend.seek(position)`` to obtain a fresh
iteration view, then streams examples from it. The original registered
backend is treated as immutable.

Anti-recursion
--------------

When the server lazy-loads a dataset via ``/v1/load`` it calls into
``forgather.ml.datasets.fast_hf_loader._local_load_iterable_dataset``,
which bypasses the ``FORGATHER_DATASET_SERVER`` env var so the server
process can have the variable set in its environment without looping
back to itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlsplit

from forgather.ml.datasets.iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


def _to_jsonable(value):
    """Coerce common HF example value types into something json.dumps
    accepts. Bytes get a base64-tagged dict so the client can recover
    them; everything else passes through and json raises on truly
    foreign types (which surfaces during testing instead of silently
    corrupting examples)."""
    import base64

    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _from_jsonable(value):
    """Inverse of `_to_jsonable`."""
    import base64

    if isinstance(value, dict):
        if "__bytes_b64__" in value and len(value) == 1:
            return base64.b64decode(value["__bytes_b64__"])
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


def _canonical_handle(load_args: Dict[str, Any]) -> str:
    """Stable short hash of a normalized load_args dict."""
    # Drop None values so callers can omit fields they don't care about
    # without producing a different handle for {x: None} vs {} omitted.
    normalized = {k: v for k, v in load_args.items() if v is not None}
    canonical = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# Fields the /v1/load endpoint accepts and forwards to the local loader.
# Anything else in the request body is silently ignored.
_LOAD_FIELDS = (
    "path",
    "name",
    "split",
    "data_files",
    "revision",
    "force_reindex",
    "num_proc",
)


class DatasetServer:
    """
    Container that owns a registry of backends and an HTTP server
    instance.

    Parameters
    ----------
    host : str, optional
        Bind address. Default ``"127.0.0.1"``.
    port : int, optional
        Port to listen on. ``0`` lets the OS pick — useful in tests.
    allow_load : bool, optional
        If ``True``, the ``/v1/load`` endpoint is enabled and the
        server will lazily load HuggingFace datasets via
        ``fast_load_iterable_dataset`` on demand. Default ``False`` —
        callers must pre-register backends via ``register()``.

    Typical usage in tests:

    >>> srv = DatasetServer(host="127.0.0.1", port=0)
    >>> srv.register("toy", InMemoryBackend(...))
    >>> srv.start()
    >>> url = srv.url
    >>> ...
    >>> srv.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        allow_load: bool = False,
    ):
        self._host = host
        self._port = port
        self._backends: dict[str, IterableDatasetBackend] = {}
        # Tracks which handles came from /v1/load so we can return
        # informative metadata.
        self._load_args: dict[str, Dict[str, Any]] = {}
        self._allow_load = allow_load
        self._load_lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def allow_load(self) -> bool:
        return self._allow_load

    def register(self, handle: str, backend: IterableDatasetBackend) -> None:
        if not handle or "/" in handle:
            raise ValueError(f"Invalid handle: {handle!r}")
        self._backends[handle] = backend

    def unregister(self, handle: str) -> None:
        self._backends.pop(handle, None)
        self._load_args.pop(handle, None)

    def get(self, handle: str) -> Optional[IterableDatasetBackend]:
        return self._backends.get(handle)

    def list_handles(self) -> list[str]:
        return sorted(self._backends.keys())

    def load_args_for(self, handle: str) -> Optional[Dict[str, Any]]:
        return self._load_args.get(handle)

    def load_on_demand(self, load_args: Dict[str, Any]) -> str:
        """
        Look up (or load + cache) a backend for ``load_args``.
        Returns the handle. Raises if ``allow_load`` is False or
        the loader fails.
        """
        if not self._allow_load:
            raise PermissionError(
                "load-on-demand is disabled on this server (allow_load=False)"
            )
        handle = _canonical_handle(load_args)
        with self._load_lock:
            if handle in self._backends:
                return handle
            # Local import — keeps `tools.dataset_server` importable
            # even when the loader isn't available (e.g. lightweight
            # client-only environments).
            from forgather.ml.datasets.fast_hf_loader import (
                _local_load_iterable_dataset,
            )

            logger.info("loading dataset on demand: %s", load_args)
            ds = _local_load_iterable_dataset(**load_args)
            # ds is a ComposableIterableDataset over an ArrowBackend —
            # serve the bare backend so the client can apply its own
            # slice / shard / map on top.
            self._backends[handle] = ds.backend
            self._load_args[handle] = dict(load_args)
        return handle

    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("Server not started")
        host, port = self._httpd.server_address[0], self._httpd.server_address[1]
        return f"http://{host}:{port}"

    @property
    def port(self) -> int:
        if self._httpd is None:
            return self._port
        return self._httpd.server_address[1]

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        if self._httpd is not None:
            raise RuntimeError("Server already started")
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self._host, self._port), handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name=f"DatasetServer:{self.port}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Dataset server listening on %s (allow_load=%s)",
            self.url,
            self._allow_load,
        )

    def stop(self, timeout: float = 5.0) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._httpd = None
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def _make_handler(server: DatasetServer):
    """Build a BaseHTTPRequestHandler subclass closed over `server`."""

    class Handler(BaseHTTPRequestHandler):
        # Quieten the default request-log noise for tests.
        def log_message(self, format, *args):
            logger.debug("%s - " + format, self.address_string(), *args)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": message})

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        # --- GET ---

        def do_GET(self):  # noqa: N802
            try:
                self._handle_get()
            except (BrokenPipeError, ConnectionResetError):
                # Normal — client closed before consuming the full stream.
                logger.debug("client disconnected mid-stream")
            except Exception as exc:
                logger.exception("internal error handling %s", self.path)
                try:
                    self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                except Exception:
                    pass

        def _handle_get(self) -> None:
            parts = urlsplit(self.path)
            segs = [s for s in parts.path.split("/") if s]
            if segs[:1] != ["v1"]:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown path: {self.path}")
                return

            # /v1/datasets
            if segs == ["v1", "datasets"]:
                self._send_json(HTTPStatus.OK, {"handles": server.list_handles()})
                return

            # /v1/datasets/{handle}/{action}
            if len(segs) == 4 and segs[1] == "datasets":
                self._handle_dataset_action(segs[2], segs[3], parts.query)
                return

            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown path: {self.path}")

        def _handle_dataset_action(self, handle: str, action: str, query: str) -> None:
            backend = server.get(handle)
            if backend is None:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown handle: {handle}")
                return

            qs = parse_qs(query)

            if action == "length":
                length = len(backend)
                logger.info(
                    "GET length handle=%s -> %d",
                    handle,
                    length,
                )
                self._send_json(HTTPStatus.OK, {"length": length})
                return

            if action == "iter":
                seed = qs.get("seed", [None])[0]
                position = qs.get("position", ["0"])[0]
                limit = qs.get("limit", [None])[0]
                try:
                    seed_v = int(seed) if seed is not None and seed != "" else None
                    pos_v = int(position)
                    limit_v = int(limit) if limit is not None and limit != "" else None
                except ValueError as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, f"Invalid query: {exc}")
                    return
                logger.info(
                    "GET iter handle=%s seed=%s position=%d limit=%s",
                    handle,
                    seed_v,
                    pos_v,
                    limit_v,
                )
                self._stream_iter(backend, seed_v, pos_v, limit_v, handle)
                return

            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown action: {action}")

        # --- POST ---

        def do_POST(self):  # noqa: N802
            try:
                self._handle_post()
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("client disconnected mid-stream")
            except Exception as exc:
                logger.exception("internal error handling %s", self.path)
                try:
                    self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                except Exception:
                    pass

        def _handle_post(self) -> None:
            parts = urlsplit(self.path)
            segs = [s for s in parts.path.split("/") if s]

            if segs == ["v1", "load"]:
                self._handle_load()
                return

            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown path: {self.path}")

        def _handle_load(self) -> None:
            if not server.allow_load:
                self._send_error(
                    HTTPStatus.FORBIDDEN,
                    "load-on-demand is disabled on this server",
                )
                return
            try:
                body = self._read_json_body()
            except json.JSONDecodeError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}")
                return

            load_args = {k: body.get(k) for k in _LOAD_FIELDS if k in body}
            if "path" not in load_args or not load_args["path"]:
                self._send_error(
                    HTTPStatus.BAD_REQUEST, "Missing required field 'path'"
                )
                return

            try:
                handle = server.load_on_demand(load_args)
            except Exception as exc:
                logger.exception("load_on_demand failed for %s", load_args)
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return

            backend = server.get(handle)
            length = len(backend) if backend is not None else 0
            logger.info(
                "POST load -> handle=%s length=%d args=%s",
                handle,
                length,
                load_args,
            )
            self._send_json(
                HTTPStatus.OK,
                {
                    "handle": handle,
                    "length": length,
                    "load_args": server.load_args_for(handle) or {},
                },
            )

        # --- streaming helpers ---

        def _stream_iter(
            self,
            backend: IterableDatasetBackend,
            seed: Optional[int],
            position: int,
            limit: Optional[int],
            handle: str = "",
        ) -> None:
            view = backend
            if seed is not None:
                view = view.shuffle(seed)
            if position:
                view = view.seek(position)

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            count = 0
            disconnected = False
            try:
                for example in view:
                    if limit is not None and count >= limit:
                        break
                    line = json.dumps(_to_jsonable(example)).encode("utf-8") + b"\n"
                    self._write_chunk(line)
                    count += 1
                self._write_chunk(b"")
            except (BrokenPipeError, ConnectionResetError):
                disconnected = True

            if disconnected:
                logger.info(
                    "iter handle=%s done: client disconnected after %d examples",
                    handle,
                    count,
                )
            else:
                logger.info(
                    "iter handle=%s done: streamed %d examples",
                    handle,
                    count,
                )

        def _write_chunk(self, data: bytes) -> None:
            """Write an HTTP chunked-transfer chunk."""
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
            if data:
                self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

    return Handler


def run_server(
    host: str = "127.0.0.1",
    port: int = 8766,
    backends: Optional[dict[str, IterableDatasetBackend]] = None,
    allow_load: bool = False,
) -> None:
    """Run a dataset server in the foreground (blocks).

    Default port is 8766 because 8765 is the forgather orchestration
    server's port.
    """
    srv = DatasetServer(host=host, port=port, allow_load=allow_load)
    for handle, backend in (backends or {}).items():
        srv.register(handle, backend)
    srv.start()
    try:
        srv._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        srv.stop()
