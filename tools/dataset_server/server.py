"""
HTTP dataset server (proof of concept).

Endpoints
---------

- ``GET /v1/datasets`` — list registered handles.
- ``GET /v1/datasets/{handle}/length`` — JSON ``{"length": int}``.
- ``GET /v1/datasets/{handle}/iter?seed=<int>&position=<int>&limit=<int>``
  — newline-delimited JSON stream of examples. ``seed`` and ``limit``
  are optional; ``position`` defaults to 0.

The server is intentionally stateless with respect to clients — every
``/iter`` call carries the seed and position to start from. This means
multiple clients can hit the same handle concurrently without
trampling each other's iteration state.

Backend handling
----------------

The handle resolves to an `IterableDatasetBackend` registered via
``DatasetServer.register``. For each ``/iter`` request the server
applies ``backend.shuffle(seed)`` (if seed given) and
``backend.seek(position)`` to obtain a fresh iteration view, then
streams examples from it. The original registered backend is treated
as immutable.
"""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
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


class DatasetServer:
    """
    Container that owns a registry of backends and an HTTP server
    instance.

    Typical usage in tests:

    >>> srv = DatasetServer(host="127.0.0.1", port=0)
    >>> srv.register("toy", InMemoryBackend(...))
    >>> srv.start()
    >>> url = srv.url  # includes the OS-assigned port
    >>> ...
    >>> srv.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._host = host
        self._port = port
        self._backends: dict[str, IterableDatasetBackend] = {}
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def register(self, handle: str, backend: IterableDatasetBackend) -> None:
        if not handle or "/" in handle:
            raise ValueError(f"Invalid handle: {handle!r}")
        self._backends[handle] = backend

    def unregister(self, handle: str) -> None:
        self._backends.pop(handle, None)

    def get(self, handle: str) -> Optional[IterableDatasetBackend]:
        return self._backends.get(handle)

    def list_handles(self) -> list[str]:
        return sorted(self._backends.keys())

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
        logger.info("Dataset server listening on %s", self.url)

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

        def do_GET(self):  # noqa: N802
            try:
                self._handle_get()
            except BrokenPipeError:
                # Client disconnected mid-stream — normal for early
                # termination. Don't log a stack trace.
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
            if segs[:2] != ["v1", "datasets"]:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown path: {self.path}")
                return

            # /v1/datasets
            if len(segs) == 2:
                self._send_json(HTTPStatus.OK, {"handles": server.list_handles()})
                return

            # /v1/datasets/{handle}/{action}
            if len(segs) != 4:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown path: {self.path}")
                return

            handle, action = segs[2], segs[3]
            backend = server.get(handle)
            if backend is None:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown handle: {handle}")
                return

            qs = parse_qs(parts.query)

            if action == "length":
                self._send_json(HTTPStatus.OK, {"length": len(backend)})
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
                self._stream_iter(backend, seed_v, pos_v, limit_v)
                return

            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown action: {action}")

        def _stream_iter(
            self,
            backend: IterableDatasetBackend,
            seed: Optional[int],
            position: int,
            limit: Optional[int],
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
            try:
                for example in view:
                    if limit is not None and count >= limit:
                        break
                    line = json.dumps(_to_jsonable(example)).encode("utf-8") + b"\n"
                    self._write_chunk(line)
                    count += 1
                # Terminating chunk.
                self._write_chunk(b"")
            except BrokenPipeError:
                logger.debug("client disconnected after %d examples", count)

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
    port: int = 8765,
    backends: Optional[dict[str, IterableDatasetBackend]] = None,
) -> None:
    """Run a dataset server in the foreground (blocks)."""
    srv = DatasetServer(host=host, port=port)
    for handle, backend in (backends or {}).items():
        srv.register(handle, backend)
    srv.start()
    try:
        srv._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        srv.stop()
