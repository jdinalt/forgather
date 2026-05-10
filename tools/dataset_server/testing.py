"""
Test helpers for the forgather dataset server.

`TestServer` wraps a uvicorn ``Server`` running in a daemon thread so
unit tests can drive the real FastAPI app over real HTTP on an
OS-assigned port. Mirrors the interface (``start`` / ``stop`` /
``url`` / ``register``) the previous PoC `DatasetServer` exposed,
so existing test code keeps compiling.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import uvicorn

from forgather.ml.datasets.iterable_backend import IterableDatasetBackend

from .app import create_app
from .state import ServerState


class TestServer:
    """uvicorn-in-a-thread helper for tests.

    Parameters
    ----------
    host
        Bind address. Default loopback.
    port
        ``0`` lets the OS pick — use ``self.url`` after ``start()`` to
        discover the assigned port.
    auth_token
        If non-empty, every gated endpoint requires
        ``Authorization: Bearer <token>``. ``None`` (or empty) — like
        passing ``--no-auth`` to the standalone server — disables auth.
    state
        Optional pre-built state container. If omitted, a fresh
        ``ServerState`` with default policy (HF cache enabled, paths
        disabled, downloads disabled, no locals) is created.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        auth_token: Optional[str] = None,
        state: Optional[ServerState] = None,
    ):
        self._host = host
        self._requested_port = port
        self._state = state or ServerState()
        self._auth_token = auth_token
        # Build the app eagerly so tests can drive it via the ASGI
        # interface as well if they want; uvicorn just needs a callable.
        self.app = create_app(self._state, auth_token=auth_token)
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    # ----- server lifecycle -----

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("TestServer already started")
        config = uvicorn.Config(
            self.app,
            host=self._host,
            port=self._requested_port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            name=f"DatasetTestServer:{self._requested_port}",
            daemon=True,
        )
        self._thread.start()
        # Wait until the server is bound so .url returns a usable
        # value. uvicorn flips `started` True after the first poll.
        deadline = time.monotonic() + 5.0
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("TestServer failed to start within 5s")
            time.sleep(0.02)

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._server = None
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ----- introspection -----

    @property
    def port(self) -> int:
        if self._server is None:
            return self._requested_port
        # uvicorn binds the socket before flipping `started`; consult
        # its server list to find the actual port (matters when
        # requested_port=0).
        for srv in self._server.servers or []:
            for sock in srv.sockets:
                return int(sock.getsockname()[1])
        return self._requested_port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    @property
    def state(self) -> ServerState:
        return self._state

    @property
    def auth_token(self) -> Optional[str]:
        return self._auth_token

    # ----- handle registry passthrough (test convenience) -----

    def register(
        self,
        handle: str,
        backend: IterableDatasetBackend,
        load_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._state.register(handle, backend, load_args=load_args, source="registered")

    def unregister(self, handle: str) -> None:
        self._state.unregister(handle)

    def get(self, handle: str) -> Optional[IterableDatasetBackend]:
        return self._state.get(handle)

    def list_handles(self) -> list[str]:
        return self._state.list_handles()
