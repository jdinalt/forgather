"""
forgather dataset server.

A Uvicorn + FastAPI server that exposes the local
``fast_load_iterable_dataset`` machinery over HTTP. Clients route
through it transparently by setting the
``FORGATHER_DATASET_SERVER`` environment variable; see
``RemoteBackend`` and ``_remote_load_iterable_dataset`` on the
client side, and the matching CLI in
``src/forgather/cli/dataset_server.py``.

Out of scope here: a web UI, LRU eviction of cached backends,
compression on the NDJSON stream, request rate limiting.
"""

from __future__ import annotations

from .app import create_app
from .auth import (
    read_standalone_token,
    standalone_token_file,
    write_standalone_token,
)
from .state import (
    HandleEntry,
    PolicyError,
    ServerState,
    canonical_handle,
)

__all__ = [
    "create_app",
    "ServerState",
    "HandleEntry",
    "PolicyError",
    "canonical_handle",
    "read_standalone_token",
    "standalone_token_file",
    "write_standalone_token",
]


def __getattr__(name: str):
    """Lazy ``TestServer`` import so the production server doesn't
    pay the uvicorn import cost just to start. Tests do
    ``from dataset_server import TestServer``."""
    if name == "TestServer":
        from .testing import TestServer

        return TestServer
    raise AttributeError(name)
