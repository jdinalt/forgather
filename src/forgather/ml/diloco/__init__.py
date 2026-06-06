"""
DiLoCo (Distributed Local SGD with Communication) for Forgather.

Enables distributed training across multiple heterogeneous machines on a LAN
using asynchronous Local-SGD. Each machine runs any existing Forgather trainer
locally and periodically synchronizes with a central parameter server.

Key components:
- DiLoCoServer: Central HTTP parameter server holding global model state
- DiLoCoClient: HTTP client for server communication
- DiLoCoWorker: Composable wrapper that hooks into any optimizer for periodic sync
- FragmentManager: Splits model into fragments for streaming sync
- HealthMonitor: Background worker health checker for fault tolerance

Heavy symbols (everything that pulls in ``torch``) are loaded lazily via
PEP 562 ``__getattr__`` so that importing a lightweight sibling module
(``forgather.ml.diloco.auth``, ``forgather.ml.diloco.env``) — as the CLI's
argument-parser modules do — does not drag ``torch`` into the process. This
keeps ``forgather --help`` / ``forgather ls`` fast. The public import surface
is unchanged: ``from forgather.ml.diloco import DiLoCoClient`` still works.
"""

import importlib
from typing import TYPE_CHECKING

# Lightweight, stdlib-only — safe to import eagerly and frequently hot-pathed.
from .env import diloco_is_enabled, diloco_server_addr

# name -> submodule (relative). Most pull in torch; load on first access only.
# (``sync_backend`` is torch-free at import, but is deferred here too for a
# uniform public surface.)
_LAZY = {
    "DiLoCoClient": ".client",
    "DiLoCoServer": ".server",
    "DiLoCoWorker": ".worker",
    "FragmentManager": ".fragments",
    "HealthMonitor": ".health",
    "OuterSyncBackend": ".sync_backend",
    "HttpStarBackend": ".sync_backend",
    "SyncResult": ".sync_backend",
}

__all__ = [
    "DiLoCoServer",
    "DiLoCoClient",
    "DiLoCoWorker",
    "FragmentManager",
    "HealthMonitor",
    "OuterSyncBackend",
    "HttpStarBackend",
    "SyncResult",
    "diloco_is_enabled",
    "diloco_server_addr",
]


def __getattr__(name):
    """Lazily resolve heavy symbols on first attribute access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    # Static type checkers / IDEs resolve the real symbols without the
    # runtime import cost.
    from .client import DiLoCoClient
    from .fragments import FragmentManager
    from .health import HealthMonitor
    from .server import DiLoCoServer
    from .sync_backend import HttpStarBackend, OuterSyncBackend, SyncResult
    from .worker import DiLoCoWorker
