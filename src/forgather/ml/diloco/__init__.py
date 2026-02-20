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
"""

from .client import DiLoCoClient
from .fragments import FragmentManager
from .server import DiLoCoServer
from .worker import DiLoCoWorker

__all__ = ["DiLoCoServer", "DiLoCoClient", "DiLoCoWorker", "FragmentManager"]
