"""
Proof-of-concept dataset server for the IterableDatasetBackend
abstraction.

Hosts a registry of named backends and exposes them over a tiny
HTTP API so a `RemoteIterableDataset` proxy on another node can iterate
without downloading the underlying data.

Out of scope for this iteration: authentication, multi-tenant
isolation, server-side caching policy, protocol negotiation. This
layer exists to validate that the backend interface is sufficient for
remote consumption.
"""

from .server import DatasetServer, run_server

__all__ = ["DatasetServer", "run_server"]
