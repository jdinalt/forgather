"""
Coordinator surface for DiLoCo.

#154's role decomposition splits the monolithic "server" into a **coordinator**
(always present — membership/heartbeat, ``/info`` negotiation, model-def staging,
control, work dispatch) and an **optional parameter authority** (the
backend-defined transport extracted as :class:`~forgather.ml.diloco.sync_backend.OuterSyncBackend`).
This module is the coordinator half: a worker holds a :class:`CoordinatorClient`
alongside its :class:`~forgather.ml.diloco.sync_backend.OuterSyncBackend`, so the
two roles are symmetric and a future backend whose parameter authority is *not*
the coordinator (a serverless collective, a shared-memory region) can still
coordinate over the HTTP server while exchanging params elsewhere.

Unlike the transport seam, coordination always speaks HTTP regardless of which
backend moves the tensors, so this is a **concrete facade** over the existing
:class:`~forgather.ml.diloco.client.DiLoCoClient` (which keeps all HTTP / TLS /
auth / bulk-url logic) rather than an ABC. It can be promoted to an ABC if a
non-HTTP coordinator ever appears; that is not needed today.

Scope (intentional): this covers the **worker-process bring-up coordination** —
``heartbeat``, ``get_info``, and model-def ``fetch_model_def``. Two other
coordination surfaces have their own purpose-built clients and are *not* folded
in here: **work-unit dispatch** (``register_dataset`` / ``request_work`` /
``complete_work``), owned by the dataset layer, and **control** (``relay_command``
/ ``save_state`` / ``shutdown`` / ``get_status`` / …), owned by the CLI. They can
adopt this surface later without an interface change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .client import DiLoCoClient


class CoordinatorClient:
    """Worker-facing coordination surface: a thin facade over ``DiLoCoClient``.

    Holds no transport state of its own — every call delegates to the wrapped
    client, so behavior is identical to calling the client directly. Its purpose
    is to give the worker a coordination object distinct from its sync backend,
    completing the coordinator-vs-parameter-authority split.
    """

    def __init__(self, client: "DiLoCoClient"):
        self._client = client

    def heartbeat(
        self,
        worker_id: str,
        steps_per_second: float = 0.0,
        stats: Optional[dict] = None,
        sync_state: Optional[dict] = None,
    ) -> dict:
        """Periodic liveness + speed report; returns server status (sync round,
        worker count, optional relayed command / DyLU ``recommended_sync_every``).

        ``sync_state`` carries the worker's own DiLoCo sync metrics so the server
        can show its progress even when it syncs off-server (shared-memory)."""
        return self._client.heartbeat(
            worker_id,
            steps_per_second=steps_per_second,
            stats=stats,
            sync_state=sync_state,
        )

    def get_info(self) -> dict:
        """Fetch the server's static configuration / negotiation payload (``/info``)."""
        return self._client.get_info()

    def fetch_model_def(self, dest_dir: str) -> str:
        """Download the model-definition bundle into ``dest_dir``; returns its hash."""
        return self._client.fetch_model_def(dest_dir)

    def register(self, worker_id: str, worker_info: Optional[dict] = None):
        """Register the worker with the coordinator for membership / diagnostics.

        Used by a worker whose sync backend does not itself register (its
        ``join`` is a tensor-path op, e.g. shared-memory). The returned global
        params are irrelevant to such a worker — the region is its source of
        truth — but registration makes it visible in the server's worker
        registry and lets its heartbeats be accepted."""
        return self._client.register(worker_id, worker_info)

    def deregister(self, worker_id: str) -> None:
        """Deregister the worker from the coordinator (best-effort)."""
        self._client.deregister(worker_id)
