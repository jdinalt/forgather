"""
Outer-synchronization backend abstraction for DiLoCo.

The DiLoCo outer step is *how* a worker and its peers agree on the next global
parameter state once per ``sync_every`` local steps: each worker contributes a
pseudo-gradient and ends up holding the averaged-and-outer-optimized weights.
Today the only mechanism is ``torch.save``-pickle over an HTTP star to a central
parameter server (:class:`~forgather.ml.diloco.client.DiLoCoClient` ↔
:class:`~forgather.ml.diloco.server.DiLoCoServer`). This module introduces the
seam that lets that mechanism be swapped — for gRPC / Arrow Flight, NCCL/gloo
collectives, or a shared-memory / RDMA parameter region — without touching the
worker's training-loop orchestration.

The seam is deliberately drawn at the **outer step**, not at a byte channel. A
byte channel (``upload(bytes)`` / ``download() -> bytes``) bakes in three
assumptions that some backends break: that a central server is authoritative,
that the worker receives weights *over a wire*, and that the outer optimizer
runs on the server. A backend that all-reduces pseudo-gradients and runs a
*replicated* outer optimizer locally returns weights it computed itself; a
shared-memory backend returns a *view* into a region rather than freshly
deserialized tensors. :class:`OuterSyncBackend` hides all of that behind
"contribute a pseudo-gradient, receive the agreed next global params," so the
worker's ``compute → synchronize → apply → broadcast`` flow is unchanged.

This module is the step-1 strangler refactor (see issue #154): it defines the
interface and reimplements *current* HTTP behavior behind :class:`HttpStarBackend`,
byte-for-byte unchanged. It does not add any new transport.

The seam is intentionally narrow:

- The **coordination plane** — heartbeat, ``/info`` negotiation, model-def
  download, control commands, work-unit dispatch — is not part of this seam; the
  worker-process coordination has its own surface
  (:class:`~forgather.ml.diloco.coordinator.CoordinatorClient`).
- The backend owns its **wire representation**.
  :meth:`~forgather.ml.diloco.param_view.ParamView.compute_pseudograds` returns
  the raw pseudo-gradient (live model dtype); :class:`HttpStarBackend` applies
  the upload cast (``upload_dtype`` / ``upload_sr``, with optional stochastic
  rounding) via :func:`~forgather.ml.diloco.wire_cast.cast_for_upload` before
  sending, so a non-HTTP backend can define its own representation (shared-mem =
  no cast; collective = packed fp8/fp4). The **download** cast is the server's
  (the central parameter authority's) concern and stays server-side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional

if TYPE_CHECKING:
    import torch

    from .client import DiLoCoClient

    # A state dict crossing the backend boundary. Tensors may be views into
    # backend-owned storage (a shared-memory region), so callers must copy out
    # rather than mutate in place — which the worker does via
    # ``ParamView.apply_global``'s ``.to(dtype, device)`` copy.
    StateDict = Dict[str, "torch.Tensor"]
    OuterOptFactory = Callable[..., object]
else:  # pragma: no cover - typing aliases only
    StateDict = dict
    OuterOptFactory = object


@dataclass
class SyncResult:
    """Outcome of one outer-synchronization round.

    ``params`` is the agreed next global parameter set the worker should apply,
    or ``None`` when the round did not commit. ``committed`` says whether this
    round produced a new global state — ``False`` means "skip this round" (a
    backend-internal failure that did not raise, e.g. a quorum that dropped this
    worker). ``round`` is the backend's sync-round counter when it exposes one,
    else ``None`` (the worker tracks its own ``_sync_count`` regardless).

    For the HTTP star backend ``synchronize`` either returns ``committed=True``
    with params, or *raises* ``ConnectionError`` (so the worker's existing
    retry/reconnect loop drives recovery); it does not produce ``committed=False``.
    That path exists for fault-tolerant collective backends (issue #154).

    ``sent_bytes`` / ``recv_bytes`` are the on-wire sizes of this round's
    upload / download, reported by the backend (which owns the wire
    representation, so it is the only place that knows the cast size). ``None``
    when the backend does not measure them; the worker then estimates from the
    tensors it holds, which is exact only for a backend that does not cast.
    """

    params: Optional[StateDict]
    committed: bool
    round: Optional[int] = None
    sent_bytes: Optional[int] = None
    recv_bytes: Optional[int] = None


class OuterSyncBackend(ABC):
    """Pluggable mechanism for the DiLoCo outer synchronization step.

    Implementations own *how* a worker joins the synchronization group,
    contributes a pseudo-gradient, and obtains the agreed next global params.
    The worker owns everything local: computing the pseudo-gradient from its
    live model, applying the returned params, broadcasting to its DDP ranks,
    and scheduling (``sync_every``, fragments, async/DyLU).

    Capability flags advertise properties the worker (or a future trainer
    callback) must honor rather than assume:

    - ``runs_outer_optimizer`` — where the outer optimizer lives:
      ``"central"`` (a server holds it; HTTP today), ``"replicated"`` (each
      worker runs an identical deterministic copy after an all-reduce), or
      ``"shared-region"`` (it operates in place on a shared parameter region).
    - ``supports_async`` — whether the backend permits apply-immediately /
      DyLU-style asynchronous sync. Bare collectives are strictly synchronous.
    - ``fault_tolerant`` — whether the backend transparently survives a *peer*
      failing mid-round (the quorum / skip-step axis). This is orthogonal to
      transport-level retry, which is owned by the caller, not the backend.
    """

    runs_outer_optimizer: Literal["central", "replicated", "shared-region"] = "central"
    supports_async: bool = False
    fault_tolerant: bool = False

    @abstractmethod
    def join(
        self,
        *,
        worker_id: str,
        worker_info: Optional[dict] = None,
        outer_opt_factory: Optional[OuterOptFactory] = None,
    ) -> StateDict:
        """Join the synchronization group and return the initial global params.

        Called at startup and on reconnection. ``outer_opt_factory`` lets
        backends that host the outer optimizer locally (replicated /
        shared-region) construct it; backends with a central optimizer (HTTP)
        ignore it.

        Returns the initial global parameter snapshot. (Step-1 simplification:
        the snapshot is returned as a state dict. A later step lets the
        coordinator hand out a *reference* that the backend resolves by the
        cheapest local path — shared-mem map, disk/object-store read, collective
        broadcast — with byte-serving as the fallback; see issue #154.)
        """

    @abstractmethod
    def synchronize(self, *, worker_id: str, pseudograds: StateDict) -> SyncResult:
        """Contribute a full-model pseudo-gradient and get the next global params.

        Blocks until the round completes (or, for synchronous backends, until
        all expected peers contribute). May raise ``ConnectionError`` to signal
        a recoverable transport failure for the caller's retry loop.
        """

    @abstractmethod
    def synchronize_fragment(
        self, *, worker_id: str, fragment_id: int, pseudograds: StateDict
    ) -> SyncResult:
        """Streaming-sync variant: contribute one fragment's pseudo-gradient.

        Typically invoked from a background thread so the per-fragment round
        overlaps continued local training.
        """

    @abstractmethod
    def current_global_params(self) -> StateDict:
        """Fetch the current global params without contributing (late-join/recovery)."""

    @abstractmethod
    def leave(self, *, worker_id: str) -> None:
        """Leave the synchronization group (best-effort; errors are swallowed)."""


def _wire_bytes(d: StateDict) -> int:
    """On-wire size of a state dict: sum of tensor byte sizes."""
    return sum(t.numel() * t.element_size() for t in d.values())


class HttpStarBackend(OuterSyncBackend):
    """Reference backend: the current HTTP central-parameter-server star.

    A thin adapter over :class:`~forgather.ml.diloco.client.DiLoCoClient` — all
    HTTP, TLS, bulk-listener and serialization logic stays inside the client.
    This backend exists to prove the seam: it reproduces today's behavior
    exactly while making the worker backend-agnostic.

    It owns the **wire representation** of the upload leg: raw pseudo-gradients
    handed to ``synchronize`` are cast to ``upload_dtype`` (with optional
    stochastic rounding) here, immediately before they are sent. The download
    cast is the server's (the central parameter authority's) concern.
    """

    runs_outer_optimizer = "central"
    supports_async = True
    # Peer/quorum fault tolerance: the HTTP server's dynamic barrier averages
    # whoever submitted and evicts the dead, so a peer failing mid-round does
    # not break the round. (Transport-level failures are orthogonal — they
    # surface as ConnectionError and are retried by the caller, not here.)
    fault_tolerant = True

    def __init__(
        self,
        client: "DiLoCoClient",
        upload_dtype: str = "bf16",
        upload_sr: bool = False,
    ):
        self.client = client
        self.upload_dtype = upload_dtype
        self.upload_sr = bool(upload_sr)

    def _cast_upload(self, pseudograds: StateDict) -> StateDict:
        # Lazy import keeps this module torch-free at import time.
        from .wire_cast import cast_for_upload

        return {
            name: cast_for_upload(pg, self.upload_dtype, self.upload_sr)
            for name, pg in pseudograds.items()
        }

    def join(
        self,
        *,
        worker_id: str,
        worker_info: Optional[dict] = None,
        outer_opt_factory: Optional[OuterOptFactory] = None,
    ) -> StateDict:
        # The central server owns the outer optimizer; the factory is ignored.
        return self.client.register(worker_id, worker_info)

    def synchronize(self, *, worker_id: str, pseudograds: StateDict) -> SyncResult:
        wire = self._cast_upload(pseudograds)
        params = self.client.submit_pseudogradients(worker_id, wire)
        # The server does not return its round counter on this leg; the worker
        # tracks its own. ConnectionError propagates to the worker's retry loop.
        return SyncResult(
            params=params,
            committed=True,
            round=None,
            sent_bytes=_wire_bytes(wire),
            recv_bytes=_wire_bytes(params),
        )

    def synchronize_fragment(
        self, *, worker_id: str, fragment_id: int, pseudograds: StateDict
    ) -> SyncResult:
        wire = self._cast_upload(pseudograds)
        params = self.client.submit_fragment_pseudogradients(
            worker_id, fragment_id, wire
        )
        return SyncResult(
            params=params,
            committed=True,
            round=None,
            sent_bytes=_wire_bytes(wire),
            recv_bytes=_wire_bytes(params),
        )

    def current_global_params(self) -> StateDict:
        return self.client.get_global_params()

    def leave(self, *, worker_id: str) -> None:
        self.client.deregister(worker_id)
