"""Tests for the OuterSyncBackend seam (issue #154, step 1).

Two things are verified here:

1. ``HttpStarBackend`` faithfully delegates each tensor-leg call to the wrapped
   ``DiLoCoClient`` and shapes ``SyncResult`` correctly (and lets
   ``ConnectionError`` propagate for the worker's retry loop).
2. ``DiLoCoWorker`` drives a full ``join -> compute -> synchronize -> apply``
   round through an injected backend with **zero HTTP** — proving the worker is
   backend-agnostic, which is the whole point of the refactor.
"""

import pytest
import torch
import torch.nn as nn

from forgather.ml.diloco.sync_backend import (
    HttpStarBackend,
    OuterSyncBackend,
    SyncResult,
)
from forgather.ml.diloco.worker import DiLoCoWorker


class TinyModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class RecordingClient:
    """Stand-in for DiLoCoClient that records calls and returns canned params."""

    def __init__(self, ret=None, raise_on_submit=None):
        self.calls = []
        self.ret = ret if ret is not None else {"linear1.weight": torch.zeros(2, 2)}
        self.raise_on_submit = raise_on_submit

    def register(self, worker_id, worker_info=None):
        self.calls.append(("register", worker_id, worker_info))
        return self.ret

    def submit_pseudogradients(self, worker_id, pseudograds):
        self.calls.append(("submit", worker_id, pseudograds))
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        return self.ret

    def submit_fragment_pseudogradients(self, worker_id, fragment_id, pseudograds):
        self.calls.append(("submit_fragment", worker_id, fragment_id, pseudograds))
        return self.ret

    def get_global_params(self):
        self.calls.append(("get_global_params",))
        return self.ret

    def deregister(self, worker_id):
        self.calls.append(("deregister", worker_id))


class TestHttpStarBackendDelegation:
    def test_capability_flags(self):
        backend = HttpStarBackend(RecordingClient())
        assert backend.runs_outer_optimizer == "central"
        assert backend.supports_async is True
        # advisory; the HTTP server's dynamic barrier survives peer churn
        assert backend.fault_tolerant is True

    def test_join_delegates_to_register(self):
        client = RecordingClient(ret={"w": torch.ones(3)})
        backend = HttpStarBackend(client)
        out = backend.join(worker_id="w0", worker_info={"k": "v"})
        assert out is client.ret
        assert client.calls == [("register", "w0", {"k": "v"})]

    def test_synchronize_delegates_and_shapes_result(self):
        client = RecordingClient(ret={"w": torch.ones(3)})
        backend = HttpStarBackend(client)
        pg = {"w": torch.zeros(3)}
        result = backend.synchronize(worker_id="w0", pseudograds=pg)
        assert isinstance(result, SyncResult)
        assert result.params is client.ret
        assert result.committed is True
        assert result.round is None
        assert client.calls == [("submit", "w0", pg)]

    def test_synchronize_propagates_connection_error(self):
        # The worker's retry/reconnect loop relies on ConnectionError surfacing.
        client = RecordingClient(raise_on_submit=ConnectionError("down"))
        backend = HttpStarBackend(client)
        with pytest.raises(ConnectionError):
            backend.synchronize(worker_id="w0", pseudograds={"w": torch.zeros(3)})

    def test_synchronize_fragment_delegates(self):
        client = RecordingClient(ret={"w": torch.ones(3)})
        backend = HttpStarBackend(client)
        pg = {"w": torch.zeros(3)}
        result = backend.synchronize_fragment(
            worker_id="w0", fragment_id=2, pseudograds=pg
        )
        assert result.params is client.ret
        assert result.committed is True
        assert client.calls == [("submit_fragment", "w0", 2, pg)]

    def test_current_global_params_delegates(self):
        client = RecordingClient(ret={"w": torch.ones(3)})
        backend = HttpStarBackend(client)
        assert backend.current_global_params() is client.ret
        assert client.calls == [("get_global_params",)]

    def test_leave_delegates_to_deregister(self):
        client = RecordingClient()
        backend = HttpStarBackend(client)
        backend.leave(worker_id="w0")
        assert client.calls == [("deregister", "w0")]


class FakeBackend(OuterSyncBackend):
    """In-memory backend: no HTTP, no server. Returns canned params."""

    runs_outer_optimizer = "replicated"
    supports_async = False
    fault_tolerant = False

    def __init__(self, init_params, sync_params):
        self.init_params = init_params
        self.sync_params = sync_params
        self.calls = []

    def join(self, *, worker_id, worker_info=None, outer_opt_factory=None):
        self.calls.append("join")
        return self.init_params

    def synchronize(self, *, worker_id, pseudograds):
        self.calls.append("synchronize")
        return SyncResult(params=self.sync_params, committed=True, round=1)

    def synchronize_fragment(self, *, worker_id, fragment_id, pseudograds):
        self.calls.append("synchronize_fragment")
        return SyncResult(params=self.sync_params, committed=True)

    def current_global_params(self):
        return self.sync_params

    def leave(self, *, worker_id):
        self.calls.append("leave")


class _UncommittedBackend(OuterSyncBackend):
    """join() seeds init params; synchronize() returns committed=False."""

    runs_outer_optimizer = "replicated"
    supports_async = False
    fault_tolerant = True

    def __init__(self, init_params, would_be_params):
        self.init_params = init_params
        self.would_be_params = would_be_params

    def join(self, *, worker_id, worker_info=None, outer_opt_factory=None):
        return self.init_params

    def synchronize(self, *, worker_id, pseudograds):
        return SyncResult(params=self.would_be_params, committed=False)

    def synchronize_fragment(self, *, worker_id, fragment_id, pseudograds):
        return SyncResult(params=self.would_be_params, committed=False)

    def current_global_params(self):
        return self.init_params

    def leave(self, *, worker_id):
        pass


class _AlwaysFailsBackend(OuterSyncBackend):
    """join() works; synchronize() always raises ConnectionError."""

    runs_outer_optimizer = "central"
    supports_async = True
    fault_tolerant = False

    def __init__(self, init_params):
        self.init_params = init_params
        self.join_calls = 0

    def join(self, *, worker_id, worker_info=None, outer_opt_factory=None):
        self.join_calls += 1
        return self.init_params

    def synchronize(self, *, worker_id, pseudograds):
        raise ConnectionError("server down")

    def synchronize_fragment(self, *, worker_id, fragment_id, pseudograds):
        raise ConnectionError("server down")

    def current_global_params(self):
        return self.init_params

    def leave(self, *, worker_id):
        pass


def _make_worker(model, backend, **kw):
    return DiLoCoWorker(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        server_addr="dummy:8512",
        sync_every=5,
        heartbeat_interval=0,  # no heartbeat thread -> no client traffic
        bf16_comm=False,
        backend=backend,
        **kw,
    )


class TestWorkerSkipSemantics:
    """The two branches the refactor adds to _sync(): a non-committed round and
    an exhausted-retries (result is None) round must both skip apply without
    raising, while still advancing the sync counter."""

    def test_uncommitted_round_skips_apply(self):
        torch.manual_seed(0)
        model = TinyModel(dim=4)
        init = {k: v.detach().clone() for k, v in model.state_dict().items()}
        would_be = {k: torch.zeros_like(v) for k, v in init.items()}
        worker = _make_worker(model, _UncommittedBackend(init, would_be))

        worker.start()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        worker._sync()  # committed=False -> must NOT apply would_be (zeros)

        for name, p in model.named_parameters():
            assert torch.equal(p.data, before[name]), name  # unchanged, not zeros
        assert worker._sync_count == 1  # counter still advances
        worker.stop()

    def test_exhausted_retries_skips_apply(self):
        torch.manual_seed(0)
        model = TinyModel(dim=4)
        init = {k: v.detach().clone() for k, v in model.state_dict().items()}
        backend = _AlwaysFailsBackend(init)
        worker = _make_worker(model, backend, max_sync_retries=0)

        worker.start()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        worker._sync()  # all attempts raise -> result None -> skip, no exception

        for name, p in model.named_parameters():
            assert torch.equal(p.data, before[name]), name
        assert worker._sync_count == 1
        assert backend.join_calls == 1  # start() only; no reconnect at retries=0
        worker.stop()


class TestWorkerIsBackendAgnostic:
    def test_default_backend_is_http_star_wrapping_client(self):
        model = TinyModel(dim=4)
        worker = DiLoCoWorker(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            server_addr="dummy:8512",
        )
        assert isinstance(worker.backend, HttpStarBackend)
        # wraps the same client the worker built
        assert worker.backend.client is worker.client

    def test_injected_backend_drives_full_round_without_http(self):
        torch.manual_seed(0)
        model = TinyModel(dim=4)
        # Initial global params = the model's own weights (join is a no-op apply).
        init_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # Post-sync params = all zeros; after a sync the model must equal these.
        sync_params = {k: torch.zeros_like(v) for k, v in init_params.items()}
        backend = FakeBackend(init_params, sync_params)

        worker = DiLoCoWorker(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            server_addr="dummy:8512",
            sync_every=100,
            heartbeat_interval=0,  # no heartbeat thread -> no client traffic
            bf16_comm=False,
            backend=backend,
        )

        worker.start()
        assert "join" in backend.calls

        # Drive one synchronization round directly (no optimizer.step needed).
        worker._sync()
        assert "synchronize" in backend.calls

        # The model now holds the backend's post-sync params.
        for name, p in model.named_parameters():
            assert torch.equal(p.data, sync_params[name]), name

        worker.stop()
        assert "leave" in backend.calls
