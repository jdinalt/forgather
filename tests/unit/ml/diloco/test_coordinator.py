"""Tests for the CoordinatorClient surface (issue #154, step 2).

Verifies the facade delegates verbatim to the wrapped DiLoCoClient, and that the
worker holds a coordinator distinct from its sync backend (default wraps its own
client; an injected coordinator is honored).
"""

import time

import torch
import torch.nn as nn

from forgather.ml.diloco.coordinator import CoordinatorClient
from forgather.ml.diloco.sync_backend import OuterSyncBackend, SyncResult
from forgather.ml.diloco.worker import DiLoCoWorker


class RecordingClient:
    """Stand-in for DiLoCoClient recording the coordination calls."""

    def __init__(self):
        self.calls = []
        self.info = {"expected_client_settings": {"sync_every": 10}}
        self.hb = {"sync_round": 3, "num_workers": 1}
        self.model_hash = "abc123"

    def heartbeat(self, worker_id, steps_per_second=0.0, stats=None, sync_state=None):
        self.calls.append(("heartbeat", worker_id, steps_per_second, stats, sync_state))
        return self.hb

    def get_info(self):
        self.calls.append(("get_info",))
        return self.info

    def fetch_model_def(self, dest_dir):
        self.calls.append(("fetch_model_def", dest_dir))
        return self.model_hash

    def register(self, worker_id, worker_info=None):
        self.calls.append(("register", worker_id, worker_info))
        return {}

    def deregister(self, worker_id):
        self.calls.append(("deregister", worker_id))


class TestCoordinatorDelegation:
    def test_heartbeat_forwards_args_and_return(self):
        client = RecordingClient()
        coord = CoordinatorClient(client)
        stats = {"loss": 1.5}
        sync_state = {"sync_count": 3}
        out = coord.heartbeat(
            "w0", steps_per_second=4.2, stats=stats, sync_state=sync_state
        )
        assert out is client.hb
        assert client.calls == [("heartbeat", "w0", 4.2, stats, sync_state)]

    def test_heartbeat_defaults(self):
        client = RecordingClient()
        CoordinatorClient(client).heartbeat("w0")
        assert client.calls == [("heartbeat", "w0", 0.0, None, None)]

    def test_register_deregister_delegate(self):
        client = RecordingClient()
        coord = CoordinatorClient(client)
        coord.register("w0", {"hostname": "h"})
        coord.deregister("w0")
        assert client.calls == [
            ("register", "w0", {"hostname": "h"}),
            ("deregister", "w0"),
        ]

    def test_get_info_delegates(self):
        client = RecordingClient()
        coord = CoordinatorClient(client)
        assert coord.get_info() is client.info
        assert client.calls == [("get_info",)]

    def test_fetch_model_def_delegates(self):
        client = RecordingClient()
        coord = CoordinatorClient(client)
        assert coord.fetch_model_def("/tmp/dest") == client.model_hash
        assert client.calls == [("fetch_model_def", "/tmp/dest")]


class TinyModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class _NoNetBackend(OuterSyncBackend):
    """Backend with no HTTP, so a worker can start()/stop() without a server."""

    runs_outer_optimizer = "central"
    supports_async = True
    fault_tolerant = False

    def __init__(self, init_params):
        self.init_params = init_params

    def join(self, *, worker_id, worker_info=None, outer_opt_factory=None):
        return self.init_params

    def synchronize(self, *, worker_id, pseudograds):
        return SyncResult(params=self.init_params, committed=True)

    def synchronize_fragment(self, *, worker_id, fragment_id, pseudograds):
        return SyncResult(params=self.init_params, committed=True)

    def current_global_params(self):
        return self.init_params

    def leave(self, *, worker_id):
        pass


class TestWorkerCoordinatorWiring:
    def test_default_coordinator_wraps_worker_client(self):
        model = TinyModel()
        worker = DiLoCoWorker(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            server_addr="dummy:8512",
        )
        assert isinstance(worker.coordinator, CoordinatorClient)
        # wraps the same underlying client the backend wraps
        assert worker.coordinator._client is worker.client
        assert worker.backend.client is worker.client

    def test_injected_coordinator_is_honored(self):
        model = TinyModel()
        injected = CoordinatorClient(RecordingClient())
        worker = DiLoCoWorker(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            server_addr="dummy:8512",
            coordinator=injected,
        )
        assert worker.coordinator is injected

    def test_heartbeat_loop_uses_injected_coordinator(self):
        """The heartbeat loop must call coordinator.heartbeat, not the raw
        client — drive the real loop and assert the recording client saw it."""
        model = TinyModel()
        init = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rec = RecordingClient()
        worker = DiLoCoWorker(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            server_addr="dummy:8512",
            sync_every=10_000,  # large -> no sync fires during the test
            heartbeat_interval=0.05,  # start() spins up the heartbeat thread
            backend=_NoNetBackend(init),  # join/leave without HTTP
            coordinator=CoordinatorClient(rec),
        )
        worker.start()
        try:
            deadline = time.time() + 3.0
            while time.time() < deadline and not any(
                c[0] == "heartbeat" for c in rec.calls
            ):
                time.sleep(0.02)
        finally:
            worker.stop()
        assert any(c[0] == "heartbeat" for c in rec.calls), rec.calls
