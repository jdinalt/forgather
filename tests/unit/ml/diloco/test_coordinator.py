"""Tests for the CoordinatorClient surface (issue #154, step 2).

Verifies the facade delegates verbatim to the wrapped DiLoCoClient, and that the
worker holds a coordinator distinct from its sync backend (default wraps its own
client; an injected coordinator is honored).
"""

import torch
import torch.nn as nn

from forgather.ml.diloco.coordinator import CoordinatorClient
from forgather.ml.diloco.worker import DiLoCoWorker


class RecordingClient:
    """Stand-in for DiLoCoClient recording the coordination calls."""

    def __init__(self):
        self.calls = []
        self.info = {"expected_client_settings": {"sync_every": 10}}
        self.hb = {"sync_round": 3, "num_workers": 1}
        self.model_hash = "abc123"

    def heartbeat(self, worker_id, steps_per_second=0.0, stats=None):
        self.calls.append(("heartbeat", worker_id, steps_per_second, stats))
        return self.hb

    def get_info(self):
        self.calls.append(("get_info",))
        return self.info

    def fetch_model_def(self, dest_dir):
        self.calls.append(("fetch_model_def", dest_dir))
        return self.model_hash


class TestCoordinatorDelegation:
    def test_heartbeat_forwards_args_and_return(self):
        client = RecordingClient()
        coord = CoordinatorClient(client)
        stats = {"loss": 1.5}
        out = coord.heartbeat("w0", steps_per_second=4.2, stats=stats)
        assert out is client.hb
        assert client.calls == [("heartbeat", "w0", 4.2, stats)]

    def test_heartbeat_defaults(self):
        client = RecordingClient()
        CoordinatorClient(client).heartbeat("w0")
        assert client.calls == [("heartbeat", "w0", 0.0, None)]

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
