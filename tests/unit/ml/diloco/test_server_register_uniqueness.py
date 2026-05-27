"""Tests for the DiLoCoServer worker_id uniqueness check on /register.

The server refuses a second registration of a worker_id that's already
in the registry with 409 (replaces the previous "re-register replaces"
semantics — that path doubled as a silent collision masker). Operators
recovering from a crashed worker either wait for heartbeat eviction or
POST /deregister.
"""

import json
import time
import urllib.error
import urllib.request

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _state_dict():
    torch.manual_seed(42)
    return {"layer.weight": torch.randn(4, 4)}


@pytest.fixture
def server(tmp_path):
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=2,
        port=0,
        heartbeat_timeout=0,  # disable health monitor — tests trigger eviction manually
    )
    s.start()
    time.sleep(0.2)
    yield s
    s.stop()


def _raw_register(server, worker_id, extra=None):
    """Hit /register directly so we can read the HTTP status on 409
    (the DiLoCoClient.register path coerces HTTPError into a
    ConnectionError, which loses the status code)."""
    body = {"worker_id": worker_id, "hostname": "test"}
    if extra is not None:
        body["extra"] = extra
    req = urllib.request.Request(
        f"http://localhost:{server.port}/register",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=5)


def test_first_register_succeeds(server):
    client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
    params = client.register("alpha")
    assert isinstance(params, dict)
    assert "layer.weight" in params


def test_duplicate_worker_id_returns_409(server):
    # First registration via the high-level client.
    client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
    client.register("alpha")

    # Second registration of the same worker_id → 409.
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_register(server, "alpha")
    assert exc_info.value.code == 409
    body = json.loads(exc_info.value.read().decode("utf-8"))
    assert "worker_id 'alpha' is already registered" in body["error"]
    # Diagnostic mentions both recovery paths so the worker's TTY
    # pane tells the operator exactly what to do.
    assert "heartbeat" in body["error"]
    assert "/deregister" in body["error"]


def test_register_after_deregister_succeeds(server):
    client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
    client.register("alpha")
    client.deregister("alpha")
    # Same worker_id can now register again — the entry was cleared.
    params = client.register("alpha")
    assert isinstance(params, dict)


def test_register_after_eviction_succeeds(server):
    """The heartbeat-eviction path (_handle_worker_death) clears the
    registry entry, so a worker_id whose previous instance died can
    re-register without operator intervention. The test triggers the
    eviction directly rather than waiting for the heartbeat timer."""
    client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
    client.register("alpha")
    # Simulate the health monitor evicting a dead worker.
    server._handle_worker_death("alpha")
    # Worker_id is free again.
    params = client.register("alpha")
    assert isinstance(params, dict)


def test_distinct_worker_ids_coexist(server):
    client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
    client.register("alpha")
    client.register("beta")
    with server._workers_lock:
        assert set(server._workers.keys()) == {"alpha", "beta"}
