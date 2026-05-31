"""Tests for the server-side trainer-control command relay.

The relay lets the CLI (and webui) drive per-worker save / save-and-stop /
abort without reaching each worker's trainer-control HTTP endpoint: a
command POSTed to ``/control/command`` is queued per worker and delivered
on that worker's next heartbeat, then cleared.
"""

import time

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import _RELAY_COMMANDS, DiLoCoServer

from .conftest import make_initial_checkpoint


def _make_state_dict(dim=4, num_layers=1, seed=0):
    torch.manual_seed(seed)
    return {f"layer{i}.weight": torch.randn(dim, dim) for i in range(num_layers)}


@pytest.fixture
def server(tmp_path):
    sd = _make_state_dict()
    ckpt = make_initial_checkpoint(sd, tmp_path)
    srv = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=2,
        port=0,
    )
    srv.start()
    time.sleep(0.2)
    try:
        yield srv
    finally:
        srv.stop()


def _client(server):
    return DiLoCoClient(f"localhost:{server.port}", timeout=5)


class TestCommandRelay:
    def test_relay_to_single_worker_delivered_on_heartbeat(self, server):
        client = _client(server)
        client.register("w0")
        client.register("w1")

        resp = client.relay_command("save_and_stop", worker_id="w0")
        assert resp["status"] == "ok"
        assert resp["workers"] == ["w0"]

        # w0 gets it; w1 does not.
        hb0 = client.heartbeat("w0")
        assert hb0.get("command") == "save_and_stop"
        hb1 = client.heartbeat("w1")
        assert "command" not in hb1

    def test_command_cleared_after_delivery(self, server):
        client = _client(server)
        client.register("w0")
        client.relay_command("save_checkpoint", worker_id="w0")

        first = client.heartbeat("w0")
        assert first.get("command") == "save_checkpoint"
        # Fires exactly once — the next heartbeat carries nothing.
        second = client.heartbeat("w0")
        assert "command" not in second

    def test_broadcast_to_all_workers(self, server):
        client = _client(server)
        client.register("w0")
        client.register("w1")

        resp = client.relay_command("abort")  # no worker_id == all
        assert set(resp["workers"]) == {"w0", "w1"}
        assert client.heartbeat("w0").get("command") == "abort"
        assert client.heartbeat("w1").get("command") == "abort"

    def test_unknown_command_rejected(self, server):
        client = _client(server)
        client.register("w0")
        with pytest.raises(Exception):
            client.relay_command("explode", worker_id="w0")
        # Nothing queued.
        assert "command" not in client.heartbeat("w0")

    def test_unknown_worker_rejected(self, server):
        client = _client(server)
        client.register("w0")
        with pytest.raises(Exception):
            client.relay_command("abort", worker_id="ghost")

    def test_relay_command_set_matches_worker_vocabulary(self):
        # Guard against drift between the relay's accepted set and the
        # worker-side trainer-control commands it maps onto.
        assert _RELAY_COMMANDS == {"save_checkpoint", "save_and_stop", "abort"}


class _FakeWorker:
    """Minimal stand-in for DiLoCoWorker for callback application tests."""

    def __init__(self, command):
        self._command = command
        self.model = None  # _command_device falls back to cpu

    def consume_pending_command(self):
        cmd, self._command = self._command, None
        return cmd


class TestCallbackApplication:
    """The callback drains the worker's relayed command and maps it onto
    TrainerControl flags (single-process path: no distributed all_reduce)."""

    def _run(self, command):
        from forgather.ml.trainer.callbacks.diloco_callback import DiLoCoCallback
        from forgather.ml.trainer.trainer_types import TrainerControl

        cb = DiLoCoCallback(server_addr="localhost:1")  # active, no worker started
        cb._worker = _FakeWorker(command)
        control = TrainerControl()
        cb.on_step_end(args=None, state=None, control=control)
        return control

    def test_save_checkpoint_sets_should_save(self):
        c = self._run("save_checkpoint")
        assert c.should_save is True
        assert c.should_training_stop is False
        assert c.should_abort_without_save is False

    def test_save_and_stop_sets_save_and_stop(self):
        c = self._run("save_and_stop")
        assert c.should_save is True
        assert c.should_training_stop is True
        assert c.should_abort_without_save is False

    def test_abort_sets_abort_without_save(self):
        c = self._run("abort")
        assert c.should_abort_without_save is True
        assert c.should_training_stop is True
        assert c.should_save is False

    def test_no_command_is_noop(self):
        c = self._run(None)
        assert c.should_save is False
        assert c.should_training_stop is False
        assert c.should_abort_without_save is False
