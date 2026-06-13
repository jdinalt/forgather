"""Tests for DiLoCo dashboard serving and control endpoints."""

import json
import time
import urllib.error
import urllib.request

import pytest
import torch

from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _make_state_dict(dim=8, num_layers=2, seed=42):
    torch.manual_seed(seed)
    return {f"layer{i}.weight": torch.randn(dim, dim) for i in range(num_layers)}


def test_compute_model_hash_is_deterministic_and_shape_sensitive():
    """The /info model_hash depends on (name, shape) only — stable across
    runs/values, changes when the parameter topology changes."""
    a = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    a2 = {"w": torch.zeros(4, 4), "b": torch.ones(4)}  # same shapes, diff values
    c = {"w": torch.randn(4, 8), "b": torch.randn(4)}  # different shape
    h = DiLoCoServer._compute_model_hash
    assert h(a) == h(a2)  # value-independent
    assert h(a) != h(c)  # shape-sensitive
    # Insertion order doesn't matter (names are sorted).
    assert h({"b": torch.randn(4), "w": torch.randn(4, 4)}) == h(a)


def _simple_sgd(params):
    return torch.optim.SGD(params, lr=1.0, momentum=0.5)


def _get(url):
    """GET request, return (status, headers, body)."""
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _post_json(url, data=None):
    """POST JSON, return (status, response_dict)."""
    body = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def server(tmp_path):
    """Create a server on a random port. (The built-in HTML dashboard
    was removed in favor of the webui's DiLoCo view; these tests now
    cover only the control endpoints + the status-field surface the
    webui consumes.)"""
    sd = _make_state_dict()
    ckpt = make_initial_checkpoint(sd, tmp_path / "initial")
    srv = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=str(ckpt),
        num_workers=2,
        port=0,
        outer_optimizer_factory=_simple_sgd,
    )
    srv.start()
    time.sleep(0.2)
    yield srv
    if srv._running:
        srv.stop()


@pytest.fixture
def server_with_save(tmp_path):
    """Create a server with output_dir configured for checkpoint saving."""
    sd = _make_state_dict()
    ckpt = make_initial_checkpoint(sd, tmp_path / "initial")
    srv = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=str(ckpt),
        num_workers=1,
        port=0,
        outer_optimizer_factory=_simple_sgd,
        save_every_n_rounds=0,  # Disable periodic saves; only save on demand
    )
    srv.start()
    time.sleep(0.2)
    yield srv, tmp_path
    if srv._running:
        srv.stop()


def test_dashboard_endpoint_removed(server):
    """The /dashboard and / endpoints used to serve the Alpine.js
    HTML dashboard. They were removed when the webui's DiLoCo view
    took over; both should now 404."""
    status_dash, _, _ = _get(f"http://localhost:{server.port}/dashboard")
    assert status_dash == 404
    status_root, _, _ = _get(f"http://localhost:{server.port}/")
    assert status_root == 404


class TestControlEndpoints:
    def test_save_state_not_dirty_is_noop(self, server):
        """save_state is a no-op when state hasn't changed (dirty=False)."""
        import os

        status, data = _post_json(f"http://localhost:{server.port}/control/save_state")
        assert status == 200
        assert data["status"] == "ok"
        # No checkpoints should have been written because _dirty is False
        assert not os.path.isdir(os.path.join(server.output_dir, "checkpoints"))

    def test_save_state_with_save_dir(self, server_with_save):
        import os

        srv, tmp_path = server_with_save
        # Dirty the server state by running an outer optimizer step
        params = srv.get_global_params()
        srv._pending_pseudograds["w0"] = {
            k: torch.zeros_like(v) for k, v in params.items()
        }
        srv._apply_outer_optimizer()

        status, data = _post_json(f"http://localhost:{srv.port}/control/save_state")
        assert status == 200
        assert data["status"] == "ok"
        # Check that a checkpoint directory was saved
        checkpoints_dir = os.path.join(str(tmp_path), "checkpoints")
        assert os.path.isdir(checkpoints_dir)
        checkpoint_dirs = os.listdir(checkpoints_dir)
        assert any(d.startswith("checkpoint-") for d in checkpoint_dirs)

    def test_kick_worker(self, server):
        """Kick a registered worker."""
        # Register a worker via the server's internal register endpoint
        body = json.dumps({"worker_id": "w0", "hostname": "test"}).encode()
        req = urllib.request.Request(
            f"http://localhost:{server.port}/register",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)

        assert "w0" in server._workers

        status, data = _post_json(
            f"http://localhost:{server.port}/control/kick_worker",
            {"worker_id": "w0"},
        )
        assert status == 200
        assert data["status"] == "ok"
        assert "w0" not in server._workers

    def test_kick_worker_not_found(self, server):
        status, data = _post_json(
            f"http://localhost:{server.port}/control/kick_worker",
            {"worker_id": "nonexistent"},
        )
        assert status == 404

    def test_kick_worker_missing_id(self, server):
        status, data = _post_json(
            f"http://localhost:{server.port}/control/kick_worker", {}
        )
        assert status == 400

    def test_update_optimizer(self, server):
        status, data = _post_json(
            f"http://localhost:{server.port}/control/update_optimizer",
            {"lr": 0.1, "momentum": 0.8},
        )
        assert status == 200
        assert data["status"] == "ok"
        assert data["lr"] == 0.1
        assert data["momentum"] == 0.8

        # Verify the optimizer was updated
        pg = server.outer_optimizer.param_groups[0]
        assert pg["lr"] == 0.1
        assert pg["momentum"] == 0.8

    def test_update_optimizer_partial(self, server):
        """Update only LR, not momentum."""
        status, data = _post_json(
            f"http://localhost:{server.port}/control/update_optimizer",
            {"lr": 0.3},
        )
        assert status == 200
        assert data["lr"] == 0.3
        assert "momentum" not in data

    def test_update_optimizer_empty(self, server):
        """Empty update returns 400."""
        status, data = _post_json(
            f"http://localhost:{server.port}/control/update_optimizer", {}
        )
        assert status == 400

    def test_update_num_workers(self, server):
        status, data = _post_json(
            f"http://localhost:{server.port}/control/update_num_workers",
            {"num_workers": 5},
        )
        assert status == 200
        assert data["num_workers"] == 5
        assert server.num_workers == 5

    def test_update_num_workers_below_min(self, server):
        """Cannot set num_workers below min_workers."""
        status, data = _post_json(
            f"http://localhost:{server.port}/control/update_num_workers",
            {"num_workers": 0},
        )
        assert status == 400

    def test_update_num_workers_missing(self, server):
        status, data = _post_json(
            f"http://localhost:{server.port}/control/update_num_workers", {}
        )
        assert status == 400

    def test_shutdown(self, server):
        status, data = _post_json(f"http://localhost:{server.port}/control/shutdown")
        assert status == 200
        assert data["status"] == "ok"
        # Wait for the background shutdown thread to finish; serve_forever() has a
        # 0.5s poll interval so it may take up to ~1s to fully stop.
        deadline = time.time() + 5.0
        while server._running and time.time() < deadline:
            time.sleep(0.1)
        assert not server._running, "Server failed to stop within timeout"

    def test_unknown_action_returns_404(self, server):
        status, data = _post_json(f"http://localhost:{server.port}/control/nonexistent")
        assert status == 404

    def test_invalid_json_returns_400(self, server):
        """Sending invalid JSON to a control endpoint returns 400."""
        req = urllib.request.Request(
            f"http://localhost:{server.port}/control/save_state",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            status = resp.status
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            status = e.code
            data = json.loads(e.read())
        assert status == 400
        assert "error" in data


class TestStatusExtensions:
    """The /status response carries the same monitoring fields the
    webui's DiLoCo view consumes (formerly read by the built-in
    HTML dashboard)."""

    def test_status_has_monitoring_fields(self, server):
        status_code, _, body = _get(f"http://localhost:{server.port}/status")
        assert status_code == 200
        data = json.loads(body)

        assert "outer_lr" in data
        assert "outer_momentum" in data
        assert "model_params" in data
        assert "model_size_mb" in data

        # Verify values
        assert data["outer_lr"] == 1.0  # from _simple_sgd
        assert data["outer_momentum"] == 0.5

        # Full one-line optimizer description (class + every hyperparameter,
        # incl. nesterov) — more informative than reconstructing SGD(lr,
        # momentum) client-side, and generalizes to other optimizers.
        assert data["outer_optimizer"].startswith("SGD(")
        assert "lr=1.0" in data["outer_optimizer"]
        assert "momentum=0.5" in data["outer_optimizer"]
        assert "nesterov=" in data["outer_optimizer"]
        assert data["model_params"] == 8 * 8 * 2  # 2 layers of 8x8
        assert isinstance(data["model_size_mb"], (int, float))

    def test_status_does_not_advertise_dashboard(self, server):
        """The dashboard_enabled field is gone (the dashboard itself
        was removed). The webui shouldn't have any reason to ask
        about it."""
        _, _, body = _get(f"http://localhost:{server.port}/status")
        data = json.loads(body)
        assert "dashboard_enabled" not in data

    def test_info_carries_output_dir(self, server, tmp_path):
        """The webui's Submit-training-job modal pre-fills
        --model-id-or-path from this field. Regression test for #52."""
        _, _, body = _get(f"http://localhost:{server.port}/info")
        data = json.loads(body)
        assert data.get("output_dir") == str(tmp_path)

    def test_info_advertises_authoritative_settings(self, server):
        """/info is the authority for the must-match worker settings, so
        every field is present and non-null (the worker takes them
        verbatim). A non-DyLU server advertises its own ``sync_every``."""
        _, _, body = _get(f"http://localhost:{server.port}/info")
        data = json.loads(body)
        assert data["settings_authority"] == "server"
        assert isinstance(data["model_hash"], str) and data["model_hash"]
        exp = data["expected_client_settings"]
        # Non-DyLU server advertises its dedicated sync_every (not None).
        assert exp["sync_every"] == server.sync_every
        assert exp["sync_every"] is not None
        assert exp["bf16_comm"] == server.bf16_comm
        assert exp["dylu"] is False
        assert exp["num_fragments_default"] == server.num_fragments
        assert exp["fragment_assignment_default"] == server.fragment_assignment
        assert exp["heartbeat_timeout"] == server.heartbeat_timeout

    def test_info_reflects_configured_group_settings(self, tmp_path):
        """A server configured with non-default group settings advertises
        exactly those — the operator's single source of truth for the
        whole group."""
        sd = _make_state_dict()
        ckpt = make_initial_checkpoint(sd, tmp_path / "initial")
        srv = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=str(ckpt),
            num_workers=1,
            port=0,
            sync_every=250,
            num_fragments=4,
            fragment_assignment="sequential",
            bf16_comm=False,
            outer_optimizer_factory=_simple_sgd,
        )
        srv.start()
        time.sleep(0.2)
        try:
            _, _, body = _get(f"http://localhost:{srv.port}/info")
            exp = json.loads(body)["expected_client_settings"]
            assert exp["sync_every"] == 250
            assert exp["num_fragments_default"] == 4
            assert exp["fragment_assignment_default"] == "sequential"
            assert exp["bf16_comm"] is False
        finally:
            srv.stop()

    def test_info_sync_every_uses_dylu_base_when_dylu_enabled(self, tmp_path):
        """Under DyLU the advertised sync_every is the DyLU base rate (the
        per-worker scaling anchor), not the plain sync_every."""
        sd = _make_state_dict()
        ckpt = make_initial_checkpoint(sd, tmp_path / "initial")
        srv = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=str(ckpt),
            num_workers=1,
            port=0,
            async_mode=True,
            dylu_enabled=True,
            dylu_base_sync_every=128,
            sync_every=999,  # ignored while DyLU is on
            outer_optimizer_factory=_simple_sgd,
        )
        srv.start()
        time.sleep(0.2)
        try:
            _, _, body = _get(f"http://localhost:{srv.port}/info")
            exp = json.loads(body)["expected_client_settings"]
            assert exp["dylu"] is True
            assert exp["sync_every"] == 128
        finally:
            srv.stop()
