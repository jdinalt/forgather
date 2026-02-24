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
    """Create a dashboard-enabled server on a random port."""
    sd = _make_state_dict()
    ckpt = make_initial_checkpoint(sd, tmp_path / "initial")
    srv = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=str(ckpt),
        num_workers=2,
        port=0,
        outer_optimizer_factory=_simple_sgd,
        dashboard_enabled=True,
    )
    srv.start()
    time.sleep(0.2)
    yield srv
    if srv._running:
        srv.stop()


@pytest.fixture
def server_no_dashboard(tmp_path):
    """Create a server with dashboard disabled."""
    sd = _make_state_dict()
    ckpt = make_initial_checkpoint(sd, tmp_path / "initial")
    srv = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=str(ckpt),
        num_workers=1,
        port=0,
        outer_optimizer_factory=_simple_sgd,
        dashboard_enabled=False,
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


class TestDashboardServing:
    def test_dashboard_returns_html(self, server):
        status, headers, body = _get(f"http://localhost:{server.port}/dashboard")
        assert status == 200
        assert "text/html" in headers.get("Content-Type", "")
        assert b"DiLoCo" in body
        assert b"alpine" in body.lower()

    def test_root_returns_dashboard(self, server):
        status, headers, body = _get(f"http://localhost:{server.port}/")
        assert status == 200
        assert b"DiLoCo" in body

    def test_dashboard_disabled_returns_404(self, server_no_dashboard):
        status, _, body = _get(f"http://localhost:{server_no_dashboard.port}/dashboard")
        assert status == 404
        data = json.loads(body)
        assert "disabled" in data.get("error", "").lower()

    def test_root_disabled_returns_404(self, server_no_dashboard):
        status, _, _ = _get(f"http://localhost:{server_no_dashboard.port}/")
        assert status == 404


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
    def test_status_has_dashboard_fields(self, server):
        status_code, _, body = _get(f"http://localhost:{server.port}/status")
        assert status_code == 200
        data = json.loads(body)

        assert "outer_lr" in data
        assert "outer_momentum" in data
        assert "model_params" in data
        assert "model_size_mb" in data
        assert "dashboard_enabled" in data

        # Verify values
        assert data["outer_lr"] == 1.0  # from _simple_sgd
        assert data["outer_momentum"] == 0.5
        assert data["model_params"] == 8 * 8 * 2  # 2 layers of 8x8
        assert isinstance(data["model_size_mb"], (int, float))
        assert data["dashboard_enabled"] is True

    def test_status_dashboard_disabled_field(self, server_no_dashboard):
        _, _, body = _get(f"http://localhost:{server_no_dashboard.port}/status")
        data = json.loads(body)
        assert data["dashboard_enabled"] is False
