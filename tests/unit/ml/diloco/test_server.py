"""Tests for DiLoCo server - outer optimizer correctness."""

import torch
import pytest

from forgather.ml.diloco.server import (
    DiLoCoServer,
    _serialize_state_dict,
    _deserialize_state_dict,
)


def _make_state_dict(dim=16, num_layers=2, seed=42):
    """Create a simple model state dict for testing."""
    torch.manual_seed(seed)
    return {
        f"layer{i}.weight": torch.randn(dim, dim)
        for i in range(num_layers)
    }


class TestSerialization:
    """Test tensor serialization round-trip."""

    def test_serialize_deserialize_roundtrip(self):
        sd = _make_state_dict()
        data = _serialize_state_dict(sd)
        restored = _deserialize_state_dict(data)

        assert set(sd.keys()) == set(restored.keys())
        for key in sd:
            assert torch.equal(sd[key], restored[key])

    def test_serialize_bf16(self):
        sd = {k: v.to(torch.bfloat16) for k, v in _make_state_dict().items()}
        data = _serialize_state_dict(sd)
        restored = _deserialize_state_dict(data)

        for key in sd:
            assert restored[key].dtype == torch.bfloat16
            assert torch.equal(sd[key], restored[key])

    def test_serialize_empty(self):
        sd = {}
        data = _serialize_state_dict(sd)
        restored = _deserialize_state_dict(data)
        assert restored == {}


class TestDiLoCoServer:
    """Test server outer optimizer logic."""

    def test_server_creation(self):
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, port=0)

        # Check global params match initial state dict
        global_params = server.get_global_params()
        for key in sd:
            assert torch.allclose(global_params[key], sd[key].float())

    def test_server_invalid_num_workers(self):
        sd = _make_state_dict()
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            DiLoCoServer(sd, num_workers=0)

    def test_outer_optimizer_step_correctness(self):
        """
        Verify the outer optimizer (SGD with Nesterov) produces correct updates.

        With a known pseudo-gradient, we can compute the expected SGD+Nesterov
        update and compare.
        """
        torch.manual_seed(0)
        dim = 4
        sd = {"w": torch.ones(dim, dim)}

        # Use simple SGD without Nesterov for easier verification
        def simple_sgd_factory(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.0)

        server = DiLoCoServer(sd, num_workers=1, outer_optimizer_factory=simple_sgd_factory)

        # Simulate: worker trained and moved params from 1.0 to 0.5
        # pseudo_grad = global - local = 1.0 - 0.5 = 0.5
        pseudo_grad = {"w": torch.full((dim, dim), 0.5)}

        # Manually apply what the server does
        server._pending_pseudograds["worker_0"] = pseudo_grad
        server._apply_outer_optimizer()

        # With SGD(lr=1.0, momentum=0), new_param = param - lr * grad = 1.0 - 1.0 * 0.5 = 0.5
        updated = server.get_global_params()
        assert torch.allclose(updated["w"], torch.full((dim, dim), 0.5))

    def test_outer_optimizer_averaging(self):
        """Verify pseudo-gradients from multiple workers are averaged."""
        dim = 4
        sd = {"w": torch.zeros(dim)}

        def simple_sgd_factory(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.0)

        server = DiLoCoServer(sd, num_workers=2, outer_optimizer_factory=simple_sgd_factory)

        # Worker 0: pseudo_grad = 1.0 (moved params by -1.0)
        # Worker 1: pseudo_grad = 3.0 (moved params by -3.0)
        # Average: 2.0
        server._pending_pseudograds["worker_0"] = {"w": torch.full((dim,), 1.0)}
        server._pending_pseudograds["worker_1"] = {"w": torch.full((dim,), 3.0)}
        server._apply_outer_optimizer()

        # new_param = 0 - 1.0 * 2.0 = -2.0
        updated = server.get_global_params()
        assert torch.allclose(updated["w"], torch.full((dim,), -2.0))

    def test_sync_round_increments(self):
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=1)

        assert server._sync_round == 0

        server._pending_pseudograds["w0"] = {k: torch.zeros_like(v) for k, v in sd.items()}
        server._apply_outer_optimizer()

        assert server._sync_round == 1

    def test_nesterov_momentum_update(self):
        """Test that Nesterov momentum works as expected over multiple rounds."""
        dim = 2
        sd = {"w": torch.zeros(dim)}

        # Use default SGD with Nesterov
        def sgd_nesterov_factory(params):
            return torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)

        server = DiLoCoServer(sd, num_workers=1, outer_optimizer_factory=sgd_nesterov_factory)

        # Apply same pseudo-gradient twice to test momentum accumulation
        for _ in range(3):
            server._pending_pseudograds["w0"] = {"w": torch.ones(dim)}
            server._apply_outer_optimizer()

        # With Nesterov momentum the params should have moved more than 3 * lr * grad
        updated = server.get_global_params()
        simple_update = 3 * 0.1 * 1.0  # Without momentum
        # With Nesterov, each step adds momentum so total movement should be larger
        assert updated["w"][0].item() < -simple_update, (
            f"Nesterov should move params further than simple SGD: "
            f"got {updated['w'][0].item()}, expected < {-simple_update}"
        )

    def test_global_params_are_float32(self):
        """Server should store params in float32 regardless of input dtype."""
        sd = {"w": torch.ones(4, dtype=torch.bfloat16)}
        server = DiLoCoServer(sd, num_workers=1)

        params = server.get_global_params()
        assert params["w"].dtype == torch.float32

    def test_save_load_state(self, tmp_path):
        """Test server state save and load."""
        sd = _make_state_dict(dim=4)

        server = DiLoCoServer(sd, num_workers=2)

        # Apply a few rounds to build up optimizer state
        for i in range(3):
            server._pending_pseudograds["w0"] = {k: torch.randn_like(v) for k, v in sd.items()}
            server._pending_pseudograds["w1"] = {k: torch.randn_like(v) for k, v in sd.items()}
            server._apply_outer_optimizer()

        # Save
        server.save_state(str(tmp_path))
        params_before = server.get_global_params()
        round_before = server._sync_round

        # Create a fresh server and load
        server2 = DiLoCoServer(sd, num_workers=2)
        server2.load_state(str(tmp_path / "diloco_server_state_latest.pt"))

        params_after = server2.get_global_params()
        assert server2._sync_round == round_before

        for key in params_before:
            assert torch.allclose(params_before[key], params_after[key])
