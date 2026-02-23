"""Tests for DiLoCo server - outer optimizer correctness."""

import os

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
        """Test server state save and load with new checkpoint format."""
        sd = _make_state_dict(dim=4)

        server = DiLoCoServer(sd, num_workers=2, save_dir=str(tmp_path))

        # Apply a few rounds to build up optimizer state
        for i in range(3):
            server._pending_pseudograds["w0"] = {k: torch.randn_like(v) for k, v in sd.items()}
            server._pending_pseudograds["w1"] = {k: torch.randn_like(v) for k, v in sd.items()}
            server._apply_outer_optimizer()

        # Save (uses next_checkpoint_path internally)
        server.save_state()
        params_before = server.get_global_params()
        round_before = server._sync_round

        # Verify checkpoint directory structure
        checkpoint_dir = tmp_path / "checkpoints" / f"checkpoint-{round_before}"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "server_state.pt").exists()
        # Should have safetensors model weights
        assert (checkpoint_dir / "model.safetensors").exists() or \
               (checkpoint_dir / "model.safetensors.index.json").exists()

        # Create a fresh server and load from checkpoint directory
        server2 = DiLoCoServer(sd, num_workers=2)
        server2.load_state(str(checkpoint_dir))

        params_after = server2.get_global_params()
        assert server2._sync_round == round_before

        for key in params_before:
            assert torch.allclose(params_before[key], params_after[key])

    def test_save_load_state_legacy_format(self, tmp_path):
        """Test loading from legacy monolithic .pt file."""
        sd = _make_state_dict(dim=4)

        server = DiLoCoServer(sd, num_workers=2)

        # Apply a few rounds
        for i in range(3):
            server._pending_pseudograds["w0"] = {k: torch.randn_like(v) for k, v in sd.items()}
            server._pending_pseudograds["w1"] = {k: torch.randn_like(v) for k, v in sd.items()}
            server._apply_outer_optimizer()

        # Create a legacy-format .pt file manually
        legacy_state = {
            "global_params": server.get_global_params(),
            "outer_optimizer": server.outer_optimizer.state_dict(),
            "sync_round": server._sync_round,
            "num_workers": server.num_workers,
            "param_names": server._param_names,
            "async_mode": server.async_mode,
            "total_submissions": server._total_submissions,
        }
        legacy_path = str(tmp_path / "diloco_server_state_legacy.pt")
        torch.save(legacy_state, legacy_path)

        params_before = server.get_global_params()
        round_before = server._sync_round

        # Load into fresh server using legacy path (file, not directory)
        server2 = DiLoCoServer(sd, num_workers=2)
        server2.load_state(legacy_path)

        assert server2._sync_round == round_before
        for key in params_before:
            assert torch.allclose(params_before[key], server2.get_global_params()[key])

    def test_load_state_model_only(self, tmp_path):
        """Test loading from a directory with only model weights (no server_state.pt)."""
        sd = _make_state_dict(dim=4)
        from forgather.ml.sharded_checkpoint import save_checkpoint

        # Save model weights only (no server_state.pt)
        model_dir = str(tmp_path / "model_only")
        save_checkpoint(model_dir, sd, safetensors=True)

        server = DiLoCoServer(sd, num_workers=2)
        # Apply some rounds so round is non-zero
        server._pending_pseudograds["w0"] = {k: torch.randn_like(v) for k, v in sd.items()}
        server._pending_pseudograds["w1"] = {k: torch.randn_like(v) for k, v in sd.items()}
        server._apply_outer_optimizer()
        assert server._sync_round == 1

        # Create fresh server and load from model-only dir
        server2 = DiLoCoServer(sd, num_workers=2)
        server2.load_state(model_dir)

        # Round should stay at 0 (no server_state.pt)
        assert server2._sync_round == 0

        # But model weights should be loaded
        for key in sd:
            assert torch.allclose(server2.get_global_params()[key], sd[key].float())

    def test_checkpoint_rotation(self, tmp_path):
        """Test that save_total_limit rotates checkpoints correctly."""
        sd = _make_state_dict(dim=4)

        server = DiLoCoServer(
            sd, num_workers=1, save_dir=str(tmp_path), save_total_limit=2
        )

        # Save 4 checkpoints
        for i in range(4):
            server._pending_pseudograds["w0"] = {k: torch.zeros_like(v) for k, v in sd.items()}
            server._apply_outer_optimizer()
            server.save_state()

        # Should only have 2 checkpoints remaining
        checkpoints_dir = tmp_path / "checkpoints"
        remaining = list(checkpoints_dir.glob("checkpoint-*"))
        assert len(remaining) == 2

        # The two remaining should be the most recent ones (rounds 3 and 4)
        remaining_names = sorted([cp.name for cp in remaining])
        assert remaining_names == ["checkpoint-3", "checkpoint-4"]

    def test_save_state_explicit_path(self, tmp_path):
        """Test saving to an explicit path bypasses rotation."""
        sd = _make_state_dict(dim=4)

        server = DiLoCoServer(
            sd, num_workers=1, save_dir=str(tmp_path / "standard"), save_total_limit=1
        )

        # Save to explicit path
        explicit_dir = str(tmp_path / "explicit")
        server.save_state(path=explicit_dir)

        # Verify checkpoint was written to explicit path
        assert os.path.exists(os.path.join(explicit_dir, "server_state.pt"))
        assert os.path.exists(os.path.join(explicit_dir, "model.safetensors")) or \
               os.path.exists(os.path.join(explicit_dir, "model.safetensors.index.json"))

        # Standard save_dir should NOT have any checkpoints
        assert not (tmp_path / "standard" / "checkpoints").exists()
