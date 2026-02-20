#!/usr/bin/env python3
"""
Tests for checkpoint replication validation.

These tests verify that the replication validation infrastructure correctly
detects when replicated state (model weights, optimizer state) diverges
across ranks in distributed training.

Tests are organized into three groups:
1. Unit tests for checksum/hash functions (no distributed required)
2. Integration tests for validate_replication() with mocked dist
3. Integration tests for CheckpointManager model validation with mocked dist
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful

from forgather.ml.distributed import StaticDistributedEnvironment
from forgather.ml.trainer.checkpoint_manager import CheckpointConfig, CheckpointManager
from forgather.ml.trainer.checkpoint_types import (
    SharingPattern,
    StateComponent,
    compute_state_hash,
)
from forgather.ml.trainer.checkpoint_utils import (
    ValidationLevel,
    compute_state_checksum,
    compute_tensor_checksum,
    validate_replication,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state_dict(**overrides):
    """Create a simple state dict with two tensors."""
    state = {
        "weight": torch.randn(4, 4),
        "bias": torch.randn(4),
    }
    state.update(overrides)
    return state


class SimpleModel(nn.Module):
    def __init__(self, in_features=10, out_features=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MockStateful(Stateful):
    """A Stateful wrapper around a dict, for use in StateComponent."""

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict


class MockStatefulProvider:
    """Minimal StatefulProvider for CheckpointManager tests."""

    def __init__(self, components):
        self._components = components

    def get_state_components(self):
        return self._components

    def get_process_groups(self):
        return {}


# ---------------------------------------------------------------------------
# 1. Unit tests for checksum / hash helpers
# ---------------------------------------------------------------------------


class TestComputeTensorChecksum(unittest.TestCase):
    """Test compute_tensor_checksum()."""

    def test_identical_tensors_produce_same_checksum(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        cs1 = compute_tensor_checksum("x", t)
        cs2 = compute_tensor_checksum("x", t.clone())
        self.assertEqual(cs1.checksum, cs2.checksum)

    def test_different_tensors_produce_different_checksum(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 4.0])
        cs1 = compute_tensor_checksum("x", t1)
        cs2 = compute_tensor_checksum("x", t2)
        self.assertNotEqual(cs1.checksum, cs2.checksum)

    def test_metadata_captured(self):
        t = torch.randn(3, 5)
        cs = compute_tensor_checksum("w", t)
        self.assertEqual(cs.name, "w")
        self.assertEqual(cs.shape, (3, 5))
        self.assertEqual(cs.numel, 15)
        self.assertIn("float", cs.dtype)

    def test_single_element_difference(self):
        """A difference in a single element should change the checksum."""
        t1 = torch.zeros(100)
        t2 = t1.clone()
        t2[50] = 1.0  # Differ at index 50 (well beyond first 10)
        cs1 = compute_tensor_checksum("x", t1)
        cs2 = compute_tensor_checksum("x", t2)
        self.assertNotEqual(cs1.checksum, cs2.checksum)


class TestComputeStateChecksum(unittest.TestCase):
    """Test compute_state_checksum() at TENSOR and FULL levels."""

    def test_identical_state_dicts_match_tensor_level(self):
        sd = make_state_dict()
        sd_clone = {k: v.clone() for k, v in sd.items()}
        cs1 = compute_state_checksum(sd, ValidationLevel.TENSOR)
        cs2 = compute_state_checksum(sd_clone, ValidationLevel.TENSOR)
        self.assertEqual(cs1.overall_hash, cs2.overall_hash)
        for key in sd:
            self.assertEqual(
                cs1.tensor_checksums[key].checksum,
                cs2.tensor_checksums[key].checksum,
            )

    def test_different_state_dicts_differ_tensor_level(self):
        sd1 = make_state_dict()
        sd2 = {k: v.clone() for k, v in sd1.items()}
        sd2["weight"][2, 2] += 1.0  # Modify a single value
        cs1 = compute_state_checksum(sd1, ValidationLevel.TENSOR)
        cs2 = compute_state_checksum(sd2, ValidationLevel.TENSOR)
        self.assertNotEqual(cs1.overall_hash, cs2.overall_hash)
        self.assertNotEqual(
            cs1.tensor_checksums["weight"].checksum,
            cs2.tensor_checksums["weight"].checksum,
        )
        # Unchanged tensor should still match
        self.assertEqual(
            cs1.tensor_checksums["bias"].checksum,
            cs2.tensor_checksums["bias"].checksum,
        )

    def test_nested_state_dict(self):
        """Optimizer-like nested state dicts are handled."""
        sd = {
            "state": {
                "step": torch.tensor([100]),
                "exp_avg": torch.randn(4, 4),
            },
            "param_groups": torch.tensor([0.001]),
        }
        cs = compute_state_checksum(sd, ValidationLevel.TENSOR)
        self.assertIn("state.step", cs.tensor_checksums)
        self.assertIn("state.exp_avg", cs.tensor_checksums)


class TestComputeStateHash(unittest.TestCase):
    """Test compute_state_hash() (QUICK level)."""

    def test_identical_state_dicts_match(self):
        sd = make_state_dict()
        sd_clone = {k: v.clone() for k, v in sd.items()}
        self.assertEqual(compute_state_hash(sd), compute_state_hash(sd_clone))

    def test_structural_difference_detected(self):
        """Shape/dtype differences are detected."""
        sd1 = {"w": torch.randn(4, 4)}
        sd2 = {"w": torch.randn(4, 5)}  # Different shape
        self.assertNotEqual(compute_state_hash(sd1), compute_state_hash(sd2))

    def test_quick_misses_late_element_difference(self):
        """
        QUICK mode only samples the first 10 elements, so a difference
        at index >= 10 goes undetected. This documents a known limitation.
        """
        t1 = torch.zeros(100)
        t2 = t1.clone()
        t2[50] = 999.0  # Differ at index 50
        sd1 = {"w": t1}
        sd2 = {"w": t2}
        # QUICK hashes should be identical (limitation)
        self.assertEqual(
            compute_state_hash(sd1),
            compute_state_hash(sd2),
        )

    def test_quick_catches_early_element_difference(self):
        """Differences within the first 10 elements are caught by QUICK."""
        t1 = torch.zeros(100)
        t2 = t1.clone()
        t2[5] = 999.0  # Differ at index 5
        sd1 = {"w": t1}
        sd2 = {"w": t2}
        self.assertNotEqual(compute_state_hash(sd1), compute_state_hash(sd2))


# ---------------------------------------------------------------------------
# 2. Tests for validate_replication() with mocked dist
# ---------------------------------------------------------------------------


class TestValidateReplication(unittest.TestCase):
    """Test validate_replication() by mocking torch.distributed."""

    def test_single_rank_always_valid(self):
        """With world_size=1 (no dist init), validation always passes."""
        sd = make_state_dict()
        is_valid, errors = validate_replication(sd, ValidationLevel.TENSOR)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_none_level_always_valid(self):
        sd = make_state_dict()
        is_valid, errors = validate_replication(sd, ValidationLevel.NONE)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    @patch("forgather.ml.trainer.checkpoint_utils.dist")
    @patch("forgather.ml.trainer.checkpoint_utils.all_gather_object_list")
    def test_tensor_level_detects_divergence(self, mock_all_gather_obj, mock_dist):
        """TENSOR level detects when one rank's tensor data differs."""
        mock_dist.is_initialized.return_value = True

        sd_rank0 = make_state_dict(weight=torch.zeros(4, 4))
        sd_rank1 = make_state_dict(weight=torch.ones(4, 4))

        cs_rank0 = compute_state_checksum(sd_rank0, ValidationLevel.TENSOR)
        cs_rank1 = compute_state_checksum(sd_rank1, ValidationLevel.TENSOR)

        # Simulate all_gather returning checksums from both ranks
        mock_all_gather_obj.return_value = [cs_rank0, cs_rank1]
        mock_dist.get_rank.return_value = 1

        # Validate from rank 1's perspective
        is_valid, errors = validate_replication(sd_rank1, ValidationLevel.TENSOR)

        self.assertFalse(is_valid)
        # Should report the specific tensor that diverged
        self.assertTrue(
            any("weight" in e and "data mismatch" in e for e in errors),
            f"Expected 'weight' data mismatch error, got: {errors}",
        )

    @patch("forgather.ml.trainer.checkpoint_utils.dist")
    @patch("forgather.ml.trainer.checkpoint_utils.all_gather_object_list")
    def test_tensor_level_passes_identical(self, mock_all_gather_obj, mock_dist):
        """TENSOR level passes when all ranks have identical state."""
        mock_dist.is_initialized.return_value = True

        sd = make_state_dict()
        cs = compute_state_checksum(sd, ValidationLevel.TENSOR)

        mock_all_gather_obj.return_value = [cs, cs]  # Two identical ranks
        mock_dist.get_rank.return_value = 0

        is_valid, errors = validate_replication(sd, ValidationLevel.TENSOR)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    @patch("forgather.ml.trainer.checkpoint_utils.dist")
    @patch("forgather.ml.trainer.checkpoint_utils.all_gather_scalar")
    def test_quick_level_detects_hash_mismatch(self, mock_all_gather, mock_dist):
        """QUICK level detects when hashes differ across ranks."""
        mock_dist.is_initialized.return_value = True

        sd1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
        sd2 = {"w": torch.tensor([1.0, 2.0, 9.0])}

        h1 = compute_state_hash(sd1)
        h2 = compute_state_hash(sd2)

        # Return different hash ints for the two "ranks"
        mock_all_gather.return_value = [
            int(h1[:15], 16),
            int(h2[:15], 16),
        ]

        is_valid, errors = validate_replication(sd2, ValidationLevel.QUICK)
        self.assertFalse(is_valid)
        self.assertTrue(any("hash mismatch" in e.lower() for e in errors))


# ---------------------------------------------------------------------------
# 3. Tests for CheckpointCoordinator._save_replicated_component()
# ---------------------------------------------------------------------------


class TestCoordinatorReplicatedValidation(unittest.TestCase):
    """Test that CheckpointCoordinator validates replicated components."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("forgather.ml.trainer.checkpoint_coordinator.validate_replication")
    def test_replicated_component_validation_failure_raises(self, mock_validate):
        """_save_replicated_component raises when validation fails for required component."""
        from forgather.ml.trainer.checkpoint_coordinator import CheckpointCoordinator

        mock_validate.return_value = (False, ["weight: data mismatch"])

        dist_env = MagicMock()
        dist_env.rank = 0
        dist_env.local_rank = 0
        dist_env.world_size = 2

        stateful = MockStateful({"w": torch.randn(4)})
        component = StateComponent(
            key="optimizer",
            stateful=stateful,
            sharing_pattern=SharingPattern.REPLICATED,
            validate_replication=True,
            validation_level="tensor",
            required=True,
        )

        coordinator = CheckpointCoordinator(
            state_components=[component],
            process_groups={},
            dist=dist_env,
            output_dir=self.test_dir,
        )

        checkpoint_path = os.path.join(self.test_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        with self.assertRaises(RuntimeError) as ctx:
            coordinator._save_replicated_component(component, checkpoint_path)

        self.assertIn("Replication validation failed", str(ctx.exception))

    @patch("forgather.ml.trainer.checkpoint_coordinator.validate_replication")
    def test_replicated_component_validation_success_saves(self, mock_validate):
        """_save_replicated_component saves normally when validation passes."""
        from forgather.ml.trainer.checkpoint_coordinator import CheckpointCoordinator

        mock_validate.return_value = (True, [])

        dist_env = MagicMock()
        dist_env.rank = 0
        dist_env.local_rank = 0
        dist_env.world_size = 2

        stateful = MockStateful({"w": torch.randn(4)})
        component = StateComponent(
            key="test_comp",
            stateful=stateful,
            sharing_pattern=SharingPattern.REPLICATED,
            validate_replication=True,
            validation_level="tensor",
            required=True,
        )

        coordinator = CheckpointCoordinator(
            state_components=[component],
            process_groups={},
            dist=dist_env,
            output_dir=self.test_dir,
        )

        checkpoint_path = os.path.join(self.test_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        manifest = coordinator._save_replicated_component(component, checkpoint_path)

        self.assertIsNotNone(manifest)
        state_path = os.path.join(checkpoint_path, "test_comp_state.pt")
        self.assertTrue(os.path.exists(state_path))


# ---------------------------------------------------------------------------
# 4. Tests for CheckpointManager model validation (Issue 1 fix)
# ---------------------------------------------------------------------------


class TestCheckpointManagerModelValidation(unittest.TestCase):
    """Test that CheckpointManager validates model replication before saving."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_manager(self, model, dist_env, model_validate=True, model_level="tensor"):
        """Create a CheckpointManager with a model component that has validation configured."""
        config = CheckpointConfig(
            output_dir=self.test_dir,
            save_total_limit=3,
        )

        model_stateful = MockStateful(model.state_dict())

        components = [
            StateComponent(
                key="model",
                stateful=model_stateful,
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=model_validate,
                validation_level=model_level,
                required=True,
            ),
        ]

        provider = MockStatefulProvider(components)

        manager = CheckpointManager(
            config=config,
            dist=dist_env,
            stateful_provider=provider,
            model=model,
        )
        return manager

    def test_model_state_component_stored(self):
        """Verify the model StateComponent is extracted and stored."""
        model = SimpleModel()
        dist_env = StaticDistributedEnvironment()
        manager = self._make_manager(model, dist_env)

        self.assertIsNotNone(manager.model_state_component)
        self.assertEqual(manager.model_state_component.key, "model")
        self.assertTrue(manager.model_state_component.validate_replication)

    def test_model_validation_skipped_single_rank(self):
        """Single-rank training skips validation (nothing to compare)."""
        model = SimpleModel()
        dist_env = StaticDistributedEnvironment()
        manager = self._make_manager(model, dist_env)

        # Should not raise even though validate_replication=True,
        # because world_size=1.
        checkpoint_path = os.path.join(self.test_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)
        with patch.object(manager, "_save_model"):
            manager.save_checkpoint(checkpoint_path=checkpoint_path)

    @patch("forgather.ml.trainer.checkpoint_manager.validate_replication")
    def test_model_validation_called_multi_rank(self, mock_validate):
        """Multi-rank training calls validate_replication for model."""
        mock_validate.return_value = (True, [])

        model = SimpleModel()
        dist_env = MagicMock()
        dist_env.rank = 0
        dist_env.local_rank = 0
        dist_env.world_size = 2
        dist_env.device = "cpu"
        dist_env.device_type = "cpu"

        manager = self._make_manager(model, dist_env)

        checkpoint_path = os.path.join(self.test_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)
        with patch.object(manager, "_save_model"):
            with patch.object(manager, "_save_training_state"):
                manager.save_checkpoint(checkpoint_path=checkpoint_path)

        # validate_replication should have been called
        mock_validate.assert_called_once()
        call_kwargs = mock_validate.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("validation_level"), ValidationLevel.TENSOR
        )

    @patch("forgather.ml.trainer.checkpoint_manager.validate_replication")
    def test_model_validation_failure_raises(self, mock_validate):
        """Model validation failure raises RuntimeError for required model."""
        mock_validate.return_value = (
            False,
            ["Tensor 'linear.weight' data mismatch (checksum differs)"],
        )

        model = SimpleModel()
        dist_env = MagicMock()
        dist_env.rank = 0
        dist_env.local_rank = 0
        dist_env.world_size = 2
        dist_env.device = "cpu"
        dist_env.device_type = "cpu"

        manager = self._make_manager(model, dist_env)

        checkpoint_path = os.path.join(self.test_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        with self.assertRaises(RuntimeError) as ctx:
            with patch.object(manager, "_save_model"):
                with patch.object(manager, "_save_training_state"):
                    manager.save_checkpoint(checkpoint_path=checkpoint_path)

        self.assertIn("diverged", str(ctx.exception))

    def test_no_model_component_no_validation(self):
        """When no model component exists, no validation occurs."""
        config = CheckpointConfig(
            output_dir=self.test_dir,
            save_total_limit=3,
        )

        # Provide only non-model components
        trainer_stateful = MockStateful({"global_step": 0})
        components = [
            StateComponent(
                key="trainer",
                stateful=trainer_stateful,
                sharing_pattern=SharingPattern.GLOBAL,
                required=False,
            ),
        ]

        provider = MockStatefulProvider(components)
        model = SimpleModel()
        dist_env = StaticDistributedEnvironment()

        manager = CheckpointManager(
            config=config,
            dist=dist_env,
            stateful_provider=provider,
            model=model,
        )

        self.assertIsNone(manager.model_state_component)

        # Should not raise
        checkpoint_path = os.path.join(self.test_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)
        with patch.object(manager, "_save_model"):
            manager.save_checkpoint(checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
