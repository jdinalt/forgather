#!/usr/bin/env python3
"""
Unit tests for distributed checkpoint types and coordination.

Tests the new state-centric checkpoint abstraction including:
- SharingPattern enum
- StateComponent validation
- CheckpointManifest serialization
- State hashing for replication validation
- CheckpointCoordinator pattern-specific handlers
"""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful

from forgather.ml.distributed import StaticDistributedEnvironment
from forgather.ml.trainer.checkpoint_types import (
    SharingPattern,
    StateComponent,
    ComponentManifest,
    CheckpointManifest,
    compute_state_hash,
    _state_dict_to_serializable,
)
from forgather.ml.trainer.checkpoint_coordinator import CheckpointCoordinator


class MockStateful(Stateful):
    """Mock stateful object for testing."""

    def __init__(self, data: dict):
        self._data = data

    def state_dict(self):
        return self._data.copy()

    def load_state_dict(self, state_dict):
        self._data = state_dict.copy()


class TestSharingPattern(unittest.TestCase):
    """Test SharingPattern enum."""

    def test_sharing_pattern_values(self):
        """Test that SharingPattern has expected values."""
        self.assertEqual(SharingPattern.GLOBAL.value, "global")
        self.assertEqual(SharingPattern.PER_RANK.value, "per_rank")
        self.assertEqual(SharingPattern.REPLICATED.value, "replicated")
        self.assertEqual(SharingPattern.PER_GROUP.value, "per_group")
        self.assertEqual(SharingPattern.PER_NODE.value, "per_node")

    def test_sharing_pattern_members(self):
        """Test that all expected patterns exist."""
        patterns = {p.value for p in SharingPattern}
        expected = {"global", "per_rank", "replicated", "per_group", "per_node"}
        self.assertEqual(patterns, expected)


class TestStateComponent(unittest.TestCase):
    """Test StateComponent dataclass and validation."""

    def test_basic_state_component(self):
        """Test creating basic StateComponent."""
        stateful = MockStateful({"key": "value"})
        component = StateComponent(
            key="test",
            stateful=stateful,
            sharing_pattern=SharingPattern.GLOBAL,
        )

        self.assertEqual(component.key, "test")
        self.assertEqual(component.stateful, stateful)
        self.assertEqual(component.sharing_pattern, SharingPattern.GLOBAL)
        self.assertTrue(component.required)
        self.assertFalse(component.validate_replication)
        self.assertEqual(component.metadata, {})

    def test_state_component_with_metadata(self):
        """Test StateComponent with metadata."""
        stateful = MockStateful({"key": "value"})
        metadata = {"version": "1.0", "config_hash": "abc123"}
        component = StateComponent(
            key="model",
            stateful=stateful,
            sharing_pattern=SharingPattern.REPLICATED,
            validate_replication=True,
            metadata=metadata,
        )

        self.assertEqual(component.metadata, metadata)
        self.assertTrue(component.validate_replication)

    def test_per_group_requires_process_group_name(self):
        """Test that PER_GROUP pattern requires process_group_name."""
        stateful = MockStateful({"key": "value"})

        with self.assertRaises(ValueError) as ctx:
            StateComponent(
                key="test",
                stateful=stateful,
                sharing_pattern=SharingPattern.PER_GROUP,
                # Missing process_group_name
            )

        self.assertIn("must specify process_group_name", str(ctx.exception))

    def test_per_group_with_process_group_name(self):
        """Test that PER_GROUP pattern works with process_group_name."""
        stateful = MockStateful({"key": "value"})
        component = StateComponent(
            key="model",
            stateful=stateful,
            sharing_pattern=SharingPattern.PER_GROUP,
            process_group_name="dp_group",
        )

        self.assertEqual(component.process_group_name, "dp_group")

    def test_validate_replication_only_for_replicated(self):
        """Test that validate_replication only applies to REPLICATED pattern."""
        stateful = MockStateful({"key": "value"})

        with self.assertRaises(ValueError) as ctx:
            StateComponent(
                key="test",
                stateful=stateful,
                sharing_pattern=SharingPattern.GLOBAL,
                validate_replication=True,  # Invalid for GLOBAL
            )

        self.assertIn("validate_replication=True", str(ctx.exception))
        self.assertIn("REPLICATED pattern", str(ctx.exception))


class TestComponentManifest(unittest.TestCase):
    """Test ComponentManifest serialization and deserialization."""

    def test_component_manifest_serialization(self):
        """Test ComponentManifest round-trip serialization."""
        manifest = ComponentManifest(
            key="model",
            sharing_pattern="replicated",
            ranks=[0],
            replicated_across=[0, 1, 2, 3],
            size_bytes=1024,
            checksum="abc123",
            metadata={"version": "1.0"},
        )

        # Serialize to dict
        data = manifest.to_dict()
        self.assertEqual(data["key"], "model")
        self.assertEqual(data["sharing_pattern"], "replicated")
        self.assertEqual(data["ranks"], [0])
        self.assertEqual(data["replicated_across"], [0, 1, 2, 3])
        self.assertEqual(data["size_bytes"], 1024)
        self.assertEqual(data["checksum"], "abc123")

        # Deserialize from dict
        restored = ComponentManifest.from_dict(data)
        self.assertEqual(restored.key, manifest.key)
        self.assertEqual(restored.sharing_pattern, manifest.sharing_pattern)
        self.assertEqual(restored.ranks, manifest.ranks)
        self.assertEqual(restored.replicated_across, manifest.replicated_across)
        self.assertEqual(restored.size_bytes, manifest.size_bytes)
        self.assertEqual(restored.checksum, manifest.checksum)
        self.assertEqual(restored.metadata, manifest.metadata)


class TestCheckpointManifest(unittest.TestCase):
    """Test CheckpointManifest serialization and file I/O."""

    def test_checkpoint_manifest_serialization(self):
        """Test CheckpointManifest round-trip serialization."""
        components = {
            "model": ComponentManifest(
                key="model",
                sharing_pattern="replicated",
                ranks=[0],
                replicated_across=[0, 1, 2, 3],
                size_bytes=1024,
            ),
            "optimizer": ComponentManifest(
                key="optimizer",
                sharing_pattern="per_rank",
                ranks=[0, 1, 2, 3],
                size_bytes=512,
            ),
        }

        manifest = CheckpointManifest(
            checkpoint_path="/path/to/checkpoint",
            world_size=4,
            timestamp=datetime.now(),
            components=components,
            training_args_hash="hash123",
            forgather_version="0.1.0",
            pytorch_version="2.0.0",
            metadata={"experiment": "test"},
        )

        # Serialize to dict
        data = manifest.to_dict()
        self.assertEqual(data["checkpoint_path"], "/path/to/checkpoint")
        self.assertEqual(data["world_size"], 4)
        self.assertEqual(len(data["components"]), 2)
        self.assertIn("model", data["components"])
        self.assertIn("optimizer", data["components"])

        # Deserialize from dict
        restored = CheckpointManifest.from_dict(data)
        self.assertEqual(restored.checkpoint_path, manifest.checkpoint_path)
        self.assertEqual(restored.world_size, manifest.world_size)
        self.assertEqual(len(restored.components), 2)
        self.assertIn("model", restored.components)
        self.assertIn("optimizer", restored.components)

    def test_checkpoint_manifest_file_io(self):
        """Test saving and loading CheckpointManifest to/from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")

            # Create manifest
            components = {
                "model": ComponentManifest(
                    key="model",
                    sharing_pattern="global",
                    ranks=[0],
                    size_bytes=1024,
                ),
            }

            manifest = CheckpointManifest(
                checkpoint_path=tmpdir,
                world_size=1,
                timestamp=datetime.now(),
                components=components,
            )

            # Save to file
            manifest.save(manifest_path)
            self.assertTrue(os.path.exists(manifest_path))

            # Load from file
            loaded = CheckpointManifest.load(manifest_path)
            self.assertEqual(loaded.checkpoint_path, manifest.checkpoint_path)
            self.assertEqual(loaded.world_size, manifest.world_size)
            self.assertEqual(len(loaded.components), 1)
            self.assertIn("model", loaded.components)


class TestStateHashing(unittest.TestCase):
    """Test state hashing for replication validation."""

    def test_state_dict_to_serializable_primitives(self):
        """Test conversion of primitive types."""
        state = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "bool": True,
            "none": None,
        }

        serializable = _state_dict_to_serializable(state)
        self.assertEqual(serializable["int"], 42)
        self.assertEqual(serializable["float"], 3.14)
        self.assertEqual(serializable["str"], "hello")
        self.assertEqual(serializable["bool"], True)
        self.assertIsNone(serializable["none"])

    def test_state_dict_to_serializable_tensors(self):
        """Test conversion of tensors to serializable format."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        state = {"tensor": tensor}

        serializable = _state_dict_to_serializable(state)
        tensor_repr = serializable["tensor"]

        self.assertEqual(tensor_repr["type"], "tensor")
        self.assertEqual(tensor_repr["shape"], [3])
        self.assertIn("torch.float32", tensor_repr["dtype"])
        self.assertEqual(len(tensor_repr["sample"]), 3)

    def test_state_dict_to_serializable_nested(self):
        """Test conversion of nested structures."""
        state = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {"key": "value"},
                "tensor": torch.tensor([1.0]),
            }
        }

        serializable = _state_dict_to_serializable(state)
        self.assertEqual(serializable["nested"]["list"], [1, 2, 3])
        self.assertEqual(serializable["nested"]["dict"]["key"], "value")
        self.assertEqual(serializable["nested"]["tensor"]["type"], "tensor")

    def test_compute_state_hash_deterministic(self):
        """Test that state hashing is deterministic."""
        state = {
            "param1": torch.tensor([1.0, 2.0, 3.0]),
            "param2": 42,
            "param3": "value",
        }

        hash1 = compute_state_hash(state)
        hash2 = compute_state_hash(state)

        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64)  # SHA256 hex digest

    def test_compute_state_hash_different_for_different_states(self):
        """Test that different states produce different hashes."""
        state1 = {"param": torch.tensor([1.0, 2.0, 3.0])}
        state2 = {"param": torch.tensor([1.0, 2.0, 4.0])}  # Different value

        hash1 = compute_state_hash(state1)
        hash2 = compute_state_hash(state2)

        self.assertNotEqual(hash1, hash2)


class TestCheckpointCoordinator(unittest.TestCase):
    """Test CheckpointCoordinator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.dist = StaticDistributedEnvironment(
            world_size=1,
            rank=0,
            local_rank=0,
            device=torch.device("cpu"),
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_coordinator_initialization(self):
        """Test basic CheckpointCoordinator initialization."""
        stateful = MockStateful({"param": 1.0})
        components = [
            StateComponent(
                key="model",
                stateful=stateful,
                sharing_pattern=SharingPattern.GLOBAL,
            )
        ]

        coordinator = CheckpointCoordinator(
            state_components=components,
            process_groups={},
            dist=self.dist,
            output_dir=self.tmpdir,
        )

        self.assertEqual(coordinator.output_dir, self.tmpdir)
        self.assertEqual(len(coordinator.state_components), 1)

    def test_coordinator_validates_duplicate_keys(self):
        """Test that coordinator rejects duplicate component keys."""
        stateful1 = MockStateful({"param": 1.0})
        stateful2 = MockStateful({"param": 2.0})

        components = [
            StateComponent(
                key="model",
                stateful=stateful1,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="model",  # Duplicate key
                stateful=stateful2,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
        ]

        with self.assertRaises(ValueError) as ctx:
            CheckpointCoordinator(
                state_components=components,
                process_groups={},
                dist=self.dist,
                output_dir=self.tmpdir,
            )

        self.assertIn("Duplicate component key", str(ctx.exception))

    def test_coordinator_validates_process_groups(self):
        """Test that coordinator validates process group references."""
        stateful = MockStateful({"param": 1.0})
        components = [
            StateComponent(
                key="model",
                stateful=stateful,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="nonexistent_group",
            )
        ]

        with self.assertRaises(ValueError) as ctx:
            CheckpointCoordinator(
                state_components=components,
                process_groups={},  # Empty process groups
                dist=self.dist,
                output_dir=self.tmpdir,
            )

        self.assertIn("unknown process group", str(ctx.exception))

    def test_save_global_component(self):
        """Test saving GLOBAL component (rank 0 only)."""
        stateful = MockStateful({"param": torch.tensor([1.0, 2.0, 3.0])})
        components = [
            StateComponent(
                key="model",
                stateful=stateful,
                sharing_pattern=SharingPattern.GLOBAL,
            )
        ]

        coordinator = CheckpointCoordinator(
            state_components=components,
            process_groups={},
            dist=self.dist,
            output_dir=self.tmpdir,
        )

        checkpoint_path = coordinator.save_checkpoint(checkpoint_id="test")

        # Verify files were created
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertTrue(
            os.path.exists(os.path.join(checkpoint_path, "model_state.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(checkpoint_path, "checkpoint_manifest.json"))
        )

        # Verify manifest
        manifest = CheckpointManifest.load(
            os.path.join(checkpoint_path, "checkpoint_manifest.json")
        )
        self.assertIn("model", manifest.components)
        self.assertEqual(
            manifest.components["model"].sharing_pattern, SharingPattern.GLOBAL.value
        )

    def test_load_global_component(self):
        """Test loading GLOBAL component."""
        stateful = MockStateful({"param": torch.tensor([1.0, 2.0, 3.0])})
        components = [
            StateComponent(
                key="model",
                stateful=stateful,
                sharing_pattern=SharingPattern.GLOBAL,
            )
        ]

        coordinator = CheckpointCoordinator(
            state_components=components,
            process_groups={},
            dist=self.dist,
            output_dir=self.tmpdir,
        )

        # Save checkpoint
        checkpoint_path = coordinator.save_checkpoint(checkpoint_id="test")

        # Modify state
        stateful._data = {"param": torch.tensor([0.0, 0.0, 0.0])}

        # Load checkpoint
        coordinator.load_checkpoint(checkpoint_path)

        # Verify state was restored
        loaded_param = stateful._data["param"]
        self.assertTrue(torch.allclose(loaded_param, torch.tensor([1.0, 2.0, 3.0])))

    def test_save_per_rank_component_single_rank(self):
        """Test saving PER_RANK component with single rank."""
        stateful = MockStateful({"rank_data": 42})
        components = [
            StateComponent(
                key="rng",
                stateful=stateful,
                sharing_pattern=SharingPattern.PER_RANK,
            )
        ]

        coordinator = CheckpointCoordinator(
            state_components=components,
            process_groups={},
            dist=self.dist,
            output_dir=self.tmpdir,
        )

        checkpoint_path = coordinator.save_checkpoint(checkpoint_id="test")

        # Verify rank-specific file was created
        self.assertTrue(
            os.path.exists(os.path.join(checkpoint_path, "rng_state_rank_0.pt"))
        )

    def test_save_replicated_component(self):
        """Test saving REPLICATED component (rank 0 only)."""
        stateful = MockStateful({"param": torch.tensor([1.0, 2.0])})
        components = [
            StateComponent(
                key="optimizer",
                stateful=stateful,
                sharing_pattern=SharingPattern.REPLICATED,
            )
        ]

        coordinator = CheckpointCoordinator(
            state_components=components,
            process_groups={},
            dist=self.dist,
            output_dir=self.tmpdir,
        )

        checkpoint_path = coordinator.save_checkpoint(checkpoint_id="test")

        # Verify only rank 0 saved (no rank suffix)
        self.assertTrue(
            os.path.exists(os.path.join(checkpoint_path, "optimizer_state.pt"))
        )

        # Verify manifest indicates replication
        manifest = CheckpointManifest.load(
            os.path.join(checkpoint_path, "checkpoint_manifest.json")
        )
        self.assertIn("optimizer", manifest.components)
        self.assertEqual(manifest.components["optimizer"].ranks, [0])
        self.assertEqual(manifest.components["optimizer"].replicated_across, [0])

    def test_optional_component_failure(self):
        """Test that optional component failures don't crash save."""

        class FailingStateful(Stateful):
            def state_dict(self):
                raise RuntimeError("Intentional failure")

            def load_state_dict(self, state_dict):
                pass

        failing_stateful = FailingStateful()
        components = [
            StateComponent(
                key="optional",
                stateful=failing_stateful,
                sharing_pattern=SharingPattern.GLOBAL,
                required=False,  # Optional component
            )
        ]

        coordinator = CheckpointCoordinator(
            state_components=components,
            process_groups={},
            dist=self.dist,
            output_dir=self.tmpdir,
        )

        # Should not raise, just log warning
        checkpoint_path = coordinator.save_checkpoint(checkpoint_id="test")
        self.assertTrue(os.path.exists(checkpoint_path))

    def test_required_component_failure(self):
        """Test that required component failures crash save."""

        class FailingStateful(Stateful):
            def state_dict(self):
                raise RuntimeError("Intentional failure")

            def load_state_dict(self, state_dict):
                pass

        failing_stateful = FailingStateful()
        components = [
            StateComponent(
                key="required",
                stateful=failing_stateful,
                sharing_pattern=SharingPattern.GLOBAL,
                required=True,  # Required component
            )
        ]

        coordinator = CheckpointCoordinator(
            state_components=components,
            process_groups={},
            dist=self.dist,
            output_dir=self.tmpdir,
        )

        # Should raise
        with self.assertRaises(RuntimeError):
            coordinator.save_checkpoint(checkpoint_id="test")


if __name__ == "__main__":
    unittest.main()
