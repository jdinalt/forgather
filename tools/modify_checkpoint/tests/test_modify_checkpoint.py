"""
Tests for checkpoint parameter modification tool.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modify_checkpoint import (
    discover_checkpoint_files,
    list_modifiable_parameters,
    modify_state_dict,
    parse_value,
    save_checkpoint_atomically,
    update_checkpoint_manifest,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_optimizer_state():
    """Create a sample optimizer state dict."""
    return {
        "state": {},
        "param_groups": [
            {
                "params": [0, 1, 2],
                "lr": 0.001,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False,
            }
        ],
    }


@pytest.fixture
def sample_scheduler_state():
    """Create a sample scheduler state dict."""
    return {
        "last_epoch": 100,
        "_step_count": 101,
        "base_lrs": [0.001],
        "verbose": False,
    }


def test_parse_value():
    """Test value parsing from strings."""
    # Numbers
    assert parse_value("0.01") == 0.01
    assert parse_value("1e-4") == 1e-4
    assert parse_value("42") == 42

    # Booleans
    assert parse_value("True") is True
    assert parse_value("False") is False

    # Tuples
    assert parse_value("(0.9,0.999)") == (0.9, 0.999)
    assert parse_value("(0.9, 0.999)") == (0.9, 0.999)

    # Lists
    assert parse_value("[1,2,3]") == [1, 2, 3]
    assert parse_value("[0.001, 0.0001]") == [0.001, 0.0001]

    # Strings
    assert parse_value("'adam'") == "adam"


def test_list_parameters_optimizer(sample_optimizer_state):
    """Test listing optimizer parameters."""
    modifiable = list_modifiable_parameters(sample_optimizer_state, "optimizer")

    assert "param_groups" in modifiable
    assert len(modifiable["param_groups"]) == 1

    pg_params = modifiable["param_groups"][0]
    assert "lr" in pg_params
    assert "weight_decay" in pg_params
    assert "betas" in pg_params
    assert "eps" in pg_params
    assert "amsgrad" in pg_params

    # Verify types and values
    assert pg_params["lr"] == ("float", 0.001)
    assert pg_params["weight_decay"] == ("float", 0.0)
    assert pg_params["betas"] == ("tuple", (0.9, 0.999))
    assert pg_params["amsgrad"] == ("bool", False)

    # 'params' should not be included
    assert "params" not in pg_params


def test_list_parameters_scheduler(sample_scheduler_state):
    """Test listing scheduler parameters."""
    modifiable = list_modifiable_parameters(sample_scheduler_state, "scheduler")

    assert "state" in modifiable
    state = modifiable["state"]

    assert "last_epoch" in state
    assert "base_lrs" in state
    assert "_step_count" in state

    assert state["last_epoch"] == ("int", 100)
    assert state["_step_count"] == ("int", 101)
    assert state["base_lrs"] == ("list", [0.001])


def test_single_file_modification(temp_dir, sample_optimizer_state):
    """Test modifying a single optimizer file."""
    checkpoint_path = Path(temp_dir) / "optimizer_state.pt"
    torch.save(sample_optimizer_state, checkpoint_path)

    # Load and modify
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    modifications = [("weight_decay", "set", 0.01)]

    modified_state, changes = modify_state_dict(
        state_dict,
        modifications,
        "optimizer"
    )

    # Verify modification
    assert modified_state["param_groups"][0]["weight_decay"] == 0.01

    # Verify changes log
    assert len(changes) == 1
    assert changes[0]["parameter"] == "weight_decay"
    assert changes[0]["old_value"] == 0.0
    assert changes[0]["new_value"] == 0.01
    assert changes[0]["operation"] == "set"


def test_scheduler_modification(temp_dir, sample_scheduler_state):
    """Test modifying scheduler state."""
    checkpoint_path = Path(temp_dir) / "scheduler_state.pt"
    torch.save(sample_scheduler_state, checkpoint_path)

    # Load and modify
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    modifications = [("last_epoch", "set", 200)]

    modified_state, changes = modify_state_dict(
        state_dict,
        modifications,
        "scheduler"
    )

    # Verify modification
    assert modified_state["last_epoch"] == 200

    # Verify changes log
    assert len(changes) == 1
    assert changes[0]["parameter"] == "last_epoch"
    assert changes[0]["old_value"] == 100
    assert changes[0]["new_value"] == 200


def test_multiple_param_groups(temp_dir):
    """Test modifying optimizer with multiple param groups."""
    state_dict = {
        "state": {},
        "param_groups": [
            {"params": [0, 1], "lr": 0.001, "weight_decay": 0.0},
            {"params": [2, 3], "lr": 0.0001, "weight_decay": 0.01},
        ],
    }

    modifications = [("weight_decay", "set", 0.02)]

    # Modify all param groups
    modified_state, changes = modify_state_dict(
        state_dict,
        modifications,
        "optimizer"
    )

    assert modified_state["param_groups"][0]["weight_decay"] == 0.02
    assert modified_state["param_groups"][1]["weight_decay"] == 0.02
    assert len(changes) == 2


def test_specific_param_group(temp_dir):
    """Test modifying specific param group."""
    state_dict = {
        "state": {},
        "param_groups": [
            {"params": [0, 1], "lr": 0.001, "weight_decay": 0.0},
            {"params": [2, 3], "lr": 0.0001, "weight_decay": 0.01},
        ],
    }

    modifications = [("weight_decay", "set", 0.02)]

    # Modify only param group 0
    modified_state, changes = modify_state_dict(
        state_dict,
        modifications,
        "optimizer",
        param_group_idx=0
    )

    assert modified_state["param_groups"][0]["weight_decay"] == 0.02
    assert modified_state["param_groups"][1]["weight_decay"] == 0.01  # Unchanged
    assert len(changes) == 1
    assert changes[0]["param_group"] == 0


def test_scale_operation(temp_dir, sample_optimizer_state):
    """Test scale operation."""
    modifications = [("lr", "scale", 0.5)]

    modified_state, changes = modify_state_dict(
        sample_optimizer_state,
        modifications,
        "optimizer"
    )

    assert modified_state["param_groups"][0]["lr"] == 0.0005  # 0.001 * 0.5
    assert changes[0]["operation"] == "scale"


def test_multiple_modifications(temp_dir, sample_optimizer_state):
    """Test multiple modifications at once."""
    modifications = [
        ("weight_decay", "set", 0.01),
        ("lr", "scale", 0.5),
        ("betas", "set", (0.9, 0.98)),
    ]

    modified_state, changes = modify_state_dict(
        sample_optimizer_state,
        modifications,
        "optimizer"
    )

    assert modified_state["param_groups"][0]["weight_decay"] == 0.01
    assert modified_state["param_groups"][0]["lr"] == 0.0005
    assert modified_state["param_groups"][0]["betas"] == (0.9, 0.98)
    assert len(changes) == 3


def test_parameter_not_found(sample_optimizer_state):
    """Test error when parameter doesn't exist."""
    modifications = [("nonexistent_param", "set", 0.01)]

    with pytest.raises(ValueError, match="Parameter 'nonexistent_param' not found"):
        modify_state_dict(
            sample_optimizer_state,
            modifications,
            "optimizer"
        )


def test_scale_non_numeric(sample_optimizer_state):
    """Test error when scaling non-numeric parameter."""
    modifications = [("betas", "scale", 2.0)]

    with pytest.raises(ValueError, match="Cannot scale non-numeric parameter"):
        modify_state_dict(
            sample_optimizer_state,
            modifications,
            "optimizer"
        )


def test_atomic_save(temp_dir, sample_optimizer_state):
    """Test atomic file save operation."""
    checkpoint_path = Path(temp_dir) / "optimizer_state.pt"
    torch.save(sample_optimizer_state, checkpoint_path)

    # Modify state
    sample_optimizer_state["param_groups"][0]["weight_decay"] = 0.01

    # Save atomically
    success = save_checkpoint_atomically(
        str(checkpoint_path),
        sample_optimizer_state,
        backup=True,
        verbose=False
    )

    assert success

    # Verify backup exists
    backup_path = Path(str(checkpoint_path) + '.bak')
    assert backup_path.exists()

    # Verify temp file doesn't exist
    temp_path = Path(str(checkpoint_path) + '.tmp')
    assert not temp_path.exists()

    # Verify modified checkpoint loads correctly
    loaded = torch.load(checkpoint_path, map_location='cpu')
    assert loaded["param_groups"][0]["weight_decay"] == 0.01

    # Verify backup has original value
    backup = torch.load(backup_path, map_location='cpu')
    assert backup["param_groups"][0]["weight_decay"] == 0.0


def test_atomic_save_without_backup(temp_dir, sample_optimizer_state):
    """Test atomic save without backup creation."""
    checkpoint_path = Path(temp_dir) / "optimizer_state.pt"
    torch.save(sample_optimizer_state, checkpoint_path)

    sample_optimizer_state["param_groups"][0]["weight_decay"] = 0.01

    success = save_checkpoint_atomically(
        str(checkpoint_path),
        sample_optimizer_state,
        backup=False,
        verbose=False
    )

    assert success

    # Verify no backup
    backup_path = Path(str(checkpoint_path) + '.bak')
    assert not backup_path.exists()

    # Verify modified checkpoint loads correctly
    loaded = torch.load(checkpoint_path, map_location='cpu')
    assert loaded["param_groups"][0]["weight_decay"] == 0.01


def test_validation_failure_cleanup(temp_dir, monkeypatch):
    """Test that temp file is deleted on validation failure."""
    checkpoint_path = Path(temp_dir) / "optimizer_state.pt"
    state_dict = {"valid": "state"}
    torch.save(state_dict, checkpoint_path)

    # Mock torch.load to fail on temp file validation
    original_load = torch.load
    temp_path_str = str(checkpoint_path) + '.tmp'

    def mock_load(path, *args, **kwargs):
        if str(path) == temp_path_str:
            raise RuntimeError("Simulated validation failure")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', mock_load)

    # Attempt save - should fail
    with pytest.raises(RuntimeError, match="Modified checkpoint failed validation"):
        save_checkpoint_atomically(
            str(checkpoint_path),
            {"modified": "state"},
            backup=True,
            verbose=False
        )

    # Verify temp file was deleted
    assert not Path(temp_path_str).exists()

    # Verify original file is unchanged
    loaded = torch.load(checkpoint_path, map_location='cpu')
    assert loaded == {"valid": "state"}

    # Verify backup exists
    backup_path = Path(str(checkpoint_path) + '.bak')
    assert backup_path.exists()


def test_discover_single_file(temp_dir, sample_optimizer_state):
    """Test discovering checkpoint files - single file pattern."""
    checkpoint_path = Path(temp_dir) / "optimizer_state.pt"
    torch.save(sample_optimizer_state, checkpoint_path)

    files = discover_checkpoint_files(str(checkpoint_path), "optimizer")
    assert len(files) == 1
    assert files[0] == str(checkpoint_path)


def test_discover_per_rank_files(temp_dir, sample_optimizer_state):
    """Test discovering checkpoint files - PER_RANK pattern."""
    # Create per-rank files
    for rank in range(4):
        file_path = Path(temp_dir) / f"optimizer_state_rank_{rank}.pt"
        torch.save(sample_optimizer_state, file_path)

    files = discover_checkpoint_files(str(temp_dir), "optimizer")
    assert len(files) == 4
    assert all("rank_" in f for f in files)


def test_discover_with_manifest(temp_dir, sample_optimizer_state):
    """Test file discovery using checkpoint manifest."""
    # Create optimizer file
    optimizer_path = Path(temp_dir) / "optimizer_state.pt"
    torch.save(sample_optimizer_state, optimizer_path)

    # Create manifest
    manifest = {
        "checkpoint_path": str(temp_dir),
        "world_size": 1,
        "components": {
            "optimizer": {
                "sharing_pattern": "replicated",
                "ranks": [0],
                "size_bytes": os.path.getsize(optimizer_path),
            }
        }
    }

    manifest_path = Path(temp_dir) / "checkpoint_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    files = discover_checkpoint_files(str(temp_dir), "optimizer")
    assert len(files) == 1
    assert files[0] == str(optimizer_path)


def test_manifest_update(temp_dir, sample_optimizer_state):
    """Test updating checkpoint manifest after modification."""
    # Create checkpoint and manifest
    optimizer_path = Path(temp_dir) / "optimizer_state.pt"
    torch.save(sample_optimizer_state, optimizer_path)

    original_size = os.path.getsize(optimizer_path)

    manifest = {
        "checkpoint_path": str(temp_dir),
        "world_size": 1,
        "components": {
            "optimizer": {
                "sharing_pattern": "replicated",
                "ranks": [0],
                "size_bytes": original_size,
                "checksum": "abc123",
            }
        }
    }

    manifest_path = Path(temp_dir) / "checkpoint_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    # Modify checkpoint (change size)
    sample_optimizer_state["param_groups"][0]["new_field"] = "test"
    torch.save(sample_optimizer_state, optimizer_path)

    # Update manifest
    update_checkpoint_manifest(
        str(temp_dir),
        "optimizer",
        [str(optimizer_path)],
        verbose=False
    )

    # Verify manifest was updated
    with open(manifest_path, 'r') as f:
        updated_manifest = json.load(f)

    optimizer_info = updated_manifest["components"]["optimizer"]

    # Size should be updated
    new_size = os.path.getsize(optimizer_path)
    assert optimizer_info["size_bytes"] == new_size

    # Checksum should be removed
    assert "checksum" not in optimizer_info

    # Metadata should be added
    assert "metadata" in optimizer_info
    assert "modified_by" in optimizer_info["metadata"]
    assert "modified_at" in optimizer_info["metadata"]

    # Backup should exist
    backup_path = Path(str(manifest_path) + '.bak')
    assert backup_path.exists()


def test_manifest_update_atomic(temp_dir):
    """Test that manifest update is atomic."""
    manifest_path = Path(temp_dir) / "checkpoint_manifest.json"
    manifest = {"test": "data"}

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    # Update manifest
    update_checkpoint_manifest(
        str(temp_dir),
        "optimizer",
        [],
        verbose=False
    )

    # Verify temp file doesn't exist
    temp_path = Path(str(manifest_path) + '.tmp')
    assert not temp_path.exists()

    # Verify manifest still loads
    with open(manifest_path, 'r') as f:
        loaded = json.load(f)
    assert loaded is not None


def test_no_manifest_graceful(temp_dir, sample_optimizer_state):
    """Test graceful handling when manifest doesn't exist."""
    optimizer_path = Path(temp_dir) / "optimizer_state.pt"
    torch.save(sample_optimizer_state, optimizer_path)

    # Should not raise error
    update_checkpoint_manifest(
        str(temp_dir),
        "optimizer",
        [str(optimizer_path)],
        verbose=False
    )

    # No manifest should be created
    manifest_path = Path(temp_dir) / "checkpoint_manifest.json"
    assert not manifest_path.exists()


def test_file_not_found_error(temp_dir):
    """Test error when checkpoint file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
        discover_checkpoint_files(str(Path(temp_dir) / "nonexistent"), "optimizer")


def test_component_not_found_error(temp_dir):
    """Test error when component files don't exist."""
    # Create directory but no files
    checkpoint_dir = Path(temp_dir) / "checkpoint"
    checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No optimizer checkpoint files found"):
        discover_checkpoint_files(str(checkpoint_dir), "optimizer")


def test_end_to_end_workflow(temp_dir):
    """Integration test: complete workflow from list to modify."""
    # Create checkpoint directory
    checkpoint_dir = Path(temp_dir) / "checkpoint-1000"
    checkpoint_dir.mkdir()

    # Create optimizer state
    optimizer_state = {
        "state": {},
        "param_groups": [
            {
                "params": [0, 1, 2],
                "lr": 0.001,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            }
        ],
    }

    optimizer_path = checkpoint_dir / "optimizer_state.pt"
    torch.save(optimizer_state, optimizer_path)

    # Create scheduler state
    scheduler_state = {
        "last_epoch": 1000,
        "_step_count": 1001,
        "base_lrs": [0.001],
    }

    scheduler_path = checkpoint_dir / "scheduler_state.pt"
    torch.save(scheduler_state, scheduler_path)

    # Step 1: List parameters
    files = discover_checkpoint_files(str(checkpoint_dir), "optimizer")
    state = torch.load(files[0], map_location='cpu')
    modifiable = list_modifiable_parameters(state, "optimizer")

    assert "param_groups" in modifiable
    assert "weight_decay" in modifiable["param_groups"][0]

    # Step 2: Modify parameters
    modifications = [
        ("weight_decay", "set", 0.01),
        ("lr", "scale", 0.5),
    ]

    modified_state, changes = modify_state_dict(
        state,
        modifications,
        "optimizer"
    )

    # Step 3: Save atomically
    save_checkpoint_atomically(
        str(optimizer_path),
        modified_state,
        backup=True,
        verbose=False
    )

    # Step 4: Verify modifications
    loaded = torch.load(optimizer_path, map_location='cpu')
    assert loaded["param_groups"][0]["weight_decay"] == 0.01
    assert loaded["param_groups"][0]["lr"] == 0.0005

    # Step 5: Verify backup
    backup_path = Path(str(optimizer_path) + '.bak')
    backup = torch.load(backup_path, map_location='cpu')
    assert backup["param_groups"][0]["weight_decay"] == 0.0
    assert backup["param_groups"][0]["lr"] == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
