"""
Core types for distributed checkpoint abstraction.

This module defines the type system for state-aware checkpoint coordination
in hybrid parallelism scenarios. State components explicitly declare their
sharing patterns (GLOBAL, PER_RANK, REPLICATED, etc.) enabling automatic
distributed checkpoint coordination.

Design Goals:
- Explicit state semantics: No guessing which ranks should save what
- Dynamic pattern resolution: Sharing can be determined at runtime
- Composability: Easy to express hybrid parallelism (DDP x Pipeline, etc.)
- Validation: Detect misconfigurations and verify checkpoint integrity
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from torch.distributed.checkpoint.stateful import Stateful


class SharingPattern(Enum):
    """
    Defines how a state component is distributed across ranks in a distributed training job.

    The sharing pattern determines which ranks should save the component and how
    it should be loaded during checkpoint restore.

    Patterns:
        GLOBAL: Single shared copy across all ranks (e.g., trainer state with centralized dispatch)
                Only rank 0 saves, all ranks load the same state.

        PER_RANK: Each rank has unique state (e.g., pipeline stage parameters, RNG state)
                  Every rank saves its own state, loads its specific state.

        REPLICATED: Identical state across all ranks (e.g., DDP model weights, scheduler)
                    Only rank 0 saves (avoiding redundancy), all ranks load the same state.
                    Can optionally validate that all ranks have identical state.

        PER_GROUP: State shared within a process group, different across groups
                   (e.g., model shard shared within DP group but different across PP stages)
                   One rank per group saves, ranks load based on group membership.

        PER_NODE: State local to each node (e.g., node-specific caches)
                  One rank per node saves, ranks load based on node membership.

    Examples:
        # Simple Trainer (no parallelism)
        - model: GLOBAL
        - optimizer: GLOBAL
        - dataset: GLOBAL (or PER_RANK if each rank has independent iterator)
        - rng: PER_RANK

        # DDP Trainer (data parallel)
        - model: REPLICATED (DDP synchronizes weights)
        - optimizer: REPLICATED (DDP synchronizes optimizer state)
        - dataset: GLOBAL (if using DataloaderDispatcher) or PER_RANK
        - rng: PER_RANK

        # Pipeline Parallel Trainer
        - model: PER_RANK (different pipeline stage per rank)
        - optimizer: PER_RANK (optimizes different parameters)
        - scheduler: REPLICATED (shared LR schedule)
        - dataset: GLOBAL (rank 0 loads and broadcasts)
        - rng: PER_RANK

        # Hybrid DDP x Pipeline
        - model: PER_GROUP (shared within PP group, different across DP)
        - optimizer: PER_GROUP (per PP group)
        - scheduler: REPLICATED
        - dataset: PER_GROUP (one per DP group)
        - rng: PER_RANK
    """

    GLOBAL = "global"
    PER_RANK = "per_rank"
    REPLICATED = "replicated"
    PER_GROUP = "per_group"
    PER_NODE = "per_node"


@dataclass
class StateComponent:
    """
    Describes a checkpointable state component with its sharing semantics.

    A StateComponent declares how a piece of training state (model, optimizer, dataset, etc.)
    is distributed across ranks, enabling the checkpoint system to automatically coordinate
    distributed save/load operations.

    Args:
        key: Unique identifier for this component (e.g., "model", "optimizer", "dataset")
        stateful: Object implementing state_dict/load_state_dict protocol
        sharing_pattern: How this state is distributed across ranks
        process_group_name: Named process group for PER_GROUP pattern (e.g., "dp_group", "pp_group")
        required: Whether this component is required for training to continue.
                  If False, missing components during load are skipped with a warning.
        validate_replication: For REPLICATED pattern, verify all ranks have identical state.
                             Catches DDP synchronization bugs.
        validation_level: How thorough to validate replication ("none", "quick", "tensor", "full").
                         - "none": No validation (fastest)
                         - "quick": Hash-based validation (fast, catches most issues)
                         - "tensor": Per-tensor checksums (moderate, more accurate)
                         - "full": Full tensor comparison (slow, catches all differences)
                         Only used when validate_replication=True.
        metadata: Optional metadata for debugging and validation (e.g., config hash, version)

    Examples:
        # Simple global optimizer
        StateComponent(
            key="optimizer",
            stateful=optimizer,
            sharing_pattern=SharingPattern.GLOBAL,
        )

        # DDP model with replication validation
        StateComponent(
            key="model",
            stateful=model,
            sharing_pattern=SharingPattern.REPLICATED,
            validate_replication=True,  # Verify all ranks match
        )

        # Pipeline stage (different per rank)
        StateComponent(
            key="model",
            stateful=pipeline_stage,
            sharing_pattern=SharingPattern.PER_RANK,
        )

        # Hybrid parallelism: model shard per DP group
        StateComponent(
            key="model",
            stateful=model_shard,
            sharing_pattern=SharingPattern.PER_GROUP,
            process_group_name="dp_group",
        )

        # Optional dataset state (may not exist for all datasets)
        StateComponent(
            key="dataset",
            stateful=train_dataloader,
            sharing_pattern=SharingPattern.GLOBAL,
            required=False,  # OK if dataset doesn't support state_dict
        )
    """

    key: str
    stateful: Stateful
    sharing_pattern: SharingPattern
    process_group_name: Optional[str] = None
    required: bool = True
    validate_replication: bool = False
    validation_level: str = "tensor"  # "none", "quick", "tensor", "full"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate StateComponent configuration."""
        if self.sharing_pattern == SharingPattern.PER_GROUP:
            if not self.process_group_name:
                raise ValueError(
                    f"StateComponent '{self.key}' with PER_GROUP pattern "
                    "must specify process_group_name"
                )

        if self.validate_replication and self.sharing_pattern != SharingPattern.REPLICATED:
            raise ValueError(
                f"StateComponent '{self.key}' has validate_replication=True "
                f"but sharing_pattern is {self.sharing_pattern.value}. "
                "Replication validation only applies to REPLICATED pattern."
            )


@dataclass
class ComponentManifest:
    """
    Manifest entry for a single state component in a checkpoint.

    Records metadata about what was saved, enabling validation during load
    and debugging of checkpoint structure.

    Args:
        key: Component identifier (matches StateComponent.key)
        sharing_pattern: How this component is distributed
        ranks: List of ranks that saved this component
        replicated_across: For REPLICATED/PER_GROUP, which ranks share this state
        group_name: Process group name for PER_GROUP components
        size_bytes: Total size of saved state in bytes (sum across all files)
        checksum: Optional hash of state for validation
        metadata: Additional metadata (e.g., component version, config hash)

    The manifest enables:
    - Validation: Verify checkpoint structure matches expected configuration
    - Debugging: Understand which ranks saved what
    - Optimization: Skip redundant loads for REPLICATED state
    - Migration: Support checkpoint format evolution
    """

    key: str
    sharing_pattern: str  # SharingPattern.value
    ranks: List[int]
    replicated_across: Optional[List[int]] = None
    group_name: Optional[str] = None
    size_bytes: int = 0
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "key": self.key,
            "sharing_pattern": self.sharing_pattern,
            "ranks": self.ranks,
            "replicated_across": self.replicated_across,
            "group_name": self.group_name,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentManifest":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class CheckpointManifest:
    """
    Complete manifest of a distributed checkpoint.

    Records comprehensive metadata about checkpoint structure, enabling validation,
    debugging, and backward compatibility.

    Args:
        checkpoint_path: Path to checkpoint directory
        world_size: Number of ranks that participated in checkpoint save
        timestamp: When checkpoint was created
        components: Manifest entries for each saved component
        training_args_hash: Optional hash of training config for validation
        forgather_version: Version of Forgather that created checkpoint
        pytorch_version: Version of PyTorch used
        metadata: Additional checkpoint-level metadata

    The manifest is saved as checkpoint_manifest.json in the checkpoint directory.
    During load, it's used to:
    - Verify checkpoint compatibility with current configuration
    - Validate all required components are present
    - Support backward compatibility with old checkpoints (no manifest)
    - Debug checkpoint structure issues

    Example manifest structure:
    {
        "checkpoint_path": "/path/to/checkpoint",
        "world_size": 8,
        "timestamp": "2025-01-24T10:30:00",
        "components": {
            "model": {
                "key": "model",
                "sharing_pattern": "per_group",
                "ranks": [0, 4],
                "group_name": "pp_group",
                ...
            },
            "optimizer": {...},
            ...
        },
        "training_args_hash": "abc123...",
        "metadata": {"model_type": "llama", ...}
    }
    """

    checkpoint_path: str
    world_size: int
    timestamp: datetime
    components: Dict[str, ComponentManifest]
    training_args_hash: Optional[str] = None
    forgather_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "world_size": self.world_size,
            "timestamp": self.timestamp.isoformat(),
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "training_args_hash": self.training_args_hash,
            "forgather_version": self.forgather_version,
            "pytorch_version": self.pytorch_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointManifest":
        """Deserialize from dictionary."""
        return cls(
            checkpoint_path=data["checkpoint_path"],
            world_size=data["world_size"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            components={
                k: ComponentManifest.from_dict(v) for k, v in data["components"].items()
            },
            training_args_hash=data.get("training_args_hash"),
            forgather_version=data.get("forgather_version"),
            pytorch_version=data.get("pytorch_version"),
            metadata=data.get("metadata", {}),
        )

    def save(self, manifest_path: str) -> None:
        """Save manifest to JSON file."""
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, manifest_path: str) -> "CheckpointManifest":
        """Load manifest from JSON file."""
        with open(manifest_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_state_hash(state_dict: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of state_dict for replication validation.

    Args:
        state_dict: State dictionary to hash

    Returns:
        Hex-encoded SHA256 hash of state_dict

    This is used to verify REPLICATED state is actually identical across ranks.
    The hash is computed over the serialized state_dict representation.

    Limitations (simplified implementation for speed):
    - Tensors: Only shape/dtype/device hashed (not actual values)
    - Floats: No precision rounding (may differ across platforms)
    - Dicts: Top-level keys sorted only (not recursive)

    This catches most synchronization bugs (shape changes, device mismatches,
    structural differences) but may miss subtle numerical differences.

    For critical validation, use ValidationLevel.TENSOR or FULL instead of QUICK.
    """
    # Convert state_dict to deterministic string representation
    # Note: Simplified hash for performance - tensors converted to metadata only
    state_str = json.dumps(
        _state_dict_to_serializable(state_dict), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(state_str).hexdigest()


def _state_dict_to_serializable(obj: Any) -> Any:
    """
    Convert state_dict to JSON-serializable format for hashing.

    Handles common PyTorch types (tensors, etc.) by converting to
    representative values.
    """
    import torch

    if isinstance(obj, dict):
        return {k: _state_dict_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_state_dict_to_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        # For hashing, use tensor metadata (shape, dtype, device) and a sample of values
        # Full tensor comparison would be too expensive
        return {
            "type": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            # Sample a few values for lightweight validation
            "sample": obj.flatten()[:min(10, obj.numel())].tolist()
            if obj.numel() > 0
            else [],
        }
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Fallback: use string representation
        return str(obj)
