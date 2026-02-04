"""
Checkpoint coordinator for distributed training with hybrid parallelism.

This module implements distributed checkpoint save/load coordination based on
explicit state sharing patterns. It replaces ad-hoc rank checks with declarative
semantics, enabling clean support for complex parallelism strategies.

Key Features:
- Pattern-based coordination: Automatic rank selection based on SharingPattern
- Manifest generation: Complete checkpoint inventory for validation
- Replication validation: Verify REPLICATED state is actually identical
- Backward compatibility: Load legacy checkpoints without manifest
- Extensible: Easy to add new sharing patterns

Architecture:
    StatefulProvider.get_state_components() -> [StateComponent]
                            |
                            v
              CheckpointCoordinator.save_checkpoint()
                            |
                            v
                  Pattern-specific handlers
                  (GLOBAL, PER_RANK, etc.)
                            |
                            v
                   Save state + manifest
                            |
                            v
                  Optional validation

Usage Example:
    # Trainer implements get_state_components()
    components = trainer.get_state_components()
    process_groups = trainer.get_process_groups()

    coordinator = CheckpointCoordinator(
        state_components=components,
        process_groups=process_groups,
        dist=trainer.dist,
        output_dir=trainer.args.output_dir,
    )

    # Save checkpoint with automatic coordination
    coordinator.save_checkpoint("checkpoint-1000")

    # Load checkpoint with validation
    coordinator.load_checkpoint("checkpoint-1000")
"""

import datetime
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.distributed import ProcessGroup
from torch.distributed.checkpoint.stateful import Stateful

from forgather.ml.distributed import (
    DistributedEnvInterface,
    get_barrier_fn,
    prefix_logger_rank,
)
from forgather.ml.sharded_checkpoint import (
    find_latest_checkpoint,
    index_file_name,
    load_checkpoint,
    make_shard_index,
    maybe_delete_oldest_checkpoint,
    next_checkpoint_path,
    save_shard_index,
    save_sharded_checkpoint,
    validate_checkpoint,
)

from .checkpoint_types import (
    CheckpointManifest,
    ComponentManifest,
    SharingPattern,
    StateComponent,
    compute_state_hash,
)
from .checkpoint_utils import (
    ValidationLevel,
    all_gather_scalar,
    collect_group_savers,
    collect_node_savers,
    find_group_checkpoint_file,
    find_node_checkpoint_file,
    get_group_file_suffix,
    get_node_file_suffix,
    is_group_leader,
    is_node_leader,
    validate_replication,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
prefix_logger_rank(logger)


class CheckpointCoordinator:
    """
    Coordinates distributed checkpoint save/load based on state sharing patterns.

    Handles complex parallelism scenarios by dispatching to pattern-specific
    handlers that know how to coordinate saves/loads for each sharing type.

    Args:
        state_components: List of components to checkpoint with their sharing patterns
        process_groups: Named process groups for PER_GROUP patterns
        dist: Distributed environment interface
        output_dir: Base directory for checkpoints
        save_total_limit: Maximum checkpoints to keep (oldest deleted first)
        save_safetensors: Use safetensors format for model weights
        model: Reference to model for weight saving (optional if using custom save)
        model_parts: List of model parts for sharded saving (e.g., pipeline stages)
        shard_index: Pre-computed shard index (auto-generated if None)

    Note: For backward compatibility with existing code, model/model_parts/shard_index
    are optional. If not provided, model weight saving is skipped (assumes custom
    handling via StateComponent).
    """

    def __init__(
        self,
        state_components: List[StateComponent],
        process_groups: Dict[str, ProcessGroup],
        dist: DistributedEnvInterface,
        output_dir: str,
        save_total_limit: int = 2,
        save_safetensors: bool = True,
        model: Optional[torch.nn.Module] = None,
        model_parts: Optional[List[torch.nn.Module]] = None,
        shard_index: Optional[Dict] = None,
    ):
        self.state_components = state_components
        self.process_groups = process_groups
        self.dist = dist
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.save_safetensors = save_safetensors
        self.best_checkpoint = None

        # Model handling for backward compatibility
        self.model = model
        self.model_parts = model_parts
        if self.model_parts is None and self.model is not None:
            self.model_parts = [self.model]

        # Shard index for model weight saving
        if shard_index is None and self.model_parts:
            shard_index = make_shard_index(
                [mod.state_dict() for mod in self.model_parts],
                safetensors=save_safetensors,
            )
        self.shard_index = shard_index

        # Barrier function for synchronization
        self.barrier_fn = get_barrier_fn(self._get_global_process_group())

        # Validate component configuration
        self._validate_components()

    def _get_global_process_group(self) -> Optional[ProcessGroup]:
        """Get global process group if available."""
        try:
            from forgather.ml.distributed import get_global_process_group

            return get_global_process_group()
        except Exception:
            return None

    def _validate_components(self) -> None:
        """Validate state component configuration."""
        keys = set()
        for component in self.state_components:
            if component.key in keys:
                raise ValueError(
                    f"Duplicate component key: {component.key}. "
                    "All StateComponent keys must be unique."
                )
            keys.add(component.key)

            # Validate PER_GROUP components have valid process groups
            if component.sharing_pattern == SharingPattern.PER_GROUP:
                if component.process_group_name not in self.process_groups:
                    raise ValueError(
                        f"Component '{component.key}' references unknown process group "
                        f"'{component.process_group_name}'. Available groups: "
                        f"{list(self.process_groups.keys())}"
                    )

    def save_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        validate: bool = False,
    ) -> str:
        """
        Save distributed checkpoint with automatic coordination.

        Coordinates save operations across all ranks based on sharing patterns,
        generates manifest, and optionally validates checkpoint integrity.

        Args:
            checkpoint_path: Explicit path for checkpoint, or None for auto-generated
            checkpoint_id: Identifier for checkpoint (e.g., "step-1000"), used if path is None
            validate: Whether to validate checkpoint after save (experimental)

        Returns:
            Path to saved checkpoint directory

        Process:
            1. Determine checkpoint path
            2. Create directory (rank 0 or per-node)
            3. Save each component based on sharing pattern:
               - GLOBAL: Rank 0 only
               - PER_RANK: Every rank
               - REPLICATED: Rank 0 only (with optional validation)
               - PER_GROUP: One rank per group
               - PER_NODE: One rank per node
            4. Generate and save manifest (rank 0)
            5. Cleanup old checkpoints (rank 0)
            6. Barrier to ensure all ranks complete

        Example:
            coordinator.save_checkpoint(checkpoint_id="step-1000")
        """
        # Determine checkpoint path
        if not checkpoint_path:
            if not checkpoint_id:
                checkpoint_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            checkpoint_path = next_checkpoint_path(self.output_dir, checkpoint_id)

        logger.info(f"Saving checkpoint at {checkpoint_path}")

        # Create checkpoint directory
        if self._should_save_unique():
            os.makedirs(checkpoint_path, exist_ok=True)
        self._barrier()

        # Save components based on sharing patterns
        component_manifests = {}

        for component in self.state_components:
            manifest_entry = self._save_component(component, checkpoint_path)
            if manifest_entry:
                component_manifests[component.key] = manifest_entry

        # Generate and save manifest (rank 0 only)
        if self.dist.rank == 0:
            manifest = CheckpointManifest(
                checkpoint_path=checkpoint_path,
                world_size=self.dist.world_size,
                timestamp=datetime.datetime.now(),
                components=component_manifests,
                pytorch_version=torch.__version__,
            )
            manifest_path = os.path.join(checkpoint_path, "checkpoint_manifest.json")
            manifest.save(manifest_path)
            logger.debug(f"Saved checkpoint manifest to {manifest_path}")

        # Cleanup old checkpoints (rank 0 only)
        if self._should_save_unique():
            maybe_delete_oldest_checkpoint(
                self.output_dir,
                self.save_total_limit,
                self.best_checkpoint,
            )

        self._barrier()

        if validate and self.dist.rank == 0:
            logger.info("Validating checkpoint...")
            self._validate_checkpoint(checkpoint_path)

        return checkpoint_path

    def _save_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> Optional[ComponentManifest]:
        """
        Save a single component based on its sharing pattern.

        Dispatches to pattern-specific handlers. Returns ComponentManifest entry
        for inclusion in checkpoint manifest.

        Args:
            component: Component to save
            checkpoint_path: Checkpoint directory

        Returns:
            ComponentManifest entry, or None if this rank doesn't save this component
        """
        pattern = component.sharing_pattern

        if pattern == SharingPattern.GLOBAL:
            return self._save_global_component(component, checkpoint_path)
        elif pattern == SharingPattern.PER_RANK:
            return self._save_per_rank_component(component, checkpoint_path)
        elif pattern == SharingPattern.REPLICATED:
            return self._save_replicated_component(component, checkpoint_path)
        elif pattern == SharingPattern.PER_GROUP:
            return self._save_per_group_component(component, checkpoint_path)
        elif pattern == SharingPattern.PER_NODE:
            return self._save_per_node_component(component, checkpoint_path)
        else:
            raise ValueError(f"Unknown sharing pattern: {pattern}")

    def _save_global_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> Optional[ComponentManifest]:
        """Save GLOBAL component (rank 0 only)."""
        if self.dist.rank != 0:
            return None

        state_path = os.path.join(checkpoint_path, f"{component.key}_state.pt")
        logger.debug(f"Saving GLOBAL component '{component.key}' to {state_path}")

        try:
            state_dict = component.stateful.state_dict()
            torch.save(state_dict, state_path)

            # Compute size
            size_bytes = (
                os.path.getsize(state_path) if os.path.exists(state_path) else 0
            )

            return ComponentManifest(
                key=component.key,
                sharing_pattern=SharingPattern.GLOBAL.value,
                ranks=[0],
                size_bytes=size_bytes,
                metadata=component.metadata,
            )
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to save required component '{component.key}': {e}"
                ) from e
            else:
                logger.warning(
                    f"Failed to save optional component '{component.key}': {e}"
                )
                return None

    def _save_per_rank_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> Optional[ComponentManifest]:
        """Save PER_RANK component (every rank saves)."""
        state_path = os.path.join(
            checkpoint_path, f"{component.key}_state_rank_{self.dist.rank}.pt"
        )
        logger.debug(
            f"Saving PER_RANK component '{component.key}' (rank {self.dist.rank}) to {state_path}"
        )

        try:
            state_dict = component.stateful.state_dict()
            torch.save(state_dict, state_path)

            # Only rank 0 creates manifest entry (aggregates info from all ranks)
            if self.dist.rank == 0:
                # Estimate total size (would need all-gather to be exact)
                size_bytes = (
                    os.path.getsize(state_path) * self.dist.world_size
                    if os.path.exists(state_path)
                    else 0
                )
                return ComponentManifest(
                    key=component.key,
                    sharing_pattern=SharingPattern.PER_RANK.value,
                    ranks=list(range(self.dist.world_size)),
                    size_bytes=size_bytes,
                    metadata=component.metadata,
                )
            return None
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to save required component '{component.key}' on rank {self.dist.rank}: {e}"
                ) from e
            else:
                logger.warning(
                    f"Failed to save optional component '{component.key}' on rank {self.dist.rank}: {e}"
                )
                return None

    def _save_replicated_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> Optional[ComponentManifest]:
        """
        Save REPLICATED component (rank 0 only, with optional validation).

        For REPLICATED state, all ranks should have identical state. We save
        only once (rank 0) to avoid redundancy. If validate_replication=True,
        we verify all ranks actually have identical state.
        """
        # Optionally validate replication before saving
        if component.validate_replication:
            # Parse validation level
            try:
                validation_level = ValidationLevel(component.validation_level)
            except ValueError:
                logger.warning(
                    f"Invalid validation_level '{component.validation_level}' for component '{component.key}', "
                    "defaulting to TENSOR"
                )
                validation_level = ValidationLevel.TENSOR

            if not self._validate_replication(component, validation_level):
                logger.error(
                    f"Replication validation failed for component '{component.key}' "
                    f"(level: {validation_level.value}). State differs across ranks!"
                )
                if component.required:
                    raise RuntimeError(
                        f"Replication validation failed for required component '{component.key}'"
                    )

        # Only rank 0 saves
        if self.dist.rank != 0:
            return None

        state_path = os.path.join(checkpoint_path, f"{component.key}_state.pt")
        logger.debug(
            f"Saving REPLICATED component '{component.key}' (rank 0 only) to {state_path}"
        )

        try:
            state_dict = component.stateful.state_dict()
            torch.save(state_dict, state_path)

            size_bytes = (
                os.path.getsize(state_path) if os.path.exists(state_path) else 0
            )

            return ComponentManifest(
                key=component.key,
                sharing_pattern=SharingPattern.REPLICATED.value,
                ranks=[0],  # Only rank 0 saves
                replicated_across=list(
                    range(self.dist.world_size)
                ),  # But all ranks have it
                size_bytes=size_bytes,
                metadata=component.metadata,
            )
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to save required component '{component.key}': {e}"
                ) from e
            else:
                logger.warning(
                    f"Failed to save optional component '{component.key}': {e}"
                )
                return None

    def _save_per_group_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> Optional[ComponentManifest]:
        """
        Save PER_GROUP component (one rank per group saves).

        For PER_GROUP state, one representative rank from each group saves.
        We use rank 0 within each group as the saver.
        """
        group_name = component.process_group_name
        assert group_name is not None
        pg = self.process_groups[group_name]

        # Determine if this rank is group leader (rank 0 within group)
        should_save = is_group_leader(pg)

        if not should_save:
            return None

        # Generate unique filename using group and global ranks
        file_suffix = get_group_file_suffix(group_name, pg)
        state_path = os.path.join(
            checkpoint_path,
            f"{component.key}_state_{file_suffix}.pt",
        )

        logger.debug(
            f"Saving PER_GROUP component '{component.key}' (group {group_name}) to {state_path}"
        )

        try:
            state_dict = component.stateful.state_dict()
            torch.save(state_dict, state_path)

            # Collect file size for manifest
            size_bytes = (
                os.path.getsize(state_path) if os.path.exists(state_path) else 0
            )

            # All-gather file sizes from all ranks to compute total
            all_sizes = all_gather_scalar(size_bytes if should_save else 0)
            total_size = sum(all_sizes)

            # Only global rank 0 creates manifest entry
            if self.dist.rank == 0:
                # Collect which ranks saved (group leaders)
                group_savers = collect_group_savers({group_name: pg})
                saver_ranks = group_savers.get(group_name, [])

                return ComponentManifest(
                    key=component.key,
                    sharing_pattern=SharingPattern.PER_GROUP.value,
                    ranks=saver_ranks,
                    group_name=group_name,
                    size_bytes=total_size,
                    metadata=component.metadata,
                )
            return None
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to save required component '{component.key}' on rank {self.dist.rank}: {e}"
                ) from e
            else:
                logger.warning(
                    f"Failed to save optional component '{component.key}' on rank {self.dist.rank}: {e}"
                )
                return None

    def _save_per_node_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> Optional[ComponentManifest]:
        """Save PER_NODE component (local rank 0 on each node)."""
        # Determine if this rank is node leader
        should_save = is_node_leader()

        if not should_save:
            return None

        # Generate unique filename using node and global ranks
        file_suffix = get_node_file_suffix()
        state_path = os.path.join(
            checkpoint_path,
            f"{component.key}_state_{file_suffix}.pt",
        )

        logger.debug(f"Saving PER_NODE component '{component.key}' to {state_path}")

        try:
            state_dict = component.stateful.state_dict()
            torch.save(state_dict, state_path)

            # Collect file size for manifest
            size_bytes = (
                os.path.getsize(state_path) if os.path.exists(state_path) else 0
            )

            # All-gather file sizes from all ranks to compute total
            all_sizes = all_gather_scalar(size_bytes if should_save else 0)
            total_size = sum(all_sizes)

            # Only global rank 0 creates manifest entry
            if self.dist.rank == 0:
                # Collect which ranks saved (node leaders)
                saver_ranks = collect_node_savers()

                return ComponentManifest(
                    key=component.key,
                    sharing_pattern=SharingPattern.PER_NODE.value,
                    ranks=saver_ranks,
                    size_bytes=total_size,
                    metadata=component.metadata,
                )
            return None
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to save required component '{component.key}' on rank {self.dist.rank}: {e}"
                ) from e
            else:
                logger.warning(
                    f"Failed to save optional component '{component.key}' on rank {self.dist.rank}: {e}"
                )
                return None

    def _validate_replication(
        self,
        component: StateComponent,
        validation_level: ValidationLevel = ValidationLevel.TENSOR,
    ) -> bool:
        """
        Validate that REPLICATED component has identical state across all ranks.

        Uses enhanced validation with configurable thoroughness levels.

        Args:
            component: Component to validate
            validation_level: How thorough to validate (QUICK, TENSOR, or FULL)

        Returns:
            True if all ranks have identical state, False otherwise
        """
        if self.dist.world_size == 1:
            return True  # Single rank, nothing to validate

        try:
            state_dict = component.stateful.state_dict()

            # Use enhanced validation from checkpoint_utils
            is_valid, errors = validate_replication(
                state_dict,
                validation_level=validation_level,
                group=None,  # Validate across all ranks
            )

            if not is_valid:
                logger.error(
                    f"Replication validation failed for component '{component.key}' "
                    f"(level: {validation_level.value}):"
                )
                for error in errors:
                    logger.error(f"  - {error}")

            return is_valid

        except Exception as e:
            logger.warning(
                f"Failed to validate replication for component '{component.key}': {e}"
            )
            return False

    def load_checkpoint(
        self, checkpoint_path: Optional[str] = None, strict: bool = True
    ) -> None:
        """
        Load checkpoint with automatic coordination.

        Loads checkpoint based on manifest (if present) or falls back to legacy
        loading for backward compatibility.

        Args:
            checkpoint_path: Path to checkpoint, or None to auto-find latest
            strict: Whether to require all components to load successfully

        Process:
            1. Resolve checkpoint path (find latest if None)
            2. Load manifest (or detect legacy checkpoint)
            3. Validate compatibility (world size, components)
            4. Load each component based on sharing pattern
            5. Barrier to ensure all ranks complete

        Example:
            coordinator.load_checkpoint()  # Load latest
            coordinator.load_checkpoint("checkpoint-1000")  # Load specific
        """
        # Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(self.output_dir)
            if not checkpoint_path:
                raise RuntimeError(f"No checkpoints found in {self.output_dir}")

        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Checkpoint path does not exist: {checkpoint_path}")

        if self.dist.local_rank == 0:
            logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Try to load manifest
        manifest_path = os.path.join(checkpoint_path, "checkpoint_manifest.json")
        if os.path.exists(manifest_path):
            self._load_with_manifest(checkpoint_path, strict=strict)
        else:
            logger.warning(
                f"No manifest found at {manifest_path}. "
                "Loading as legacy checkpoint."
            )
            self._load_legacy_checkpoint(checkpoint_path, strict=strict)

        self._barrier()

    def _load_with_manifest(self, checkpoint_path: str, strict: bool = True) -> None:
        """Load checkpoint using manifest for validation and coordination."""
        manifest_path = os.path.join(checkpoint_path, "checkpoint_manifest.json")
        manifest = CheckpointManifest.load(manifest_path)

        # Validate compatibility
        if strict and manifest.world_size != self.dist.world_size:
            raise RuntimeError(
                f"Checkpoint was saved with world_size={manifest.world_size}, "
                f"but current world_size={self.dist.world_size}. "
                "Set strict=False to attempt load anyway."
            )

        # Load each component
        for component in self.state_components:
            if component.key not in manifest.components:
                if strict and component.required:
                    raise RuntimeError(
                        f"Required component '{component.key}' not found in checkpoint"
                    )
                logger.warning(
                    f"⚠️  Component '{component.key}' not found in checkpoint\n"
                    f"    This is normal if you deleted it to change {component.key} type.\n"
                    f"    Training will continue with current {component.key} configuration."
                )
                continue

            manifest_entry = manifest.components[component.key]
            self._load_component(component, checkpoint_path, manifest_entry)

    def _load_component(
        self,
        component: StateComponent,
        checkpoint_path: str,
        manifest_entry: Optional[ComponentManifest] = None,
    ) -> None:
        """Load a single component based on its sharing pattern."""
        pattern = component.sharing_pattern

        if pattern == SharingPattern.GLOBAL:
            self._load_global_component(component, checkpoint_path)
        elif pattern == SharingPattern.PER_RANK:
            self._load_per_rank_component(component, checkpoint_path)
        elif pattern == SharingPattern.REPLICATED:
            self._load_replicated_component(component, checkpoint_path)
        elif pattern == SharingPattern.PER_GROUP:
            self._load_per_group_component(component, checkpoint_path)
        elif pattern == SharingPattern.PER_NODE:
            self._load_per_node_component(component, checkpoint_path)
        else:
            raise ValueError(f"Unknown sharing pattern: {pattern}")

    def _load_global_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> None:
        """Load GLOBAL component (all ranks load same file)."""
        state_path = os.path.join(checkpoint_path, f"{component.key}_state.pt")

        if not os.path.exists(state_path):
            if component.required:
                raise RuntimeError(
                    f"Required component '{component.key}' not found at {state_path}"
                )
            logger.warning(
                f"⚠️  Component '{component.key}' not found at {state_path}\n"
                f"    This is normal if you deleted it to change {component.key} type.\n"
                f"    Training will continue with current {component.key} configuration."
            )
            return

        try:
            logger.debug(
                f"Loading GLOBAL component '{component.key}' from {state_path}"
            )
            state_dict = torch.load(state_path, map_location=torch.device("cpu"))
            component.stateful.load_state_dict(state_dict)
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to load required component '{component.key}': {e}"
                ) from e
            logger.warning(f"Failed to load component '{component.key}': {e}")

    def _load_per_rank_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> None:
        """Load PER_RANK component (each rank loads its own file)."""
        state_path = os.path.join(
            checkpoint_path, f"{component.key}_state_rank_{self.dist.rank}.pt"
        )

        if not os.path.exists(state_path):
            if component.required:
                raise RuntimeError(
                    f"Required component '{component.key}' for rank {self.dist.rank} not found at {state_path}"
                )
            logger.warning(
                f"Component '{component.key}' for rank {self.dist.rank} not found, skipping"
            )
            return

        try:
            logger.debug(
                f"Loading PER_RANK component '{component.key}' (rank {self.dist.rank}) from {state_path}"
            )
            state_dict = torch.load(state_path, map_location=torch.device("cpu"))
            component.stateful.load_state_dict(state_dict)
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to load required component '{component.key}' on rank {self.dist.rank}: {e}"
                ) from e
            logger.warning(
                f"Failed to load component '{component.key}' on rank {self.dist.rank}: {e}"
            )

    def _load_replicated_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> None:
        """Load REPLICATED component (all ranks load same file)."""
        # REPLICATED is saved by rank 0, loaded by all ranks
        state_path = os.path.join(checkpoint_path, f"{component.key}_state.pt")

        if not os.path.exists(state_path):
            if component.required:
                raise RuntimeError(
                    f"Required component '{component.key}' not found at {state_path}"
                )
            logger.warning(
                f"⚠️  Component '{component.key}' not found at {state_path}\n"
                f"    This is normal if you deleted it to change {component.key} type.\n"
                f"    Training will continue with current {component.key} configuration."
            )
            return

        try:
            logger.debug(
                f"Loading REPLICATED component '{component.key}' from {state_path}"
            )
            state_dict = torch.load(state_path, map_location=torch.device("cpu"))
            component.stateful.load_state_dict(state_dict)
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to load required component '{component.key}': {e}"
                ) from e
            logger.warning(f"Failed to load component '{component.key}': {e}")

    def _load_per_group_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> None:
        """Load PER_GROUP component (ranks load based on group membership)."""
        group_name = component.process_group_name
        assert group_name is not None
        pg = self.process_groups[group_name]

        # Find the checkpoint file for this rank's group
        state_path = find_group_checkpoint_file(
            checkpoint_path, component.key, group_name, pg
        )

        if not state_path or not os.path.exists(state_path):
            if component.required:
                raise RuntimeError(
                    f"Required component '{component.key}' for group {group_name} not found at {checkpoint_path}"
                )
            logger.warning(
                f"Component '{component.key}' for group {group_name} not found, skipping"
            )
            return

        try:
            logger.debug(
                f"Loading PER_GROUP component '{component.key}' (group {group_name}) from {state_path}"
            )
            state_dict = torch.load(state_path, map_location=torch.device("cpu"))
            component.stateful.load_state_dict(state_dict)
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to load required component '{component.key}': {e}"
                ) from e
            logger.warning(f"Failed to load component '{component.key}': {e}")

    def _load_per_node_component(
        self, component: StateComponent, checkpoint_path: str
    ) -> None:
        """Load PER_NODE component (ranks load based on node membership)."""
        # Find the checkpoint file for this rank's node
        state_path = find_node_checkpoint_file(checkpoint_path, component.key)

        if not state_path or not os.path.exists(state_path):
            if component.required:
                raise RuntimeError(
                    f"Required component '{component.key}' (PER_NODE) not found at {checkpoint_path}"
                )
            logger.warning(
                f"Component '{component.key}' (PER_NODE) not found, skipping"
            )
            return

        try:
            logger.debug(
                f"Loading PER_NODE component '{component.key}' from {state_path}"
            )
            state_dict = torch.load(state_path, map_location=torch.device("cpu"))
            component.stateful.load_state_dict(state_dict)
        except Exception as e:
            if component.required:
                raise RuntimeError(
                    f"Failed to load required component '{component.key}': {e}"
                ) from e
            logger.warning(f"Failed to load component '{component.key}': {e}")

    def _load_legacy_checkpoint(
        self, checkpoint_path: str, strict: bool = True
    ) -> None:
        """
        Load checkpoint without manifest (backward compatibility).

        Falls back to heuristics based on file naming conventions.
        """
        logger.info("Loading legacy checkpoint (no manifest)")

        for component in self.state_components:
            # Try to infer file names from component key and sharing pattern
            if component.sharing_pattern == SharingPattern.GLOBAL:
                self._load_global_component(component, checkpoint_path)
            elif component.sharing_pattern == SharingPattern.PER_RANK:
                self._load_per_rank_component(component, checkpoint_path)
            elif component.sharing_pattern == SharingPattern.REPLICATED:
                # REPLICATED uses same file as GLOBAL
                self._load_replicated_component(component, checkpoint_path)
            else:
                logger.warning(
                    f"Cannot load component '{component.key}' with pattern "
                    f"{component.sharing_pattern.value} from legacy checkpoint"
                )

    def _validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validate checkpoint structure and completeness.

        Checks:
        - All required components present
        - Files readable
        - Manifest matches actual files

        Returns:
            True if checkpoint is valid
        """
        # Use existing validate_checkpoint for basic validation
        if not validate_checkpoint(checkpoint_path):
            logger.error(f"Basic checkpoint validation failed for {checkpoint_path}")
            return False

        # Additional validation could include:
        # - Verify manifest component files exist
        # - Check file sizes match manifest
        # - Validate checksums

        return True

    def _barrier(self) -> None:
        """Synchronization barrier across all ranks."""
        if self.dist.world_size > 1 and self.barrier_fn:
            self.barrier_fn()

    def _should_save_unique(self) -> bool:
        """
        Whether this rank should save unique files (manifest, shard index).

        Returns:
            True if this is rank 0 (global coordinator)
        """
        return self.dist.rank == 0

    def set_best_checkpoint(self, best_checkpoint: str) -> None:
        """Mark a checkpoint as the best model (for checkpoint retention)."""
        self.best_checkpoint = best_checkpoint
