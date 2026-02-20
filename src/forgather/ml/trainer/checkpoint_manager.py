import datetime
import logging
import os
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

import torch
from torch.distributed.checkpoint.stateful import Stateful

from forgather.ml.distributed import (
    DistributedEnvInterface,
    get_barrier_fn,
    get_global_process_group,
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

from .checkpoint_coordinator import CheckpointCoordinator
from .checkpoint_types import StateComponent
from .checkpoint_utils import ValidationLevel, validate_replication
from .trainer_types import CheckpointInterface, StatefulProvider

if TYPE_CHECKING:
    from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
prefix_logger_rank(logger)


def default_checkpoint_id():
    """
    Generate checkpoint id from timestamp
    """
    return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


Statefuls = Dict[str, Stateful]
ModelParts = Iterable[torch.nn.Module]


@dataclass(kw_only=True)
class CheckpointConfig:
    output_dir: str
    save_total_limit: int
    save_on_each_node: bool = False
    save_safetensors: bool = True
    # When saving on node, which local-rank should be save on?
    save_on_local_rank: int = 0
    # Save weights and Statefuls on all ranks
    # If more than one rank will be saving, the rank is added to the file name
    # WRT, save_on_local_rank, this rank still only saves "unique" data, like the shard index.
    save_on_all_ranks: bool = False


class CheckpointManager(CheckpointInterface):
    def __init__(
        self,
        config: CheckpointConfig,
        dist: DistributedEnvInterface,
        stateful_provider: StatefulProvider,
        model: torch.nn.Module,
        model_parts: ModelParts | None = None,
        model_preprocessor: Any = None,
        shard_index=None,
    ):

        self.dist = dist
        self.config = config
        self.stateful_provider = stateful_provider

        assert model is not None

        if model_parts is None:
            model_parts = [model]

        self.model = model
        self.model_parts = model_parts
        self.model_preprocessor = model_preprocessor

        if not shard_index:
            shard_index = make_shard_index(
                [mod.state_dict() for mod in model_parts],
                safetensors=config.save_safetensors,
            )
        self.shard_index = shard_index
        self.best_checkpoint = None  # Deprecated - use best_checkpoints instead
        self.best_checkpoints: List[Tuple[str, float]] = (
            []
        )  # List of (path, metric_value)
        self.preserve_n_best: int = 1
        self.trainer: "BaseTrainer | None" = None  # Set by trainer for callback access
        self.barrier_fn = get_barrier_fn(get_global_process_group())

        # Initialize CheckpointCoordinator for state component handling
        # Try new API first, fall back to old API if not implemented
        state_components = stateful_provider.get_state_components()
        if state_components is not None:
            # Extract model component - CheckpointManager handles model saving
            # separately via sharded checkpoint, but we still need the
            # StateComponent metadata for replication validation.
            model_components = [
                comp for comp in state_components if comp.key == "model"
            ]
            self.model_state_component: StateComponent | None = (
                model_components[0] if model_components else None
            )

            non_model_components = [
                comp for comp in state_components if comp.key != "model"
            ]

            if non_model_components:
                process_groups = stateful_provider.get_process_groups() or {}
                self.coordinator = CheckpointCoordinator(
                    state_components=non_model_components,
                    process_groups=process_groups,
                    dist=dist,
                    output_dir=config.output_dir,
                )
            else:
                # No non-model components to coordinate
                self.coordinator = None
        else:
            # StatefulProvider must implement get_state_components()
            raise RuntimeError(
                "StatefulProvider does not implement get_state_components(). "
                "All trainers must use the new checkpoint API."
            )

    def save_checkpoint(
        self,
        checkpoint_path: str | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        if not checkpoint_path:
            if not checkpoint_id:
                checkpoint_id = default_checkpoint_id()
            checkpoint_path = next_checkpoint_path(
                self.config.output_dir, checkpoint_id
            )

        logger.info(f"Saving checkpoint at {checkpoint_path}")

        if self._should_save_unique():
            # Ensure the checkpoint directory exists
            os.makedirs(checkpoint_path, exist_ok=True)
        self._barrier()

        # Validate model replication before saving (all ranks must participate)
        if (
            self.model_state_component is not None
            and self.model_state_component.validate_replication
        ):
            self._validate_model_replication(self.model_state_component)

        # Save model weights (only on ranks that should save)
        if self._should_save_common():
            self._save_model(checkpoint_path)

        # Save training state
        # ALL ranks must call CheckpointCoordinator (has barriers)
        if self.coordinator is not None:
            self._save_training_state(checkpoint_path)

        # Save stateful callback states and best checkpoints list
        if self._should_save_common():
            checkpoint_metadata = {}

            # Save best checkpoints list for preservation on resume
            if self.best_checkpoints:
                checkpoint_metadata["best_checkpoints"] = self.best_checkpoints
                logger.debug(
                    f"Saving best checkpoints list: {[cp[0] for cp in self.best_checkpoints]}"
                )

            # Save callback states
            if self.trainer and hasattr(self.trainer, "callbacks"):
                callback_states = {}
                for i, callback in enumerate(self.trainer.callbacks):
                    if isinstance(callback, Stateful):
                        callback_states[f"callback_{i}_{type(callback).__name__}"] = (
                            callback.state_dict()
                        )
                if callback_states:
                    checkpoint_metadata["callback_states"] = callback_states
                    logger.debug(f"Saved {len(callback_states)} callback states")

            # Save metadata if we have anything to save
            if checkpoint_metadata:
                metadata_path = os.path.join(checkpoint_path, "checkpoint_metadata.pt")
                torch.save(checkpoint_metadata, metadata_path)
                logger.debug(f"Saved checkpoint metadata to {metadata_path}")

        # At most, one process per node should delete excess checkpoints
        if self._should_save_unique():
            # Build list of preserved checkpoint paths
            preserved_paths = [cp[0] for cp in self.best_checkpoints]
            # Also preserve old-style best_checkpoint for backward compatibility
            if self.best_checkpoint and self.best_checkpoint not in preserved_paths:
                preserved_paths.append(self.best_checkpoint)

            maybe_delete_oldest_checkpoint(
                self.config.output_dir,
                self.config.save_total_limit,
                preserved_checkpoints=preserved_paths,
            )
        self._barrier()
        return checkpoint_path

    def update_best_checkpoints(
        self,
        checkpoint_path: str,
        metrics: dict[str, float],
        metric_key: str,
        greater_is_better: bool | None,
        preserve_n_best: int,
    ) -> bool:
        """
        Update best checkpoints list with new checkpoint if it qualifies.

        This should be called BEFORE save_checkpoint() so the preserved list
        is accurate when deletion happens.

        Args:
            checkpoint_path: Path to checkpoint being evaluated
            metrics: Dictionary of evaluation metrics
            metric_key: Name of metric to use for comparison
            greater_is_better: Whether higher metric values are better
            preserve_n_best: Number of best checkpoints to keep
            is_world_process_zero: Whether this is rank 0 (for logging)

        Returns:
            True if this checkpoint qualifies as one of the best
        """
        # Extract metric value
        metric_value = metrics.get(metric_key) or metrics.get(f"eval_{metric_key}")

        if metric_value is None:
            logger.warning(
                f"Metric '{metric_key}' not found in evaluation metrics. "
                f"Available: {list(metrics.keys())}"
            )
            return False

        # Auto-detect comparison direction if not specified
        if greater_is_better is None:
            greater_is_better = metric_key not in ["loss", "eval_loss"]

        # Determine if this checkpoint should be preserved
        is_best = False

        if len(self.best_checkpoints) < preserve_n_best:
            # Have room for more best checkpoints
            is_best = True
        else:
            # Compare against worst of current best checkpoints
            worst_best = (max if greater_is_better else min)(
                self.best_checkpoints, key=lambda x: x[1]
            )
            is_best = (
                (metric_value > worst_best[1])
                if greater_is_better
                else (metric_value < worst_best[1])
            )

        if is_best:
            logger.info(
                f"New best checkpoint: {checkpoint_path} ({metric_key}={metric_value:.4f})"
            )

            # Add to list
            self.best_checkpoints.append((checkpoint_path, metric_value))

            # Sort (best to worst)
            self.best_checkpoints.sort(key=lambda x: x[1], reverse=greater_is_better)

            # Trim to N best
            self.best_checkpoints = self.best_checkpoints[:preserve_n_best]

            # Log the updated list with metrics
            logger.info("Best checkpoints:")
            for cp_path, cp_metric in self.best_checkpoints:
                logger.info(f"  {cp_path} ({metric_key}={cp_metric:.4f})")

        return is_best

    def get_best_checkpoints_summary(self, metric_key: str = "loss") -> str:
        """Get formatted summary of best checkpoints with metrics."""
        if not self.best_checkpoints:
            return "No best checkpoints tracked"

        lines = [f"Best checkpoints (N={len(self.best_checkpoints)}):"]
        for cp_path, cp_metric in self.best_checkpoints:
            lines.append(f"  {cp_path}: {metric_key}={cp_metric:.4f}")
        return "\n".join(lines)

    def set_best_checkpoint(self, best_checkpoint: str) -> None:
        """
        Mark checkpoint as best (deprecated single-checkpoint API).

        This is kept for backward compatibility with CheckpointInterface.
        Use update_best_checkpoints() for the new N-best API.
        """
        self.best_checkpoint = best_checkpoint

    def resolve_checkpoint_path(self, checkpoint_path: str | None) -> str | None:
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(self.config.output_dir)
            if not checkpoint_path:
                logger.warning(
                    f"No model checkpoints found in {self.config.output_dir}"
                )
                return None
        else:
            # Explicit path provided
            if os.path.exists(checkpoint_path):
                if not validate_checkpoint(checkpoint_path):
                    logger.warning(f"Invalid checkpoint at: {checkpoint_path}")
                    return None
            else:
                logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
                return None
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        checkpoint_path = self.resolve_checkpoint_path(checkpoint_path)
        if checkpoint_path is None:
            raise RuntimeError("Could not load checkpoint")
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        self._load_model_from_checkpoint(checkpoint_path)
        self._load_training_state(checkpoint_path)

        # Load checkpoint metadata (best checkpoints list + callback states)
        metadata_path = os.path.join(checkpoint_path, "checkpoint_metadata.pt")
        if os.path.exists(metadata_path):
            try:
                checkpoint_metadata = torch.load(
                    metadata_path, map_location=torch.device("cpu")
                )

                # Restore best checkpoints list, filtering out non-existent paths
                if "best_checkpoints" in checkpoint_metadata:
                    restored_checkpoints = checkpoint_metadata["best_checkpoints"]
                    # Filter out checkpoints that no longer exist on disk
                    self.best_checkpoints = [
                        (cp_path, metric)
                        for cp_path, metric in restored_checkpoints
                        if os.path.exists(cp_path)
                    ]

                    # Log what was restored and what was filtered
                    if len(self.best_checkpoints) < len(restored_checkpoints):
                        filtered = len(restored_checkpoints) - len(
                            self.best_checkpoints
                        )
                        logger.warning(
                            f"Filtered out {filtered} non-existent checkpoints from best_checkpoints list"
                        )

                    if self.best_checkpoints:
                        logger.info(
                            f"Restored best checkpoints list: "
                            f"{[os.path.basename(cp[0]) for cp in self.best_checkpoints]}"
                        )
                    else:
                        logger.info(
                            "Best checkpoints list was empty after filtering non-existent paths"
                        )

                # Restore callback states
                if (
                    "callback_states" in checkpoint_metadata
                    and self.trainer
                    and hasattr(self.trainer, "callbacks")
                ):
                    callback_states = checkpoint_metadata["callback_states"]
                    for i, callback in enumerate(self.trainer.callbacks):
                        key = f"callback_{i}_{type(callback).__name__}"
                        if isinstance(callback, Stateful) and key in callback_states:
                            callback.load_state_dict(callback_states[key])
                            logger.info(f"Restored state for {type(callback).__name__}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
        else:
            # Try legacy callback_states.pt for backward compatibility
            callback_path = os.path.join(checkpoint_path, "callback_states.pt")
            if (
                os.path.exists(callback_path)
                and self.trainer
                and hasattr(self.trainer, "callbacks")
            ):
                try:
                    callback_states = torch.load(
                        callback_path, map_location=torch.device("cpu")
                    )

                    for i, callback in enumerate(self.trainer.callbacks):
                        key = f"callback_{i}_{type(callback).__name__}"
                        if isinstance(callback, Stateful) and key in callback_states:
                            callback.load_state_dict(callback_states[key])
                            logger.info(f"Restored state for {type(callback).__name__}")
                except Exception as e:
                    logger.warning(f"Failed to load callback states: {e}")

    def save_model(
        self,
        output_dir: str | os.PathLike | None = None,
        overwrite_output_dir: bool = False,
    ) -> None:
        """
        Save model and tokenizer to output_dir
        """
        model = self.model
        if model is None:
            return
        if output_dir is None:
            output_dir = self.config.output_dir
        else:
            output_dir = str(output_dir)
        if self._should_save_unique():
            if not overwrite_output_dir and validate_checkpoint(output_dir):
                raise Exception(
                    "Would overwrite output model in output directory. "
                    f"Set 'args.overwrite_output_dir' to override: {output_dir}"
                )
            os.makedirs(output_dir, exist_ok=True)

            config = getattr(model, "config", None)
            assert config
            if hasattr(config, "save_pretrained"):
                config.save_pretrained(output_dir)
            if self.model_preprocessor and hasattr(
                self.model_preprocessor, "save_pretrained"
            ):
                self.model_preprocessor.save_pretrained(output_dir)
        self._barrier()
        if self._should_save_common():
            self._save_model(output_dir)
        self._barrier()

    def _barrier(self):
        if self.dist.world_size > 1:
            self.barrier_fn()

    def _should_save_unique(self):
        # Returns True, if rank should save "unique" files, like the shard index -- or
        # cleaning up checkpoints.
        return self.dist.rank == self.config.save_on_local_rank or (
            self.config.save_on_each_node
            and self.dist.local_rank == self.config.save_on_local_rank
        )

    def _should_save_common(self):
        # Should save parameters and Stateful objects. If more than one rank is saving
        # on the same node, each file will be named after the rank. Useful for things
        # like Pipeline parallel.
        if (
            not self.config.save_on_all_ranks
            and self.config.save_on_local_rank != self.dist.local_rank
        ):
            return False
        return True

    def _validate_model_replication(self, model_component: StateComponent):
        """Validate that model weights are identical across all ranks.

        Called before saving model weights when the model StateComponent
        has validate_replication=True (e.g., DDP training).
        """
        if self.dist.world_size <= 1:
            return

        try:
            level_str = model_component.validation_level
            try:
                level = ValidationLevel(level_str)
            except ValueError:
                logger.warning(
                    f"Invalid validation_level '{level_str}' for model, "
                    "defaulting to TENSOR"
                )
                level = ValidationLevel.TENSOR

            state_dict = model_component.stateful.state_dict()
            is_valid, errors = validate_replication(
                state_dict,
                validation_level=level,
                group=None,
            )

            if not is_valid:
                logger.error(
                    f"Model replication validation failed "
                    f"(level: {level.value}). "
                    f"Model weights differ across ranks!"
                )
                for error in errors:
                    logger.error(f"  - {error}")
                if model_component.required:
                    raise RuntimeError(
                        "Model replication validation failed: "
                        "DDP model weights have diverged across ranks"
                    )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Failed to validate model replication: {e}")

    def _save_model(self, output_dir: str):
        shard_index = self.shard_index
        save_safetensors = self.config.save_safetensors

        # The primary process on each saves the common state
        if self._should_save_unique():
            # Save the shard index
            save_shard_index(shard_index, output_dir, index_file_name(save_safetensors))

        for mod in self.model_parts:
            save_sharded_checkpoint(
                output_dir,
                shard_index,
                mod,
                safetensors=save_safetensors,
            )

    def _dict_name(self, key):
        if self.dist.world_size > 1 and self.config.save_on_all_ranks:
            return f"{key}_state_rank_{self.dist.rank}.pt"
        else:
            return f"{key}_state.pt"

    def _save_state_dict(self, key: str, obj: Stateful, output_dir: str):
        state_path = os.path.join(output_dir, self._dict_name(key))
        logger.debug(f"Saving key {key} to {state_path}")
        torch.save(obj.state_dict(), state_path)

    def _load_state_dict(self, key: str, obj: Stateful, output_dir: str):
        state_path = os.path.join(output_dir, self._dict_name(key))
        logger.debug(f"Loading key {key} from {state_path}")
        state = torch.load(state_path, map_location=torch.device("cpu"))

        obj.load_state_dict(state)

    def _load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint using the sharded checkpoint loader."""

        # Use the sharded checkpoint loader to handle all checkpoint formats
        logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")

        for mod in self.model_parts:
            load_checkpoint(
                checkpoint_path,
                mod,
                device=self.dist.device,
                strict=True,
            )

    def _save_training_state(self, output_dir: str) -> None:
        """Save all training state components to separate files."""
        if self.coordinator is not None:
            # Use CheckpointCoordinator API
            # IMPORTANT: ALL ranks must call this (coordinator has barriers)
            try:
                self.coordinator.save_checkpoint(output_dir, validate=False)
            except Exception as e:
                logger.error(
                    f"Failed to save training state via CheckpointCoordinator\n{e}"
                )
                traceback.print_exc()
                raise

    def _load_training_state(self, checkpoint_path: str) -> None:
        """Load all training state components from separate files."""
        if self.coordinator is not None:
            # Use CheckpointCoordinator API
            # The coordinator handles per-component errors internally and logs them.
            # We only catch unexpected errors (e.g. filesystem failures, manifest corruption).
            try:
                self.coordinator.load_checkpoint(checkpoint_path, strict=False)
            except Exception as e:
                logger.error(
                    f"Failed to load training state via CheckpointCoordinator: {e}\n"
                    f"Training will continue WITHOUT any restored training state "
                    f"(optimizer, scheduler, etc.). This is likely to cause training instability.",
                    exc_info=True,
                )


class RNGState(Stateful):
    """
    A stateful for saving and restoring the random number generator states
    """

    def load_state_dict(self, state_dict):
        # Restore CPU RNG state
        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])
            logger.debug("Restored CPU RNG state from checkpoint")

        # Restore CUDA RNG state if available
        if "cuda_rng_state" in state_dict and torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            saved_device = state_dict.get("cuda_device", current_device)

            torch.cuda.set_rng_state(
                state_dict["cuda_rng_state"], device=current_device
            )
            logger.debug(
                f"Restored CUDA RNG state for device {current_device} from checkpoint"
            )

    def state_dict(self):
        rng_state = {
            "torch_rng_state": torch.get_rng_state(),
            "initial_seed": torch.initial_seed(),
        }

        # Save CUDA RNG state if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            current_device = torch.cuda.current_device()
            rng_state["cuda_rng_state"] = torch.cuda.get_rng_state(
                device=current_device
            )
            rng_state["cuda_device"] = current_device

        return rng_state
