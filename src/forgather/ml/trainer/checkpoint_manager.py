import datetime
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
from torch.distributed.checkpoint.stateful import Stateful

from forgather.ml.distributed import (
    DistributedEnvInterface,
    get_barrier_fn,
    get_global_process_group,
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
from .trainer_types import CheckpointInterface, StatefulProvider

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        self.best_checkpoint = None
        self.barrier_fn = get_barrier_fn(get_global_process_group())

        # Initialize CheckpointCoordinator for state component handling
        # Try new API first, fall back to old API if not implemented
        state_components = stateful_provider.get_state_components()
        if state_components is not None:
            # Filter out model component - CheckpointManager handles model separately
            # via sharded checkpoint (which can handle large models efficiently)
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
            # Old API - will use get_statefuls_for_save/load
            self.coordinator = None

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

        if self.dist.local_rank == 0:
            logger.info(f"Saving checkpoint at {checkpoint_path}")

        if self._should_save_unique():
            # Ensure the checkpoint directory exists
            os.makedirs(checkpoint_path, exist_ok=True)
        self._barrier()

        # Save model weights (only on ranks that should save)
        if self._should_save_common():
            self._save_model(checkpoint_path)

        # Save training state
        # If using CheckpointCoordinator, ALL ranks must call it (has barriers)
        # If using old API, only saving ranks call it
        if self.coordinator is not None:
            # New API: all ranks participate
            self._save_training_state(checkpoint_path)
        else:
            # Old API: only saving ranks
            if self._should_save_common():
                self._save_training_state(checkpoint_path)

        # At most, one process per node should delete excess checkpoints
        if self._should_save_unique():
            maybe_delete_oldest_checkpoint(
                self.config.output_dir,
                self.config.save_total_limit,
                self.best_checkpoint,
            )
        self._barrier()
        return checkpoint_path

    def set_best_checkpoint(self, best_checkpoint: str) -> None:
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
        if self.dist.local_rank == 0:
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        self._load_model_from_checkpoint(checkpoint_path)
        self._load_training_state(checkpoint_path)

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
        if self.dist.local_rank == 0:
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
            # Use new CheckpointCoordinator API
            # IMPORTANT: ALL ranks must call this (coordinator has barriers)
            try:
                self.coordinator.save_checkpoint(output_dir, validate=False)
            except Exception as e:
                logger.error(f"Failed to save training state via CheckpointCoordinator\n{e}")
                traceback.print_exc()
                raise
        else:
            # Fall back to old API
            for key, obj in self.stateful_provider.get_statefuls_for_save().items():
                if obj:
                    try:
                        self._save_state_dict(key, obj, output_dir)
                    except Exception as e:
                        logger.error(f"Failed to save {key}\n{e}")
                        traceback.print_exc()

    def _load_training_state(self, checkpoint_path: str) -> None:
        """Load all training state components from separate files."""
        if self.coordinator is not None:
            # Use new CheckpointCoordinator API
            try:
                self.coordinator.load_checkpoint(checkpoint_path, strict=False)
            except Exception as e:
                logger.warning(f"Failed to load training state via CheckpointCoordinator\n{e}")
                traceback.print_exc()
        else:
            # Fall back to old API
            for key, obj in self.stateful_provider.get_statefuls_for_load().items():
                if obj:
                    try:
                        self._load_state_dict(key, obj, checkpoint_path)
                    except Exception as e:
                        logger.warning(f"Failed to load {key}\n{e}")
                        # traceback.print_exc()


class RNGState(Stateful):
    """
    A stateful for saving and restoring the random number generator states
    """

    def load_state_dict(self, rng_state):
        # Restore CPU RNG state
        if "torch_rng_state" in rng_state:
            torch.set_rng_state(rng_state["torch_rng_state"])
            logger.debug("Restored CPU RNG state from checkpoint")

        # Restore CUDA RNG state if available
        if "cuda_rng_state" in rng_state and torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            saved_device = rng_state.get("cuda_device", current_device)

            torch.cuda.set_rng_state(rng_state["cuda_rng_state"], device=current_device)
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
