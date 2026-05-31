import datetime
import logging
import os
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple

import torch
from torch.distributed.checkpoint.stateful import Stateful

from forgather.ml.distributed import (
    DistributedEnvInterface,
    get_barrier_fn,
    get_global_process_group,
    prefix_logger_rank,
)
from forgather.ml.sharded_checkpoint import (
    MODEL_EXCLUDED_MARKER,
    create_sharing_metadata,
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
prefix_logger_rank(logger, show_all_ranks=True)


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
        model_save_fn: Callable[[str], None] | None = None,
        model_load_fn: Callable[[str], None] | None = None,
        model_weights_external: bool = False,
    ):

        self.dist = dist
        self.config = config
        self.stateful_provider = stateful_provider
        # When True, model weights are supplied by an external authority (e.g.
        # a DiLoCo parameter server) and must NOT be saved or loaded here. This
        # is distinct from ``model_state_component is None``: an FSDP2 trainer
        # legitimately registers no "model" component yet still saves/loads via
        # model_save_fn/model_load_fn, so the absence of the component does NOT
        # mean the model is excluded — only this flag does.
        self.model_weights_external = model_weights_external

        assert model is not None

        if model_parts is None:
            model_parts = [model]

        self.model = model
        self.model_parts = model_parts
        self.model_preprocessor = model_preprocessor
        # Optional hooks that take over model save/load when set. All ranks
        # call the hook; the hook is responsible for its own rank gating,
        # since collective ops (e.g. DTensor full-state-dict gather) need to
        # run on every rank.
        self.model_save_fn = model_save_fn
        self.model_load_fn = model_load_fn

        if not shard_index:
            shard_index = make_shard_index(
                [mod.state_dict() for mod in model_parts],
                safetensors=config.save_safetensors,
            )
        self.shard_index = shard_index

        # Validate: safetensors cannot save tensors that share storage (tied
        # weights).  Fail early so the user learns about the incompatibility at
        # startup rather than hours later at the first checkpoint save.
        if config.save_safetensors:
            sharing_metadata = create_sharing_metadata(model)
            if sharing_metadata:
                tied_desc = "; ".join(" <-> ".join(group) for group in sharing_metadata)
                raise ValueError(
                    f"save_safetensors=True is incompatible with models that have "
                    f"tied (shared) weights. The safetensors format cannot "
                    f"represent tensors that share storage.\n"
                    f"Tied weight groups: {tied_desc}\n"
                    f"Set save_safetensors=False in your training configuration, "
                    f"or use --save-safetensors false on the command line."
                )
        self.best_checkpoint = None  # Deprecated - use best_checkpoints instead
        self.best_checkpoints: List[Tuple[str, float]] = (
            []
        )  # List of (path, metric_value)
        self.preserve_n_best: int = 1
        self.trainer: "BaseTrainer | None" = None  # Set by trainer for callback access
        self.barrier_fn = get_barrier_fn(get_global_process_group())

        # Initialize CheckpointCoordinator for state component handling.
        # Prefer the filtered accessor (honors args.checkpoint_components) so
        # a run can exclude components — notably "model", which makes the
        # model-weight save/load below a no-op (DiLoCo: the server owns the
        # weights). Falls back to the raw accessor for providers that predate
        # the filter. Try new API first, fall back to old API if not implemented.
        get_components = getattr(
            stateful_provider,
            "get_active_state_components",
            stateful_provider.get_state_components,
        )
        state_components = get_components()
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

        # Save model weights. When a model_save_fn hook is set (e.g. FSDP2's
        # full-state-dict gather), every rank must call it because the hook
        # runs a collective op; the hook itself handles rank gating for the
        # actual file write. In the legacy shard-index path, only the "save
        # common" rank writes.
        # Skip model-weight save entirely when weights are externally managed
        # (e.g. DiLoCo, where the parameter server owns them). All ranks
        # evaluate the same flag, so the model_save_fn collective is
        # consistently skipped. Drop a marker so validate_checkpoint accepts
        # this model-less checkpoint while still rejecting a partial/corrupt
        # normal one. NB: gate on model_weights_external, NOT
        # model_state_component — an FSDP2 trainer has no "model" component but
        # still saves via model_save_fn.
        if self.model_weights_external:
            if self._should_save_common():
                with open(
                    os.path.join(checkpoint_path, MODEL_EXCLUDED_MARKER), "w"
                ) as fh:
                    fh.write(
                        "Model weights are supplied externally (e.g. a DiLoCo "
                        "parameter server) and are intentionally not "
                        "checkpointed.\n"
                    )
        elif self.model_save_fn is not None:
            self._save_model(checkpoint_path)
        elif self._should_save_common():
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

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint being evaluated.
        metrics : dict of str to float
            Dictionary of evaluation metrics.
        metric_key : str
            Name of metric to use for comparison.
        greater_is_better : bool or None
            Whether higher metric values are better.
        preserve_n_best : int
            Number of best checkpoints to keep.

        Returns
        -------
        bool
            ``True`` if this checkpoint qualifies as one of the best.
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
        # Skip model-weight load when weights are externally managed (DiLoCo
        # supplies them from the server). The non-model training state
        # (optimizer/scheduler/trainer/rng) still restores. Gate on
        # model_weights_external, NOT model_state_component (FSDP2 has no
        # "model" component but still loads via model_load_fn).
        if not self.model_weights_external:
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
        """Save model and tokenizer to ``output_dir``."""
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
        # Decides whether *this* rank participates in writing model
        # shard files. Three regimes:
        #
        #   save_on_all_ranks=True
        #     PP / FSDP — every rank holds a different subset of weights
        #     so every rank writes its own non-overlapping shards.
        #
        #   save_on_each_node=True (default-False)
        #     DDP across nodes that *don't* share a filesystem — every
        #     node's chosen local rank writes a full copy locally.
        #     ``save_on_local_rank`` selects which local rank does that
        #     (defaults to 0 — a non-zero value lets the operator route
        #     the write to a non-rank-0 GPU per node, e.g. one with
        #     more disk I/O headroom).
        #
        #   save_on_each_node=False (the documented default for shared
        #     storage) — only one global writer. Hardcoded to global
        #     rank 0; ``save_on_local_rank`` is *not* consulted here
        #     because that field describes a local-rank index and
        #     comparing it to a global rank silently produces nonsense
        #     for any non-zero value (e.g. setting save_on_local_rank=1
        #     and running a 4-rank DDP would route writes to global
        #     rank 1, which on a 2-node 2x2 layout is the second rank
        #     of the first node, not "the first node's rank 1"). If a
        #     future operator wants to pick a non-rank-0 global writer,
        #     add a separate ``save_on_global_rank`` field rather than
        #     overloading the local-rank knob.
        if self.config.save_on_all_ranks:
            return True
        if self.config.save_on_each_node:
            return self.dist.local_rank == self.config.save_on_local_rank
        return self.dist.rank == 0

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
        if self.model_save_fn is not None:
            # Hook takes over model save entirely. Every rank calls it; the
            # hook is responsible for any collective op plus its own rank
            # gating for the actual file writes.
            self.model_save_fn(output_dir)
            return

        shard_index = self.shard_index
        save_safetensors = self.config.save_safetensors

        # The primary process on each saves the common state
        if self._should_save_unique():
            # Save the shard index
            save_shard_index(shard_index, output_dir, index_file_name(save_safetensors))

        # Shard *files* must be gated too — without this, plain DDP
        # across multiple nodes with shared storage has multiple
        # ranks writing the same file path at the same time, racing
        # on the bytes. Pipeline-parallel sets save_on_all_ranks=True
        # so every rank's call writes a non-overlapping subset; that
        # case is preserved by _should_save_common.
        if self._should_save_common():
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

        logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")

        if self.model_load_fn is not None:
            # Hook takes over model load entirely. Typical use: FSDP2 trainer
            # reads the full state dict on rank 0 from HF safetensors and
            # broadcasts via set_model_state_dict. The hook is responsible
            # for re-tying weights if applicable.
            self.model_load_fn(checkpoint_path)
            return

        # Use the sharded checkpoint loader to handle all checkpoint formats
        for mod in self.model_parts:
            load_checkpoint(
                checkpoint_path,
                mod,
                device=self.dist.device,
                strict=True,
            )

        # Re-establish tied weights after loading.
        #
        # When loading from safetensors format, tied weights are stored as
        # separate tensors (the format cannot represent shared storage, so HF
        # deduplicates on save and re-ties on load).  With PyTorch format the
        # sharing is already intact, but calling tie_weights() is idempotent.
        #
        # Only call on the full model, not on individual pipeline stage
        # modules.  In pipeline-parallel, tied weights (e.g. embedding / head)
        # are intentionally on separate nodes; the pipeline trainer handles
        # within-stage sharing via retie_parameters() separately.
        is_pipeline = len(self.model_parts) > 1
        if not is_pipeline and hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

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
        """Load all training state components from separate files.

        Raises on failure: by the time this is called, the user has
        explicitly requested a checkpoint load (resume_from_checkpoint was
        resolved to a concrete path), so silently continuing with freshly
        initialized optimizer / scheduler / dataset state would turn the
        user's finetune into training-from-scratch. The coordinator already
        tolerates missing optional components internally; anything that
        escapes here is either a required-component failure or an
        unexpected filesystem / serialization error, both of which must be
        fatal.
        """
        if self.coordinator is not None:
            try:
                self.coordinator.load_checkpoint(checkpoint_path, strict=False)
            except Exception:
                logger.error(
                    f"Failed to load training state via CheckpointCoordinator "
                    f"from {checkpoint_path}. Aborting training.",
                    exc_info=True,
                )
                raise


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
