from typing import (
    Callable,
    Optional,
    List,
    Dict,
)
from types import NoneType
import os
import json
from abc import abstractmethod
from contextlib import ExitStack

import logging
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
)

from .trainer_types import (
    ExtensibleTrainer,
    TrainingArguments,
    TrainOutput,
    TrainerControl,
)
from ..sharded_checkpoint import (
    load_checkpoint,
    validate_checkpoint,
    find_latest_checkpoint,
    next_checkpoint_path,
    maybe_delete_oldest_checkpoint,
    save_checkpoint_metrics,
    load_checkpoint_metrics,
)

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseTrainer(ExtensibleTrainer):
    """
    Implements the common aspects of the ExtensibleTrainer class,
        but is also an abstract-base-class, with the meat of the "trainer"
        implementation needing to be filled in.
    """

    @classmethod
    def default_callbacks(cls):
        """
        Returns a list of default callbacks
        """
        return []

    def __init__(
        self,
        model: PreTrainedModel | torch.nn.Module = None,
        args: Optional[dict | TrainingArguments] = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: List = None,
        # Depreicated; use processing_class
        tokenizer=None,
        compute_loss_func: Callable = None,
    ):
        if callbacks is None:
            callbacks = []

        assert (
            model or model_init
        ), "Either a model or a model constructor must be specified"

        # Try to maintain backward compatability for now.
        if processing_class is None and tokenizer is not None:
            processing_class = tokenizer

        # Init args
        self.model = model
        if args is None:
            args = TrainingArguments()
        elif isinstance(args, dict):
            args = TrainingArguments(**args)

        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.model_init = model_init
        self.callbacks = self.default_callbacks()
        self.callbacks.extend(callbacks)
        self.loss_fn = compute_loss_func

        # Init attributes
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.state = None
        self.is_local_process_zero = True
        self.is_world_process_zero = True
        self.num_processes = 1

        # Silence annoying Huggingface FastTokenizer warnings
        # If knows if it is safe or not, and does the right thing, why
        # do I need to hear about it and create a janky workaround for
        # a non-issue!?
        if self.args.dataloader_num_workers > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.args.detect_anomaly:
            logger.warning(
                "Enabling autograd detect anomaly; expect performance degradation"
            )
            torch.autograd.set_detect_anomaly(True)

        if self.args.float32_matmul_precision is not None:
            logger.info(
                f'Setting float32_matmul_precision to "{self.args.float32_matmul_precision}"'
            )
            torch.set_float32_matmul_precision(self.args.float32_matmul_precision)

        self._post_init()
        # self._validate_dirs()

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"model={self.model},"
            f"args={self.args},"
            f"data_collator={self.data_collator},"
            f"train_dataset={self.train_dataset},"
            f"eval_dataset={self.eval_dataset},"
            f"processing_class={self.processing_class},"
            f"model_init={self.model_init},"
            f"callbacks={self.callbacks},"
            ")"
        )

    def train(self, **kwargs) -> TrainOutput:
        """
        The main entry point to start training the model.
        """
        with ExitStack() as stack:
            backends = self._get_sdpa_backends(self.args.sdpa_backend)
            if backends:
                logger.info(
                    f"sdpa_backends={backends}, set_priority={self.args.sdpa_set_priority}"
                )
                stack.enter_context(
                    sdpa_kernel(backends, set_priority=self.args.sdpa_set_priority)
                )

            self._prepare(
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset
            )
            return self._train_loop()

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, **kwargs
    ) -> dict[str, float]:
        """
        The main entry point to evaluate the model.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        with ExitStack() as stack:
            backends = self._get_sdpa_backends(self.args.sdpa_backend)
            if backends:
                stack.enter_context(
                    sdpa_kernel(backends, set_priority=self.args.sdpa_set_priority)
                )
            self._prepare(train_dataset=None, eval_dataset=eval_dataset)
            return self._eval_loop()

    def save_model(self, output_dir: os.PathLike | str = None) -> None:
        """
        Save model and tokenizer to output_dir
        """
        if self.model is None:
            return
        if output_dir is None:
            output_dir = self.args.output_dir
        if self._should_save_unique():
            if not self.args.overwrite_output_dir and self.model_exists(output_dir):
                raise Exception(
                    "Would overwrite output model in output directory. "
                    f"Set 'args.overwrite_output_dir' to override: {output_dir}"
                )
            os.makedirs(output_dir, exist_ok=True)
            self._save_model_config(output_dir)
            self._save_model_preprocessor(output_dir)
        self._barrier()
        self._save_model(output_dir)
        self._barrier()

    def add_callback(self, callback):
        if isinstance(callback, type):
            callback = callback()
        self.callbacks.append(callback)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            compare = lambda a, b: type(a) == b
        else:
            compare = lambda a, b: id(a) == id(b)
        for i, cb in enumerate(self.callbacks):
            if compare(cb, callback):
                return self.callbacks.pop(i)

    def log(self, logs: Dict[str, float]):
        self.state.log_history.append(logs)

        self._dispatch_event(
            "on_log",
            logs=logs,
        )

    def unwrapped_model(self):
        """
        Unwrap model for saving
        Some sub-classes may 'wrap' the model in another object.
        This method should return the base model, given the wrapped model.
        """
        return self.model

    def model_exists(self, output_dir):
        """
        Return True, if a saved model exists in the output_dir
        """
        output_artifacts = (
            WEIGHTS_NAME,
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        for artifact_name in output_artifacts:
            if os.path.exists(os.path.join(output_dir, artifact_name)):
                return True
        return False

    def _validate_dirs(self):
        """
        TODO: Review logic
        """
        output_dir = self.args.output_dir
        if os.path.isdir(output_dir):
            if self.model_exists(output_dir):
                if not self.args.overwrite_output_dir:
                    logger.warning(
                        f"Model exists in output dir '{output_dir}' and 'args.overwrite_output_dir' "
                        "is not 'True.' Model can not be saved! Set args.overwrite_output_dir=True "
                        "to override."
                    )
                else:
                    logger.warning(
                        f"Model exists in output dir '{output_dir}' and model may be overwritten!"
                    )
        elif os.path.exists(output_dir):
            raise Exception(
                f"Something other than a directory already exists at the output path! {output_dir}"
            )
        else:
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        logging_dir = self.args.logging_dir
        if not os.path.isdir(logging_dir):
            os.makedirs(logging_dir, exist_ok=True)

    def _barrier(self):
        """
        Wait for all processes before continuing
        """
        pass

    def _should_save_unique(self):
        """
        Should this process save a unique file?
        """
        return True

    def _save_model(self, output_dir):
        model = self.unwrapped_model()
        if isinstance(model, PreTrainedModel):
            model.save_pretrained(
                save_directory=output_dir,
                is_main_process=True,
                safe_serialization=self.args.save_safetensors,
            )
        else:
            logger.info("Saving model as state-dictionary")
            torch.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

    def _save_model_config(self, output_dir):
        if isinstance(self.model, PreTrainedModel):
            self.model.config.save_pretrained(output_dir)

    def _save_model_preprocessor(self, output_dir):
        if self.processing_class and hasattr(self.processing_class, "save_pretrained"):
            self.processing_class.save_pretrained(output_dir)

    def _resolve_checkpoint_path(self) -> str | None:
        """Resolve the checkpoint path based on resume_from_checkpoint setting."""
        if not self.args.resume_from_checkpoint:
            return None

        if isinstance(self.args.resume_from_checkpoint, str):
            # Explicit path provided
            if os.path.exists(self.args.resume_from_checkpoint):
                if validate_checkpoint(self.args.resume_from_checkpoint):
                    return self.args.resume_from_checkpoint
                else:
                    logger.warning(
                        f"Invalid checkpoint at: {self.args.resume_from_checkpoint}"
                    )
                    return None
            else:
                logger.warning(
                    f"Checkpoint path does not exist: {self.args.resume_from_checkpoint}"
                )
                return None
        elif self.args.resume_from_checkpoint is True:
            # Auto-discover latest checkpoint
            return find_latest_checkpoint(self.args.output_dir)

        return None

    def save_checkpoint(self, checkpoint_path=None) -> None:
        if not checkpoint_path:
            checkpoint_path = next_checkpoint_path(
                self.args.output_dir, self.state.global_step
            )

        logger.info(f"Saving checkpoint at {checkpoint_path}")

        if self._should_save_unique():
            # Ensure the checkpoint directory exists
            os.makedirs(checkpoint_path, exist_ok=True)
            self._save_model_config(self.args.output_dir)
            self._save_model_preprocessor(self.args.output_dir)

        self._barrier()
        self._save_model(checkpoint_path)
        self._save_training_state(checkpoint_path)

        # At most, one process per node should delete excess checkpoints
        if self._should_save_unique():
            maybe_delete_oldest_checkpoint(
                self.args.output_dir,
                self.args.save_total_limit,
                self.state.best_model_checkpoint,
            )

        self._dispatch_event(
            "on_save",
            checkpoint_path=checkpoint_path,
        )
        # Make available to sub-class for saving additional state
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path=None) -> None:
        if not checkpoint_path:
            checkpoint_path = self._resolve_checkpoint_path()

        if checkpoint_path:
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            self._load_model_from_checkpoint(checkpoint_path)
            self._load_training_state(checkpoint_path)
        else:
            logger.warn("No checkpoints found")

    def save_metrics(
        self, split: str, metrics: Dict[str, float], combined: bool = True
    ) -> None:
        """
        Save metrics to JSON files following HuggingFace conventions.

        Args:
            split: The dataset split (e.g., "train", "eval", "test")
            metrics: Dictionary of metrics to save
            combined: Whether to also save to all_results.json
        """
        if not self._should_save_unique():
            return

        # Save split-specific results
        results_file = os.path.join(self.args.output_dir, f"{split}_results.json")
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Save to combined results if requested
        if combined:
            combined_file = os.path.join(self.args.output_dir, "all_results.json")

            # Load existing combined results if they exist
            combined_results = {}
            if os.path.exists(combined_file):
                try:
                    with open(combined_file, "r") as f:
                        combined_results = json.load(f)
                except (json.JSONDecodeError, IOError):
                    logger.warning(
                        f"Could not read existing combined results from {combined_file}"
                    )

            # Add current metrics with split prefix
            for key, value in metrics.items():
                combined_results[f"{split}_{key}"] = value

            # Save combined results
            with open(combined_file, "w") as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False)

    def log_metrics(self, split: str, metrics: Dict[str, float]) -> None:
        """
        Log metrics in formatted output following HuggingFace conventions.

        Args:
            split: The dataset split (e.g., "train", "eval", "test")
            metrics: Dictionary of metrics to log
        """
        if not self._should_save_unique():
            return

        logger.info(f"***** {split} metrics *****")
        for key in sorted(metrics.keys()):
            logger.info(f"  {key} = {metrics[key]}")

    def _update_best_model(
        self, checkpoint_path: str, metrics: Dict[str, float]
    ) -> None:
        """Update best model tracking based on current metrics."""
        if not self.args.load_best_model_at_end:
            return

        # Try exact match first, then with eval_ prefix
        metric_value = metrics.get(self.args.metric_for_best_model)
        if metric_value is None:
            metric_value = metrics.get(f"eval_{self.args.metric_for_best_model}")

        if metric_value is None:
            logger.warning(
                f"Metric '{self.args.metric_for_best_model}' not found in evaluation metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )
            return

        # Initialize best metric on first evaluation
        if self.state.best_metric is None:
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = checkpoint_path
            logger.info(
                f"First evaluation - setting best {self.args.metric_for_best_model}: {metric_value}"
            )
            return

        # Check if current metric is better
        is_better = (
            self.args.greater_is_better and metric_value > self.state.best_metric
        ) or (not self.args.greater_is_better and metric_value < self.state.best_metric)

        if is_better:
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = checkpoint_path
            logger.info(
                f"New best {self.args.metric_for_best_model}: {metric_value} (previous: {self.state.best_metric})"
            )
            logger.info(f"Saving new best model checkpoint to {checkpoint_path}")

    def load_best_model(self) -> None:
        """Load the best model from the best checkpoint."""
        if not self.state.best_model_checkpoint:
            logger.warning("No best model checkpoint available to load")
            return

        if not os.path.exists(self.state.best_model_checkpoint):
            logger.warning(
                f"Best model checkpoint path does not exist: {self.state.best_model_checkpoint}"
            )
            return

        logger.info(f"Loading best model from {self.state.best_model_checkpoint}")
        self._load_model_from_checkpoint(self.state.best_model_checkpoint)

    def _state_dict(self) -> Dict:
        """Save optimizer and scheduler state. Subclasses can override for custom behavior."""
        training_state = {}

        if self.args.save_optimizer_state and self.optimizer is not None:
            training_state["optimizer"] = self.optimizer.state_dict()
            logger.debug("Saved optimizer state to checkpoint")

        if self.args.save_scheduler_state and self.lr_scheduler is not None:
            training_state["lr_scheduler"] = self.lr_scheduler.state_dict()
            logger.debug("Saved LR scheduler state to checkpoint")

        # Save global step for proper resume functionality
        if self.args.save_dataset_state:
            training_state["global_step"] = self.state.global_step
            logger.debug(f"Saved global step {self.state.global_step} to checkpoint")

            # Save dataloader state if available
            dataloader_state = self._save_dataloader_state()
            if dataloader_state:
                training_state["dataloader_state"] = dataloader_state
                logger.info(
                    f"Saved dataloader state to checkpoint with keys: {list(dataloader_state.keys())}"
                )
            else:
                logger.warning("No dataloader state to save to checkpoint")

        # Save RNG state for reproducibility
        if self.args.save_rng_state:
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

            training_state["rng_state"] = rng_state
            logger.debug("Saved RNG state to checkpoint")
        return training_state

    def _save_training_state(self, output_dir: str) -> None:
        training_state = self._state_dict()
        if training_state:
            training_state_path = os.path.join(output_dir, "training_state.pt")
            torch.save(training_state, training_state_path)
            logger.info(f"Saved training state to {training_state_path}")
        else:
            logger.warning("No training state saved!")

    def _load_state_dict(self, training_state) -> None:
        if not training_state:
            logger.warning("Training state was not loaded!")
            return

        if self.args.restore_optimizer_state and "optimizer" in training_state:
            if self.optimizer is not None:
                self.optimizer.load_state_dict(training_state["optimizer"])
                logger.info("Restored optimizer state from checkpoint")
            else:
                logger.warning(
                    "Cannot restore optimizer state: optimizer not initialized"
                )

        if self.args.restore_scheduler_state and "lr_scheduler" in training_state:
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                logger.info("Restored LR scheduler state from checkpoint")
            else:
                logger.warning(
                    "Cannot restore LR scheduler state: scheduler not initialized"
                )

        # Also restore global step if present in training state
        if self.args.restore_dataset_state and "global_step" in training_state:
            self.state.global_step = training_state["global_step"]
            logger.info(f"Restored global step to {self.state.global_step}")

            # Restore dataloader state if available
            if "dataloader_state" in training_state:
                self._load_dataloader_state(training_state["dataloader_state"])
                logger.info("Restored dataloader state from checkpoint")

        # Restore RNG state for reproducibility
        if self.args.restore_rng_state and "rng_state" in training_state:
            rng_state = training_state["rng_state"]

            # Restore CPU RNG state
            if "torch_rng_state" in rng_state:
                torch.set_rng_state(rng_state["torch_rng_state"])
                logger.debug("Restored CPU RNG state from checkpoint")

            # Restore CUDA RNG state if available
            if "cuda_rng_state" in rng_state and torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                saved_device = rng_state.get("cuda_device", current_device)

                if current_device != saved_device:
                    logger.warning(
                        f"CUDA device mismatch: current={current_device}, saved={saved_device}. "
                        "Restoring RNG state anyway (should be fine with identical GPU models)."
                    )

                torch.cuda.set_rng_state(
                    rng_state["cuda_rng_state"], device=current_device
                )
                logger.debug(
                    f"Restored CUDA RNG state for device {current_device} from checkpoint"
                )

            logger.info("Restored RNG state from checkpoint")
        elif self.args.restore_rng_state:
            logger.info("No RNG state found in checkpoint - using current RNG state")

    def _save_dataloader_state(self):
        """
        Save dataloader state for checkpointing. Subclasses should override.
        Returns dict or None if no state to save.
        """
        return None

    def _load_dataloader_state(self, dataloader_state):
        """
        Load dataloader state from checkpoint. Subclasses should override.
        """
        pass

    def _load_training_state(self, checkpoint_path: str) -> None:
        """Load optimizer and scheduler state. Subclasses can override for custom behavior."""
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")

        if not os.path.exists(training_state_path):
            logger.info(f"No training state file found at: {training_state_path}")
            return None

        try:
            training_state = torch.load(
                training_state_path, map_location=torch.device("cpu")
            )

        except Exception as e:
            logger.error(
                f"Failed to load training state from {training_state_path}: {e}"
            )
        self._load_state_dict(training_state)

    def _load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint using the sharded checkpoint loader."""
        assert self.model is not None

        # Handle case where device might be None
        device = self.args.device if self.args.device is not None else "cpu"

        # Use the sharded checkpoint loader to handle all checkpoint formats
        logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")
        load_checkpoint(
            checkpoint_path, self.model, device=torch.device(device), strict=True
        )

    @staticmethod
    def _get_sdpa_backends(
        backend: List[str | SDPBackend] | str | SDPBackend | None,
    ) -> List[SDPBackend] | SDPBackend | None:
        """
        Normalize various SDPA backend specification types
        """
        if backend is None:
            return None

        sdpa_mapping = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }

        def get_backend(b):
            if isinstance(b, SDPBackend):
                return b
            return sdpa_mapping[b]

        if isinstance(backend, str):
            return get_backend(backend)
        elif isinstance(backend, list):
            return [get_backend(i) for i in backend]
        else:
            raise ValueError("sdpa-backend must be a List[str] or str")

    def _dispatch_event(self, event: str, **kwargs):
        """
        Dispatch event to all callbacks
        """
        control = TrainerControl()
        # Dispatch to call backkbacks in list
        for callback in self.callbacks:
            event_handler = getattr(callback, event, None)
            # If handler is undefined, skip to next.
            if event_handler is None:
                continue

            new_control = event_handler(
                self.args,
                self.state,
                control,
                model=self.unwrapped_model(),
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )

            if new_control is not None:
                control = new_control
        return control

    @abstractmethod
    def _post_init(self) -> None:
        """
        This hook is intended to be used for an implementation which needs to wrap the components,
        load things to devices, etc.

        For example, Torch DDP and Accelerate.
        """
        ...

    @abstractmethod
    def _prepare(
        self, train_dataset: Dataset | NoneType, eval_dataset: Dataset | NoneType
    ) -> None:
        """
        Prepare for training and/or evaluation

        The dataloaders shoud be constructed for the provided datasets, which MAY be None.
        If train_dataset is not None, prepare for training:
            Init optimizer, lr_schedulr, etc.

        Subclasses of a concrete implementation may use this to 'wrap' objects.
        e.g. Accelerate or DDP.
        """
        ...

    @abstractmethod
    def _train_loop(self) -> TrainOutput:
        """
        The inner training loop
        """
        ...

    @abstractmethod
    def _eval_loop(self) -> dict[str, float]:
        """
        The inner evaluation loop
        """
        ...
