from typing import (
    Callable,
    Optional,
    List,
    Dict,
    Tuple,
)

import os
import platform
import time
from abc import abstractmethod
from contextlib import ExitStack
import logging
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import Dataset
from torch.distributed.checkpoint.stateful import Stateful

from .trainer_types import (
    ExtensibleTrainer,
    MinimalTrainingArguments,
    TrainerState,
    TrainOutput,
    TrainerControl,
    CheckpointInterface,
    StatefulProvider,
    IntervalStrategy,
)

from .checkpoint_manager import RNGState

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BaseTrainingArguments(MinimalTrainingArguments):
    # These arguments are NOT HF Compatible!
    # Checkpointing options
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_dataset_state: bool = True
    save_rng_state: bool = True

    restore_optimizer_state: bool = True
    restore_scheduler_state: bool = True
    restore_dataset_state: bool = True
    restore_rng_state: bool = True

    # Default torch dtype
    default_dtype: str | None = None

    # Limit maximum validaiton/eval steps
    max_eval_steps: int = -1

    # Offload activation tensors to CPU memory -- best combined with some form of activation checkpointing.
    # https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#saving-tensors-to-cpu
    enable_activation_offloading: bool = False

    # Set torch.autograd.set_detect_anomaly
    # https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection
    detect_anomaly: bool = False

    # Set SDPA Kernel backend(s)
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel
    sdpa_backend: List[str] | str | None = (
        None  # "math" | "flash" | "efficient" | "cudnn"
    )
    sdpa_set_priority: bool = False  # If list, interpret as priority order

    # Set on NVIDIA Ampere or later GPUs to "high" when training in 32-bit precision for a significant speedup
    # https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    float32_matmul_precision: str | None = None  # "highest" | "high" | "medium"

    dynamo_recompile_limit: int | None = None

    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(
                self.output_dir, "runs", f"{time.time_ns()}_{platform.node()}"
            )

        # As per https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        if self.dataloader_prefetch_factor is None and self.dataloader_num_workers > 0:
            self.dataloader_prefetch_factor = 2
        if self.torch_compile_backend is None:
            self.torch_compile_backend = "inductor"
        if self.torch_compile_mode is None:
            self.torch_compile_backend = "default"

        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}

        # Auto-determine greater_is_better from metric name if not set
        if self.greater_is_better is None:
            # Common metrics where higher is better
            higher_is_better_metrics = {
                "accuracy",
                "f1",
                "precision",
                "recall",
                "auc",
                "roc_auc",
                "ap",
                "map",
                "bleu",
                "rouge",
                "meteor",
                "bertscore",
                "exact_match",
                "squad_f1",
            }
            # Check if metric name contains any higher-is-better patterns
            metric_lower = self.metric_for_best_model.lower()
            self.greater_is_better = any(
                pattern in metric_lower for pattern in higher_is_better_metrics
            )

        # Validate alignment requirements for load_best_model_at_end
        if self.load_best_model_at_end:
            if self.save_strategy != self.eval_strategy:
                raise ValueError(
                    "load_best_model_at_end requires save_strategy and eval_strategy to be the same. "
                    f"Got save_strategy={self.save_strategy}, eval_strategy={self.eval_strategy}"
                )

            if (
                self.save_strategy == IntervalStrategy.STEPS
                and self.eval_strategy == IntervalStrategy.STEPS
            ):
                if self.save_steps % self.eval_steps != 0:
                    raise ValueError(
                        "load_best_model_at_end requires save_steps to be a multiple of eval_steps when using step-based strategies. "
                        f"Got save_steps={self.save_steps}, eval_steps={self.eval_steps}"
                    )


class BaseTrainer(ExtensibleTrainer, Stateful, StatefulProvider):
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
        args: BaseTrainingArguments,
        model: torch.nn.Module | None = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init: Optional[Callable[[], torch.nn.Module]] = None,
        callbacks: List | None = None,
        # Depreicated; use processing_class
        tokenizer=None,
        compute_loss_func: Callable | None = None,
    ):
        if callbacks is None:
            callbacks = []

        assert (
            model or model_init
        ), "Either a model or a model constructor must be specified"

        assert (
            args.gradient_accumulation_steps > 0
        ), "gradient_accumulation_steps must be > 0"

        # Try to maintain backward compatability for now.
        if processing_class is None and tokenizer is not None:
            processing_class = tokenizer

        self.model = model
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
        self.is_local_process_zero = True
        self.is_world_process_zero = True
        self.num_processes = 1
        self.checkpoint_manager: CheckpointInterface | None = None

        self.state = TrainerState(
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            train_batch_size=args.per_device_train_batch_size,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            max_eval_steps=args.max_eval_steps,
        )
        self.control = TrainerControl()

        # Silence annoying Huggingface FastTokenizer warnings
        # If knows if it is safe or not, and does the right thing, why
        # do I need to hear about it and create a janky workaround for
        # a non-issue!?
        if self.args.dataloader_num_workers > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.args.dynamo_recompile_limit:
            logger.info(
                f"Setting torch._dynamo.config.recompile_limit = {self.args.dynamo_recompile_limit}"
            )
            torch._dynamo.config.recompile_limit = self.args.dynamo_recompile_limit

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

    # AbstractBaseTrainer
    # @override
    def train(self, **kwargs) -> TrainOutput:
        """
        The main entry point to start training the model.
        """
        with ExitStack() as exit_stack:
            backends = self._get_sdpa_backends(self.args.sdpa_backend)
            if backends:
                logger.info(
                    f"sdpa_backends={backends}, set_priority={self.args.sdpa_set_priority}"
                )
                exit_stack.enter_context(
                    sdpa_kernel(backends, set_priority=self.args.sdpa_set_priority)
                )
            if self.args.enable_activation_offloading:
                exit_stack.enter_context(
                    torch.autograd.graph.save_on_cpu(pin_memory=True)
                )
            self._prepare(
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset
            )
            return self._train_loop()

    # AbstractBaseTrainer
    # @override
    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, **kwargs
    ) -> dict[str, float]:
        """
        The main entry point to evaluate the model.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        with ExitStack() as exit_stack:
            backends = self._get_sdpa_backends(self.args.sdpa_backend)
            if backends:
                exit_stack.enter_context(
                    sdpa_kernel(backends, set_priority=self.args.sdpa_set_priority)
                )
            self._prepare(train_dataset=None, eval_dataset=eval_dataset)
            return self._eval_loop()

    # AbstractBaseTrainer
    # @override
    def add_callback(self, callback):
        if isinstance(callback, type):
            callback = callback()
        self.callbacks.append(callback)

    # AbstractBaseTrainer
    # @override
    def pop_callback(self, callback):
        if isinstance(callback, type):
            compare = lambda a, b: type(a) == b
        else:
            compare = lambda a, b: id(a) == id(b)
        for i, cb in enumerate(self.callbacks):
            if compare(cb, callback):
                return self.callbacks.pop(i)

    # AbstractBaseTrainer
    # @override
    def remove_callback(self, callback):
        self.pop_callback(callback)

    def log(self, logs: Dict[str, float]):
        self.state.log_history.append(logs)

        return self._dispatch_event(
            "on_log",
            logs=logs,
        )

    @staticmethod
    def _get_sdpa_backends(
        backend: List[str | SDPBackend] | str | SDPBackend | None,  # type: ignore[valid-type]
    ) -> List[SDPBackend] | SDPBackend | None:  # type: ignore[valid-type]
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
        # Dispatch to call backkbacks in list
        unwrapped_model = self.unwrapped_model()
        for callback in self.callbacks:
            event_handler = getattr(callback, event, None)
            # If handler is undefined, skip to next.
            if event_handler is None:
                continue

            new_control = event_handler(
                self.args,
                self.state,
                self.control,
                model=unwrapped_model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )

            if new_control is not None:
                self.control = new_control
        return self.control

    def unwrapped_model(self) -> torch.nn.Module:
        # Gen unwrapped model. e.g., if wrapped with DDP, return the base model
        assert self.model
        return self.model

    # AbstractBaseTrainer
    # @override
    def save_model(self, output_dir: Optional[os.PathLike | str] = None) -> None:
        assert self.checkpoint_manager
        self.checkpoint_manager.save_model(
            output_dir=output_dir, overwrite_output_dir=self.args.overwrite_output_dir
        )

    # AbstractBaseTrainer
    # @override
    def save_checkpoint(self, checkpoint_path=None) -> None:
        assert self.checkpoint_manager
        self.checkpoint_manager.save_checkpoint(checkpoint_path)

    # AbstractBaseTrainer
    # @override
    def load_checkpoint(self, checkpoint_path=None) -> None:
        assert self.checkpoint_manager
        self.checkpoint_manager.load_checkpoint(checkpoint_path)

    # StatefulProvider
    # @override
    def get_statefuls_for_save(self):
        statefuls = {}
        save_dataset_state = False

        # Not all dataloaders are stateful and not all dataloader with
        # a state_dict() method are an instance of Stateful.
        if self.args.save_dataset_state:
            if hasattr(self.train_dataloader, "state_dict"):
                save_dataset_state = True
            else:
                logger.warning("train_dataloader doesn't have state_dict method")

        for key, obj, save in (
            ("optimizer", self.optimizer, self.args.save_optimizer_state),
            ("scheduler", self.lr_scheduler, self.args.save_scheduler_state),
            ("trainer", self, self.args.save_dataset_state),
            ("dataset", self.train_dataloader, save_dataset_state),
            ("rng", RNGState(), self.args.save_rng_state),
        ):
            if not save:
                continue
            assert obj, f"{key} is not initialized"
            statefuls[key] = obj

        return statefuls

    # StatefulProvider
    # @override
    def get_statefuls_for_load(self):
        statefuls = {}
        restore_dataset_state = False
        if self.args.restore_dataset_state:
            if hasattr(self.train_dataloader, "load_state_dict"):
                restore_dataset_state = True
            else:
                logger.warning(
                    "Could not restored Dataloader state, as it does not have a load method"
                )

        for key, obj, load in (
            ("optimizer", self.optimizer, self.args.restore_optimizer_state),
            ("scheduler", self.lr_scheduler, self.args.restore_scheduler_state),
            ("trainer", self, self.args.restore_dataset_state),
            ("dataset", self.train_dataloader, restore_dataset_state),
            ("rng", RNGState(), self.args.restore_rng_state),
        ):
            if not load:
                continue
            assert obj, f"{key} is not initialized"
            statefuls[key] = obj

        return statefuls

    # Stateful
    # @override
    def load_state_dict(self, state_dict):
        self.state.global_step = state_dict["global_step"]

    # Stateful
    # @override
    def state_dict(self):
        return {"global_step": self.state.global_step}

    @abstractmethod
    def _post_init(self) -> None:
        """
        This hook is intended to be used for an implementation which needs to wrap the components,
        load things to devices, etc.

        For example, Torch DDP and Accelerate.
        """
        pass

    @abstractmethod
    def _prepare(
        self, train_dataset: Dataset | None, eval_dataset: Dataset | None
    ) -> None:
        """
        Prepare for training and/or evaluation

        The dataloaders shoud be constructed for the provided datasets, which MAY be None.
        If train_dataset is not None, prepare for training:
            Init optimizer, lr_schedulr, etc.

        Subclasses of a concrete implementation may use this to 'wrap' objects.
        e.g. Accelerate or DDP.
        """
        pass

    @abstractmethod
    def _train_loop(self) -> TrainOutput:
        """
        The inner training loop
        """
        pass

    @abstractmethod
    def _eval_loop(self) -> dict[str, float]:
        """
        The inner evaluation loop
        """
        pass


def logits_from_outputs(outputs) -> Tensor:
    if not isinstance(outputs, Tensor):
        assert hasattr(outputs, "logits")
        return outputs.logits
    return outputs


def loss_from_outputs(outputs) -> Tensor:
    if isinstance(outputs, tuple):
        loss = outputs[0]
        assert isinstance(loss, Tensor)
        return loss
    assert hasattr(outputs, "loss")
    return outputs.loss


def loss_and_logits_from_outputs(outputs) -> Tuple[Tensor, Tensor]:
    if isinstance(outputs, tuple):
        loss, logits = outputs
        assert isinstance(loss, Tensor) and isinstance(logits, Tensor)
        return loss, logits
    assert hasattr(outputs, "loss") and hasattr(outputs, "logits")
    return outputs.loss, outputs.logits
