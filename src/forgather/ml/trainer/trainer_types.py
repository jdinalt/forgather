import os
from typing import Any, List, Dict, NamedTuple, Optional
from abc import ABC, abstractmethod
from typing import Protocol
from dataclasses import dataclass, field
from pprint import pformat

from torch.utils.data import Dataset
from torch.distributed.checkpoint.stateful import Stateful

from ..utils import ConversionDescriptor, DiagnosticEnum

OUTPUTDIR_NAME = "tmp_trainer"


class TrainOutput(NamedTuple):
    global_step: int
    metrics: Dict[str, float]


class IntervalStrategy(DiagnosticEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


@dataclass(kw_only=True)
class TrainerState:
    """
    Trainer global state to be passed to callbacks.
    This is the same API as used by the HF Trainer class, for compatibility.
    Not all values are implemented at present.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#

    """

    logging_steps: int
    eval_steps: int
    train_batch_size: int
    max_steps: int
    epoch: float = 0.0
    global_step: int = 0
    num_train_epochs: int
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    log_history: list[Dict[str, float]] = field(default_factory=lambda: [])
    save_steps: int = 0
    best_metric: float | None = None
    best_model_checkpoint: str | None = None
    # Unimplemented in Trainer; included for consistency with HF Trainer
    num_input_tokens_seen: int = 0
    total_flos: float = 0.0
    is_hyper_param_search: bool = False
    stateful_callbacks: List["TrainerCallback"] = field(default_factory=lambda: [])


@dataclass(slots=True)
class TrainerControl:
    """
    Controls the execution flow of the Trainer class
    This is the same API as used by the HF Trainer class, for compatibility.
    This is only partially implemented at present.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    # Forgather extension: abort without saving
    should_abort_without_save: bool = False


@dataclass(kw_only=True)
class MinimalTrainingArguments:
    """
    Stores training arguments, independent of model/dataset/etc.

    A sub-set of the TrainingArguments from transformers.TrainingArguments
    As a minimal sub-set, this should not be "everything-for-everyone."
    Additional arguments can be added via sub-classing.
    """

    output_dir: str = OUTPUTDIR_NAME
    logging_dir: str | None = None
    per_device_eval_batch_size: int = 16
    per_device_train_batch_size: int = 16
    num_train_epochs: int = 1
    device: Any = None

    seed: int = -1
    use_cpu: bool = False

    # Not if HF trainer; number of train-batches in an epoch, when dataset does not support len()
    # This just becomes a relative value for book-keeping.
    epoch_train_steps: int = 100000
    max_steps: int = -1

    dataloader_num_workers: int = 0
    dataloader_pin_memory: int = True
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int | None = None
    dataloader_drop_last: bool = False

    # Strategy may also be: "no" | "steps" | "epoch"
    eval_strategy: ConversionDescriptor = ConversionDescriptor(
        IntervalStrategy, default=IntervalStrategy.NO
    )
    eval_steps: int = 100
    eval_delay: int = 0

    logging_strategy: ConversionDescriptor = ConversionDescriptor(
        IntervalStrategy, default=IntervalStrategy.STEPS
    )
    logging_steps: int = 50
    logging_first_step: bool = False

    torch_compile: bool = False
    torch_compile_backend: str | None = None
    torch_compile_mode: str | None = "default"
    torch_compile_dynamic: bool = True
    torch_compile_full_graph: bool = False

    max_grad_norm: float | None = None
    gradient_accumulation_steps: int = 1

    # Checkpointing options
    save_strategy: ConversionDescriptor = ConversionDescriptor(
        IntervalStrategy, default=IntervalStrategy.STEPS
    )
    save_steps: int = 1000
    save_total_limit: int = 2
    save_safetensors: bool = True
    save_on_each_node: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: bool | str = False

    # Best model tracking and loading options
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    greater_is_better: bool | None = None  # Auto-determined from metric name

    # Compatibility with HF Trainer -- would be better if they took a factory arg...
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: dict | None = None
    warmup_steps: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8

    # Enable gradient checkpointing (a.k.a activation checkpointing) on models which support the HF API
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict | None = None

    def __str__(self):
        return pformat(self)


class AbstractBaseTrainer(Protocol):
    """
    A minimal subset of core "Trainer" methods, based upon the HF Trainer API

    We are trying to keep this down to a minimum, as all "Trainers" should not
    need to need to support every conceivable use-case. That's what class specialization
    is for!

    A "Trainer," at a minimum, should be able to "train," "evaluate', and "save" models.
    """

    @abstractmethod
    def train(self, **kwargs) -> TrainOutput:
        pass

    @abstractmethod
    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, **kwargs
    ) -> dict[str, float]:
        """
        Perform evaluation, either from the default eval dataset or from a specified dataset.

        Returns: A dictionary of metrics.
        """
        pass

    @abstractmethod
    def save_model(self, output_dir: Optional[os.PathLike | str] = None) -> None:
        """
        Save the model, either to the default location or to the specified location.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_path=None) -> None:
        """
        Save model / trainer checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path=None) -> None:
        """
        Load model / trainer checkpoint
        """
        pass


class ExtensibleTrainer(AbstractBaseTrainer):
    """
    A slightly extended abstract Trainer, which supports the TrainerCallback API.
    """

    @abstractmethod
    def add_callback(self, callback):
        """
        Add callback to the list of callbacks
        Either a type (instantiate it) or an instance
        """
        pass

    @abstractmethod
    def pop_callback(self, callback):
        """
        Callback may either be and instance or a type
        Remove the first match and return it
        """
        pass

    @abstractmethod
    def remove_callback(self, callback):
        """
        Like pop, but don't return it.
        This seems redundant, but API consistency...
        """
        pass


class TrainerCallback(Protocol):
    """
    Abstract trainer callback for handling various events.
    This interface is intended to be compatible with the HF Trainer, as to ease porting.
    Not all callbacks are implemented at present.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#
    """

    def on_init_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_train_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_train_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_epoch_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_epoch_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_step_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_optimizer_step(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_substep_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_step_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_evaluate(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_predict(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_save(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_log(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_prediction_step(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_pre_optimizer_step(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass


class CheckpointInterface(Protocol):
    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_path: str | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        pass

    @abstractmethod
    def save_model(
        self,
        output_dir: str | os.PathLike | None = None,
        overwrite_output_dir: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def set_best_checkpoint(self, best_checkpoint: str) -> None:
        pass

    @abstractmethod
    def resolve_checkpoint_path(self, checkpoint_path: str | None) -> str | None:
        pass


class StatefulProvider(Protocol):
    @abstractmethod
    def get_statefuls_for_save(self) -> Dict[str, Stateful]:
        pass

    @abstractmethod
    def get_statefuls_for_load(self) -> Dict[str, Stateful]:
        pass
