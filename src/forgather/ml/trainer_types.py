import os
from typing import Any, List, Dict, NamedTuple, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import namedtuple
import platform
import time

from torch.utils.data import DataLoader, Dataset
import numpy as np

from .utils import ConversionDescriptor, DiagnosticEnum

OUTPUTDIR_NAME = "tmp_trainer"
LR_SCHEDULER_NAME = None


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
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
    num_train_epochs: int = 0
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    log_history: list[Dict[str, float]] = field(default_factory=lambda: [])
    save_steps: int = 0

    # Unimplemented in Trainer; included for consistency with HF Trainer
    num_input_tokens_seen: int = 0
    total_flos: float = 0.0
    best_metric: float = 0.0
    best_model_checkpoint: str = None
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


@dataclass(kw_only=True)
class MinimalTrainingArguments:
    """
    A minimal sub-set of the TrainingArguments from transformers.TrainingArguments
    """

    output_dir: str = OUTPUTDIR_NAME
    logging_dir: str = None
    logging_steps: int = 500
    per_device_eval_batch_size: int = 16
    per_device_train_batch_size: int = 16
    learning_rate: float = 1.0e-3
    num_train_epochs: int = 1
    device: Any = None

    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(
                self.output_dir, "runs", f"{time.time_ns()}_{platform.node()}"
            )


@dataclass(kw_only=True)
class TrainingArguments(MinimalTrainingArguments):
    """
    Stores training arguments, independent of model/dataset/etc.

    A sub-set of the TrainingArguments from transformers.TrainingArguments
    As a minimal sub-set, this should not be "everything-for-everyone."
    Additional arguments can be added via sub-classing.
    """

    max_steps: int = -1
    lr_scheduler_type: str = LR_SCHEDULER_NAME
    warmup_steps: int = 0
    eval_steps: int = 500
    save_steps: int = 500
    dataloader_num_workers: int = 0
    dataloader_pin_memory: int = True
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = None
    dataloader_drop_last: bool = False
    overwrite_output_dir: bool = False
    seed: int = -1

    eval_strategy: ConversionDescriptor = ConversionDescriptor(
        IntervalStrategy, default=IntervalStrategy.NO
    )
    logging_strategy: ConversionDescriptor = ConversionDescriptor(
        IntervalStrategy, default=IntervalStrategy.STEPS
    )
    save_strategy: ConversionDescriptor = ConversionDescriptor(
        IntervalStrategy, default=IntervalStrategy.STEPS
    )

    logging_first_step: bool = False
    eval_delay: int = 0
    save_total_limit: int = 2
    use_cpu: bool = False

    torch_compile: bool = False
    torch_compile_backend: str | None = None
    torch_compile_mode: str | None = None

    # Not if HF trainer; number of train-batches in an epoch, when dataset does not support len()
    # This just becomes a relative value for book-keeping.
    epoch_train_steps: int = 100000

    def __post_init__(self):
        super().__post_init__()
        # As per https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        if self.dataloader_prefetch_factor is None and self.dataloader_num_workers > 0:
            self.dataloader_prefetch_factor = 2
        if (
            self.torch_compile_backend is not None
            or self.torch_compile_mode is not None
        ):
            self.torch_compile = True
        if self.torch_compile_backend is None:
            self.torch_compile_backend = "inductor"


class AbstractBaseTrainer(ABC):
    """
    A minimal subset of core "Trainer" methods, based upon the HF Trainer API

    We are trying to keep this down to a minimum, as all "Trainers" should not
    need to need to support every conceivable use-case. That's what class specialization
    is for!

    A "Trainer," at a minimum, should be able to "train," "evaluate', and "save" models.
    """

    @abstractmethod
    def train(self, **kwargs) -> TrainOutput: ...
    @abstractmethod
    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, **kwargs
    ) -> dict[str, float]:
        """
        Perform evaluation, either from the default eval dataset or from a specified dataset.

        Returns: A dictionary of metrics.
        """
        ...

    @abstractmethod
    def save_model(self, output_dir: Optional[os.PathLike | str] = None) -> None:
        """
        Save the model, either to the default location or to the specified location.
        """
        ...


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
        ...

    @abstractmethod
    def pop_callback(self, callback):
        """
        Callback may either be and instance or a type
        Remove the first match and return it
        """
        ...

    def remove_callback(self, callback):
        """
        Like pop, but don't return it.
        This seems redundant, but API consistency...
        """
        self.pop_callback(self, callback)


class TrainerCallback(ABC):
    """
    Abstract trainer callback for handling various events.
    This interface is intended to be compatible with the HF Trainer, as to ease porting.
    Not all callbacks are implemented at present.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#
    """

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        pass

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass
