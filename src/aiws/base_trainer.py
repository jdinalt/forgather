from typing import (
    Callable,
    Optional,
    List,
    Type,
    Dict,
)
from types import NoneType
import os
from abc import abstractmethod

from loguru import logger
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedModel,
)

from .trainer_types import (
    ExtensibleTrainer,
    TrainingArguments,
    TrainOutput,
    TrainerControl,
)

WEIGHTS_NAME = "pytorch_model.bin"


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
        tokenizer=None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: List = None,
    ):
        if callbacks is None:
            callbacks = []
        # Init args
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.callbacks = self.default_callbacks()
        self.callbacks.extend(callbacks)

        # Init attributes
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.state = None
        self.is_local_process_zero = True
        self.is_world_process_zero = True
        self.num_processes = 1

        self._post_init()
        self._validate_dirs()
        self._dispatch_event("on_init_end")

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"model={self.model},"
            f"args={self.args},"
            f"data_collator={self.data_collator},"
            f"train_dataset={self.train_dataset},"
            f"eval_dataset={self.eval_dataset},"
            f"tokenizer={self.tokenizer},"
            f"model_init={self.model_init},"
            f"callbacks={self.callbacks},"
            ")"
        )

    def train(self, **kwargs) -> TrainOutput:
        """
        The main entry point to start training the model.
        """
        self._prepare(train_dataset=self.train_dataset, eval_dataset=self.eval_dataset)
        self.model = self.model.to(self.args.device)
        return self._train_loop()

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, **kwargs
    ) -> dict[str, float]:
        """
        The main entry point to evaluate the model.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        else:
            assert isinstance(eval_dataset, Dataset)

        self._prepare(train_dataset=None, eval_dataset=eval_dataset)
        self.model = self.model.to(self.args.device)
        return self._eval_loop()

    def save_model(self, output_dir: os.PathLike | str = None) -> None:
        """
        Save model and tokenizer to output_dir
        """
        if self.model is None:
            return
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.overwrite_output_dir and self.model_exists(output_dir):
            raise Exception(
                "Would overwrite output model in output directory. "
                f"Set 'args.overwrite_output_dir' to override: {output_dir}"
            )
        os.makedirs(output_dir, exist_ok=True)
        self._save(output_dir)

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
        if not self.is_local_process_zero:
            return
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
        else:
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        logging_dir = self.args.logging_dir
        if not os.path.isdir(logging_dir):
            os.makedirs(logging_dir, exist_ok=True)

    def _save(self, output_dir):
        if self.is_world_process_zero:
            model = self.unwrapped_model()
            if isinstance(model, PreTrainedModel):
                model.save_pretrained(
                    save_directory=output_dir,
                    safe_serialization=True,
                )
            else:
                logger.info("Saving model as state-dictionary")
                torch.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

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
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )

            if new_control is not None:
                control = new_control
        return control

    def _save_checkpoint(self):
        logger.warning("Trainer._save_checkpoint() is unimplemented!")

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
