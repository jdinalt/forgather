import os
from typing import Optional
from collections.abc import Sequence

import torch
import datetime
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .trainer_types import AbstractBaseTrainer, TrainOutput, MinimalTrainingArguments


class SimpleTrainer(AbstractBaseTrainer):
    """
    A stripped-down version of the Forgather Trainer class

    And like that class, it supports a sub-set of the Huggingface Trainer API,
    thus is somewhat interchangable.

    This Trainer is intended as learning tool, to illustrate how to implement a Trainer,
    than it is a practical implementation. Despite this, it may still be faster for
    tiny models than the basic Trainer class.

    Overall, this code is probably easier to understand, but it less flexible.
    """

    def __init__(
        self,
        model
        model: torch.nn.Module = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        args: Optional[dict | MinimalTrainingArguments] = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        optimizer_factory=None,
    ):
        assert model or model_init, "Either a model or a model constructor must be specified"

        if model_init:
            self.model = model_init()
        else:
             self.model = model
        
        if args is None:
            args = TrainingArguments()
        elif isinstance(args, dict):
            args = TrainingArguments(**args)

        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer_factory = optimizer_factory

        # Determine which device to use.
        if self.args.device is None:
            self.args.device = (
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            )

        # Wrap our dataset in DataLoaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
        )

        # Default to SGD, if unspecified.
        if optimizer_factory is None:
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.learning_rate
            )
        else:
            self.optimizer = optimizer_factory(
                model.parameters(), lr=self.args.learning_rate
            )

        self.total_steps = len(self.train_dataloader) * self.args.num_train_epochs
        self.summary_writer = None

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"model={self.model},"
            f"args={self.args},"
            f"data_collator={self.data_collator},"
            f"train_dataset={self.train_dataset},"
            f"eval_dataset={self.eval_dataset},"
            f"optimizer_factory={self.optimizer_factory},"
            ")"
        )

    def train(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.global_step = 0

        # https://tqdm.github.io/docs/tqdm/
        # https://tqdm.github.io/docs/shortcuts/
        self.train_progress_bar = tqdm(total=self.total_steps, dynamic_ncols=True)

        # https://www.tensorflow.org/tensorboard
        # https://pytorch.org/docs/stable/tensorboard.html
        self.summary_writer = SummaryWriter(self.args.logging_dir)

        try:
            for i in range(self.args.num_train_epochs):
                self._train_loop()
                self._eval_loop()

        finally:
            self.train_progress_bar.close()
            self.train_progress_bar = None
            self.summary_writer.close()
            self.summary_writer = None

        # Not fully implemented
        return TrainOutput(self.global_step, 0.0, {})

    def save_model(self, output_dir: Optional[os.PathLike | str] = None) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")
        )

    def evaluate(self) -> dict[str, float]:
        return self._eval_loop()

    def _train_loop(self):
        self.model = self.model.to(self.args.device)
        self.model.train()

        # Train for a  full epoch.
        for batch in self.train_dataloader:
            args, kwargs = self._prepare_batch(batch)

            # Forward pass
            loss, logits = self.model(*args, **kwargs)

            # Backward step
            loss.backward()

            # Optimize
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.global_step += 1
            self.train_progress_bar.update()
            if self.global_step % self.args.logging_steps == 0:
                self.train_progress_bar.write(self._format_train(loss.item()))
                self.summary_writer.add_scalar(
                    "train-loss", loss.item(), global_step=self.global_step
                )

    # Don't compute gradients in eval mode
    @torch.no_grad()
    def _eval_loop(self):
        self.model.eval()
        correct = 0
        total_loss = 0

        eval_progress_bar = tqdm(
            total=len(self.eval_dataloader),
            leave=self.train_progress_bar is None,
            dynamic_ncols=True,
        )

        try:
            for batch in self.eval_dataloader:
                args, kwargs = self._prepare_batch(batch)
                loss, logits = self.model(*args, **kwargs)
                total_loss += loss.item()

                # Note: This metric function makes assumptions about the format of the
                # dataset, which are not true for all datasets.
                correct += (logits.argmax(1) == args[1]).type(torch.float).sum().item()
                eval_progress_bar.update()
        finally:
            mean_loss = total_loss / len(self.eval_dataloader)
            accuracy = correct / len(self.eval_dataloader.dataset)
            metrics = dict(mean_loss=mean_loss, accuracy=accuracy)
            eval_progress_bar.write(self._format_eval(mean_loss, accuracy))

            # Write resutls to Tensorboard, if available.
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(
                    "eval-loss", mean_loss, global_step=self.global_step
                )
                self.summary_writer.add_scalar(
                    "accuracy", accuracy, global_step=self.global_step
                )

            eval_progress_bar.close()
        return metrics

    def _prepare_batch(self, batch):
        """
        Move the batch to the device and returns (args, kwargs) in batch
        """
        if isinstance(batch, Sequence):
            return (tuple(x.to(self.args.device) for x in batch), {})
        else:
            return (tuple(), {k: v.to(self.args.device) for k, v in batch.items()})

    def _record_header(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch = self.global_step / len(self.train_dataloader)
        s = f"{timestamp:<22}{self.global_step:>10,d}  {round(epoch, 2):<5.3}"
        return s

    def _format_train(self, loss):
        header = self._record_header()
        return f"{header} train-loss: {round(loss, 5):<10}"

    def _format_eval(self, eval_loss, accuracy):
        header = self._record_header()
        return f"{header} eval-loss:  {round(eval_loss, 5):<10}accuracy: {(accuracy * 100):>0.1f}"
