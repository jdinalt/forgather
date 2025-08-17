from typing import Any, Callable, Iterable, Optional, Dict
import os
import math

from pprint import pformat
from torch.utils.tensorboard import SummaryWriter

from ...utils import format_train_info, format_mapping
from ..trainer_types import TrainerCallback


# Log loss to TensorBoard
class TBLogger(TrainerCallback):
    """
    A Trainer callbacks which implements Tensorboard logging.
    """

    def __init__(self, summary_writer, **kwargs):
        super().__init__()
        self.summary_writer = summary_writer
        self.last_step = -1
        self.kwargs = self.mapping_as_markdown(kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if len(self.kwargs):
            self.summary_writer.add_text("experiment", self.kwargs)

        info, extra_info = format_train_info(args, state, control, **kwargs)
        self.summary_writer.add_text(
            "training_info", self.mapping_as_markdown(info | extra_info)
        )

    @staticmethod
    def mapping_as_markdown(mapping):
        """
        Format dictionary as markdown

        Tensorboard expects text to be in markdown format...
        """
        s = "```\n"
        s += format_mapping(mapping)
        s += "```"
        return s

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not state.is_world_process_zero:
            return
        global_step = state.global_step
        if self.last_step == global_step:
            return
        self.last_step = global_step
        self.summary_writer.add_scalar(
            "eval-loss", metrics["eval_loss"], global_step=global_step
        )
        if metrics.get("eval_accuracy") is not None:
            self.summary_writer.add_scalar(
                "eval-accuracy", metrics["eval_accuracy"], global_step=global_step
            )
        try:
            pp = math.exp(metrics["eval_loss"])
        except OverflowError:
            pp = float("inf")
        self.summary_writer.add_scalar("eval-perplexity", pp, global_step=global_step)

    def on_log(self, args, state, control, logs, **kwargs):
        if not state.is_world_process_zero:
            return
        global_step = state.global_step
        if "loss" in logs:
            self.summary_writer.add_scalar(
                "train-loss", logs["loss"], global_step=global_step
            )
        if "learning_rate" in logs:
            self.summary_writer.add_scalar(
                "learning-rate", logs["learning_rate"], global_step=global_step
            )
        self.summary_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero and len(state.log_history):
            return
        self.summary_writer.add_text(
            "train_results", self.mapping_as_markdown(state.log_history[-1])
        )
