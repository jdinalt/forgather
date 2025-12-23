import math
import os
from pprint import pformat
from typing import Any, Callable, Dict, Iterable, Optional

from torch.utils.tensorboard import SummaryWriter

from ...utils import format_mapping, format_train_info
from ..trainer_types import TrainerCallback


def get_perplexity(value, metrics):
    try:
        pp = math.exp(value)
    except OverflowError:
        pp = float("inf")
    return pp


# Log loss to TensorBoard
class TBLogger(TrainerCallback):
    """
    A Trainer callbacks which implements Tensorboard logging.
    summary_writer: A Tensor Board Summary Writer instance
    scalars: A list of tuples describing scalars to log.
        (METRIC_KEY, LABLE) or (METRIC_NAME, LABEL, CALLABLE)
        In the case of the third tuple element, this is a callable of the
        form "my_callable(value, metrics)", where value is the value from METRIC_KEY,
        and metrics is a dictionary of all the reported metrics. Returns a new value
        to log, derived from metrics.
    """

    def __init__(
        self,
        summary_writer,
        scalars=None,
        **kwargs,
    ):
        super().__init__()
        if not scalars:
            scalars = [
                ("loss", "train-loss"),
                ("learning_rate", "learning-rate"),
                ("grad_norm", "grad-norm"),
                ("max_grad_norm", "grad-norm[max]"),
                ("eval_accuracy", "eval-accuracy"),
                ("eval_loss", "eval-loss"),
                ("eval_loss", "eval-perplexity", get_perplexity),
            ]
        self.scalars = scalars
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

    def _log_metrics(self, global_step, metrics):
        for scalar in self.scalars:
            metric_key = scalar[0]
            label = scalar[1]
            value = metrics.get(metric_key, None)
            if not value:
                continue
            if len(scalar) > 2:
                value = scalar[2](value, metrics)
                if value is None:
                    continue

            self.summary_writer.add_scalar(label, value, global_step=global_step)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not state.is_world_process_zero:
            return
        global_step = state.global_step
        if self.last_step == global_step:
            return
        self.last_step = global_step

        self._log_metrics(global_step, metrics)

    def on_log(self, args, state, control, logs, **kwargs):
        if not state.is_world_process_zero:
            return

        self._log_metrics(state.global_step, logs)
        self.summary_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero and len(state.log_history):
            return
        self.summary_writer.add_text(
            "train_results", self.mapping_as_markdown(state.log_history[-1])
        )
