# A light-weight replacement for the HF Trainer class
from typing import Any, Callable, Iterable, Optional, List, Dict
from tqdm.auto import tqdm
import os
import datetime

from ...utils import format_train_info, format_mapping

from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback,
)


class ProgressCallback:
    """
    A TQDM progress-bar callback class based upon:
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py
    In theory, the original should work. This version provides a working reference implementation,
    should a future change to the original break compatibility.
    """

    def __init__(self):
        super().__init__()
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.last_step = state.global_step
        self.train_progress_bar = tqdm(
            initial=state.global_step, total=state.max_steps, dynamic_ncols=True
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.train_progress_bar.close()
        self.train_progress_bar = None

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self.train_progress_bar.update(state.global_step - self.last_step)
        self.last_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.eval_progress_bar is None:
            max_eval_steps = getattr(state, "max_eval_steps", -1)
            total = max(len(eval_dataloader), max_eval_steps, 1)
            self.eval_progress_bar = tqdm(
                initial=1,
                total=total,
                leave=self.train_progress_bar is None,
                dynamic_ncols=True,
            )
        else:
            self.eval_progress_bar.update(1)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.write(self._format_eval(state, metrics))
            self.eval_progress_bar.close()
            self.eval_progress_bar = None

    def on_log(self, args, state, control, logs, **kwargs):
        if not state.is_world_process_zero or self.train_progress_bar is None:
            return
        self.train_progress_bar.write(self._format_train(state, logs))

    @staticmethod
    def _record_header(state):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        s = f"{timestamp:<22}{state.global_step:>10,d}  {round(state.epoch, 2):<5.3}"
        return s

    @staticmethod
    def format_mapping(header, mapping):
        s = header + " "
        for key, value in mapping.items():
            if isinstance(value, int):
                value = f"{value:,}"
            elif isinstance(value, float):
                value = f"{value:.4}"
            elif not isinstance(value, str):
                value = pformat(value)
            if len(value) > 80:
                s += f"{key}:\n{value}\n"
            else:
                s += f"{key}: {value} "
        return s

    @staticmethod
    def _format_train(state, record):
        header = ProgressCallback._record_header(state)
        if "loss" in record and "learning_rate" in record and "grad_norm" in record:
            return f"{header} train-loss: {round(record['loss'], 5):<10}grad-norm: {round(record['grad_norm'], 5):<10}learning-rate: {record['learning_rate']:1.2e}"
        # Fallback to generic formatting
        else:
            return ProgressCallback.format_mapping(header, record)

    @staticmethod
    def _format_eval(state, record):
        header = ProgressCallback._record_header(state)
        if "eval_loss" in record:
            return f"{header} eval-loss:  {round(record['eval_loss'], 5):<10}"
        else:
            return ProgressCallback.format_mapping(header, record)


class InfoCallback:
    def on_train_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        info, extra_info = format_train_info(args, state, control, **kwargs)
        print(format_mapping(info))
        # logger.debug(format_mapping(extra_info))
        # logger.info("IPY", hasattr(__builtins__,'__IPYTHON__'))
