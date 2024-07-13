# A light-weight replacement for the HF Trainer class
from typing import Any, Callable, Iterable, Optional, List, Dict
import copy
from dataclasses import dataclass, field
import inspect
import torch
from torch import Tensor
from tqdm.auto import tqdm
import json
import os
import datetime

from .utils import alt_repr, format_train_info, format_mapping

from .trainer_types import (
    TrainingArguments, TrainerState, TrainerControl, TrainerCallback)

class ProgressCallback(TrainerCallback):
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
        self.step = 0
        self.train_progress_bar = tqdm(
            total=state.max_steps,
            dynamic_ncols=True
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.train_progress_bar.close()
        self.train_progress_bar = None
        
    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.train_progress_bar.update(state.global_step - self.step)
        self.step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.eval_progress_bar is None:
            self.eval_progress_bar = tqdm(
                total=len(eval_dataloader),
                leave=self.train_progress_bar is None,
                dynamic_ncols=True
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
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        s = f"{timestamp:<22}{state.global_step:>10,d}  {round(state.epoch, 2):<5.3}"
        return s

    @staticmethod
    def _format_train(state, record):
        header = ProgressCallback._record_header(state)
        if 'loss' in record and 'learning_rate' in record:
            return f"{header} train-loss: {round(record['loss'], 5):<10}learning-rate: {record['learning_rate']:1.2e}"
        else:
            return format_mapping(record)
        
    @staticmethod
    def _format_eval(state, record):
        header = ProgressCallback._record_header(state)
        if 'eval_loss' in record:
            return f"{header} eval-loss:  {round(record['eval_loss'], 5):<10}"
        else:
            return format_mapping(record)

class InfoCallback(TrainerCallback):
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if not state.is_world_process_zero:
            return
        info, extra_info = format_train_info(args, state, control, **kwargs)
        print(format_mapping(info))
        #logger.debug(format_mapping(extra_info))
        #logger.info("IPY", hasattr(__builtins__,'__IPYTHON__'))
            

class JsonLogger(TrainerCallback):
    """
    A very simple JSON  logger callback

    It just writes a JSON record to a file, adding a UTC timestamp, each time on_log or on_evaluate are called.
    """
    def __init__(self):
        super().__init__()
        self.log_file = None
        self.log_path = None
        self.prefix = ""

    def __del__(self):
        self.close()

    def close(self):
        if self.log_file is not None:
            self.log_file.write('\n]')
            self.log_file.close()
            self.log_file = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or args.logging_dir is None:
            return

        os.makedirs(args.logging_dir, exist_ok=True)
        self.log_path = os.path.join(args.logging_dir, "trainer_logs.json")
        self.log_file = open(self.log_path, 'x')
        self.log_file.write("[\n")
        info, extra_info = format_train_info(args, state, control, **kwargs)
        self._write_log(state, info | extra_info)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.log_file is None:
            return
        self._write_log(state, metrics)

    def on_log(self, args, state, control, logs, **kwargs):
        if self.log_file is None:
            return
        self._write_log(state, logs)
        
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.close()

    def _write_log(self, state, data: dict):
        assert(self.log_file is not None)
        new_fields = dict(
            timestamp = datetime.datetime.utcnow().timestamp(),
            global_step = state.global_step,
            epoch = state.epoch,
        )
        self.log_file.write(self.prefix + json.dumps(new_fields | data))
        self.prefix = ",\n"