# A light-weight replacement for the HF Trainer class
from typing import Any, Callable, Iterable, Optional
import copy
from dataclasses import dataclass, field
import inspect
import torch
from torch import Tensor
from tqdm.auto import tqdm

@dataclass
class TrainerControl:
    """
    Controls the execution flow of the Trainer class
    This is the same API as used by the HF Trainer class, for compatibility.
    This is only partially implemented at present.
    """
    should_training_stop: bool = False # Implemented for on_step_end()
    should_epoch_stop: bool = False # Implemented for on_epoch_end()
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

class TrainerCallback:
    pass

class TrainingArguments:
    pass

@dataclass
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
    log_history: list[TrainerCallback] = field(default_factory=lambda: [])
    ########################
    # Non-standard values
    ########################

    # The total number of processes used for training
    num_processes: int = 1
    

class TrainerCallback:
    """
    Abstract trainer callback for handling various events.
    This interface is intended to be compatible with the HF Trainer, as to ease porting.
    Not all callbacks are implemented at present.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#
    """
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Unimplemented
        """
        pass
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Unimplemented
        """
        pass
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Unimplemented
        """
        pass
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

class ProgressCallback(TrainerCallback):
    """
    A TQDM progress-bar callback class based upon:
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py
    """
    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.train_progress_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.train_progress_bar.update(1)

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
            self.eval_progress_bar.close()
            self.eval_progress_bar = None
        print(f"\nEval: {metrics}")
    
    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero and self.train_progress_bar is not None:
            # avoid modifying the logs object as it is shared between callbacks
            logs = copy.deepcopy(logs)
            _ = logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in logs:
                logs["epoch"] = round(logs["epoch"], 2)
            self.train_progress_bar.write(str(logs))

class InfoCallback(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return
        print("***** Running training *****")
        print(f"args: {args}")
        print(f"state: {state}")
                    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return
        print("\n\nTraining completed")
        print(f"{state.log_history[-1]}")
