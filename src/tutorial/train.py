# A light-weight replacement for the HF Trainer class
from typing import Any, Callable, Iterable, Optional
from dataclasses import dataclass, field
import inspect
import time
import datetime
import copy

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm.auto import tqdm
import evaluate
import numpy as np
import transformers
from transformers import (
    set_seed,
    default_data_collator
)

from accelerate import Accelerator

from tutorial_code.trainer_callback import (
    TrainerState,
    TrainerControl,
    TrainerCallback,
    ProgressCallback,
)

from transformers.utils import ExplicitEnum
class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"

@dataclass
class TrainingArguments:
    """
    Stores training arguments, independent of model/dataset/etc.
    Can init with dictionary: TrainingArguments(**my_dict)
    """
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    max_steps: int = -1
    log_steps: int = 500
    eval_steps: int = 500
    learning_rate: float = 1.0e-3
    num_train_epochs: int = 1
    seed: int = -1
    lr_scheduler_name: str = "constant"
    num_warmup_steps: int = 0
    device: str = 'cpu'

    # For compatibility with NotebookProgressCallback, which requires this attribute
    # Note: Even though the arguments specify epochs, internally this is always
    # converted to steps.
    eval_strategy: IntervalStrategy = IntervalStrategy.STEPS

def default_optimizer_factory(params, training_args):
    """
    Construct the default optimizer
    """
    return torch.optim.AdamW(
        params,
        lr=training_args.learning_rate
    )

def default_lr_scheduler_factory(optimizer, num_training_steps, training_args):
    """
    Construct the default learning-rate scheduler
    """
    return transformers.get_scheduler(
        name=training_args.lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

def default_data_collator_factory(tokenizer, training_arguments):
    """
    Construct the default learning-rate scheduler
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors='pt',
    )

class PeriodicFunction:
    '''
    A periodic function caller, which calls 'f' every 'period' steps
    '''
    def __init__(self, period: int, f: Callable):
        self.period = period
        assert(period > 0)
        self.f = f
        self.reset()

    def reset(self) -> None:
        self.counter = 0

    def step(self, *args, **kwargs) -> None:
        self.counter += 1
        if self.counter == self.period:
            self.f(*args, **kwargs)
            self.reset()

    def count(self) -> int:
        return self.counter
    
class Trainer:
    """
    This transformer trainer is a simplified version of the HF Trainer class
    The intent is to hopefully make the workings of such a class more comprehensible and
    easier to customize.
    """
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        callbacks = [ ProgressCallback() ],
        training_arguments: TrainingArguments=TrainingArguments(),
        data_collator_factory=default_data_collator_factory,
        optimizer_factory=default_optimizer_factory, 
        lr_scheduler_factory=default_lr_scheduler_factory,
    ):
        #print(training_arguments)
        self.args = training_arguments
        self.model = model
        self.tokenizer = tokenizer
        self.callbacks = callbacks
        self.device = self.args.device

        # Set the random seed
        if self.args.seed != -1:
            set_seed(self.args.seed)

        data_collator=data_collator_factory(tokenizer, self.args)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=data_collator,
            drop_last=True,
            #pin_memory=True,
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=True,
            #pin_memory=True,
        )

        self._update_training_steps()
        self.optimizer = optimizer_factory(self.model.parameters(), self.args)
        self.lr_scheduler = lr_scheduler_factory(self.optimizer, self.max_steps, self.args)
        self.state = None
        self._prepare()
        self._dispatch_event("on_init_end")

    def _update_training_steps(self) -> None:
        """
        Estimate the training steps from the train data-loader
        This value can potentially change when using parallel training.
        Should this occur, update the value be calling this again.
        """
         # The number of training steps in a single epoch
        self.epoch_train_steps = len(self.train_dataloader)
        
        # If limit is specified, constrain to limit.
        if self.args.max_steps >= 0:
            self.max_steps = min(self.args.max_steps, self.epoch_train_steps)
        else:
            # The total number of training steps in all epochs
            self.max_steps = self.args.num_train_epochs * self.epoch_train_steps
            
    def _prepare(self) -> None:
        """
        This hook is intended to be used for an implementation which needs to wrap the components,
        load things to devices, etc.
        
        For example, Torch DDP and Accelerate.
        """
        self.model = self.model.to(self.device)

    def train(self) -> None:
        """
        The main entry point to start training the model.
        """
        self._train_loop()
        self.eval()

    def eval(self) -> None:
        """
        The main entry point to evaluate the model.
        """
        return self._eval_loop()

    def _train_loop(self) -> None:
        """
        The inner training loop
        """
        start_time = time.perf_counter()
        self.model.zero_grad()
        periodic_log = PeriodicFunction(period=self.args.log_steps, f=self._log_step)
        periodic_eval = PeriodicFunction(period=self.args.eval_steps, f=self._eval_loop)
        total_loss = torch.zeros(1, device=self.device)
        self.state = self._init_state()
        self._dispatch_event("on_train_begin")
        if self.state.is_world_process_zero:
            print(f"Dataloader len={len(self.train_dataloader)}")
        # Epoch loop
        while True:
            self._dispatch_event("on_epoch_begin")
            # Batch within epoch loop
            for batch in self.train_dataloader:
                self._dispatch_event("on_step_begin")
                loss = self._train_step(self._prepare_batch(batch))
                total_loss = self._accumulate_loss(total_loss, loss)
                self.state.global_step += 1
                
                # Compute epoch as continous value from steps
                self.state.epoch = float(self.state.global_step) / float(self.epoch_train_steps)
                
                # Log every 'log_steps'
                periodic_log.step(total_loss, periodic_log.count())
                
                # Eval every 'eval_steps'
                periodic_eval.step()
                
                # Stop, if requested by callback.
                control = self._dispatch_event("on_step_end")
                if control is not None and control.should_training_stop:
                    break
                
                # Break both loops when we reach the target global steps
                if self.state.global_step >= self.max_steps:
                    break
                    
            else: # Continue, if loop exits normally
                control = self._dispatch_event("on_epoch_end")
                if control is not None and control.should_epoch_stop:
                    break
                continue
            break # Break, if inner-loop breaks
        # Training complete
        self._log_step(total_loss, periodic_log.count())
        self.train_time = datetime.timedelta(seconds=time.perf_counter() - start_time)
        self._dispatch_event("on_train_end")

    def _eval_loop(self) -> dict:
        """
        The inner evaluation loop
        """
        total_loss = torch.zeros(1, device=self.device)
        for step, batch in enumerate(self.eval_dataloader):
            loss, _, _ = self._prediction_step(self._prepare_batch(batch))
            total_loss = self._accumulate_loss(total_loss, loss)
            self._dispatch_event("on_prediction_step")
        metrics = { "eval_loss": (total_loss / step).item() }
        self._dispatch_event("on_evaluate", metrics=metrics)
        return metrics

    def _train_step(self, batch: Tensor) -> Tensor:
        """
        Perform a single training step
        """
        self.model.train()
        outputs = self.model(**batch)
        loss = outputs[0]
        self._backward(loss)
        self.optimizer.step()
        self._dispatch_event("on_optimizer_step")
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return loss.mean().detach()

    def _backward(self, loss):
        loss.backward()

    @torch.no_grad()
    def _prediction_step(self, batch: Tensor) -> Tensor:
        """
        Perform a single batch of predictions
        """
        self.model.eval()
        outputs = self.model(**batch)
        logits = outputs[1]
        loss = outputs[0]
        return loss.mean().detach(), logits.detach(), batch["labels"]

    def _log_step(self, total_loss: Tensor, log_steps: int):
        """
        Log training progress; called every 'log_steps' and once more at the end of training.
        Note: mean loss must be gathered if multi-process
        """
        if log_steps == 0:
            return
        mean_loss = (total_loss / log_steps).item()
        
        # Reset training loss and log step counter
        total_loss -= total_loss

        logs = {
            "epoch": self.state.epoch,
            "loss": mean_loss,
        }

        self.state.log_history.append(logs)
        
        self._dispatch_event(
            "on_log",
            logs=logs,
        )

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Performs any required steps to ready the data for the model to process.
        For example, normal torch training requires moving the batch to the target device.
        """
        return {k: v.to(self.device) for k, v in batch.items()}

    def _init_state(self) -> TrainerState:
        """
        Init training state
        """
        return TrainerState(
            max_steps=self.max_steps,
            logging_steps=self.args.log_steps,
            eval_steps=self.args.eval_steps,
            num_train_epochs=int(self.args.num_train_epochs),
            train_batch_size=self.train_dataloader.batch_size
        )

    def _accumulate_loss(self, total_loss: Tensor, loss: Tensor) -> Tensor:
        total_loss += loss
        return total_loss

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
                model=self.model,
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

class AccelTrainer(Trainer):
    """
    Modify the base Trainer to use the Accelerate library.
    """
    def __init__(self, accelerator_args={}, *args, **kwargs):
        self.accelerator_args = accelerator_args
        super().__init__(*args, **kwargs)
        self.first = True

    def _prepare(self) -> None:
        """
        Wrap relevant componens with accelerator
        """
        self.accelerator = Accelerator(**self.accelerator_args)
        self.train_dataloader, self.eval_dataloader, self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.train_dataloader,
            self.eval_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )
        # Accelerate modifies the dataloaders, which can change both the length and the batch size.
        self._update_training_steps()
        # Accel uses a special device target
        self.device = self.accelerator.device
        self.accelerator.wait_for_everyone()
        

    def _backward(self, loss):
        self.accelerator.backward(loss)

    def _accumulate_loss(self, total_loss: Tensor, loss: Tensor):
        """
        Reduces loss accross processes
        """
        total_loss += self.accelerator.reduce(loss, "mean")
        return total_loss

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.first and self.state.is_world_process_zero:
            print("Batch Shape:")
            print({k: v.shape for k, v in batch.items()})
            self.first = False
        return batch

    def _init_state(self) -> TrainerState:
        """
        Modifies parent state by setting process rank info
        """
        state = super()._init_state()
        state.is_local_process_zero = self.accelerator.is_local_main_process
        state.is_world_process_zero = self.accelerator.is_main_process
        state.num_processes = self.accelerator.num_processes
        return state
