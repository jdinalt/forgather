import datetime
import sys
from pprint import pformat
from typing import Any, Literal

from forgather.ml.trainer.trainer_types import TrainerState
from forgather.ml.utils import alt_repr

Mapping = dict[str, Any]


def format_train_info(
    args,
    state,
    control,
    model,
    processing_class,
    optimizer,
    lr_scheduler,
    train_dataloader,
    eval_dataloader,
    **kwargs,
):
    """
    Given objects passed to TrainerCallback, generate nice representations for logging

    This returns two dictionaries, info and extra_info, for basic and verbose logging.
    """
    if hasattr(state, "num_processes"):
        total_train_batch_size = state.num_processes * state.train_batch_size
        total_train_samples = total_train_batch_size * state.max_steps
        total_examples = state.epoch_train_steps * total_train_batch_size
        total_train_batch_size = f"{total_train_batch_size:,}"
        total_train_samples = f"{total_train_samples:,}"
        total_examples = f"{total_examples:,}"
    else:
        # TODO: The HF Trainer does not pass these values. Is there a way to compute this
        # from the available information?
        total_train_batch_size = "Unavailable"
        total_train_samples = "Unavailable"
        total_examples = "Unavailable"

    total_parameters = sum(t.numel() for t in model.parameters())
    trainable_parameters = sum(
        t.numel() if t.requires_grad else 0 for t in model.parameters()
    )
    num_params = lambda x: f"{x/1000000:.1f}M"

    info = {
        "total_examples": f"{total_examples}",
        "total_train_samples": f"{total_examples}",
        "per_device_train_batch_size": f"{args.per_device_train_batch_size:,}",
        "actual_per_device_batch_size": f"{state.train_batch_size:,}",
        "total_train_batch_size": f"{total_train_batch_size}",
        "max_steps": f"{state.max_steps:,}",
        "total_parameters": f"{num_params(total_parameters)}",
        "trainable_parameters": f"{num_params(trainable_parameters)}",
        "max_steps": f"{state.max_steps:,}",
    }

    extra_info = {
        "args": pformat(args),
        "state": pformat(state),
        "processing_class": pformat(processing_class),
        "optimizer": alt_repr(optimizer),
        "lr_schedulerr": alt_repr(lr_scheduler),
        "train_dataloader": alt_repr(train_dataloader),
        "eval_dataloader": alt_repr(eval_dataloader),
        "model": str(model),
    }
    return info, extra_info


def format_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp:<22}"


def format_log_header(state: TrainerState):
    s = f"{state.global_step:>10,d}  {round(state.epoch, 2):<5.3}"
    return s


def format_train_log(state: TrainerState, mapping: Mapping):
    header = format_log_header(state)
    if "loss" in mapping and "learning_rate" in mapping and "grad_norm" in mapping:
        return f"{header} train-loss: {round(mapping['loss'], 5):<10}grad-norm: {round(mapping['grad_norm'], 5):<10}learning-rate: {mapping['learning_rate']:1.2e}"
    # Fallback to generic formatting
    else:
        return header + format_mapping(mapping)


def format_eval_log(state, mapping: Mapping):
    header = format_log_header(state)
    if "eval_loss" in mapping:
        return f"{header} eval-loss:  {round(mapping['eval_loss'], 5):<10}"
    else:
        return header + format_mapping(mapping)


def format_mapping(mapping: Mapping):
    """
    Format a mapping for pretty-printing

    This is intended for formatting the mappings returned by format_train_info() as strings
    for console logging, but may be useful for formatting other datatypes as well.
    """
    s = ""
    for key, value in mapping.items():
        if isinstance(value, int):
            value = f"{value:,}"
        elif isinstance(value, float):
            value = f"{value:.4}"
        elif not isinstance(value, str):
            value = pformat(value)
        if len(value) > 80:
            s += f"{key}:\n{value}\n\n"
        else:
            s += f"{key}: {value}\n"
    return s


EvnType = Literal["file", "tty", "notebook"]


def get_env_type() -> EvnType:
    """
    Determine if output environment is a notebook, a TTY, or file/pipe
    """
    # Check if we are even in an IPython environment
    ipython = sys.modules.get("IPython")
    if ipython:
        try:
            shell = ipython.get_ipython()
            # Check for the Kernel config as TQDM does
            if shell and "IPKernelApp" in shell.config:
                return "notebook"
        except (AttributeError, NameError):
            pass

    # Check if we are outputting to a real terminal
    if sys.stdout.isatty():
        return "tty"

    # Default to file/redirection
    return "file"
