import types
from typing import Callable
from pprint import pformat
from enum import Enum


def alt_repr(obj):
    """
    Alternative __repr__ implementation for objects which have
    not implemented it for their class.

    Some object are relatively opaque. This should expose more info.
    """
    # If __repr__ is a wrapper, they have not implemented it.
    if isinstance(obj.__repr__, types.MethodWrapperType):
        attrs = {}
        for key in dir(obj):
            # Ignore protected, private, etc.
            if key.startswith("_"):
                continue
            value = getattr(obj, key)

            # Ignore methods and other callables.
            if isinstance(value, Callable):
                continue

            attrs[key] = value
        return pformat(attrs)
    else:
        return repr(obj)


def format_train_info(
    args,
    state,
    control,
    model,
    tokenizer,
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
        "model": str(model),
    }

    extra_info = {
        "args": pformat(args),
        "state": pformat(state),
        "tokenizer": pformat(tokenizer),
        "optimizer": alt_repr(optimizer),
        "lr_schedulerr": alt_repr(lr_scheduler),
        "train_dataloader": alt_repr(train_dataloader),
        "eval_dataloader": alt_repr(eval_dataloader),
    }
    return info, extra_info


def format_mapping(mapping):
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


class ConversionDescriptor:
    """
    A descriptor for automatically converting types.

    Example:
    ```
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    @dataclass
    class Data:
        color: ConversionDescriptor = ConversionDescriptor(Color, default=Color.RED)

    data = Data(color="blue")
    print(data)
    > Data(color=<Color.BLUE: 'blue'>)
    ```
    """

    def __init__(self, cls, *, default):
        self._cls = cls
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default
        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        setattr(obj, self._name, self._cls(value))


class DiagnosticEnum(Enum):
    """
    An extension to Enum which provides better diagnostic info when an invalid value is set.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}; choose one of {cls._value2member_map_.keys()}"
        )
