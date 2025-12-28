import types
from contextlib import contextmanager
from enum import Enum
from pprint import pformat
from typing import Callable

import torch


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


def count_parameters(model) -> dict[str, str]:
    """
    Get model 'total' and 'trainable' parameters in a model as formatted strings

    The primary use-case is for displaying / logging this information.
    """
    total_parameters = sum(t.numel() for t in model.parameters())
    trainable_parameters = sum(
        t.numel() if t.requires_grad else 0 for t in model.parameters()
    )
    num_params = lambda x: f"{x/1000000:.2f}M"
    return dict(
        total=num_params(total_parameters), trainable=num_params(trainable_parameters)
    )


@contextmanager
def default_dtype(dtype: torch.dtype):
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(prev_dtype)
