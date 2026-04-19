from enum import Enum
from functools import wraps
from typing import Any, Callable, Set, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


class ConversionDescriptor:
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
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}; choose one of {cls._value2member_map_.keys()}"
        )


def format_line_numbers(text: str) -> str:
    """Format a string with line-numbers"""
    return "".join(map(lambda x: f"{x[0]+1:>6}: {x[1]}\n", enumerate(text.split("\n"))))


def add_exception_notes(error: Exception, *args) -> Exception:
    """
    Add notes to an exception using the native Python 3.11+ add_note() API.
    """
    for arg in args:
        error.add_note(str(arg))
    return error


class AutoName:
    NAMES = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        i = self.i
        self.i += 1
        name = ""
        while True:
            next_name = self.NAMES[i % len(self.NAMES)]
            i = i // len(self.NAMES)
            if len(name):
                name = next_name + "_" + name
            else:
                name = next_name
            if i == 0:
                break
        return name + "_"


def track_depth(method: _F) -> _F:
    """Decorator to track the recursion depth of a class method."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Increment the recursion level
        self.level += 1
        try:
            # Call the actual method
            result = method(self, *args, **kwargs)
        finally:
            # Decrement the recursion level regardless of method success
            self.level -= 1
        return result

    return wrapper  # type: ignore[return-value]


def indent_block(block, indent_level=4):
    indent = " " * indent_level
    s = "".join(map(lambda s: indent + s + "\n", block.split("\n")))
    return s[:-1]
