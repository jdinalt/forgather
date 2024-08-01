from enum import Enum
from typing import Any, Set


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


def add_exception_notes(error: Exception, *args):
    """
    Add a note to an exception.

    Python 3.11 supports adding additional details to exceptions via notes.
    Unfortunately, our present target is Python 3.10...

    We will try to add a note to an exception anyway via adding it to something
    in the exception which will be printed with the exception.

    This is most definitely a hack, but it's the 'least-bad-thing,' as the line
    numbers reported by Yaml will likely correspond to a pre-processed Yaml file,
    making the line-numbers relatively useless, unless you have access to the
    preprocessed data, with line-numbers.

    Ideally, at some later point, when we upgrade to a newer version of Python, this
    should be replaced with the native method of adding notes.
    """
    note = ""
    for arg in args:
        note += "\n\n" + str(arg)

    # Some Yaml exceptions have a 'note' attribute, which will be printed with
    # the exception. If it has this, use it.
    if hasattr(error, "note"):
        if isinstance(error, str):
            error.note += note
        else:
            error.note = note
        return error
    # Try appending the note to the first str in the exception's args.
    else:
        # Not all exception have a string in the first arg. For example, "yaml.MarkedYAMLError"
        # We try to be generic here by find the first string in args, but it's impossible to
        # know if there could be side effects -- probably not, but committed to Python 3.10 at present,
        # so "add_not() is not an option, so do the least-bad-thing.
        error_args = list(error.args)
        for i, arg in enumerate(error_args):
            if isinstance(arg, str):
                error_args[i] += "\n" + note
                error.args = tuple(error_args)
                return error
    # Fallback to generic exception, which at least should chain them.
    raise Exception(note)


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


def track_depth(method):
    """Decorator to track the recursion depth of a class method."""

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

    return wrapper


def indent_block(block):
    indent = " " * indent_level
    s = "".join(map(lambda s: indent + s + "\n", block.split("\n")))
    return s[:-1]
