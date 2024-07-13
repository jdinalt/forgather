from typing import (
    Any,
    Callable,
    List,
    Dict,
    Sequence,
    Set,
)
from collections.abc import Sequence, Mapping
import os
import sys
from yaml import SafeLoader
import platform
from pathlib import Path

from importlib.metadata import version
from jinja2 import Environment, meta

from pprint import pformat
from .latent import Latent
from .preprocess import PPEnvironment
from .yaml_utils import callable_constructor, load_depth_first, tuple_constructor


def format_line_numbers(text: str) -> str:
    """Format a string with line-numbers"""
    return "".join(map(lambda x: f"{x[0]+1:>6}: {x[1]}\n", enumerate(text.split("\n"))))


class ConfigText(str):
    """
    A simple str sub-class, which can fromat the string with line-numbers
    """

    def with_line_numbers(self, show_line_numbers=True):
        return format_line_numbers(self) if show_line_numbers else self


def fconfig(obj, sort_items=True, indent_level=2):
    """
    Recursively pretty-format a configuration

    TODO: Rewrite using reprlib
    """

    def indent_block(block):
        indent = " " * indent_level
        s = "".join(map(lambda s: indent + s + "\n", block.split("\n")))
        return s[:-1]

    if isinstance(obj, ConfigText):
        return obj.with_line_numbers()
    elif isinstance(obj, Config):
        return fconfig(
            dict(config=obj.config, pp_config=obj.pp_config), sort_items, indent_level
        )
    elif isinstance(obj, str):
        return f"'{obj}'"
    elif isinstance(obj, Set):
        return fconfig(tuple(obj), sort_items, indent_level)
    elif isinstance(obj, Mapping):
        s = ""
        items = obj.items()
        if sort_items:
            items = dict(sorted(items)).items()
        for key, value in items:
            fmt_value = fconfig(value, sort_items, indent_level)
            if "\n" in fmt_value or len(fmt_value) > 80:
                s += f"{key}:\n" + indent_block(fmt_value) + "\n"
            else:
                s += f"{key}: " + fmt_value + "\n"
        return s[:-1]
    elif isinstance(obj, Sequence):
        s = ""
        items = obj
        if sort_items:
            sortable = True
            for value in items:
                if not isinstance(value, str):
                    sortable = False
                    break
            if sortable:
                items = sorted(items)
        for value in items:
            s += "- " + fconfig(value, sort_items, indent_level) + "\n"
        return s[:-1]
    elif isinstance(obj, Latent):
        s = f"Latent '{obj.constructor}'"
        if len(obj.args):
            s += "\n" + indent_block(fconfig(obj.args, sort_items, indent_level))
        if len(obj.kwargs):
            s += "\n" + indent_block(fconfig(obj.kwargs, sort_items, indent_level))
        return s
    else:
        return pformat(obj)


def pconfig(obj, /, *args, **kwargs):
    """
    Print a config
    """
    print(fconfig(obj, *args, **kwargs))


def _add_exception_notes(error: Exception, *args):
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


def default_pp_globals():
    return dict(
        script_args="N/A",
        world_size=1,
        rank=0,
        local_rank=0,
        hostname=platform.node(),
        uname=platform.uname(),
        versions={"python": platform.python_version()}
        | {
            lib: version(lib)
            for lib in (
                "torch",
                "transformers",
                "accelerate",
            )
        },
    )


# We will be adding custom YAML tags to the loader; create a new class,
# as we don't want these applied to all instances of SafeLoader.
class ConfigLoader(SafeLoader):
    pass


ConfigLoader.add_multi_constructor("!callable", callable_constructor)
ConfigLoader.add_constructor("!tuple", tuple_constructor)


class Config:
    """Congiguration Container w/ orginal pre-processed data"""

    def __init__(self, config, pp_config):
        self.config = config
        self.pp_config = pp_config

    def __repr__(self):
        return (
            f"{type(self).__name__}(config={self.config}, pp_config={self.pp_config})"
        )

    def materialize(self, **kwargs):
        """
        Materialize the configuration

        This is essentially just a wrapper for Latent.materialize(), but we feed the
        preprocessed config into the configuration for logging and run this in a
        try block, which will dump the pre-processed context on exception.
        """
        try:
            kwargs["pp_config"] = lambda: config_out.pp_config
            materialized_config = Latent.materialize(self.config, **kwargs)
            if isinstance(materialized_config, dict):
                materialized_config = ConfigDict(materialized_config)
            return materialized_config
        except Exception as error:
            raise _add_exception_notes(error, self.pp_config)


class ConfigDict(dict):
    """
    A simple dictionary wrapper for a configuration

    This filters out ".key" dictionary keys and exposes the keys
    as properties to make it cleaner to access the keys.
    """

    def __getattr__(self, name):
        return self[name]

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        iter = filter(lambda item: not item[0].startswith("."), dct.items())
        for key, value in iter:
            self[key] = value


class ConfigEnvironment:
    """
    Contains the configuration envrionment
    """

    def __init__(
        self,
        searchpath: List[str | os.PathLike] | str | os.PathLike = Path("."),
        pp_environment: Environment = None,
        globals: Dict[str, Any] = {},
    ):
        # Implicitly add CWD to search path
        if isinstance(searchpath, str) or isinstance(searchpath, os.PathLike):
            searchpath = [Path("."), searchpath]
        else:
            searchpath.insert(0, Path("."))

        if pp_environment is None:
            pp_environment = PPEnvironment(searchpath=searchpath)
        self.pp_environment = pp_environment
        self.pp_environment.globals |= globals

    def preprocess(
        self,
        config_path: os.PathLike | str,
        /,
        **kwargs,
    ) -> ConfigText:
        """
        Preprocess a configuration file and return it

        returns: ConfigText, a 'str' sub-type with a 'with_line_numbers()' method.
        """
        template = self.pp_environment.get_template(config_path)
        return ConfigText(template.render(**kwargs))

    def preprocess_from_string(
        self,
        config: str,
        /,
        **kwargs,
    ) -> ConfigText:
        """
        Preprocess a configuration in a string and return it.
        """
        template = self.pp_environment.from_string(config)
        return ConfigText(template.render(**kwargs))

    def load(
        self,
        config_path: os.PathLike | str,
        /,
        **kwargs,
    ) -> Config:
        """
        Preprocess and parse a configuration file
        """
        pp_config = self.preprocess(config_path, **kwargs)
        return self.load_from_ppstring(pp_config)

    def load_from_string(
        self,
        config: str,
        /,
        **kwargs,
    ) -> Config:
        """
        Preprocess and load a configuration from a string
        """
        pp_config = self.preprocess_from_string(config, **kwargs)
        return self.load_from_ppstring(pp_config)

    def load_from_ppstring(self, pp_config: str) -> Config:
        """
        Load a configuration from a preprocessed string.
        """
        try:
            loaded_config = load_depth_first(pp_config, Loader=ConfigLoader)
        except Exception as error:
            raise _add_exception_notes(error, pp_config)
        if isinstance(loaded_config, dict):
            loaded_config = ConfigDict(loaded_config)
        return Config(loaded_config, pp_config)


def load_config(config_path: str | os.PathLike) -> ConfigDict:
    """
    Just load a simple configuration and return it

    This is intended for loading a very basic configuration, like a meta-config.

    The search path is relative to the CWD and returns a ConfigDict.
    """
    environment = ConfigEnvironment()
    config = environment.load(config_path)
    return config.config
