from typing import (
    Any,
    Callable,
    List,
    Dict,
    Set,
    Iterator,
    Tuple,
)
from collections.abc import Sequence, Mapping
import os
import sys
from yaml import SafeLoader
from pathlib import Path

from jinja2 import Environment, meta

from pprint import pformat
from .latent import Latent
from .preprocess import PPEnvironment
from .yaml_utils import callable_constructor, load_depth_first, tuple_constructor
from .utils import format_line_numbers, add_exception_notes


class ConfigText(str):
    """
    A simple str sub-class, which can fromat the string with line-numbers
    """

    def with_line_numbers(self, show_line_numbers=True):
        return format_line_numbers(self) if show_line_numbers else self


class Config:
    """Congiguration Container w/ orginal pre-processed data"""

    def __init__(self, config, pp_config):
        self.config = config
        self.pp_config = pp_config

    def __repr__(self):
        return (
            f"{type(self).__name__}(config={self.config}, pp_config={self.pp_config})"
        )


def fconfig(obj, sort_items=True, indent_level=2, visited=set()):
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
        if id(obj) in visited:
            s = f"Latent *{id(obj)}"
        else:
            visited.add(id(obj))
            s = f"Latent &{id(obj)} '{obj.constructor}'"
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


# We will be adding custom YAML tags to the loader; create a new class,
# as we don't want these applied to all instances of SafeLoader.
class ConfigLoader(SafeLoader):
    pass


ConfigLoader.add_multi_constructor("!callable", callable_constructor)
ConfigLoader.add_constructor("!tuple", tuple_constructor)


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
        # if isinstance(searchpath, str) or isinstance(searchpath, os.PathLike):
        #    searchpath = [Path("."), searchpath]
        # else:
        #    searchpath.insert(0, Path("."))

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
            raise add_exception_notes(error, pp_config)
        if isinstance(loaded_config, dict):
            loaded_config = ConfigDict(loaded_config)
        return Config(loaded_config, pp_config)

    def find_referenced_templates(
        self, template_name: os.PathLike | str, level=0
    ) -> Iterator[Tuple[int, str, str]]:
        environment = self.pp_environment
        source, filename, _ = environment.loader.get_source(environment, template_name)
        yield (level, template_name, filename)
        ast = environment.parse(source, name=template_name, filename=filename)
        iter = meta.find_referenced_templates(ast)
        for template in iter:
            if template is None:
                continue
            for t in self.find_referenced_templates(template, level + 1):
                yield t


def load_config(config_path: str | os.PathLike, /, **kwargs) -> ConfigDict:
    """
    Just load a simple configuration and return it

    This is intended for loading a very basic configuration, like a meta-config.

    The search path is relative to the CWD and returns a ConfigDict.
    """
    environment = ConfigEnvironment()
    config = environment.load(config_path, **kwargs)
    return config.config
