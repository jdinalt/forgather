from typing import (
    Any,
    Callable,
    List,
    Dict,
    Set,
    Iterator,
    Iterable,
    Tuple,
)
from collections.abc import Sequence, Mapping
import os
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pformat

from yaml import SafeLoader
from jinja2 import Environment, meta
from platformdirs import user_config_dir


from .latent import (
    Latent,
    Node,
    CallableNode,
    VarNode,
    FactoryNode,
    SingletonNode,
    LambdaNode,
    MetaNode,
)

from .preprocess import PPEnvironment
from .yaml_utils import (
    CallableConstructor,
    load_depth_first,
    tuple_constructor,
    list_constructor,
    var_constructor,
    dict_constructor,
)

from .utils import (
    format_line_numbers,
    add_exception_notes,
    AutoName,
    track_depth,
    indent_block,
)


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

    def get(self):
        """
        Returns the config as a tuple

        (self.config, self.pp_config)
        """
        return self.config, self.pp_config

    def __repr__(self):
        return (
            f"{type(self).__name__}(config={self.config}, pp_config={self.pp_config})"
        )


def fconfig(obj, sort_items=True, indent_level=2, visited=None):
    """
    Recursively pretty-format a configuration

    TODO: Rewrite using reprlib
    """
    if visited is None:
        visited = set()

    def indent_block(block):
        indent = " " * indent_level
        s = "".join(map(lambda s: indent + s + "\n", block.split("\n")))
        return s[:-1]

    if isinstance(obj, ConfigText):
        return obj.with_line_numbers()
    elif isinstance(obj, Config):
        return fconfig(
            dict(config=obj.config, pp_config=obj.pp_config),
            sort_items,
            indent_level,
            visited,
        )
    elif isinstance(obj, str):
        return f"'{obj}'"
    elif isinstance(obj, Set):
        return fconfig(tuple(obj), sort_items, indent_level, visited)
    elif isinstance(obj, Mapping):
        s = ""
        items = obj.items()
        if sort_items:
            items = dict(sorted(items)).items()
        for key, value in items:
            fmt_value = fconfig(value, sort_items, indent_level, visited)
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
            s += "- " + fconfig(value, sort_items, indent_level, visited) + "\n"
        return s[:-1]
    elif isinstance(obj, Node):
        s = ""
        if isinstance(obj, VarNode):
            return f"var {obj.constructor}={obj.value}\n"
        elif isinstance(obj, SingletonNode):
            s += "singleton "
        elif isinstance(obj, LambdaNode):
            s += "lambda "
        elif isinstance(obj, CallableNode):
            s += "callable "
        else:
            s += "node "
        s += f"{repr(obj.identity)} {obj.constructor}"
        if obj.identity in visited:
            s += "elided ..."
            return s

        visited.add(obj.identity)
        if isinstance(obj, CallableNode):
            if len(obj.submodule_searchpath):
                s += f" searchpath={obj.submodule_searchpath}"
            if len(obj.args):
                s += "\n" + indent_block(
                    fconfig(obj.args, sort_items, indent_level, visited)
                )
            if len(obj.kwargs):
                s += "\n" + indent_block(
                    fconfig(obj.kwargs, sort_items, indent_level, visited)
                )
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


ConfigLoader.add_multi_constructor("!factory", CallableConstructor(FactoryNode))
ConfigLoader.add_multi_constructor("!singleton", CallableConstructor(SingletonNode))
ConfigLoader.add_multi_constructor("!lambda", CallableConstructor(LambdaNode))
ConfigLoader.add_multi_constructor("!meta", CallableConstructor(MetaNode))
ConfigLoader.add_constructor("!var", var_constructor)
ConfigLoader.add_multi_constructor("!tuple", tuple_constructor)
ConfigLoader.add_multi_constructor("!list", list_constructor)
ConfigLoader.add_multi_constructor("!dict", dict_constructor)


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
        searchpath: Iterable[str | os.PathLike] | str | os.PathLike = tuple("."),
        pp_environment: Environment = None,
        global_vars: Dict[str, Any] = None,
    ):
        if global_vars is None:
            global_vars = {}
        # Convert search path to tuple, if str or os.PathLike
        if isinstance(searchpath, os.PathLike) or isinstance(searchpath, str):
            searchpath = [searchpath]
        assert isinstance(searchpath, Iterable), "searchpath must be Iterable"

        # Remove non-existent directories from searchpath
        searchpath = list(filter(lambda path: os.path.isdir(path), searchpath))

        if pp_environment is None:
            pp_environment = PPEnvironment(searchpath=searchpath)
        self.pp_environment = pp_environment
        self.pp_environment.globals |= global_vars

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
            Latent.check(loaded_config)
        except Exception as error:
            raise add_exception_notes(error, format_line_numbers(pp_config))
        if isinstance(loaded_config, dict):
            loaded_config = ConfigDict(loaded_config)
        return Config(loaded_config, pp_config)

    def find_referenced_templates(
        self,
        template_name: os.PathLike | str,
    ) -> Iterator[Tuple[int, str, str]]:
        """
        Iterate over the template hierarchy

        We try to yield templates from root to leaf, while skipping ones
        we have already visited.
        """
        environment = self.pp_environment
        queue = [(template_name, 0)]
        visited = set(template_name)

        while len(queue):
            template_name, level = queue.pop(-1)
            source, filename, _ = environment.loader.get_source(
                environment, template_name
            )

            yield (level, template_name, filename)

            ast = environment.parse(source, name=template_name, filename=filename)

            # Filter out 'None' items, then sort items with those ending in '.yaml' last.
            # As we draw in LIFO order, this ensures that all 'file' templates are traversed
            # before the sub-templates defined in files.
            # Without doing so, it's possible to try to load a sub-template, before the file
            # which defines it has been loaded.
            iterator = sorted(
                filter(lambda x: x is not None, meta.find_referenced_templates(ast)),
                key=lambda a: 1 if a.endswith(".yaml") else -1,
            )

            for t in iterator:
                if t not in visited:
                    queue.append((t, level + 1))
                    visited.add(t)
