import yaml
import pprint

from loguru import logger

from .latent import Latent


def callable_constructor(loader, tag_suffix, node):
    """
    A Yaml constuctor for Latent objects -- which are all Callables
    """
    if isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node, deep=True)

        args = value.get("args", None)
        kwargs = value.get("kwargs", None)
        # Only args
        if len(value) == 1 and args is not None:
            kwargs = {}
        # Only kwargs
        elif len(value) == 1 and kwargs is not None:
            args = tuple()
        # Exactly args and kwargs
        elif len(value) == 2 and args is not None and kwargs is not None:
            pass
        # Everything else; use the value as shorthand for kwargs
        else:
            args = tuple()
            kwargs = value

    elif isinstance(node, yaml.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
        kwargs = {}
    else:
        args = loader.construct_scalar(node)
        kwargs = {}

    assert isinstance(kwargs, dict), f"Expected dict, but found {type(kwargs)}"
    tag_suffix = tag_suffix[1:]
    return Latent(tag_suffix, *args, **kwargs)


def key_constructor(loader, node):
    """
    A Yaml constuctor for Latent 'key' objects
    """
    assert isinstance(node, yaml.ScalarNode), "Key tags must be singleton nodes."
    value = loader.construct_scalar(node)
    assert isinstance(value, str), f"Keys must be string, found {type(value)}"
    assert ":" not in value, f"Keys may not include the character ':', found {value}"
    return Latent(value, identity=0)


def tuple_constructor(loader, node):
    assert isinstance(node, yaml.SequenceNode)
    value = loader.construct_sequence(node, deep=True)
    return tuple(value)


def load_depth_first(stream, Loader=yaml.SafeLoader):
    """
    Load yaml document "depth-first"

    "yaml.load()" instantiates objects breadth-first, which can result in missing objects when
    used with anchors and custom tags. This appears to be a bug in the PyYaml, but the library
    does allow forcing objects to be constructed "depth-first," which resolves the issue.

    This call can directly replace a call to yaml.load(), where this is an issue.
    """
    loader = Loader(stream)
    try:
        node = loader.get_single_node()
        if node is not None:
            data = loader.construct_object(node, deep=True)
        else:
            data = None
    finally:
        loader.dispose()
    return data
