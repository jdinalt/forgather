from typing import Any
import re
import bisect

import yaml

from .latent import (
    VarNode,
    CallableNode,
    SingletonNode,
)


def split_tag_idenity(tag_suffix):
    if len(tag_suffix) == 0:
        return None, None
    split_suffix = tag_suffix[1:].split("@", maxsplit=1)
    tag_value = split_suffix[0]
    tag_identity = split_suffix[1] if len(split_suffix) > 1 else None
    return tag_value, tag_identity


positional_re = re.compile(r"arg(\d+)")


def split_args(kwargs) -> tuple[tuple[Any], dict[Any]]:
    """
    Split out args and kwargs from kwargs, where "args" have keys matching match_positional
    and sort args. This allows encoding of both positonal and keyword args as a single dictionary,
    which makes it far more practical to extend or modify the args with a YAML config. e.g.

    kwargs = { "arg5": 5, "arg15": 15, "arg0": 0, "alpha": "a", "beta": "b" }
    split_args(kwargs)
    ((0, 5, 15), {'alpha': 'a', 'beta': 'b'})
    """
    sorted_args = []
    delete_keys = []

    for key, value in kwargs.items():
        match = positional_re.fullmatch(key)
        if match:
            bisect.insort(sorted_args, (int(match.group(1)), value))
            delete_keys.append(key)
    for k in delete_keys:
        del kwargs[k]
    args = (v for _, v in sorted_args)

    return args, kwargs


class CallableConstructor:
    def __init__(self, node_type: CallableNode):
        self.node_type = node_type

    def __call__(self, loader, tag_suffix, node):
        constructor, identity = split_tag_idenity(tag_suffix)

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
                args, kwargs = split_args(value)

        elif isinstance(node, yaml.SequenceNode):
            args = loader.construct_sequence(node, deep=True)
            kwargs = {}
        else:
            args = loader.construct_scalar(node)
            kwargs = {}

        assert isinstance(kwargs, dict), f"Expected dict, but found {type(kwargs)}"
        return self.node_type(constructor, *args, _identity=identity, **kwargs)


def var_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return VarNode(**loader.construct_mapping(node))
    elif isinstance(node, yaml.ScalarNode):
        return VarNode(loader.construct_scalar(node))
    else:
        raise TypeError(f"Var nodes may not be sequences. Found {node}")


def list_constructor(loader, tag_suffix, node):
    constructor, identity = split_tag_idenity(tag_suffix)
    if isinstance(node, yaml.SequenceNode):
        return SingletonNode(
            "named_list", loader.construct_sequence(node), _identity=identity
        )
    elif isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
        if not isinstance(value, str) or value != "":
            raise TypeError(f"list node sequence or empty. Found {type(value)}={value}")
        return SingletonNode("named_list", _identity=identity)
    else:
        raise TypeError(f"list nodes must be sequencess or empty. Found {node}")


def dlist_constructor(loader, tag_suffix, node):
    constructor, identity = split_tag_idenity(tag_suffix)
    if isinstance(node, yaml.MappingNode):
        sequence = list(
            filter(
                lambda item: not item is None, loader.construct_mapping(node).values()
            )
        )

        return SingletonNode(
            "named_list",
            sequence,
            _identity=identity,
        )
    elif isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
        if not isinstance(value, str) or value != "":
            raise TypeError(
                f"dlist node must be mapping or empty. Found {type(value)}={value}"
            )
        return SingletonNode("named_list", _identity=identity)
    else:
        raise TypeError(f"dlist nodes must be mappings. Found {node}")


def tuple_constructor(loader, tag_suffix, node):
    constructor, identity = split_tag_idenity(tag_suffix)
    if isinstance(node, yaml.SequenceNode):
        return SingletonNode(
            "named_tuple", loader.construct_sequence(node), _identity=identity
        )
    else:
        raise TypeError(f"tuple nodes must be sequencess. Found {node}")


def dict_constructor(loader, tag_suffix, node):
    constructor, identity = split_tag_idenity(tag_suffix)
    if isinstance(node, yaml.MappingNode):
        return SingletonNode(
            "named_dict", _identity=identity, **loader.construct_mapping(node)
        )
    elif isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
        if not isinstance(value, str) or value != "":
            raise TypeError(
                f"dict node scalar type must be ''. Found {type(value)}={value}"
            )
        return SingletonNode("named_dict", _identity=identity)
    else:
        raise TypeError(f"dict nodes must be mappings. Found {node}")


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
