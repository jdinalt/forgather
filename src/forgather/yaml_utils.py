import yaml
import pprint

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
                args = tuple()
                kwargs = value

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
    else:
        raise TypeError(f"list nodes must be sequencess. Found {node}")


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
