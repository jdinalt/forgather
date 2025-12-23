import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from enum import Enum
from types import NoneType
from typing import Any

from .latent import CallableNode, Latent, MetaNode, VarNode, prune_node_type
from .utils import AutoName, track_depth


class NamePolicy(Enum):
    # Only generate names for nodes which occur more than once in the graph.
    REQUIRED = (0,)
    # As above, but also include all nodes given explicit names.
    NAMED = (1,)
    # Generate names for all nodes.
    ALL = (2,)


class NodeNameDict(dict):
    """
    Generates a human-readable map of node.identity -> str
    """

    def __init__(self, obj, name_policy: NamePolicy = None, prune_meta=False):
        # If str, convert to enum
        if isinstance(name_policy, str):
            match name_policy:
                case "required":
                    name_policy = NamePolicy.REQUIRED
                case "named":
                    name_policy = NamePolicy.NAMED
                case "all":
                    name_policy = NamePolicy.ALL
                case _:
                    raise ValueError(
                        "name_policy must be one of: 'required', 'named', 'all'"
                    )

        # Default
        if name_policy is None:
            name_policy = NamePolicy.NAMED

        # Count the occurances of each node in the graph
        idcount = defaultdict(int)

        for level, node, sub_nodes in Latent.walk(obj):
            if prune_meta:
                prune_node_type(sub_nodes, MetaNode)
            if not isinstance(node, CallableNode):
                continue
            idcount[node.identity] += 1
            if idcount[node.identity] > 1:
                # If we have visited this node more than once, skip all children.
                sub_nodes.clear()

        match name_policy:
            case NamePolicy.REQUIRED:
                filter_fn = lambda t: t[1] > 1
            case NamePolicy.NAMED:
                filter_fn = lambda t: isinstance(t[0], str) or t[1] > 1
            case NamePolicy.ALL:
                filter_fn = lambda t: True
            case _:
                raise TypeError(
                    f"Unknown naming policy encountered [{type(name_policy)}]={name_policy}"
                )

        # Create a mapping of id -> human-readable-name for all items with a count > 1
        name_gen = iter(AutoName())
        for key, _ in filter(filter_fn, idcount.items()):
            # Use explicit name, if specified
            if isinstance(key, str):
                self[key] = key
            else:
                self[key] = next(name_gen)


class GraphEncoder(metaclass=ABCMeta):
    """
    An abstract class for converting a node graph into... something

    For example, serializing it or coverting it to another language.
    """

    # Deliberately 'init' not '__init__' the object state is initialized
    # each time it is called.
    def init(self, obj: Any, name_policy, prune_meta=False):
        self.level = -1
        self.name_map = NodeNameDict(
            obj, name_policy=name_policy, prune_meta=prune_meta
        )
        self.defined_ids = set()
        self.split_text_re = re.compile(r"(\n)")

    """
    Given a node-graph, return 'something'
    """

    @abstractmethod
    def __call__(self, obj: Any, name_policy=None) -> Any: ...

    def _indent(self, offset: int = 0) -> str:
        """
        Generate an indent for the current level.
        """
        return " " * 4 * (self.level + offset)

    def split_text(self, txt):
        """
        Split text on newlines, retaining the newlines
        """
        chunk = ""
        if not len(txt):
            yield chunk
            return
        for i in self.split_text_re.split(txt):
            chunk += i
            if i == "\n":
                yield chunk
                chunk = ""
        if len(chunk):
            yield chunk

    def _encode(self, obj: Any) -> str:
        """
        Main type dispatcher

        This may be recursive.
        """
        if isinstance(obj, str):
            return self._str(obj)

        elif isinstance(obj, int):
            return self._int(obj)

        elif isinstance(obj, float):
            return self._float(obj)

        elif isinstance(obj, bool):
            return self._bool(obj)

        elif obj is None:
            return self._none()

        elif isinstance(obj, list):
            return self._list(obj)

        elif isinstance(obj, dict):
            return self._dict(obj)

        elif isinstance(obj, tuple):
            return self._tuple(obj)

        elif isinstance(obj, VarNode):
            return self._var(obj)

        elif isinstance(obj, CallableNode):
            return self._callable(obj)

    """
    Enocde the specified type

    Note: Use the track_depth() method decorator to automatically increment
    and decrement the present level in the graph.
    """

    @abstractmethod
    def _list(self, obj: list): ...

    @abstractmethod
    def _dict(self, obj: dict): ...

    @abstractmethod
    def _tuple(self, obj: tuple): ...

    @abstractmethod
    def _var(self, obj: VarNode): ...

    @abstractmethod
    def _callable(self, obj: CallableNode): ...

    @abstractmethod
    def _none(self): ...

    @abstractmethod
    def _str(self, obj: str): ...

    @abstractmethod
    def _int(self, obj: int): ...

    @abstractmethod
    def _float(self, obj: float): ...

    @abstractmethod
    def _bool(self, obj: bool): ...
