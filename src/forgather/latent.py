from typing import Any, Callable, Tuple, Optional, Hashable, List, Final
from types import NoneType
from itertools import chain
from collections import defaultdict
from enum import Enum
import sys
import os
from abc import ABCMeta, abstractmethod
from functools import partial
import re

from loguru import logger

# Disable debug logging
# Enable with: logger.enable("forgather.latent")
logger.disable(__name__)


from .dynamic import dynamic_import, get_builtin

from .utils import (
    AutoName,
    track_depth,
    add_exception_notes,
)


class UndefinedType:
    def __repr__(self):
        return "Undefined"


Undefined: Final[UndefinedType] = UndefinedType()


class UnboundVarError(NameError):
    def __init__(self, name):
        super().__init__(f"Variable {repr(name)} is undefined.", name=name)


class Node(metaclass=ABCMeta):
    def __init__(
        self,
        constructor: str,
        _identity: Hashable = None,
    ):
        if not isinstance(constructor, str):
            raise ValueError(
                f"Constructor must be of type 'str', but found {type(constructor)}"
            )
        self.constructor = constructor
        if _identity is not None:
            self._identity = _identity

    def __repr__(self):
        return f"{type(self).__name__}({self._arg_repr()}{self._kwarg_repr()})"

    def _arg_repr(self):
        return repr(self.constructor)

    def _kwarg_repr(self):
        return f", identity={repr(self.identity)}"

    def __str__(self):
        """
        Simplified representation, without arg-recursion.
        """
        return f"{repr(self.identity)}[{type(self).__name__}]={self.constructor}"

    @property
    def identity(self):
        return getattr(self, "_identity", id(self))

    def __call__(self, *args, **kwargs):
        return Latent.materialize(self, *args, **kwargs)


class VarNode(Node):
    def __init__(
        self,
        name: str,
        _identity: Hashable = None,
        default: Any = Undefined,
    ):
        super().__init__(name, _identity=_identity)
        self.value = default

    def _kwarg_repr(self):
        return super()._kwarg_repr() + f", value={repr(self.value)}"


class CallableNode(Node, metaclass=ABCMeta):
    """
    Call a function

    Useful built-ins:
    https://docs.python.org/3/library/functions.html
    https://docs.python.org/3/library/operator.html
    """

    BUILT_INS = dict(
        items=lambda x: x.items(),
        keys=lambda x: x.keys(),
        values=lambda x: x.values(),
    )

    def __init__(
        self,
        constructor: str,
        *args,
        _identity: Hashable = None,
        submodule_searchpath: Optional[List[str | os.PathLike]] = None,
        **kwargs,
    ):
        super().__init__(constructor, _identity=_identity)
        self.args = args
        self.kwargs = kwargs
        if submodule_searchpath is not None:
            self._submodule_searchpath = submodule_searchpath

    def _arg_repr(self):
        return super()._arg_repr() + f", *{self.args}"

    def _kwarg_repr(self):
        if len(self.submodule_searchpath):
            submod_str = f", submodule_searchpath={self.submodule_searchpath}"
        else:
            submod_str = ""
        return super()._kwarg_repr() + f"{submod_str}, **{self.kwargs}"

    @property
    def submodule_searchpath(self):
        return getattr(self, "_submodule_searchpath", [])

    @property
    def callable(self):
        # The object has not yet been constructed. Verify that the
        # constructor has been resolved.
        callable = getattr(self, "_callable", None)
        if callable is not None:
            return callable

        logger.debug(f"Resolving Callable: {self.constructor}")
        if ":" in self.constructor:
            try:
                callable = dynamic_import(
                    self.constructor, searchpath=self.submodule_searchpath
                )
            except Exception as e:
                raise add_exception_notes(
                    e,
                    f"Exception occured while resolving symbol {repr(self.constructor)} in {self} ",
                )
        else:
            callable = get_builtin(self.constructor)
            if callable is None:
                callable = self.BUILT_INS.get(self.constructor, None)
            if callable is None:
                raise NameError(f"Built-in callable {self.constructor} was not found")

        if not isinstance(callable, Callable):
            raise TypeError(
                f"Imported constructor is not Callable: [{type(callable)}] {self.constructor}"
            )
        # Cache the resolved symbol
        setattr(self, "_callable", callable)
        return callable


# While these nodes types are a CallableNodes, without overrides,
# their type is used to determine how they are processed in a graph.
class FactoryNode(CallableNode):
    """
    A FactoryNode generates a new object instance for every occurance in a graph.
    """

    pass


class SingletonNode(CallableNode):
    """
    A SingletonNode only generates a single instance of an object. All other occurances
    are references to the same constructed object.
    """

    pass


class MetaNode(SingletonNode):
    """
    When a MetaNode is encountered in a graph, the nodes areguments are not constructed
    before calling the node.

    This allows for MetaNode to do things like modifying the graph, before continuing, or
    to convert the graph to an alternate representation (Python code or Yaml, for example).
    """

    pass


class LambdaNode(CallableNode):
    """
    A LambdaNode returns a Callable for constructing the downstream graph, rather than
    as a constructed object.

    This can be used construct a Callable object, for example as a factory argument to
    another object.

    Note: This is very similar to s FactoryNode, with the difference being that s
    FactoryNode is implicitly called, while a LambdaNode must be explicitly called.
    """

    pass


class Latent:
    @staticmethod
    def materialize(obj: Any, /, *args, **kwargs):
        materializer = Materializer()
        return materializer(obj, *args, **kwargs)

    @staticmethod
    def walk(
        node: Any, top_down: bool = True, level: int = 0
    ) -> tuple[int, Any, dict[int | str, Any]]:
        """
        Generator for walking the node-graph

        node: The root-node to walk

        top_down: if True, yields levels from lowest to highest and the map of
            sub-nodes may be edited to prune the graph
            if False, yields levels from highest to lowest; editing sub-nodes
            has no effect, as sub-nodes have already been walked.

        level: the returned level is relative to this value, with the root
            being the given level.

        yields (level, node, sub_nodes)
            level: The present recursion depth, relative to 'level' given for root
            node: The present node
            sub_nodes: A map of sub-nodes within the 'node'
                if a CallableNode, the args have 'int' keys and the kwargs have 'str' keys.
                When top_down is True, you can edit the map. For example, pruning nodes
                you don't wish to visit or by reordering the items.
                This is roughtly similar to pathlib.Path.walk()
        """
        if isinstance(node, list):
            generator = enumerate(node)
        elif isinstance(node, tuple):
            generator = enumerate(node)
        elif isinstance(node, dict):
            generator = node.items()
        elif isinstance(node, CallableNode):
            generator = chain(enumerate(node.args), node.kwargs.items())
        else:
            generator = []

        sub_nodes = dict(generator)
        if top_down:
            yield (level, node, sub_nodes)

        for sub_node in sub_nodes.values():
            yield from Latent.walk(sub_node, top_down, level + 1)

        if not top_down:
            yield (level, node, sub_nodes)

    @staticmethod
    def to_yaml(obj):
        encoder = YamlEncoder()
        return encoder(obj)

    @staticmethod
    def to_py(obj, name_policy=None):
        if name_policy is None:
            policy = NamePolicy.NAMED
        else:
            match name_policy:
                case "required":
                    policy = NamePolicy.REQUIRED
                case "named":
                    policy = NamePolicy.NAMED
                case "all":
                    policy = NamePolicy.ALL
                case _:
                    raise ValueError(
                        "name_policy must be one of: 'required', 'named', 'all'"
                    )
        encoder = PyEncoder()
        return encoder(obj, name_policy=policy)


def prune_node_type(node_map, node_type):
    """
    A helper function to remove the nodes of a given type from a map

    The intended use-case is removing nodes of a type while using Latent.walk()
    """
    prune_list = []
    for key, sub_node in node_map.items():
        if isinstance(sub_node, node_type):
            prune_list.append(key)
    for key in prune_list:
        del node_map[key]


class Materializer:
    def __call__(self, obj, /, *args, **kwargs):
        self.level = -1
        self.idmap = dict()

        # Merge args with kwargs
        for i, arg in enumerate(args):
            key = "arg" + str(i)
            kwargs[key] = arg
        self.kwargs = kwargs
        return self._materialize(obj)

    @track_depth
    def _materialize(self, obj):
        if isinstance(obj, list):
            return [self._materialize(value) for value in obj]

        elif isinstance(obj, dict):
            return {key: self._materialize(value) for key, value in obj.items()}

        elif isinstance(obj, tuple):
            return tuple(self._materialize(value) for value in obj)

        elif isinstance(obj, VarNode):
            value = self.kwargs.get(obj.constructor, obj.value)
            if value is Undefined:
                raise UnboundVarError(obj.constructor)
            return value

        elif isinstance(obj, CallableNode):
            # If not the root-node, stop traversal and return callable.
            if self.level > 0 and isinstance(obj, LambdaNode):
                return partial(obj, **self.kwargs)

            if isinstance(obj, SingletonNode):
                # Have we already constructed this object?
                if (value := self.idmap.get(obj.identity, None)) is not None:
                    return value

            if isinstance(obj, MetaNode):
                value = obj.callable(*obj.args, **obj.kwargs)
            else:
                args = self._materialize(obj.args)
                kwargs = self._materialize(obj.kwargs)
                try:
                    value = obj.callable(*args, **kwargs)
                except Exception as e:
                    raise add_exception_notes(
                        e,
                        f"Exception occured while calling {obj.constructor}(*{repr(args)}, **{repr(kwargs)})",
                    )

            if isinstance(obj, SingletonNode):
                # Store object in map to return if called again.
                self.idmap[obj.identity] = value
            return value
        else:
            return obj


class NamePolicy(Enum):
    REQUIRED = (0,)
    NAMED = (1,)
    ALL = (2,)


class NodeNameDict(dict):
    def __init__(self, obj, name_policy: NamePolicy = None, prune_meta=False):
        if name_policy is None:
            name_policy = NamePolicy.NAMED

        idcount = defaultdict(int)
        visited = set()
        # Count the occurances of each id in the graph
        for level, node, sub_nodes in Latent.walk(obj):
            if prune_meta:
                prune_node_type(sub_nodes, MetaNode)
            if isinstance(node, CallableNode) and node.identity not in visited:
                visited.add(node.identity)
                idcount[node.identity] += 1

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
    # Deliberately 'init' not '__init__' the object state is initialized
    # each time it is called.
    def init(self, obj: Any, name_policy, prune_meta=False):
        self.level = -1
        self.name_map = NodeNameDict(
            obj, name_policy=name_policy, prune_meta=prune_meta
        )
        self.defined_ids = set()
        self.split_text_re = re.compile(r"(\n)")

    @abstractmethod
    def __call__(self, obj: Any, name_policy=None) -> Any: ...

    def _indent(self, offset: int = 0) -> str:
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


class YamlEncoder(GraphEncoder):
    def __call__(self, obj: Any, name_policy=None) -> str:
        self.init(obj, name_policy)
        # It's not possible to use the '.define: ...' syntax when
        # the root-node is not a mapping
        if not isinstance(obj, dict):
            s = self._encode(obj)
        else:
            s = self._definitions(obj) + self._encode(obj)
        return s.strip()

    @track_depth
    def _definitions(self, obj):
        """
        Encode the definition of all nodes with names in the name_map

        At a minimum, these nodes occur in more than one place in the graph. Depending
        upon the naming-policy, this could also include all nodes with explicit names or
        even all nodes in the graph.

        We only define ids for callable nodes which have an idenity in the name_map,
            which has pruned MetaNodes, and which have not already been defined.
        """

        def node_filter(t):
            node = t[1]
            return (
                isinstance(node, CallableNode)
                and node.identity in self.name_map
                and node.identity not in self.defined_ids
            )

        s = ""
        # Walk nodes, bottom-up
        for _, node, _ in filter(node_filter, Latent.walk(obj, top_down=False)):
            name = self.name_map[node.identity]
            s += f".define: "
            s += self._encode(node) + "\n\n"
            self.defined_ids.add(node.identity)
        return s

    @track_depth
    def _list(self, obj: list):
        s = ""
        for value in obj:
            s += "\n" + self._indent() + "- " + self._encode(value)
        return s

    @track_depth
    def _dict(self, obj: dict):
        s = ""
        for key, value in obj.items():
            s += "\n" + self._indent() + str(key) + ": " + self._encode(value)
        return s

    @track_depth
    def _tuple(self, obj: tuple):
        s = "!tuple"
        for value in obj:
            s += "\n" + self._indent() + "- " + self._encode(value)
        return s

    @track_depth
    def _var(self, obj: VarNode):
        if obj.value is Undefined:
            return f"!var {obj.constructor}"
        else:
            return f"!var {{ name: {obj.constructor}, default: {repr(obj.value)} }}"

    def _callable(self, obj: CallableNode):
        if obj.identity in self.defined_ids:
            return self._identity(obj)
        elif obj.identity in self.name_map:
            self.defined_ids.add(obj.identity)
            return f"&{self.name_map[obj.identity]} " + self._callable_main(obj)
        else:
            return self._callable_main(obj)

    def _identity(self, obj):
        name = self.name_map[obj.identity]
        return f"*{name}"

    def _callable_main(self, obj):
        s = ""
        if isinstance(obj, MetaNode):
            s += "!meta"
        elif isinstance(obj, SingletonNode):
            s += "!singleton"
        elif isinstance(obj, LambdaNode):
            s += "!lambda"
        elif isinstance(obj, FactoryNode):
            s += "!factory"
        else:
            raise ValueError(
                f"Encountered unrepresentable node type in graph {type(obj)}"
            )

        s += ":" + obj.constructor

        if isinstance(obj.identity, str):
            s += f"@{obj.identity}"

        match obj.constructor:
            case "list":
                return s + self._named_list(obj)
            case "dict":
                return s + self._named_dict(obj)

        if len(obj.args) + len(obj.kwargs):
            if len(obj.args) == 0:
                s += self._dict(obj.kwargs)
            elif len(obj.kwargs) == 0:
                s += self._list(obj.args)
            else:
                s += self._args(obj)
        else:
            s += " []"
        return s

    def _named_list(self, obj):
        return self._list(obj.args)

    def _named_dict(self, obj):
        return self._dict(obj.kwargs)

    @track_depth
    def _args(self, obj):
        s = "\n"
        s += self._indent() + "args:"
        s += self._list(obj.args) + "\n"
        s += self._indent() + "kwargs:"
        s += self._dict(obj.kwargs)
        return s

    @track_depth
    def _none(self):
        return "null"

    def _textblock(self, obj):
        return "".join(map(lambda line: self._indent() + line, self.split_text(obj)))

    @track_depth
    def _str(self, obj: str):
        if "\n" in obj:
            s = "|\n"
            s += self._textblock(obj)
            return s
        else:
            return repr(obj)

    @track_depth
    def _int(self, obj: int):
        return repr(obj)

    @track_depth
    def _float(self, obj: float):
        return repr(obj)

    @track_depth
    def _bool(self, obj: bool):
        return repr(obj)


class PyEncoder(GraphEncoder):
    def init(self, obj: Any, name_policy):
        super().init(obj, name_policy, True)
        self.imports = set()
        self.dynamic_imports = set()
        self.vars = set()

    def __call__(self, obj: Any, name_policy=None) -> dict:
        self.init(obj, name_policy)
        self.level = 0
        definitions = self._encode_definitions(obj).strip()
        main_body = self._encode(obj).strip()
        return dict(
            definitions=definitions,
            main_body=main_body,
            # And convert these from sets to lists
            imports=list(self.imports),
            dynamic_imports=[
                (module, callable_name, list(searchpath))
                for module, callable_name, searchpath in self.dynamic_imports
            ],
            variables=list(self.vars),
        )

    def _encode_definitions(self, obj):
        """
        Encode the definition of all nodes with names in the name_map

        At a minimum, these nodes occur in more than one place in the graph. Depending
        upon the naming-policy, this could also include all nodes with explicit names or
        even all nodes in the graph.

        We only define ids for callable nodes which have an idenity in the name_map,
            which has pruned MetaNodes, and which have not already been defined.
        """

        def node_filter(t):
            node = t[1]
            return (
                isinstance(node, CallableNode)
                and node.identity in self.name_map
                and node.identity not in self.defined_ids
            )

        s = ""
        # Walk nodes, bottom-up
        for _, node, _ in filter(node_filter, Latent.walk(obj, top_down=False)):
            s += self.name_map[node.identity]
            s += " = "
            if isinstance(node, FactoryNode) or isinstance(node, LambdaNode):
                s += "lambda: "
            s += self._encode(node) + "\n\n"
            self.defined_ids.add(node.identity)
        return s

    @track_depth
    def _list(self, obj: list):
        s = "[\n"
        for value in obj:
            s += self._indent() + self._encode(value) + ",\n"
        s += self._indent(-1) + "]"
        return s

    @track_depth
    def _dict(self, obj: dict):
        s = "{\n"
        for key, value in obj.items():
            s += self._indent() + repr(key) + ": " + self._encode(value) + ",\n"
        s += self._indent(-1) + "}"
        return s

    @track_depth
    def _tuple(self, obj: tuple):
        s = "(\n"
        for value in obj:
            s += self._indent() + self._encode(value) + ",\n"
        s += self._indent(-1) + ")"
        return s

    def _var(self, obj: VarNode):
        # We include a bool for if the var is undefined, as this detection
        # is otherwise complicated in a Jinja template.
        self.vars.add((obj.constructor, obj.value != Undefined, obj.value))
        return obj.constructor

    def _callable(self, obj: CallableNode):
        if obj.identity in self.defined_ids:
            return self._identity(obj)
        else:
            return self._callable_main(obj)

    @track_depth
    def _identity(self, obj):
        name = self.name_map[obj.identity]
        if type(obj) is FactoryNode:
            return name + "()"
        else:
            return name

    @track_depth
    def _callable_main(self, obj):
        if isinstance(obj, MetaNode):
            return self._encode(obj.callable(*obj.args, **obj.kwargs))
        if ":" in obj.constructor:
            if obj.constructor == "operator:getitem":
                return self._getitem(obj)
            module, callable_name = obj.constructor.split(":")
            if module.endswith(".py"):
                self.dynamic_imports.add(
                    (module, callable_name, tuple(obj.submodule_searchpath))
                )
            else:
                self.imports.add((module, callable_name))
        else:
            # We have a handful of special cases built-ins.
            # These are mostly for readability.
            match obj.constructor:
                case "getattr":
                    return self._getattr(obj)
                case "values":
                    return self._values(obj)
                case "keys":
                    return self._keys(obj)
                case "items":
                    return self._items(obj)
                case "list":
                    return self._named_list(obj)
                case "dict":
                    return self._named_dict(obj)
                case _:
                    callable_name = obj.constructor

        s = ""
        if self.level > 1:
            if isinstance(obj, LambdaNode):
                s += "lambda: "
                # TODO: implement lambda, with arguments
                # This appears to be a difficult one, so deferring for now.
                # return self._encode_lambda(obj)
            elif type(obj) == FactoryNode and obj.identity in self.name_map:
                s += "lambda: "

        s += callable_name + "("
        if len(obj.kwargs) + len(obj.args) == 0:
            s += ")"
            return s
        s += "\n"
        for arg in obj.args:
            s += self._indent() + self._encode(arg) + ",\n"
        for key, value in obj.kwargs.items():
            s += self._indent() + str(key) + "=" + self._encode(value) + ",\n"
        s += self._indent(-1) + ")"
        return s

    def _getitem(self, obj):
        mapping = obj.args[0]
        key = obj.args[1]
        s = self._encode(mapping) + f"[{repr(key)}]"
        return s

    def _getattr(self, obj):
        o = obj.args[0]
        attribute = obj.args[1]
        assert isinstance(
            attribute, str
        ), f"Attribute is expected to be a string, found{type(attribute)}"
        s = self._encode(o) + f".{attribute}"
        return s

    def _values(self, obj):
        o = obj.args[0]
        s = self._encode(o) + f".values()"
        return s

    def _keys(self, obj):
        o = obj.args[0]
        s = self._encode(o) + f".keys()"
        return s

    def _encode_items(self, obj):
        o = obj.args[0]
        s = self._encode(o) + f".items()"
        return s

    def _named_list(self, obj):
        try:
            self.level -= 1
            s = self._list(obj.args)
        finally:
            self.level += 1
        return s

    def _named_dict(self, obj):
        try:
            self.level -= 1
            s = self._dict(obj.kwargs)
        finally:
            self.level += 1
        return s

    def _none(self):
        return "None"

    def _str(self, obj: str):
        if "\n" in obj:
            s = "(\n"
            s += self.encode_textblock(obj)
            s += "\n" + self._indent(-1) + ")"
            return s
        else:
            return repr(obj)

    def encode_textblock(self, obj):
        return "\n".join(
            map(lambda line: self._indent() + repr(line), self.split_text(obj))
        )

    def _int(self, obj: int):
        return repr(obj)

    def _float(self, obj: float):
        return repr(obj)

    def _bool(self, obj: bool):
        return repr(obj)
