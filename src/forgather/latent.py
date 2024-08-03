from typing import Any, Callable, Tuple, Optional, Hashable, List, Final
from types import NoneType
from itertools import chain
from collections import defaultdict
from enum import Enum
import sys
import os
from abc import ABCMeta

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
        identity: Hashable = None,
    ):
        if not isinstance(constructor, str):
            raise ValueError(
                f"Constructor must be of type 'str', but found {type(constructor)}"
            )
        self.constructor = constructor
        if identity is not None:
            self._identity = identity

    def __repr__(self):
        return f"{type(self).__name__}({self._arg_repr()}{self._kwarg_repr()})"

    def _arg_repr(self):
        return repr(self.constructor)

    def _kwarg_repr(self):
        return f", identity={repr(self.identity)}"

    @property
    def identity(self):
        return getattr(self, "_identity", id(self))

    def __call__(self, *args, **kwargs):
        return Latent.materialize(self, *args, **kwargs)


class VarNode(Node):
    def __init__(
        self,
        name: str,
        identity: Hashable = None,
        default: Any = Undefined,
    ):
        super().__init__(name, identity=identity)
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
        identity: Hashable = None,
        submodule_searchpath: Optional[List[str | os.PathLike]] = None,
        **kwargs,
    ):
        super().__init__(constructor, identity=identity)
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


class NodeNameDict(dict):
    def __init__(self, obj):
        idcount = defaultdict(int)
        # Count the occurances of each id in the graph
        for node in Latent.all_of_type(obj, CallableNode):
            idcount[node.identity] += 1

        # Create a mapping of id -> human-readable-name for all items with a count > 1
        self.name_map = dict()
        name_gen = iter(AutoName())
        for key, _ in filter(lambda t: t[1] > 1, idcount.items()):
            # Use explicit name, if specified
            if isinstance(key, str):
                self[key] = key
            else:
                self[key] = next(name_gen)


class Latent:
    @staticmethod
    def materialize(obj: Any, /, *args, **kwargs):
        materializer = Materializer()
        return materializer(obj, *args, **kwargs)

    @staticmethod
    def all_nodes(value, key=None, level=0) -> Tuple[int, int | str | NoneType, Any]:
        """
        Iterate over all objects in graph

        yields tuple(level: int, key: int|str|NoneType, value: Any)
        """
        yield (level, key, value)

        # Supported sequence types
        if isinstance(value, list) or isinstance(value, tuple):
            generator = enumerate(value)
        # Supported mapping types
        elif isinstance(value, dict):
            generator = value.items()
        elif isinstance(value, CallableNode):
            generator = chain(enumerate(value.args), value.kwargs.items())
        else:
            return
        for k, v in generator:
            yield from Latent.all_nodes(v, k, level + 1)

    @staticmethod
    def all_of_type(value: Any, obj_type: type) -> Any:
        for _, _, value in filter(
            lambda x: isinstance(x[2], obj_type), Latent.all_nodes(value)
        ):
            yield value

    @staticmethod
    def to_yaml(obj):
        dumper = YamlDumper()
        return dumper(obj)

    @staticmethod
    def to_py(obj):
        encoder = PyEncoder()
        return encoder(obj)


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
                return obj

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


class YamlDumper:
    def __call__(self, obj):
        self.level = -1
        self.name_map = NodeNameDict(obj)
        self.visited = set()
        return self._dump(obj).strip()

    def _indent(self, offset=0):
        return " " * 4 * (self.level + offset)

    def _dump(self, obj):
        if isinstance(obj, list):
            return self._list(obj)
        elif isinstance(obj, dict):
            return self._dict(obj)
        elif isinstance(obj, tuple):
            return self._tuple(obj)
        elif isinstance(obj, VarNode):
            return self._var(obj)
        elif isinstance(obj, CallableNode):
            if obj.identity in self.name_map:
                return self._node_identity(obj)
            else:
                return self._node(obj)
        else:
            return self._other(obj)

    @track_depth
    def _list(self, obj):
        s = ""
        for value in obj:
            s += "\n" + self._indent() + "- " + self._dump(value)
        return s

    @track_depth
    def _dict(self, obj):
        s = ""
        for key, value in obj.items():
            s += "\n" + self._indent() + str(key) + ": " + self._dump(value)
        return s

    @track_depth
    def _tuple(self, obj):
        s = "!tuple"
        for value in obj:
            s += "\n" + self._indent() + "- " + self._dump(value)
        return s

    def _var(self, obj):
        if obj.value is Undefined:
            return f"!var {obj.constructor}"
        else:
            return f"!var {{ name: {obj.constructor}, default: {repr(obj.value)} }}"

    def _node_identity(self, obj):
        name = self.name_map[obj.identity]
        if obj.identity in self.visited:
            return f"*{name}"
        else:
            self.visited.add(obj.identity)
            return f"&{name} " + self._node(obj)

    @track_depth
    def _node(self, obj):
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

        s += ":" + obj.constructor + "\n"
        if len(obj.args):
            s += self._indent() + "args:"
            s += self._list(obj.args)
        if len(obj.kwargs):
            s += self._indent() + "kwargs:"
            s += self._dict(obj.kwargs)
        return s

    @track_depth
    def _other(self, obj):
        if obj is None:
            return "null"
        elif (
            isinstance(obj, bool)
            or isinstance(obj, int)
            or isinstance(obj, float)
            or isinstance(obj, str)
        ):
            return repr(obj)
        raise TypeError(f"Found unexpected type in graph: {type(obj)}")


class PyEncoder:
    def __call__(self, obj):
        self.level = 0
        self.name_map = NodeNameDict(obj)
        self.defined_ids = set()
        self.imports = set()
        self.dynamic_imports = set()
        self.vars = set()
        s = self._encode(obj).strip()
        return dict(
            main_body=s,
            # And convert these from sets to lists
            imports=list(self.imports),
            dynamic_imports=[
                (module, callable_name, list(searchpath))
                for module, callable_name, searchpath in self.dynamic_imports
            ],
            variables=list(self.vars),
        )

    def _indent(self, offset=0):
        return " " * 4 * (self.level + offset)

    def _is_intrinsic(self, obj):
        module, callable_name = obj.constructor.split(":")
        return module == "forgather.construct" and callable_name in self.INTRINSICS

    def _encode(self, obj):
        """
        Dispatch by type
        """
        if isinstance(obj, list):
            return self._encode_list(obj)
        elif isinstance(obj, dict):
            return self._encode_dict(obj)
        elif isinstance(obj, tuple):
            return self._encode_tuple(obj)
        elif isinstance(obj, VarNode):
            return self._encode_var(obj)
        elif isinstance(obj, CallableNode):
            return self._encode_callable(obj)
        else:
            return self._encode_other(obj)

    @track_depth
    def _encode_list(self, obj):
        s = "[\n"
        for value in obj:
            s += self._indent() + self._encode(value) + ",\n"
        s += self._indent(-1) + "]"
        return s

    @track_depth
    def _encode_dict(self, obj):
        s = "{\n"
        for key, value in obj.items():
            s += self._indent() + repr(key) + ": " + self._encode(value) + ",\n"
        s += self._indent(-1) + "}"
        return s

    @track_depth
    def _encode_tuple(self, obj):
        s = "(\n"
        for value in obj:
            s += self._indent() + self._encode(value) + ",\n"
        s += self._indent(-1) + ")"
        return s

    def _encode_var(self, obj):
        # We include a bool for if the var is undefined, as this detection
        # is otherwise complicated in a Jinja template.
        self.vars.add((obj.constructor, obj.value != Undefined, obj.value))
        return obj.constructor

    def _encode_callable(self, obj):
        if obj.identity in self.name_map:
            return self._encode_identity(obj)
        else:
            return self._encode_callable_main(obj)

    @track_depth
    def _encode_identity(self, obj):
        name = self.name_map[obj.identity]
        if obj.identity in self.defined_ids:
            if type(obj) is FactoryNode:
                return name + "()"
            else:
                return name
        else:
            self.defined_ids.add(obj.identity)
            s = (
                "(\n"
                + self._indent()
                + name
                + " := "
                + self._encode_callable_main(obj)
                + "\n"
            )
            s += self._indent(-1) + ")"
            # When multiple instances of a FactoryNode exist, we define it as a lambda, so
            # it can be reused, but we need to call it here to instantiate the first instance.
            if type(obj) is FactoryNode:
                s += "()"
            return s

    @track_depth
    def _encode_callable_main(self, obj):
        if isinstance(obj, MetaNode):
            return self._encode(obj.callable(*obj.args, **obj.kwargs))
        if ":" in obj.constructor:
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
                case "getitem":
                    return self._encode_getitem(obj)
                case "getattr":
                    return self._encode_getattr(obj)
                case "values":
                    return self._encode_values(obj)
                case "keys":
                    return self._encode_keys(obj)
                case "items":
                    return self._encode_items(obj)
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

    def _encode_getitem(self, obj):
        mapping = obj.args[0]
        key = obj.args[1]
        s = self._encode(mapping) + f"[{repr(key)}]"
        return s

    def _encode_getattr(self, obj):
        o = obj.args[0]
        attribute = obj.args[1]
        assert isinstance(
            attribute, str
        ), f"Attribute is expected to be a string, found{type(attribute)}"
        s = self._encode(o) + f".{attribute}"
        return s

    def _encode_values(self, obj):
        o = obj.args[0]
        s = self._encode(o) + f".values()"
        return s

    def _encode_keys(self, obj):
        o = obj.args[0]
        s = self._encode(o) + f".keys()"
        return s

    def _encode_items(self, obj):
        o = obj.args[0]
        s = self._encode(o) + f".items()"
        return s

    @track_depth
    def _encode_other(self, obj):
        if (
            obj is None
            or isinstance(obj, bool)
            or isinstance(obj, int)
            or isinstance(obj, float)
            or isinstance(obj, str)
        ):
            return repr(obj)
        raise TypeError(f"Found unexpected type in graph: {type(obj)}")
