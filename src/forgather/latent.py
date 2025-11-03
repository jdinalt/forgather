from typing import Any, Callable, Tuple, Optional, Hashable, List, Final
from itertools import chain
import sys
import os
from abc import ABCMeta
from functools import partial
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from .dynamic import dynamic_import, get_builtin
from .utils import track_depth, add_exception_notes


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
        getitem=lambda x, y: x[y],
        named_list=list,
        named_tuple=tuple,
        named_dict=dict,
        call=lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )

    def __init__(
        self,
        constructor: str,
        *args,
        _identity: Hashable = None,
        submodule_searchpath: Optional[List[str | os.PathLike]] = None,
        latent_args: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(constructor, _identity=_identity)
        self.args = args
        self.kwargs = kwargs
        self.latent_args = latent_args
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance = None


class MetaNode(SingletonNode):
    """
    When a MetaNode is encountered in a graph, the nodes arguments are not constructed
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


class Materializer:
    def __call__(self, obj, /, *args, **kwargs):
        self.level = -1

        mtargets = kwargs.pop("mtargets", None)
        self.context_vars = kwargs.pop("context_vars", {})
        # if len(args):
        #    print(f"args={args}, kwargs={kwargs}")
        # Merge args with kwargs
        # for i, arg in enumerate(args):
        #    key = "arg" + str(i)
        #    context_vars[key] = arg

        self.kwargs = kwargs
        self.args = args

        if mtargets is not None:
            if isinstance(mtargets, str):
                objects = self._selective_materialize(obj, set((mtargets,)))
                return objects[mtargets]
            else:
                return self._selective_materialize(obj, set(mtargets))
        else:
            return self._materialize(obj)

    @track_depth
    def _selective_materialize(self, obj: Any, mtargets: set[str]):
        if not isinstance(obj, dict):
            raise TypeError(
                f"Root node is not a dictionary ({type(obj)}) and mtargets was specified."
            )
        iterator = filter(lambda x: x[0] in mtargets, obj.items())
        return {key: self._materialize(value) for key, value in iterator}

    @track_depth
    def _materialize(self, obj):
        if isinstance(obj, list):
            return [self._materialize(value) for value in obj]

        elif isinstance(obj, dict):
            return {key: self._materialize(value) for key, value in obj.items()}

        elif isinstance(obj, tuple):
            return tuple(self._materialize(value) for value in obj)

        elif isinstance(obj, VarNode):
            value = self.context_vars.get(obj.constructor, obj.value)
            if value is Undefined:
                raise UnboundVarError(obj.constructor)
            return value

        elif isinstance(obj, CallableNode):
            try:
                logger.debug(str(obj))
                # If not the root-node, stop traversal and return callable.
                if self.level > 0 and isinstance(obj, LambdaNode):
                    return partial(obj, context_vars=self.context_vars)

                if isinstance(obj, SingletonNode):
                    # Have we already constructed this object?
                    if (value := obj.instance) is not None:
                        logger.debug(f"Found Singleton {str(obj)}")
                        return value

                if isinstance(obj, MetaNode):
                    value = obj.callable(*obj.args, **obj.kwargs)
                else:
                    args = self._materialize(obj.args)
                    kwargs = self._materialize(obj.kwargs)
                    if obj.latent_args is not None:
                        latent_args = self._materialize(obj.latent_args)
                        assert isinstance(latent_args, dict)
                        kwargs |= latent_args
                    if self.level == 0 and isinstance(obj, LambdaNode):
                        args = list(args)
                        args.extend(self.args)
                        kwargs |= self.kwargs
                    value = obj.callable(*args, **kwargs)

                if isinstance(obj, SingletonNode):
                    # Store object in map to return if called again.
                    logger.debug(f"Constructed new Singleton {str(obj)}")
                    obj.instance = value
                return value
            except Exception as e:
                raise add_exception_notes(
                    e,
                    f"Exception occured while constructing {obj.constructor}",
                )
        else:
            return obj


class DuplicateNameError(RuntimeError):
    pass


class Latent:
    """
    A namespace class for processing node graphs

    TODO: Depricate the name-space; it no longer servers its original purpose.
    """

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
    def check(obj: Any):
        """
        Check for issues in a node graph
        """
        names = set()
        visisted = set()

        for _, node, _ in Latent.walk(obj):
            if (
                isinstance(node, str)
                or isinstance(node, int)
                or isinstance(node, float)
                or isinstance(node, bool)
                or node is None
                or isinstance(node, list)
                or isinstance(node, dict)
                or isinstance(node, tuple)
            ):
                continue
            elif not isinstance(node, Node):
                raise TypeError(f"Found unsupported type in graph: {type(node)}")

            if id(node) in visisted:
                continue
            visisted.add(id(node))
            if node.identity in names:
                raise DuplicateNameError(
                    f"Multiple definitions found for node {node.identity}"
                )
            names.add(node.identity)


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
