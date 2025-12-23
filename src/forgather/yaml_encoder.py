from typing import Any

from .graph_encoder import GraphEncoder
from .latent import (
    CallableNode,
    FactoryNode,
    LambdaNode,
    Latent,
    MetaNode,
    Node,
    SingletonNode,
    Undefined,
    VarNode,
)
from .utils import track_depth


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
            return f"!var {repr(obj.constructor)}"
        else:
            return (
                f"!var {{ name: {repr(obj.constructor)}, default: {repr(obj.value)} }}"
            )

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
            case "named_list":
                return s + self._named_list(obj)
            case "named_dict":
                return s + self._named_dict(obj)
            case "named_tuple":
                return s + self._named_tuple(obj)

        if len(obj.args) + len(obj.kwargs):
            if len(obj.args) == 0:
                s += self._dict(obj.kwargs)
            elif len(obj.kwargs) == 0:
                s += self._list(obj.args)
            else:
                s += self._args(obj)
        return s

    def _named_list(self, obj):
        return self._list(obj.args[0] if obj.args else list())

    def _named_tuple(self, obj):
        return self._tuple(obj.args[0] if obj.args else list())

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


def to_yaml(obj):
    """
    Encode graph as YAML
    """
    encoder = YamlEncoder()
    return encoder(obj)
