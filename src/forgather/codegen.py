from typing import List, Any, Optional
import os

from .preprocess import PPEnvironment
from .graph_encoder import GraphEncoder, NamePolicy
from .latent import (
    Latent,
    Undefined,
    VarNode,
    CallableNode,
    FactoryNode,
    MetaNode,
    LambdaNode,
)
from .utils import track_depth


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

        # We used sorted, as sets do not have a deterministic order, which can result
        # in symantically equivalent, but reordered output, accross invocations. This
        # makes it difficult to compare the output files for actual differences.
        return dict(
            definitions=definitions,
            main_body=main_body,
            # And convert these from sets to lists
            imports=sorted(list(self.imports)),
            dynamic_imports=sorted(
                [
                    (module, callable_name, tuple(searchpath))
                    for module, callable_name, searchpath in self.dynamic_imports
                ]
            ),
            variables=sorted(list(self.vars)),
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
                empty_args = len(node.kwargs) + len(node.args) == 0
                if not empty_args:
                    s += "partial("
            s += self._encode(node) + "\n\n"
            self.defined_ids.add(node.identity)
        return s

    @track_depth
    def _list(self, obj: list):
        if len(obj) == 0:
            return "[]"
        else:
            s = "[\n"
            for value in obj:
                s += self._indent() + self._encode(value) + ",\n"
            s += self._indent(-1) + "]"
            return s

    @track_depth
    def _dict(self, obj: dict):
        if len(obj) == 0:
            return "{}"
        else:
            s = "{\n"
            for key, value in obj.items():
                s += self._indent() + repr(key) + ": " + self._encode(value) + ",\n"
            s += self._indent(-1) + "}"
            return s

    @track_depth
    def _tuple(self, obj: tuple):
        if len(obj) == 0:
            return "tuple()"
        else:
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
        is_dynamic = False
        if isinstance(obj, MetaNode):
            return self._encode(obj.callable(*obj.args, **obj.kwargs))
        if ":" in obj.constructor:
            if obj.constructor == "operator:getitem":
                return self._getitem(obj)
            elif obj.constructor == "operator:call":
                return self._call(obj)
            module, callable_name = obj.constructor.split(":")
            if module.endswith(".py"):
                is_dynamic = True
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
                case "call":
                    return self._call(obj)
                case "values":
                    return self._values(obj)
                case "keys":
                    return self._keys(obj)
                case "items":
                    return self._items(obj)
                case "getitem":
                    return self._getitem(obj)
                case "named_list":
                    return self._named_list(obj)
                case "named_dict":
                    return self._named_dict(obj)
                case "named_tuple":
                    return self._named_tuple(obj)
                case _:
                    callable_name = obj.constructor

        s = ""

        in_name_map = type(obj) == FactoryNode and obj.identity in self.name_map
        is_partial = in_name_map or isinstance(obj, LambdaNode)
        empty_args = len(obj.kwargs) + len(obj.args) == 0

        if self.level > 1 and is_partial and not empty_args:
            s += "partial("

        s += callable_name
        if is_dynamic:
            s += "()"
        if is_partial:
            if empty_args:
                return s
            s += ", "
        else:
            s += "("

        if empty_args:
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

    def _call(self, obj):
        s = self._encode(obj.args[0]) + "("
        if len(obj.kwargs) + len(obj.args) - 1 == 0:
            s += ")"
            return s
        s += "\n"
        for arg in obj.args[1:]:
            s += self._indent() + self._encode(arg) + ",\n"
        for key, value in obj.kwargs.items():
            s += self._indent() + str(key) + "=" + self._encode(value) + ",\n"
        s += self._indent(-1) + ")"
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

    def _getitem(self, obj):
        o = obj.args[0]
        key = obj.args[1]
        s = self._encode(o) + "[" + self._encode(key) + "]"
        return s

    def _encode_items(self, obj):
        o = obj.args[0]
        s = self._encode(o) + f".items()"
        return s

    def _named_list(self, obj):
        try:
            self.level -= 1
            s = self._list(obj.args[0])
        finally:
            self.level += 1
        return s

    def _named_tuple(self, obj):
        try:
            self.level -= 1
            s = self._tuple(obj.args[0])
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


DEFAULT_CODE_TEMPLATE = """
## Set default factory name, if not provided
from functools import partial
-- if factory_name is undefined:
    -- set factory_name = "construct"
-- endif
-- for module, name in imports:
from {{ module }} import {{ name }}
-- endfor
-- if dynamic_imports|length
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys

# Import a dynamic module.
def dynimport(module, name, searchpath):
    module_path = module
    module_name = os.path.basename(module).split(".")[0]
    module_spec = spec_from_file_location(
        module_name,
        module_path,
        submodule_search_locations=searchpath,
    )
    mod = module_from_spec(module_spec)
    sys.modules[module_name] = mod
    module_spec.loader.exec_module(mod)
    for symbol in name.split("."):
        mod = getattr(mod, symbol)
    return mod

    -- for module, name, searchpath in dynamic_imports:
{{ name.split('.')[-1] }} = lambda: dynimport("{{ module }}", "{{ name }}", {{ searchpath }})
    -- endfor
-- endif

def {{ factory_name }}(
-- for var, has_default, default in variables:
    {{ var }}{% if has_default %}={{ repr(default) }}{% endif %},
-- endfor
-- if relaxed_kwargs is defined:
    **kwargs
-- endif
):
    {{ definitions|indent(4) }}
    
    return {{ main_body|indent(4) }}
""".strip()


def generate_code(
    obj,
    template_name: Optional[str] = None,
    template_str: Optional[str] = None,
    searchpath: Optional[List[str | os.PathLike] | str | os.PathLike] = ".",
    env=None,  # jinja2 environment or compatible API
    name_policy: str | NamePolicy = None,
    **kwargs,
) -> Any:
    """
    Generate Python code from a Forgather graph

    When used as such, the node-type should be a MetaNode.
    ```yaml
    code: !metanode:forgather.ml.construct:generate_code *model_def
    ```

    Outside of this context, it can be used directly to help understand how a graph is being
    interpreted, by expressing it as executable Python code.

    ```python
    print(generate_code(config))
    ```

    obj: A Forgather graph
    template_name: The template name; this is interpreted by Environment's Loader.
        By default, this is a file-name within searchpath, but is specific to the Loader type.
        If not None, it overrides 'template_str'
    template_str: A string containing a code template.
        If both template_str and template_name are None, the default code template is used.
    searchpath: The search path to locate Jinja templates.
        Only applicable when using the default Jinja environment.
    name_policy: When to name variabes. [default='named']
        'required': Only when more than one instance exists
        'named': If given an explicit name or more than one instance exists
        'all': Given a name to all CallableNodes
    kwargs: Any remaining kwargs are passed to the template's 'render' method.
        That is, these arguments are visible within the template.

    returns: If return_value is not Undefined, return_value, else the generated code as a str.

    The default template accepts the following kwargs:

    factory_name: Optional[str]="construct", ; The name of the generated factory function.
    relaxed_kwargs: Optional[bool]=Undefined, ; if defined, **kwargs is added to the arg list
    """

    # Convert the input to Python code
    encoder = PyEncoder()
    py_output = encoder(obj, name_policy=name_policy)

    if env is None:
        if isinstance(searchpath, os.PathLike) or isinstance(searchpath, str):
            searchpath = [searchpath]

        # Removes paths which don't exist
        searchpath = list(filter(lambda path: os.path.isdir(path), searchpath))
        env = PPEnvironment(searchpath=searchpath)

    if template_name is None:
        if template_str is None:
            template = env.from_string(DEFAULT_CODE_TEMPLATE)
        else:
            template = template_str
    else:
        template = env.get_template(template_name)

    return template.render(**kwargs | py_output).strip()
