from typing import List, Any, Optional
import os

from .preprocess import PPEnvironment
from .latent import Latent, Undefined

DEFAULT_CODE_TEMPLATE = """
## Set default factory name, if not provided
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
    name_policy: str = None,
    **kwargs,
) -> Any:
    """
    Generate Python code from a Forgather graph

    When used as such, the node-type should be a MetaNode.
    ```yaml
    code: !metanode:aiws.construct:generate_code *model_def
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
    py_output = Latent.to_py(obj, name_policy=name_policy)

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