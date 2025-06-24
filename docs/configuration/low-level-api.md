# Low Level API

*Basic Usage*

```python
# Imports
from forgather.config import ConfigEnvironment

# Construct a configuration environment
env = ConfigEnvironment()

# Define a configuration
document = """
!call:torch:randn [ 2, 2 ]
"""

# Convert the configuration to a graph
graph = env.load_from_string(document).config

# Construct the graph
graph()
tensor([[ 0.0090,  0.0064],
        [-1.1638,  0.7066]])
```

## Create Config Environment

A configuration environment is required to construct configurations from YAML/Jinja2 inputs; it conains the infromation needed to located Jina2 templates by name as well as defining the global variables available to templates.

```python
from forgather.config import ConfigEnvironment
...
ConfigEnvironment(
    searchpath: Iterable[str | os.PathLike] | str | os.PathLike = tuple("."),
    pp_environment: Environment = None,
    global_vars: Dict[str, Any] = None,
):
```

- searchpath: A list of directories to search for templates in.
- pp_environment: Override the default Jinja2 environment class with another implementation.
- global_vars: Jinja2 global variables visible to all templates.

```python
env = ConfigEnvironment("./templates/")
```

## Define Input

A configuration document consists of a combination of YAML and Jinja2 syntax. Typically, a config template would be loaded from a file, but for testing we can create a template directly from a Python string.

Both the Jinja2 template and the configuration may accept variables.

## Convert Document to Graph

```python
class ConfigEnvironment:
... 
    def load(
        self,
        config_path: os.PathLike | str,
        /,
        **kwargs,
    ) -> Config:
...
    def load_from_string(
        self,
        config: str,
        /,
        **kwargs,
    ) -> Config:
```

- load: Load a template from a path; all paths relative to 'searchpaths' are searched for the template.
    - config_path: The relative (to searchpaths) template path.
    - kwargs: These are passed into the context of the template.
- load_from_string: As with load, but a Python string defines the template body; Note that this bypasses the template loader.
    - config: A Python string with a Jinja2 template.
    - kwargs: Passed to the template.

## Materializing the Graph

Construct the objects directly from the graph.

```python
from forgather.latent import Latent
...
def materialize(obj: Any, /, *args, context_vars: Dict=None, **kwargs):
```

Construct all object in the graph, returning the constructed root-node.

context_vars: The global variables, which will be substitued by '!var' nodes.

If the root node is a partial funciton, *args and **kwargs are forwarded to the function.

Alternatively, if the root-node is not a dictionary, the following are equivalnt:

```python
Latent.materialize(graph)

# Performs the same action, if the root-node is not a dictionary.
graph()
```

If the root-node is a dictionary...

```yaml
main: !partial:math:sqrt []
```

The dictionary elements can be accessed using dot-notation and costructed individually.

```python
graph.main(16)
4.0
```

## Convert Graph to YAML

Convert the node-graph to a YAML representation. This may not be exactly the same as it was in the source template, but should be symantically equivalent.

```python
from forgather.yaml_encoder import to_yaml
...
def to_yaml(obj: Any):
```

## Convert Graph to Python

This function takes the output from Latent.to_py(graph) and uses it to render Pyhon code using a Jinja2 template. If the template is unspecified, an implicit "built-in" template is used, which will generate appropriate import and dynamic import statements, where required.

```python
from forgather.codegen import generate_code
...
def generate_code(
    obj,
    template_name: Optional[str] = None,
    template_str: Optional[str] = None,
    searchpath: Optional[List[str | os.PathLike] | str | os.PathLike] = ".",
    env=None,  # jinja2 environment or compatible API
    **kwargs,
) -> Any:
```

The default template accepts the following additional kwargs:

    factory_name: Optional[str]="construct", ; The name of the generated factory function.
    relaxed_kwargs: Optional[bool]=Undefined, ; if defined, **kwargs is added to the arg list
    
See 'help(generate_code)' for details.