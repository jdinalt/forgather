# Project System

The project system is the central abstraction in Forgather. A `Project` resolves a configuration file through the template inheritance chain and provides access to all configured components.

**Related documentation:**

- [Core Concepts](../core-concepts/README.md) — projects, templates, and the configuration pipeline
- [Configuration Overview](../configuration/README.md) — template system and YAML configuration
- [Syntax Reference](../configuration/syntax-reference.md) — complete reference for line statements and YAML tags
- [Low-level API](../configuration/low-level-api.md) — the API underlying the `Project` abstraction

## Quick Example

```python
from forgather.project import Project

proj = Project("train_tiny_llama.yaml")

# Materialize the full training script
training_script = proj()

# Materialize individual components
model_factory = proj("model")
train_dataset  = proj("train_dataset")

model = model_factory()
```

---

### `Project` {#forgather-project-project}

`forgather.project.Project`

```python
class Project(config_name: Optional[str] = '', project_dir: Optional[str | os.PathLike] = '.', **kwargs)
```

Central user-facing abstraction for a Forgather ML experiment.

A ``Project`` loads a YAML configuration file through a Jinja2 template
inheritance chain, parses it into a node graph, and can materialise any
named target from that graph into live Python objects.  It is the primary
entry point for interactive experiment development and for training scripts.

**Parameters**

- `config_name` (str) — Name of the configuration template to load (e.g. ``"train_tiny_llama.yaml"``).
An empty string or ``None`` loads the project's default configuration as
declared in ``meta.yaml``.
- `project_dir` (str or os.PathLike) — Path to the project directory.  Must contain a ``meta.yaml`` file and a
``templates/`` sub-directory.  Defaults to the current working directory.
- `**kwargs` — Additional keyword arguments forwarded to the Jinja2 preprocessor as
template variables.

**Attributes**

- `config_name` (str) — Name of the selected configuration; automatically set to the project
default when *config_name* is empty or ``None``.
- `project_dir` (str) — Absolute path to the project directory.
- `meta` (MetaConfig) — Parsed project metadata (search paths, default config, etc.).
- `environment` (ConfigEnvironment) — Jinja2 + YAML preprocessing environment used to load templates.
- `config` (Any) — The parsed node graph produced from the preprocessed YAML.  ``None``
when no configuration has been loaded yet.
- `pp_config` (str) — The fully preprocessed YAML text (after Jinja2 rendering), useful for
debugging template issues.

**Examples**

Load a project from the current directory using the default configuration:

```python
>>> proj = Project()
>>> training_script = proj()
```

Load a specific configuration and materialise individual targets:

```python
>>> proj = Project("train_tiny_llama.yaml", "examples/tutorials/tiny_llama")
>>> model = proj("model")
>>> model, tokenizer = proj("model", "tokenizer")
```

> **Note**
>
> When debugging a configuration it is usually easier to construct the project
> incrementally for better diagnostic messages.  See ``project_config.ipynb``
> for a step-by-step notebook example.

**Attributes**

- `config_name` (str)
- `project_dir` (str)
- `meta` (MetaConfig)
- `environment` (ConfigEnvironment)
- `config` (Any)
- `pp_config` (str)

**Methods**

#### `load_config` {#forgather-project-project-load_config}

```python
def load_config(config_name: str, **kwargs)
```

Load and parse the named configuration template.

Preprocesses the template through the Jinja2 environment, then parses
the resulting YAML into a node graph.  The results are stored in
``self.config`` and ``self.pp_config``.

**Parameters**

- `config_name` (str) — Name of the configuration template to load, relative to the
project's ``config_prefix`` directory (e.g. ``"train.yaml"``).
- `**kwargs` — Additional keyword arguments forwarded to the Jinja2 preprocessor
as template variables.

#### `add_template` {#forgather-project-project-add_template}

```python
def add_template(name, data)
```

Add an in-memory template definition to the Jinja2 loader.

**Parameters**

- `name` (str) — Template name used to reference this template from other templates
(e.g. via ``-- extends`` or ``-- include``).
- `data` (str) — Raw template source text.

---

### `MetaConfig` {#forgather-meta_config-metaconfig}

`forgather.meta_config.MetaConfig`

```python
class MetaConfig(project_dir = '.', meta_name = PROJECT_META_NAME)
```

Project metadata loaded from ``meta.yaml``.

``MetaConfig`` reads and parses the ``meta.yaml`` file that sits at the
root of every Forgather project.  It resolves template search paths,
locates the workspace root by walking up the directory tree, and exposes
the configuration values needed by :class:`~forgather.project.Project` to
set up its :class:`ConfigEnvironment`.

**Parameters**

- `project_dir` (str or os.PathLike) — Path to the project directory containing ``meta.yaml``.  Defaults to
the current working directory (``"."``)
- `meta_name` (str) — Name of the metadata file to load.  Defaults to ``"meta.yaml"``.

**Attributes**

- `project_dir` (str) — Path to the project directory as supplied to ``__init__``.
- `name` (str) — Name of the meta file (e.g. ``"meta.yaml"``).
- `project_name` (str or None) — Human-readable project name declared in ``meta.yaml``.
- `description` (str or None) — Short project description declared in ``meta.yaml``.
- `meta_path` (str) — Absolute path to the meta file.
- `searchpath` (list of str) — Ordered list of absolute directory paths searched for config templates.
Derived from the ``searchdir`` key in ``meta.yaml``, defaulting to
``[project_dir/templates]``.
- `system_path` (str or None) — Optional system-level template search path from ``meta.yaml``.
- `config_prefix` (str) — Sub-directory inside the search path where leaf configuration files
live.  Defaults to ``"configs"``.
- `default_cfg` (str or None) — Name of the default configuration file as declared in ``meta.yaml``.
When ``None``, :meth:`default_config` picks the first template found
under *config_prefix*.
- `config_dict` (dict) — Raw dictionary parsed from ``meta.yaml``.
- `workspace_root` (str) — Absolute path to the workspace root directory (the directory that
contains ``forgather_workspace/``), found by walking up from
*project_dir*.

**Raises**

- `ValueError` — If the project directory does not exist, ``meta.yaml`` is not found, or
no ``forgather_workspace/`` directory exists in the ancestor hierarchy.

**Examples**

```python
>>> meta = MetaConfig("/path/to/my_project")
>>> print(meta.project_name)
My Project
>>> print(meta.searchpath)
['/path/to/my_project/templates', '/path/to/workspace/forgather_workspace']
```

**Attributes**

- `project_dir` (str)
- `name` (str)
- `project_name` (Optional[str])
- `description` (Optional[str])
- `meta_path` (str)
- `searchpath` (List[str])
- `system_path` (Optional[str])
- `config_prefix` (str)
- `default_cfg` (Optional[str])
- `config_dict` (dict)
- `workspace_root` (str)

**Methods**

#### `norm_path` {#forgather-meta_config-metaconfig-norm_path}

```python
def norm_path(path)
```

_No documentation._

#### `default_config` {#forgather-meta_config-metaconfig-default_config}

```python
def default_config()
```

Return the name of the default configuration template.

**Returns**

- `str` — The value of ``default_config`` from ``meta.yaml`` if set;
otherwise the name of the first template discovered under
*config_prefix* across all search paths.

#### `config_path` {#forgather-meta_config-metaconfig-config_path}

```python
def config_path(config_template = None)
```

Return the template-relative path for the given configuration name.

**Parameters**

- `config_template` (str or None) — Name of the configuration template (e.g. ``"train.yaml"``).
When ``None`` or an empty string, the default configuration is used.

**Returns**

- `str` — Path of the form ``"{config_prefix}/{config_template}"`` suitable
for passing to :meth:`ConfigEnvironment.load`.

#### `find_templates` {#forgather-meta_config-metaconfig-find_templates}

```python
def find_templates(prefix = '', suffix = '.yaml')
```

Iterate over all templates in the search path matching a prefix and suffix.

Walks every directory in :attr:`searchpath`, descending into the
sub-directory given by *prefix*, and yields ``(name, path)`` pairs for
every file whose name ends with *suffix*.  Hidden directories
(names starting with ``"."``) are skipped.

**Parameters**

- `prefix` (str) — Sub-directory to search within each search-path entry.  Defaults to
``""`` (search from the root of each search-path entry).
- `suffix` (str) — File extension filter.  Defaults to ``".yaml"``.

**Yields**

- `template_name` `str` — Template name relative to the prefixed search directory, suitable
for use with :meth:`ConfigEnvironment.load`.
- `template_path` `str` — Filesystem path to the template file.

**Examples**

Find all templates under a ``models`` directory in any search-path entry:

```python
>>> for template_name, template_path in meta.find_templates("models"):
...     print(template_name, template_path)
```

#### `find_workspace_dir` {#forgather-meta_config-metaconfig-find_workspace_dir}

```python
def find_workspace_dir(project_dir)
```

Walk up the directory tree to find the Forgather workspace root.

The workspace root is the nearest ancestor directory that contains a
``forgather_workspace/`` sub-directory.

**Parameters**

- `project_dir` (str) — Starting directory for the upward search.

**Returns**

- `str` — Absolute path to the workspace root directory.

**Raises**

- `ValueError` — If no ``forgather_workspace/`` directory is found in any ancestor.

#### `find_project_dir` {#forgather-meta_config-metaconfig-find_project_dir}

```python
def find_project_dir(project_dir)
```

Walk up the directory tree to find the nearest Forgather project directory.

A project directory is one that directly contains a ``meta.yaml`` file.

**Parameters**

- `project_dir` (str) — Starting directory for the upward search.

**Returns**

- `str` — Absolute path to the nearest project directory that contains
``meta.yaml``.

**Raises**

- `ValueError` — If no project directory is found at or above *project_dir*.

---

### `ConfigEnvironment` {#forgather-config-configenvironment}

`forgather.config.ConfigEnvironment`

```python
class ConfigEnvironment(searchpath: Iterable[str | os.PathLike] | str | os.PathLike = tuple('.'), pp_environment: Optional[Environment] = None, global_vars: Optional[Dict[str, Any]] = None)
```

Jinja2 preprocessing and YAML parsing environment for Forgather configurations.

``ConfigEnvironment`` wraps a :class:`~forgather.preprocess.PPEnvironment`
(a customised Jinja2 environment) and a suite of custom YAML constructors
that translate ``!call``, ``!singleton``, ``!factory``, ``!partial``, and
``!var`` tags into :class:`~forgather.latent.Node` objects.  The result of
:meth:`load` is a :class:`Config` containing both the parsed node graph and
the preprocessed YAML text.

**Parameters**

- `searchpath` (str, os.PathLike, or iterable of str/PathLike) — Directories searched for templates, in priority order.  Non-existent
directories are silently ignored.  Defaults to ``(".",)``.
- `pp_environment` (jinja2.Environment or None) — A pre-configured Jinja2 environment to use instead of the default
:class:`~forgather.preprocess.PPEnvironment`.  When ``None`` (default)
a fresh ``PPEnvironment`` is created from *searchpath*.
- `global_vars` (dict or None) — Variables injected into the Jinja2 global namespace and available in
every template.  Merged with any variables already present in
*pp_environment*.  Defaults to ``{}``.

**Examples**

```python
>>> env = ConfigEnvironment(searchpath=["/path/to/templates"])
>>> config = env.load("configs/train.yaml")
>>> node_graph, pp_text = config.get()
```

**Attributes**

- `pp_environment`

**Methods**

#### `get_pp_environment` {#forgather-config-configenvironment-get_pp_environment}

```python
def get_pp_environment()
```

_No documentation._

#### `get_loader` {#forgather-config-configenvironment-get_loader}

```python
def get_loader()
```

_No documentation._

#### `preprocess` {#forgather-config-configenvironment-preprocess}

```python
def preprocess(config_path: os.PathLike | str, **kwargs, /)
```

Render a configuration template through Jinja2 and return the result.

Locates *config_path* in the search path, renders it with the
configured global variables plus any extra *kwargs*, and returns the
resulting YAML text.

**Parameters**

- `config_path` (str or os.PathLike) — Template path relative to the search path (e.g. ``"configs/train.yaml"``).
- `**kwargs` — Additional keyword arguments passed as Jinja2 template variables,
overriding globals for this render.

**Returns**

- `ConfigText` — The rendered YAML text.  :class:`ConfigText` is a ``str`` subclass
that additionally exposes :meth:`~ConfigText.with_line_numbers`.

#### `preprocess_with_trace` {#forgather-config-configenvironment-preprocess_with_trace}

```python
def preprocess_with_trace(config_path: os.PathLike | str, **kwargs, /)
```

Preprocess *config_path* and also return the per-template trace.

Runs :meth:`preprocess` inside :func:`capture_pp` so the second element
of the returned tuple is the ordered list of
``(template_name, preprocessed_source)`` pairs that participated in the
render — the same data that ``pp_verbose`` prints to stdout, but
returned programmatically.

**Returns**

- `(ConfigText, list[tuple[str, str]])` — The fully rendered text plus the per-template trace (load order).

#### `preprocess_from_string` {#forgather-config-configenvironment-preprocess_from_string}

```python
def preprocess_from_string(config: str, **kwargs, /)
```

Render a configuration template supplied as a string through Jinja2.

**Parameters**

- `config` (str) — Raw template source text (may contain Jinja2 directives).
- `**kwargs` — Additional keyword arguments passed as Jinja2 template variables.

**Returns**

- `ConfigText` — The rendered YAML text.

#### `load` {#forgather-config-configenvironment-load}

```python
def load(config_path: os.PathLike | str, **kwargs, /)
```

Preprocess and parse a configuration file into a node graph.

Combines :meth:`preprocess` and :meth:`load_from_ppstring` into a
single call.

**Parameters**

- `config_path` (str or os.PathLike) — Template path relative to the search path.
- `**kwargs` — Additional Jinja2 template variables forwarded to :meth:`preprocess`.

**Returns**

- `Config` — Container holding the parsed node graph and the preprocessed YAML
text.

**Raises**

- `Exception` — Any YAML or node-graph parse error, annotated with the numbered
preprocessed source for easier debugging.

#### `load_from_string` {#forgather-config-configenvironment-load_from_string}

```python
def load_from_string(config: str, **kwargs, /)
```

Preprocess and parse a configuration supplied as a string.

**Parameters**

- `config` (str) — Raw template source text.
- `**kwargs` — Additional Jinja2 template variables forwarded to
:meth:`preprocess_from_string`.

**Returns**

- `Config` — Container holding the parsed node graph and the preprocessed YAML
text.

#### `load_from_ppstring` {#forgather-config-configenvironment-load_from_ppstring}

```python
def load_from_ppstring(pp_config: str)
```

Parse an already-preprocessed YAML string into a node graph.

Parses *pp_config* with the custom YAML constructors (``!call``,
``!singleton``, ``!factory``, ``!partial``, ``!var``, etc.) and
validates the resulting graph with :meth:`~forgather.latent.Latent.check`.

**Parameters**

- `pp_config` (str) — Fully rendered YAML text (output of Jinja2 preprocessing).

**Returns**

- `Config` — Container holding the parsed node graph and *pp_config*.

**Raises**

- `Exception` — Any YAML parse error or node-graph validation error, annotated with
line-numbered source text.

#### `render_code` {#forgather-config-configenvironment-render_code}

```python
def render_code(config_path: os.PathLike | str, *, target: Optional[str] = 'main', /, **kwargs)
```

Render *config_path* as Python source via :func:`forgather.codegen.generate_code`.

Mirrors the ``forgather code`` CLI: preprocesses + parses the config,
looks up *target* (default ``"main"``) in the resulting node graph,
and runs the codegen template. When *target* is ``None`` the entire
config graph is rendered (useful for reviewing every materialisable
target in one document).

**Raises**

- `PreprocessError` — Jinja2 preprocessing failed (delegated from :meth:`preprocess`).
- `YamlParseError` — The preprocessed text was not valid YAML.
- `CodeGenError` — *target* was not found in the config or codegen itself raised.

#### `find_referenced_templates` {#forgather-config-configenvironment-find_referenced_templates}

```python
def find_referenced_templates(template_name: os.PathLike | str, **kwargs, /)
```

Iterate over the full template inheritance hierarchy for a given template.

Traces actual template loading at render time so that dynamic
``extends`` / ``include`` references (those whose targets are computed
by Jinja2 expressions) are captured in addition to statically declared
ones.

**Parameters**

- `template_name` (str or os.PathLike) — Name of the root template to analyse (relative to the search path).
- `**kwargs` — Forwarded to the inner :meth:`load` call so dynamic-args
conditional includes (e.g. ``include "trainers/" + trainer_type +
".yaml"``) resolve to the right files. Omit for the static-default
view.

**Yields**

- `level` `int` — Depth of this template in the hierarchy (0 = root).
- `name` `str` — Template name as it appears in the loader.
- `filename` `str` — Filesystem path to the template file.

#### `get_template_dependencies` {#forgather-config-configenvironment-get_template_dependencies}

```python
def get_template_dependencies(template_name: os.PathLike | str, **kwargs, /)
```

Return raw dependency relationships for a template, suitable for graph generation.

**Parameters**

- `template_name` (str or os.PathLike) — Name of the root template to analyse.
- `**kwargs` — Forwarded to the inner :meth:`load` so the trace reflects which
templates are actually included given those Jinja variables.

**Returns**

- `load_sequence` `list of tuple[str, str]` — Ordered list of ``(template_name, filename)`` pairs in the order
templates were loaded during rendering.
- `dependencies_dict` `dict[str, set[str]]` — Mapping from each template name to the set of template names it
directly references (via ``extends`` or ``include``).
