# Core Concepts

This page describes the mental model behind Forgather. Understanding these concepts
makes everything else -- configuration syntax, template design, training workflows --
much easier to reason about.

## The configuration pipeline

Forgather transforms YAML template files into running Python objects through a
three-stage pipeline:

```
Template files (.yaml)
    │
    ▼  Preprocessing (Jinja2)
Plain YAML
    │
    ▼  Parsing (custom YAML tags)
Node graph (lazy)
    │
    ▼  Materialization (depth-first construction)
Python objects (model, trainer, dataset, ...)
```

The node graph is the central representation. In the normal case, it is materialized
directly into Python objects. But because it is a structured intermediate
representation, it can also be converted to other forms -- equivalent Python code,
back to YAML, or in principle to any serialization format like JSON. The `forgather
code` command uses this to show the Python equivalent of a configuration, which can
be easier to read than the YAML + Jinja2 source.

Each stage is independent and inspectable:

| Stage | CLI command | What you see |
|-------|------------|--------------|
| Preprocessing | `forgather pp` | Fully-expanded YAML after Jinja2 rendering |
| Parsing | `forgather graph` | Node graph structure |
| Materialization | `forgather construct` | Constructed Python objects |
| (Alternate output) | `forgather code` | Node graph rendered as equivalent Python |

### Stage 1: Preprocessing

Template files are Jinja2 templates with syntactic sugar for readability. The
preprocessor expands inheritance (`-- extends`), blocks (`[block_name]`), includes
(`-- include`), and variables (`-- set`, `{{ expr }}`), producing plain YAML.

The sugar is straightforward:

| Sugar | Jinja2 equivalent |
|-------|-------------------|
| `-- statement` | `{% statement %}` |
| `== expression` | `{{ expression }}` |
| `[block_name]` | `{% block block_name %}` ... `{% endblock %}` |
| `== super()` | `{{ super() }}` |
| `##` (full-line) | Comment (stripped) |

### Stage 2: YAML parsing

The preprocessed YAML is parsed with custom tags that represent deferred
construction. Instead of building objects immediately, the parser creates a *node
graph* -- a tree of instructions describing what to build and how.

The key tags and their semantics:

| Tag | Behavior |
|-----|----------|
| `!singleton` / `!call` | Construct once, cache and reuse on subsequent references |
| `!factory` | Construct a new instance every time it is referenced |
| `!partial` | Create a partial function (bind some arguments now, supply the rest later) |
| `!meta` | Pass the raw graph (not materialized) to the callable for transformation |
| `!var` | Variable reference, resolved at materialization time |

Tag format: `!tag:module.path:ClassName@optional_name`

```yaml
optimizer: !partial:torch.optim:AdamW@optimizer
    lr: 1.0e-3
    weight_decay: 0.01
```

A [partial function](https://docs.python.org/3/library/functools.html#functools.partial)
captures some arguments up front and returns a callable that accepts the remaining
ones later. Here, the configuration binds `lr` and `weight_decay` to AdamW, producing
a callable that only needs the model's parameters to create a fully configured
optimizer. The training code can then call `optimizer(model.parameters())` without
knowing anything about which optimizer was chosen or how it was configured -- that
decision lives entirely in the configuration file.

### Stage 3: Materialization

`Latent.materialize()` walks the node graph depth-first, constructing real Python
objects. Singletons are cached so that YAML anchors and aliases behave as expected
-- multiple references to the same `!singleton` node yield the same object.

Materialization is *selective*: you can request specific targets (named root-level
keys) and only the subgraph needed for those targets is constructed.

```python
proj = Project("train_tiny_llama.yaml")
model = proj("model")          # Only materializes the "model" target
model, tokenizer = proj("model", "tokenizer")  # Two targets
training_script = proj()        # Default: materializes "main"
```

### Code generation for models

The `!meta` tag enables a special path. When a `MetaNode` is encountered during
materialization, its child nodes are *not* constructed. Instead, the raw node graph
is passed to the callable, which can inspect or transform it.

Forgather uses this for model definitions: the `!meta` callable encodes the model's
node graph into standalone Python source code and writes it to `output_models/`.
The generated code has no dependency on Forgather at runtime -- the model can be
loaded and used with standard PyTorch and HuggingFace APIs. This also serves as a
readable reference for understanding exactly what a configuration will construct,
since most people find Python easier to read than the configuration language.

## Projects and workspaces

### Projects

A **project** is a directory containing a `meta.yaml` file and a `templates/`
subdirectory. It is the unit of work in Forgather -- one project encapsulates one
experiment or model definition.

```
my_project/
├── meta.yaml                    # Project metadata
└── templates/
    ├── project.yaml             # Root template (defines the experiment)
    └── configs/
        ├── baseline.yaml        # Leaf configurations (what you select with -t)
        └── variant.yaml
```

- `meta.yaml` names the project, sets the default config, and defines the template
  search path.
- `templates/project.yaml` is the root template. It typically extends a base
  template from the template library and overrides blocks to customize the
  experiment.
- `templates/configs/*.yaml` are leaf configurations. Each one extends
  `project.yaml` (or another template) and makes targeted changes -- a different
  optimizer, a different dataset, a different model size. These are what you pass
  to `forgather -t`.

### Workspaces

A **workspace** is a directory containing a `forgather_workspace/` subdirectory.
It groups related projects and provides shared configuration.

```
my_workspace/
├── forgather_workspace/
│   ├── meta_defaults.yaml       # Shared defaults (search paths, base directories)
│   └── base_directories.yaml    # Points to Forgather installation
├── project_a/
│   ├── meta.yaml                # extends meta_defaults.yaml
│   └── templates/...
└── project_b/
    ├── meta.yaml
    └── templates/...
```

Each project's `meta.yaml` extends the workspace's `meta_defaults.yaml` via Jinja2
inheritance. The workspace's `base_directories.yaml` typically sets `ns.forgather_dir`
so that all projects can find the template library.

Forgather discovers the workspace by searching upward from the project directory for
a `forgather_workspace/` folder.

### Template search path

When a template uses `-- extends` or `-- include`, Forgather searches for the
referenced file in this order:

1. The project's `templates/` directory (highest priority)
2. The workspace's `forgather_workspace/` directory
3. The Forgather template library (`templatelib/`)

This layering lets projects override any template from the library while inheriting
everything else.

## Template inheritance

Templates use Jinja2 inheritance to eliminate configuration duplication. A child
template extends a parent and overrides only what changes.

### Extending with `== super()`

The `== super()` directive includes the parent block's content, allowing you to
add to or selectively override parts of a block rather than replacing it entirely.

**Parent** defines an optimizer block:
```yaml
[optimizer]
optimizer: &optimizer !partial:torch.optim:AdamW
    lr: 1.0e-4
    weight_decay: 0.01
```

**Child** overrides just the learning rate:
```yaml
-- extends 'projects/lm_training_project.yaml'

[optimizer]
    == super()
    lr: 1.0e-3
```

This works because of how YAML handles duplicate keys: when a key appears more
than once in a mapping, the last value wins. The `== super()` directive first emits
the parent block (including `lr: 1.0e-4`), then the child appends `lr: 1.0e-3`,
which overrides the earlier value. This pattern is used extensively throughout the
template library.

Note that this approach works well for overriding values at the top level of a
block. When you need to override something nested more deeply, it is often better
to factor that part into its own block, define it with a YAML anchor (`&name`),
and reference it via an alias (`*name`). This gives the child template a clean
override point without having to restate the surrounding structure.

### Replacing a block

When you need to change more than a few values, you can replace a block entirely
by omitting `== super()`:

```yaml
-- extends 'projects/lm_training_project.yaml'

[optimizer]
optimizer: &optimizer !partial:forgather.ml.optim:Adafactor
    lr: 1.0e-3
```

### Inline template definitions

Jinja2 does not support multiple inheritance. If your template extends one parent
and includes content from a template that extends a *different* parent, you cannot
directly override blocks from the included template. The workaround is to create
a separate template file that performs the override, then include that.

Inline template definitions avoid the need for that extra file. A single config
file can contain multiple sub-templates, separated by a header line:

```yaml
## Main template
-- extends 'projects/lm_training_project.yaml'

[construct_new_model]
    -- include 'project.model_config'

#-------------------- project.model_config --------------------
## This is an inline sub-template named 'project.model_config'
-- extends 'models/llama.yaml'

[model_config]
    == super()
    hidden_size: 512
```

The separator line (`#--- name ---`) splits the file into independently addressable
templates. The main template can `-- include` or `-- extends` the sub-template by
name. This keeps related configuration in one file while working around Jinja2's
single-inheritance constraint.

This pattern is used extensively in project configurations -- particularly for
model definitions, where the model template chain has a different parent than the
training template chain.

For a complete syntax reference, see [Configuration Syntax](../configuration/syntax-reference.md).

## Trainers

Forgather provides a hierarchy of trainer classes for different training scenarios:

| Trainer | Use case |
|---------|----------|
| `Trainer` | Single-GPU or basic distributed training |
| `AccelTrainer` | Multi-GPU via HuggingFace Accelerate |
| `DDPTrainer` | DistributedDataParallel |
| `PipelineTrainer` | Pipeline parallelism (split model across GPUs) |

All trainers share the same callback system, checkpoint coordination, and
configuration interface. The trainer is selected in the configuration template --
switching from single-GPU to pipeline parallel is a template change, not a code
change.

For the complete list of training arguments (batching, compile, checkpointing,
memory, DDP, pipeline, ...) and each trainer class's constructor parameters,
see the [Trainer Options Reference](../trainers/trainer_options.md).

## Further reading

- [Getting Started](../getting-started/README.md) -- Install and train your first model
- [Configuration Syntax](../configuration/syntax-reference.md) -- Complete tag and directive reference
- [Template Inheritance](../configuration/inheritance.md) -- Detailed inheritance patterns
- [Model Architecture](../model-architecture.md) -- Transformer component inventory
- [Trainer Options Reference](../trainers/trainer_options.md) -- All training arguments and trainer constructor parameters
- [Checkpointing](../checkpointing/README.md) -- Distributed checkpoint system
