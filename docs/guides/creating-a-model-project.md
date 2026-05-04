# Creating a Model Project

This guide walks through creating a new model project from scratch. A model
project defines a model architecture as a Forgather configuration, which the
code generator then turns into standalone Python code.

For existing model projects to use as reference, see
[`examples/models/`](../examples/models/README.md). For the full `forgather model`
command reference — constructing, testing, checkpoint handling, and rebuilding from
modified sources — see [Model CLI Reference](model-cli.md).

## Project structure

A model project has this layout:

```
my_model/
├── meta.yaml                    # Project metadata
├── templates/
│   ├── project.yaml             # Root template (optional if configs are self-contained)
│   └── configs/
│       ├── small.yaml           # Model variant (e.g., 4M params)
│       └── large.yaml           # Another variant (e.g., 30M params)
├── modelsrc/                    # Optional: custom components
│   └── my_attention.py
└── src/                         # Optional: HF converter
    └── converter.py
```

Only `meta.yaml` and at least one config are required. The `modelsrc/` and `src/`
directories are only needed if you have custom components or want HuggingFace
conversion support.

## Creating a project with the CLI

The quickest way to get started is with the CLI scaffolding commands. If you
don't already have a workspace, create one first:

```bash
forgather ws create \
    --name "My Workspace" \
    --description "Model experiments" \
    --forgather-dir /path/to/forgather \
    -l base -l examples \
    my_workspace
```

The `-l base -l examples` flags add both the base and examples template libraries
to the search path. The `examples` library is needed for
`models/transformers/dynamic_causal_transformer.yaml`, which provides a complete
Llama-style architecture as a starting point.

Then create the model project inside the workspace:

```bash
cd my_workspace
forgather project create \
    --name "My Model" \
    --description "A custom transformer model"
```

This creates `my_model/` with `meta.yaml`, a default config, and the templates
directory. The default config is a minimal skeleton that you'll flesh out in the
next steps.

## Writing the model config

Each config defines a concrete model variant. The simplest approach is a single
file that combines the project-level template with the model definition using an
inline template.

Edit `templates/configs/default.yaml`:

```yaml
-- extends "models/model_type.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "My Model"
    -- set ns.config_description = "A custom transformer model"
    -- set ns.model_name = model_name | default("my_model")

[model_definition]
    -- include "config.default.model"

[dynamic_args]
    == super()
    model_name:
        names: "--model-name"
        default: "my_model"
        help: "Name for the output model"

#-------------------- config.default.model --------------------
-- extends "models/transformers/dynamic_causal_transformer.yaml"

[model_tokenizer]
## Load tokenizer from the wikitext tokenizer project (2K vocab)
tokenizer: &tokenizer !call:forgather:from_project
    project_dir: "{{ joinpath(ns.forgather_dir, 'examples', 'tokenizers', 'wikitext') }}"
    config_template: "2k.yaml"

[model_config]
    == super()
    hidden_size: 128
    num_attention_heads: 4
    num_key_value_heads: 2
    num_hidden_layers: 2
    intermediate_size: 512
    max_position_embeddings: 512
    vocab_size: 2000
    tie_word_embeddings: false
```

The file has two halves separated by the `#----` line. The top half is the
project-level template (metadata, CLI args). The bottom half is the inline model
template, which extends `dynamic_causal_transformer.yaml` and provides a complete
Llama-style architecture. The `[model_config]` block sets the hyperparameters.

Tokenizers are loaded using `!call:forgather:from_project`, which constructs the
tokenizer from a Forgather tokenizer project. The wikitext tokenizer project
provides pre-built BPE tokenizers in several vocabulary sizes.

## Verify

```bash
# Check that configs parse
forgather ls

# Preview the fully expanded configuration
forgather pp

# Show the generated Python code
forgather code

# Test forward and backward pass on GPU
forgather model --device cuda test
```

The `forgather model test` command constructs the model, runs a forward and
backward pass with random data, and reports the loss. This validates that the
model definition is correct end-to-end.

For the complete reference on `forgather model` subcommands -- including how to
save checkpoints, load weights, rebuild after source changes, test with real
data, and control hyperparameters via dynamic CLI args -- see
[examples/models/README.md](../examples/models/README.md).

## Customizing the architecture

The architecture is composed from components defined in `modelsrc/transformer/`.
Each component is referenced via a factory in the `[model_bits]` block of the
base template. Override individual factories to swap components.

### Available components

| Component | Default | Alternatives |
|-----------|---------|-------------|
| Layer norm | `RMSNorm` | `LayerNorm` |
| Layer type | `PreLNLayer` (pre-layer-norm) | `PostLNLayer` (post-layer-norm) |
| Attention | `CausalMultiheadAttn` (RoPE) | `CausalAlibiAttn` (ALiBi) |
| Feedforward | `GLUFeedforwardLayer` (SiLU gated) | `FeedforwardLayer` (basic) |
| Positional encoding | `RotaryEmbedding` (RoPE) | `SinusoidalPE`, `null` (none) |
| Initialization | `llama_init_weights` | `init_weights_isqrt_dmodel`, `simple_weight_init` |

### Overriding a component

To change a component, override the relevant block in the inline model template.
For example, to disable RoPE (for use with ALiBi or no positional encoding):

```yaml
#-------------------- config.small.model --------------------
-- extends "models/transformers/dynamic_causal_transformer.yaml"

[rel_positional_encoder]
.define: &relative_pe null

[model_config]
    == super()
    hidden_size: 256
    # ...
```

To switch from pre-layer-norm to post-layer-norm with a different feedforward:

```yaml
#-------------------- config.small.model --------------------
-- extends "models/transformers/dynamic_causal_transformer.yaml"

[model_bits]
    == super()

    [layer_factory]
    layer_factory: &layer_factory !partial:.post_ln_layer:PostLNLayer@layer_factory
        feedforward_factory: *feedforward_factory
        attention_factory: *attention_factory
        norm_factory: *layer_norm_factory

    [feedforward_factory]
    feedforward_factory: &feedforward_factory !partial:.feedforward_layer:FeedforwardLayer@feedforward_factory
        activation: !partial:torch.nn:ReLU
```

The leading dot in `.post_ln_layer:PostLNLayer` means "import from the model
submodule search path" (`modelsrc/transformer/` by default).

### Adding new components

Overriding an existing component is straightforward -- you redefine the block
with the same name. But *adding* a new component to an existing block is
trickier, because of how Jinja2 blocks work. Consider this (incorrect) attempt:

```yaml
[model_bits]
    [my_new_component]
.define: &my_new_component ...
```

This does **not** add `my_new_component` to the parent's `[model_bits]` block --
it completely replaces the entire `[model_bits]` block with just your new
component, losing all the existing bits (loss_fn, layer_norm_factory,
attention_factory, etc.).

To make extension possible, Forgather's base templates include **empty
extension blocks** at each level of the hierarchy. For example,
`dynamic_causal_transformer.yaml` defines `[causal_bits]` inside `[model_bits]`
for this purpose:

```yaml
# In dynamic_causal_transformer.yaml:
[model_bits]
    [causal_bits]
## Override to add more bits to the base model
    [loss_fn]
    # ...
```

A child template can then add new components by overriding the empty extension
block:

```yaml
[causal_bits]
    == super()
    [my_new_component]
.define: &my_new_component ...
```

The `== super()` directive includes the (empty) parent content, then your
additions are appended. To allow the extension pattern to continue downstream,
add your own empty extension block:

```yaml
[causal_bits]
    == super()
    [my_custom_bits]
## Override to add more bits in a child template
    [my_new_component]
.define: &my_new_component ...
```

Templates that extend yours can then add their own components via
`[my_custom_bits]` without disturbing anything you or the base templates defined.

Without this pattern, the only way to add new components would be to modify a
parent template -- which is a problem when the parent lives in the template
library.

### Adding custom components

If you need a component that doesn't exist in `modelsrc/transformer/`, create a
`modelsrc/` directory in your model project and add your implementation there.
Then register the search path so the code generator can find it:

```yaml
# In your config's inline model template:
[model_submodule_searchpath]
    - "{{ joinpath(project_dir, 'modelsrc') }}"
    == super()
```

Your custom module can then be referenced with the dot-import syntax:
```yaml
[attention_factory]
attention_factory: &attention_factory !partial:.my_attention:MyAttention@attention_factory
    d_model: !var "hidden_size"
    num_heads: !var "num_attention_heads"
    # ... all required constructor arguments
```

When overriding a factory, you must include **all required constructor arguments**
that the original factory binds. Check the default factory definition in
`dynamic_causal_transformer.yaml` to see which arguments are needed.

**Important constraints for custom components:**

1. **Custom modules must be self-contained.** They cannot import from other
   modelsrc modules (e.g., `from .causal_multihead_attn import ...`). This is
   because HuggingFace's dynamic module loader only resolves imports one level
   deep from the main model file. Use dependency injection instead -- receive
   any shared objects through constructor parameters.

2. **Custom modules with parameters must implement `reset_parameters()`.** The
   weight initialization system calls this method on every module that has
   learnable parameters. Without it, initialization will fail with a
   `ValueError`.

3. **The interface must be compatible.** If you're replacing the attention
   module, your `forward()` signature must accept the same arguments as the
   module it replaces (hidden_states, attention_mask, past_key_values, etc.).

See `examples/models/singlehead/` for a working example of a model with a
custom attention component.

## Using the model in a training project

Training projects reference model projects via `ns.model_project_dir` and
`ns.model_project_config`. In a training config:

```yaml
-- extends 'project.yaml'

[config_metadata]
    == super()
    -- set ns.model_project_dir = "/path/to/my_model"
    -- set ns.model_project_config = "small.yaml"
```

Or set these in the training project's `project.yaml` to make them the default
for all configs in that project. See `examples/tutorials/tiny_llama/` for a
complete example of a training project that imports a model from another project.

## Model templates reference

The key base templates in the inheritance chain:

| Template | Location | Purpose |
|----------|----------|---------|
| `model_type.yaml` | `templatelib/base/models/` | Root: defines CLI args, output structure |
| `base_language_model.yaml` | `templatelib/base/models/` | Adds tokenizer, config, constructor blocks |
| `custom.yaml` | `templatelib/base/models/causal_lm/` | Causal LM with config and constructor |
| `custom_dynamic.yaml` | `templatelib/base/models/causal_lm/` | Adds code generation via `!meta` |
| `dynamic_causal_transformer.yaml` | `templatelib/examples/models/transformers/` | Complete Llama-style architecture with all component factories |
