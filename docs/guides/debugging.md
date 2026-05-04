# Debugging Configuration Errors

This guide covers common errors and a systematic approach to diagnosing
configuration problems. For the full debugging command reference with detailed
examples, see [Configuration Debugging](../configuration/debugging.md).

## Debugging workflow

When something goes wrong, work through the pipeline stages in order. Each
command tests a different stage, so the first one that fails tells you where
the problem is.

**Step 1: Do all configs parse?**

```bash
forgather ls
```

Failing configs show as `PARSE ERROR` instead of their name and description.
Add `-d` to stop at the first error with a full traceback:

```bash
forgather ls -d
```

**Step 2: Inspect the preprocessed output**

```bash
forgather -t config.yaml pp -d
```

The `-d` flag preserves line numbers and dumps all templates as Jinja2 sees
them. This is essential because Forgather's line-statement sugar (`-- if`,
`[block_name]`, etc.) is converted before Jinja2 processes it, which can make
line numbers in error messages inaccurate without debug mode.

**Step 3: Check template inheritance**

```bash
forgather -t config.yaml trefs --format tree
```

Or for a graphical view (requires `graphviz`):

```bash
forgather -t config.yaml trefs --format svg -e
```

**Step 4: Verify YAML parsing**

```bash
forgather -t config.yaml graph --format yaml
```

If `pp` succeeds but `graph` fails, the problem is in the YAML layer (malformed
syntax, bad tag usage, etc.).

**Step 5: Test object construction**

```bash
forgather -t config.yaml construct --target trainer_args
forgather -t config.yaml construct --target model --call
```

Use `forgather -t config.yaml targets` to list all available targets.

## Common errors

### PARSE ERROR in `forgather ls`

This means preprocessing or YAML parsing failed. Run `forgather ls -d` for the
traceback.

**Most common cause**: `jinja2.exceptions.UndefinedError` -- a variable or
namespace attribute doesn't exist.

```
jinja2.exceptions.UndefinedError: 'Namespace' has no attribute 'pipeline_layers'
```

Fix: check that the variable is defined in a parent template, or provide a
default with `| default(value)`.

### RecursionError (infinite template loop)

```
RecursionError: maximum recursion depth exceeded
```

**Cause**: A template extends itself. This happens when multiple projects in the
search path have configs with the same name. For example, if both
`llama/configs/4M.yaml` and `llama_canon/configs/4M.yaml` exist in the search
path, a child config named `4M.yaml` that extends `configs/4M.yaml` resolves to
itself.

**Fix**: Use a distinct config name (e.g., `nope_4M.yaml`), or isolate the
search paths using separate model sub-projects.

### ModuleNotFoundError for modelsrc components

```
ModuleNotFoundError: No module named 'my_custom_module'
```

**Cause**: When extending a model project that has a `modelsrc/` directory, the
`project_dir` variable in `[model_submodule_searchpath]` resolves to the
*current* project, not the base model project. The base model's custom modules
are not in the search path.

**Fix**: Override `[model_submodule_searchpath]` to include the base model's
modelsrc directory:

```yaml
[model_submodule_searchpath]
    - "{{ joinpath(ns.forgather_dir, 'examples/models/base_model/modelsrc') }}"
    == super()
```

### Block override replaces entire parent block

```yaml
# This replaces ALL of [model_bits], not just [my_new_component]
[model_bits]
    [my_new_component]
.define: &my_part ...
```

**Cause**: Nesting a new block inside a parent block without `== super()`
replaces the entire parent block content.

**Fix**: Use the extension block pattern. Override the designated empty
extension block (e.g., `[causal_bits]`) with `== super()` + your additions.
See [Creating a Model Project: Adding new components](creating-a-model-project.md#adding-new-components).

### Wrong main_feature for dataset

Tokenization produces empty results or the model trains on garbage.

**Cause**: The dataset's text field is not named `"text"` (the default). Code
datasets often use `"code"`, Wikipedia datasets may use `"page"`, etc.

**Fix**: Set `ns.main_feature` in the dataset config's `[config_metadata]`:

```yaml
[config_metadata]
    == super()
    -- set ns.main_feature = "code"
```

Check the dataset's HuggingFace page or inspect it directly:

```python
from datasets import load_dataset
ds = load_dataset("author/dataset-name")
print(ds)  # Shows features
```

### Custom modelsrc module import errors

```
ModuleNotFoundError: No module named 'causal_multihead_attn'
```

**Cause**: Custom modules in `modelsrc/` cannot import from other modelsrc
components. HuggingFace's dynamic module loader only resolves imports one level
deep from the main model file.

**Fix**: Make custom modules self-contained. Use dependency injection (pass
objects through constructor parameters) instead of importing from sibling
modules. See [Creating a Model Project: Adding custom components](creating-a-model-project.md#adding-custom-components).

### Missing `reset_parameters()` on custom module

```
ValueError: Module of type 'MyModule' has parameters, but lacks a 'reset_parameters()' method
```

**Fix**: Add a `reset_parameters()` method to your custom `nn.Module`.

## Useful flags

| Flag | Commands | Effect |
|------|----------|--------|
| `-d` / `--debug` | `ls`, `pp`, `graph` | Verbose output, preserved line numbers, stop on first error |
| `-e` / `--edit` | `pp`, `trefs`, `code`, `construct` | Open output in VS Code or vim |
| `-r` / `--refresh-model` | `model` | Force regeneration from sources (needed after modelsrc changes) |
| `--target TARGET` | `graph`, `code`, `construct` | Examine a specific target instead of the whole config |
| `--format` | `trefs`, `graph` | Output format (`tree`, `svg`, `yaml`, `python`, etc.) |

## Syntax highlighting

Install the Forgather syntax highlighting plugins for your editor. They make
template files significantly easier to read and errors easier to spot.

- [Vim](../syntax_highlighting/vim/vim-syntax-install.md)
- [VS Code](../syntax_highlighting/vscode/README.md)
