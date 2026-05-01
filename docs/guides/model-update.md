# Updating a Saved Model to Newer Forgather Sources

When the source code or templates for a Forgather model architecture
change -- a parameter is renamed, a layer refactored, a config field
renamed -- existing saved models become incompatible with the new code:
loading the checkpoint with `strict=True` fails on renamed FQNs, and
`strict=False` silently leaves the renamed parameters uninitialised.

`forgather update` migrates a saved Forgather model in place: it
regenerates the model code from the *current* sources and applies a
chain of versioned migrations to the saved config and weights, so the
result is a working model on the new schema with the original
hyperparameters preserved.

This is the in-Forgather counterpart to
[`forgather convert`](model-conversion.md). Where `convert` round-trips
through HuggingFace, `update` stays inside Forgather and does not
require an HF counterpart for the architecture.

## Quick start

```bash
# Migrate a saved model to the current schema for its arch
forgather update output_models/my_llama out/my_llama_v2
```

The tool reads the source model's `forgather_arch` and
`forgather_arch_version` from `config.json`, looks up the registered
converter for that arch, walks its migration chain end-to-end, and
writes the result to the destination directory.

## When you need this

`forgather update` solves a different problem from
[`forgather model construct`](model-cli.md). `model construct` rebuilds a
model from a project + config template, which only works when you still
have a config that matches the saved hyperparameters. Real saved models
routinely carry customisations that have drifted from any template
default -- an extended RoPE base for long-context training, a hand-tuned
sliding window, a non-default head dim. Those values must come from the
saved `config.json`, not from a fresh template.

`forgather update`:

- Reads hyperparameters from the *saved* config and translates them
  through versioned config-translators.
- Applies parameter-FQN renames to the saved state_dict.
- Optionally applies weight-level transforms (head reshapes,
  permutations) when a step requires them.
- Stamps the new schema version into the destination's `config.json`.

It does **not** re-train the model, change vocabulary, or rebuild the
tokenizer. Tokenizer files, `generation_config.json`, and any chat
template are carried forward verbatim.

## CLI reference

```bash
forgather update [OPTIONS] SRC_MODEL_PATH DST_MODEL_PATH
```

**Provenance overrides** (use when the source model predates schema
stamping):

| Option | Description |
|--------|-------------|
| `--arch NAME` | Converter registry key (e.g. `llama`); overrides `forgather_arch` from source config |
| `--from-version N` | Source schema version; overrides `forgather_arch_version` from source config |
| `--to-version N` | Stop the chain at version `N` (default: the converter's current `arch_version`) |

**Checkpoint:**

| Option | Description |
|--------|-------------|
| `-c, --checkpoint PATH` | Specific checkpoint directory to migrate (default: latest under `SRC/checkpoints/`, falling back to `SRC` root) |

**Model properties:**

| Option | Description |
|--------|-------------|
| `--device {cpu,cuda,...}` | Device used during migration (default: `cpu`) |
| `--dtype {bfloat16,float16,float32}` | Override saved dtype (default: keep checkpoint dtype) |
| `--safetensors` | Save weights as safetensors (default: PyTorch `.bin`) |
| `--no-strict` | Allow missing/unexpected keys when loading the migrated state_dict (default: strict) |

**Other:**

| Option | Description |
|--------|-------------|
| `--converter-path PATH` | Additional directory to search for converter plugins (repeatable) |
| `--dry-run` | Resolve and print the migration plan; write nothing |
| `--log-level LEVEL` | Logging level (default: `INFO`) |

## How updates work

### Schema provenance

Forgather model code generation stamps two fields into every newly
saved `config.json`:

- `forgather_arch` -- string, the converter registry key for the
  architecture (e.g. `"llama"`). Defaults to `ns.model_short_name` from
  the model template, which already matches the converter key for
  archs under `examples/models/`.
- `forgather_arch_version` -- integer, the schema version of the
  Forgather sources at the time of save. Defaults to `1`.

`forgather update` reads these fields to identify the migration source.
Models saved before this stamping was added simply lack the fields;
pass `--arch` and `--from-version` on the CLI to identify them
manually.

### Versioned migrations

Each architecture's converter (the same Python class that
[`forgather convert`](model-conversion.md) uses for HF↔FG conversion)
declares two class attributes for the in-Forgather update path:

```python
@register_converter("llama")
class LlamaConverter(HFConverter):
    arch = "llama"
    arch_version = 2  # current schema version
    forgather_migrations = {
        # 1 -> 2: rename attention.query_linear to attention.q_proj
        1: VersionMigration(
            description="rename attention.query_linear -> attention.q_proj",
            migrate_config=lambda cfg: cfg,  # no config field rename here
            param_subs=(
                (r"attention\.query_linear", r"attention.q_proj", ()),
            ),
        ),
    }
```

A `VersionMigration` step has three parts:

| Field | Purpose |
|-------|---------|
| `migrate_config` | Function that translates the (already partially migrated) config dict into the next-version dict. Field renames, default backfills, and removals all happen here. |
| `param_subs` | Recursive regex substitution list (same format as
[`forgather convert`](model-conversion.md#parameter-remapping) and
`forgather.ml.remap_params.remap_state_dict`). Rewrites parameter FQNs. |
| `transform_state_dict` | Optional weight-level transform applied after `param_subs`. Receives the renamed state_dict + migrated config, returns a new state_dict. Use for shape changes, head-dim reshapes, permutations. |

### Migrations chain end-to-end

`forgather update` is designed for arbitrary version gaps. Maintainers
register one step per schema bump (`1 -> 2`, `2 -> 3`, ...). The tool
composes the chain at runtime:

```
v1 -> v2 -> v3 -> ... -> v_current
```

Each step's output becomes the next step's input. If any step in the
range is missing from `forgather_migrations`, the tool fails before
touching weights with a clear diagnostic naming the missing step(s).
Pass `--to-version` to stop the chain at a version the converter
*can* reach.

Backwards updates (downgrades) are not supported.

### What gets carried forward

- Model weights (after rename + optional transform).
- Hyperparameters from the saved `config.json` (translated through
  `migrate_config`).
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, ...) -- saved by the project
  materialisation against the source directory.
- `generation_config.json` and any `chat_template.jinja` /
  `chat_template.json` -- copied verbatim if the regenerated tree
  doesn't already produce them.
- `hf_model_type` and `dtype` Forgather-specific config fields.

### What gets regenerated

- All Python source files (`*.py`) from the current Forgather
  templates and modelsrc.
- `auto_map`, `architectures`, and other code-generator-derived fields
  in `config.json`.
- `forgather_arch_version` -- bumped to the target version.

### Audit log

Every successful update writes `DST/forgather_update.json`:

```json
{
  "schema": "forgather_update.v1",
  "timestamp": "2026-05-01T12:34:56+00:00",
  "source": "/path/to/output_models/my_llama",
  "arch": "llama",
  "from_version": 1,
  "to_version": 3,
  "migrations": [
    "1->2: rename attention.query_linear -> attention.q_proj",
    "2->3: head_dim split"
  ],
  "dtype": "bfloat16",
  "missing_keys": [],
  "unexpected_keys": []
}
```

Use this to verify which migration steps actually ran when debugging
a strict-load failure.

## Examples

**Stamped provenance, no overrides:**
```bash
forgather update output_models/llama_4m out/llama_4m_v2
```

**Older model without stamped metadata:**
```bash
forgather update legacy/llama_old out/llama_new \
    --arch llama --from-version 1
```

**Stop at an intermediate version** (useful when you want to verify a
specific migration step in isolation):
```bash
forgather update SRC DST --to-version 2
```

**Dry run** (resolve the plan and print it; no writes):
```bash
forgather update SRC DST --dry-run
```

**Specific (non-latest) checkpoint:**
```bash
forgather update SRC DST -c SRC/checkpoints/checkpoint-385440
```

**Permissive load** (e.g. when developing a new migration and the chain
is incomplete):
```bash
forgather update SRC DST --no-strict
```

## Authoring a migration

When you make a non-backwards-compatible change to an arch's source
code or config schema, register a migration entry on the arch's
converter so existing saved models can be brought forward.

1. Open the converter, e.g. `examples/models/llama/src/converter.py`.
2. Bump `arch_version` by one.
3. Add an entry to `forgather_migrations` keyed by the *previous*
   version, describing the v_prev -> v_new step.

Example: renaming `attention.query_linear` -> `attention.q_proj`
between v1 and v2:

```python
from forgather.ml.model_conversion import VersionMigration

@register_converter("llama")
class LlamaConverter(HFConverter):
    arch = "llama"
    arch_version = 2
    forgather_migrations = {
        1: VersionMigration(
            description="rename attention.query_linear -> attention.q_proj",
            migrate_config=lambda cfg: cfg,
            param_subs=(
                (r"attention\.query_linear", r"attention.q_proj", ()),
            ),
        ),
    }
```

Example with a config field rename (`mlp_dim` -> `intermediate_size`):

```python
def _v2_to_v3_config(cfg):
    new = dict(cfg)
    if "mlp_dim" in new:
        new["intermediate_size"] = new.pop("mlp_dim")
    return new

forgather_migrations = {
    1: ...,
    2: VersionMigration(
        description="rename mlp_dim -> intermediate_size",
        migrate_config=_v2_to_v3_config,
    ),
}
```

Verify your migration with a real saved model before shipping. The
golden test is bit-equivalent logits between source and destination on
a fixed input -- `forgather update` runs strict load, so a missing
rename surfaces immediately.

## See also

- **[Model Conversion](model-conversion.md)** -- bidirectional HF ↔ FG
  conversion, including the parameter-remapping engine and converter
  plugin pattern that `forgather update` reuses.
- **[Model Architecture](model-architecture.md)** -- how Forgather
  generates self-contained model code; the
  `forgather_arch` / `forgather_arch_version` provenance fields are
  stamped during that code generation.
- **[Finalize Model](finalize-model.md)** -- build a clean handoff
  directory after pre-training. `finalize` and `update` are
  complementary: `finalize` produces a polished release artifact from a
  training run, `update` brings an older release artifact forward to a
  new schema.
- **[Model CLI](model-cli.md)** -- `forgather model construct` and
  related commands. Note: `model construct` rebuilds from a project
  template, which is *not* the same as updating a saved model -- use
  `forgather update` when the saved hyperparameters must be preserved.
