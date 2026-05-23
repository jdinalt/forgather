# Meta-templates

Scaffolds for creating new configuration files. When the webui's
**New Config…** / **New Template…** modal opens, it offers these as a
"pick a starting point" tree alongside the **Blank file** option.

A meta-template is a pair of files in the same directory:

- `<name>.yaml` — the **body**: the actual config text, with
  `$VAR` / `${VAR}` markers for the variables the user will fill in.
- `<name>.meta.yaml` — the **manifest**: title, description,
  `target_kind` (`config` or `template`), and a `fields[]` list
  declaring each variable.

Both files must exist for the meta-template to be discovered.

## Why `$VAR` and not Jinja syntax

A Forgather config *is* a Jinja template — using Jinja markers for
meta-template substitution would force escaping `{{` and `{%` in every
body, which is painful and bug-prone. We use Python's `string.Template`
syntax (`$VAR`, `${VAR}`) instead. The meta-template engine substitutes
once at file-creation time, writes the result to disk as a normal
config, and Forgather's preprocessor sees the file the same way it sees
any hand-written config.

## Hierarchy

The directory path under `templatelib/meta/` is the picker's tree:

```
templatelib/meta/
  _category.yaml                 (optional: catalog-wide intro)
  datasets/
    _category.yaml               (optional: group label + description)
    huggingface/
      _category.yaml
      with_config.yaml + with_config.meta.yaml
      single.yaml      + single.meta.yaml
  models/
    ...
```

`_category.yaml` is optional. Without one, the picker displays the
title-cased directory name (`huggingface` → "Huggingface").

```yaml
# _category.yaml
title: "Hugging Face Hub"
description: "Datasets loaded via the `datasets` library."
```

Templates and sub-categories can coexist at the same level — common
cases at the top, exotic ones nested below.

## Manifest format

```yaml
title: "HuggingFace dataset (with config name)"
description: |
  Multi-line description, shown when the user selects this scaffold.
  Markdown is fine; the picker renders it as plain text.
target_kind: "config"      # "config" → writes under configs_dir
                           # "template" → writes under templates_dir
fields:
  - name: CONFIG_NAME      # The marker name in the body ($CONFIG_NAME).
    label: "Config name"   # Optional. Falls back to `name` if omitted.
    description: "Helper text shown under the input."
    placeholder: "C4"      # Optional HTML placeholder.
    required: true         # Empty/missing → server returns 400.
  - name: VALIDATION_SPLIT
    default: "validation[:1000]"   # Pre-filled in the form; used when
                                   # the user submits the value empty.
```

**Resolution order** for each field when the new file is rendered:

1. User-supplied value (non-empty)
2. Manifest `default`
3. If `required`, render fails with a list of missing fields
4. Otherwise the marker is substituted with the empty string

Extra keys in the submitted values that aren't declared in the manifest
are silently ignored — the manifest is the source of truth for which
fields a scaffold accepts.

A `$VAR` in the body that isn't declared in the manifest raises an
error at render time. That's intentional: it catches typos in the
meta-template instead of silently leaving the marker in the user's
file.

## Adding a new meta-template

1. Pick (or create) a directory under `templatelib/meta/`.
2. Write the body file with `$VAR` markers. The body should
   `-- extends "..."` a normal templatelib config so the generated
   file slots into Forgather's inheritance graph correctly.
3. Write the matching `.meta.yaml` declaring every `$VAR` in the body.
4. Verify the discovery picks it up:

   ```python
   from tools.forgather_server import meta_templates
   for cat in meta_templates.discover():
       print(cat)
   ```

5. Open the webui's New Config modal — the new entry should appear in
   the picker.

## Conventions

- Use `UPPER_SNAKE_CASE` for variable names so they're visually
  distinct from Jinja's `{{ lower_case_args }}`.
- **Make `$VAR` markers only for fields that are universal to every
  instance of this scaffold's category.** For a packed-dataset scaffold,
  the universals are `CONFIG_NAME`, `DESCRIPTION`, and the dataset
  `path:` — every config of that shape has all three. Things like
  `source:`, `name:` (HF variant), or `main_feature:` are *common* but
  not universal, so they belong in the body as commented examples, not
  as form fields. Form fields aren't free: each one is a question the
  user has to answer up front, and a wrong-shape question is friction.
- **Prefer one verbose scaffold over many narrow ones.** The model is:
  show every knob someone might want, with reasonable defaults inherited
  from `== super()`, and put the *optional* knobs in as commented
  examples the user uncomments and edits. The user reads the file,
  keeps what they need, and deletes the rest. This scales better than
  trying to enumerate every permutation in the picker.
- The body must remain valid after substitution. If a field is
  optional and the resulting empty value would break the config,
  either give it a sensible `default` or make it `required`.
