# Template development

Definitive syntax reference: `docs/configuration/syntax-reference.md`.
This page covers the common patterns and the override gotchas worth
remembering.

## Inheritance

Single-parent only (Jinja2 limitation). For composition across
multiple bases, use the "include and extend" pattern.

```yaml
-- extends "types/training_script/causal_lm/causal_lm.yaml"

-- block construct_new_model
    -- include 'project.model_config'
-- endblock construct_new_model

-- block optimizer
optimizer: &optimizer !lambda:torch:optim.AdamW
    lr: 1.0e-3
-- endblock optimizer

#--- project.model_config ---
-- block model_config
    == super()
    # Project overrides
    hidden_size: 512
-- endblock model_config
```

## Common override patterns

- **Model-chain blocks** (`[rel_positional_encoder]`,
  `[attention_factory]`, etc.) must be overridden in an **inline model
  template** (after `#--- config.*.model ---`), not in the main config.
- **Disable RoPE**: set `[rel_positional_encoder]` to
  `.define: &relative_pe null`. Attention modules guard with
  `if self.pos_encoder:`.
- **Disable a factory-based component**: set the anchor to `null` in
  the appropriate block.
- **Reference a models sub-project from training**: use
  `-- set ns.model_project_dir = abspath(joinpath(project_dir, "models"))`.

## Quick metadata override

```yaml
-- extends 'types/training_script/causal_lm/causal_lm.yaml'
-- block config_metadata
    == super()
    -- set ns.config_name = "My Experiment"
-- endblock
```

## Cross-project model inheritance

When an experiment extends a model project that has its own
`modelsrc/`, the experiment's baseline config must override
`[model_submodule_searchpath]` because `project_dir` resolves to the
*current* project, not the base model.

```yaml
-- extends "configs/base_config.yaml"
[config_metadata]
    == super()
    -- set ns.model_name = "my_baseline"
[model_definition]
    -- include "config.baseline.model"

#--- config.baseline.model ---
-- extends "config.base.model"
[model_submodule_searchpath]
    - "{{ joinpath(ns.forgather_dir, 'examples/models/base_model/modelsrc') }}"
    == super()
```

Without the override: `ModuleNotFoundError` at codegen. Base models
without their own `modelsrc/` need no override.

## Workflow

- After editing templates, **always** run `forgather ls` to check that
  every config still parses (PARSE ERROR rows mean failure).
- Use `forgather -t cfg.yaml pp` to inspect preprocessor output when
  diagnosing config bugs.
- Use `forgather meta` to inspect workspace/project structure +
  template search paths.
- Notebooks (`project_index.ipynb`) are first-class for interactive
  exploration; materialize with `Project("config.yaml")(...)`.

## Working examples

Refer to these when creating new projects (canonical reference for
common patterns):

| Pattern | Example |
|---|---|
| Bare harness over `projects/lm_training_project.yaml` | `examples/base_lm_project/` |
| Projects overview tutorial | `examples/tutorials/projects_overview/` |
| Project composition tutorial | `examples/tutorials/project_composition/` |
| Model training tutorial | `examples/tutorials/tiny_llama/` |
| Cross-project inheritance with `modelsrc/` | `examples/tiny_experiments/canon/` |
| Cross-project inheritance without `modelsrc/` | `examples/pretrain/small-llm/custom_canon/` |
