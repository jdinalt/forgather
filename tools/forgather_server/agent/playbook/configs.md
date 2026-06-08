# configs — write, validate, and debug Forgather configs

Key docs (read_file, or search_docs for more):
`docs/configuration/syntax-reference.md` (template/config syntax),
`docs/project-templates/lm-training-projects.md`,
`docs/guides/creating-a-model-project.md`,
`docs/guides/creating-a-dataset-project.md`,
`docs/guides/debugging.md`, `docs/configuration/debugging.md`.

CLI → tool equivalents (you don't run the CLI):
`forgather pp` → render_config_pp; `forgather graph` (validate) → check_config;
`forgather code` → render_config_code; `forgather targets` → inspect_config's
code_targets; `forgather tlist` → list_config_templates; `forgather trefs` →
config_template_refs; `forgather ls` → list_projects / list_configs.

The config pipeline has three stages: (1) preprocess templates (Jinja2
inheritance) → render_config_pp; (2) parse into a node graph (YAML + `!` tags)
→ check_config; (3) materialize objects (runs constructors — not exposed).
render_config_code is a DEBUG/export tool (equivalent stand-alone Python), NOT
a pipeline stage.

After writing/editing a config, validate with check_config — it runs preprocess
+ parse-to-node-graph and returns {ok:true, targets} or {ok:false, error}. That
is the "does it compile?" check; do NOT use render_config_code to validate. Use
render_config_pp to inspect resolved text. Fix and re-check before reporting done.

Creating: propose_new_project scaffolds dir + meta.yaml + a default config (the
project must exist before propose_new_config). No workspace? propose_new_workspace
first. For a better start, seed propose_new_config/propose_new_project from a
scaffold (list_meta_templates → meta_template + values) or copy an existing
config (copy_from a path from list_configs) rather than an empty stub.
