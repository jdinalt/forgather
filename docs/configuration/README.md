# Configuration

Forgather configurations are YAML files preprocessed by Jinja2, parsed with custom YAML tags, and materialized into Python objects through a node graph. This section is a reference for the configuration language and the system that runs it.

For the conceptual overview (what a project is, how the pipeline works), start with [Core Concepts](../core-concepts/README.md).

## Pages in this section

- **[Syntax Reference](syntax-reference.md)** — complete reference for line statements (`-- extends`, `-- block`, `-- if`, `-- set`) and YAML tags (`!call`, `!factory`, `!partial`, `!singleton`, `!var`).
- **[Model Initialization](model-initialization.md)** — regex-based parameter initialization patterns.
- **[Debugging](debugging.md)** — preprocessor walkthrough, common error patterns, and tools for diagnosing template-resolution issues.
- **[Low-level API](low-level-api.md)** — the `ConfigEnvironment` and node-graph API the `Project` abstraction is built on.

## Related

- [Project Templates](../project-templates/lm-training-projects.md) — reusable project bases (LM Training, Auto LR).
- [Project System API](../api/project.md) — the `Project` class and its programmatic interface.
