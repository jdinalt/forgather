# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this
repository. **Keep this file short** — anything load-on-demand belongs
under `CLAUDE.d/` or in `docs/`.

Forgather is a configuration-driven ML framework built on template
inheritance and code generation. The central abstraction is the
**Project**: an ML experiment defined by templates that materialize
into standalone Python in `output_models/`.

## When you're not sure where to look

| Question | Read this |
|---|---|
| What does this project do, end-to-end? | `README.md`, then `CLAUDE.d/architecture.md` |
| How do I run the `forgather` CLI? | `CLAUDE.d/cli.md`, or just `forgather <sub> --help` |
| How do templates and config inheritance work? | `CLAUDE.d/templates.md`, then `docs/configuration/syntax-reference.md` |
| How does checkpointing work? | `CLAUDE.d/checkpointing.md`, then `docs/checkpointing/user_guide.md` |
| Multi-node training / smoke tests | `CLAUDE.d/multinode.md`, then `docs/operations/tls.md` |
| Common pitfalls and their fixes | `CLAUDE.d/gotchas.md` |
| Server backend / webui (before editing) | `tools/forgather_server/README.md` |
| Inference server / client | `tools/inference_server/README.md` |
| Tests — organization, fixtures, workflows | `docs/development/testing.md` |
| TLS / mTLS for cluster auth | `docs/operations/tls.md` |

`docs/` is the source of truth for user-facing and operator-facing
documentation; `CLAUDE.d/` is agent-targeted condensed reference.
`tools/<component>/README.md` and `tools/<component>/ARCHITECTURE.md`
are the source of truth for the component's internals.

## Conventions

**Style.** Follow the conventions already in the file you're editing.
Avoid emojis.

**Documentation ships with the feature.** A feature that isn't
documented effectively doesn't exist for anyone but the person who
wrote it. When adding or substantially changing a component, update in
the same PR (or the very next commit):

- The component's `tools/<component>/ARCHITECTURE.md` and
  `tools/<component>/README.md` (technical truth).
- Each affected hit under `docs/` (user-facing — grep for the
  feature/CLI/endpoint name and update every reference).

Out-of-date docs are worse than absent ones because they confidently
mislead.

**Validate after editing configs or templates.** Run `forgather ls` —
a failed config renders as `PARSE ERROR` instead of its description.
Use `forgather -t cfg.yaml pp` to inspect preprocessor output when
diagnosing a specific config.

**Test instructions.** See `docs/development/testing.md` for the full
guide. Tests live under `tests/`; pytest is the runner.

## Working notes — `.claude_notes/`

Use `.claude_notes/` (gitignored) for cross-session working notes:
in-progress implementation summaries, design-decision logs, debugging
trails. **Never commit them.** Treat that directory like a scratchpad
that survives context resets but doesn't belong in main.

Examples of what belongs there: `PHASE1_IMPLEMENTATION_SUMMARY.md`,
`CHECKPOINT_INTEGRATION_SUMMARY.md`.

## Things to remember without looking

These are the gotchas you'll waste real time on if you forget them.
Full explanations in `CLAUDE.d/gotchas.md`.

- **vLLM is currently broken** (Transformers v5 not yet supported).
- **YAML tags are not interchangeable**: `!partial` = Callable;
  `!singleton` = lazy, cached; `!factory` = called every access.
  With no args, pass `[]`.
- **`forgather ls` after every config edit** — catches PARSE ERRORs.
- **Cross-project model inheritance with `modelsrc/`** needs the
  baseline config to override `[model_submodule_searchpath]` — else
  `ModuleNotFoundError` at codegen.
- **Template name shadowing → infinite recursion**. Don't name a
  child config the same as a parent in another model project's search
  path.
