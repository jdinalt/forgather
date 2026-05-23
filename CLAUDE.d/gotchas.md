# Gotchas

Things that bite anyone editing this repo. Each item lists the symptom
and the fix.

## YAML tag confusion

`!partial`, `!singleton`, `!factory` are not interchangeable:

| Tag | Behavior |
|---|---|
| `!partial` | Constructs a `functools.partial`-style Callable (object is a function, not a value) |
| `!singleton` | Lazy object; called once on first access, cached |
| `!factory` | Called on **every** access, never cached |

If you want "a function object the caller invokes", use `!partial`.
If you want "an instance shared by every consumer", `!singleton`. If
you want "a fresh instance per access", `!factory`.

When invoking with no arguments, pass empty list `[]` — the parser
needs the explicit call-site marker.

Full reference: `docs/configuration/syntax-reference.md`.

## Template name shadowing → RecursionError

When multiple model projects sit in the search path and share a
config name (e.g., both `llama/configs/4M.yaml` and
`llama_canon/configs/4M.yaml`), a child config named `4M.yaml` that
extends `configs/4M.yaml` will resolve to itself → infinite recursion.

**Fix**: rename the child (e.g., `nope_4M.yaml`) or put each base
model in its own models sub-project with isolated search paths.
`examples/tiny_experiments/canon/llama_nope/` is the worked example.

## `ModuleNotFoundError` when extending a model project with `modelsrc/`

`project_dir` in `[model_submodule_searchpath]` resolves to the
**current** project, not the base model. If you extend a model that
has its own `modelsrc/`, you must override the search path in the
models sub-project's baseline config to point back at the base model.
See `CLAUDE.d/templates.md` ("Cross-project model inheritance") for the
exact snippet.

## Missing imports (`Callable` etc.)

Generated code may reference types whose imports the generator didn't
emit. If a `NameError`/`ImportError` mentions a stdlib type, add the
import in the affected file. Not a deep bug — just a generation gap.

## Complex64 / RoPE checkpointing

RoPE models can fail to save: `safetensors` doesn't support complex
tensors. If you hit this, the work-around is to compute RoPE buffers
at load time instead of persisting them.

## vLLM is currently broken

Transformers v5 (which Forgather moved to) is not yet supported by
vLLM. The `vllm serve` commands and `tp_plan`/`pp_plan` machinery are
preserved (`docs/inference/vllm_integration.md`;
`templatelib/examples/models/transformers/dynamic_llama.yaml` has a
worked `base_model_tp_plan` / `base_model_pp_plan`) but won't run
end-to-end today.

## Validate after every config edit

After editing any template or config, run **`forgather ls`** — failed
configs render as `PARSE ERROR` instead of their description. This is
the cheapest way to catch regressions across the whole project.

For diagnosing a specific config's preprocessor output:
`forgather -t cfg.yaml pp`.
