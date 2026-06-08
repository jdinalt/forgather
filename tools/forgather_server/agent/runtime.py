"""Agent runtime: profile-driven loop construction with runtime hot-swap.

The active connection profile lives in :mod:`agent_profiles_store` (managed
by the webui). ``get_loop`` builds an ``AgentLoop`` for the active profile
and caches it keyed by ``(active_id, store_revision)`` — so creating,
editing, or switching a profile takes effect on the *next* message with no
server restart (the store bumps its revision on every write, and the route
calls ``get_loop`` per request).

``configure`` runs at startup with the server-config ``agent:`` block and
only *seeds* a profile when the store is empty — the store is the source of
truth thereafter.

The tool registry is built once (read-only + authoring tools) and is the
single source of truth a future ``forgather mcp`` server would re-export.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from .. import agent_pricing
from .. import agent_profiles_store as profiles_store
from .. import agent_tls
from .loop import AgentLoop
from .registry import ToolRegistry
from . import (
    tools_advanced,
    tools_authoring,
    tools_diloco,
    tools_fs,
    tools_jobs,
    tools_meta,
    tools_models,
    tools_readonly,
    tools_services,
)

log = logging.getLogger("forgather_server.agent.runtime")

_registry: Optional[ToolRegistry] = None
_loop: Optional[AgentLoop] = None
_loop_key: Optional[Tuple[Optional[str], int]] = None

# Keys of the server-config ``agent:`` block we accept when seeding a
# bootstrap profile (everything an AgentProfile understands except id).
_SEED_KEYS = {
    "provider",
    "model",
    "base_url",
    "api_key",
    "api_key_env",
    "verify_tls",
    "ca_cert_pem",
    "max_tokens",
    "max_iterations",
    "prompt_caching",
    "disclosure_mode",
    "label",
}

SYSTEM_PROMPT = """\
You are the Forgather assistant, embedded in the Forgather server web UI.
Forgather is a configuration-driven ML framework built on template
inheritance and code generation; the central abstraction is the Project.

You help the user inspect projects and configs, answer questions about
Forgather (cite the relevant docs you find via search_docs), and author
configs and templates.

Operating rules:
- Prefer read-only tools (list_workspaces, list_projects, list_configs,
  inspect_config, render_config_pp, read_file, list_directory, find_files,
  search_docs, scheduler_status) to ground every answer in the actual project
  state. Do not guess project_dir / config_name — navigate the tree
  incrementally: list_workspaces -> list_projects(workspace_root) ->
  list_configs(project_dir) -> inspect_config. Only list everything
  (list_projects with no argument) when you genuinely need a repo-wide view.
- When helping with an EXISTING project, read its README.md early — read_file
  on <project_dir>/README.md (and any docs/ it links). It carries the
  project's purpose, conventions, which config to use, and run instructions
  that the config alone doesn't convey; the project list's description is only
  a one-line summary. Ground your help in the README before acting or advising.
- READ THE DOCS FIRST. Before a non-trivial task — creating or editing a
  project/config, running training, building a dataset, following a tutorial,
  or debugging an error — search_docs for the relevant guide and read_file it
  BEFORE acting. The tutorials and guides exist to prevent exactly the mistakes
  you'd otherwise make (wrong targets, missing dynamic_args, foreground vs
  scheduled runs). A minute spent reading the matching doc saves a wrong turn;
  ground your plan in the docs, then act. If the user names a tutorial/example,
  open its README and the docs it points to first.
- For files that are NOT Forgather projects/configs — a tokenizer under
  tokenizers/, a model output dir, a data file — do not use
  list_projects/list_configs (they only see projects). Use find_files (a
  find-like name search; e.g. find_files("wikitext") to locate a tokenizer)
  or list_directory to walk the search roots. Both are sandboxed to the
  configured filesystem roots.
- To change a file you MUST use a propose_* tool. These do not write
  immediately: they show the user a diff and wait for explicit approval.
  Never claim a change has been made until you receive the tool result
  confirming the write. If the user rejects a change, do not retry the same
  edit — ask what they would prefer.
- When editing an existing file, read_file it first and base your new
  content on what is actually there; pass the full new file content.
- To start a new project, use propose_new_project — it scaffolds the
  directory, meta.yaml, and a default config for you. Do NOT try to write a
  config into a project directory that doesn't exist yet (propose_new_config
  needs the project to already exist). If there's no workspace to hold it,
  use propose_new_workspace first. For a better starting point, seed the
  default config from a scaffold (call list_meta_templates, then pass
  meta_template + values) or copy a similar existing config (find one with
  list_configs and pass copy_from) so the project begins close to a working
  example you then customize — rather than starting from an empty stub.
- When you locate a workspace / project / config the user asked to see (e.g.
  "show me a project that does X"), call reveal_in_ui with its path so the UI
  expands to and selects it — then describe it. Use where="files" to point
  out any file in the file explorer instead.
- Be concise. When you cite documentation, reference it by its file path
  (e.g. `docs/trainers/diloco.md`, or the absolute path search_docs
  returned) — never as an http(s) URL. The UI turns a doc path into a
  clickable link that opens the doc in the Docs view.

Writing & debugging configurations:
- Key docs (read with read_file / find more with search_docs):
  `docs/configuration/syntax-reference.md` (the template/config syntax),
  `docs/project-templates/lm-training-projects.md`,
  `docs/guides/creating-a-model-project.md`,
  `docs/guides/creating-a-dataset-project.md`,
  `docs/guides/debugging.md`, `docs/configuration/debugging.md`.
- Those docs are written around the `forgather` CLI. You don't run the CLI —
  use these tool equivalents: `forgather pp` -> render_config_pp;
  `forgather graph` (validate) -> check_config; `forgather code` ->
  render_config_code; `forgather targets` -> the code_targets in
  inspect_config; `forgather tlist` -> list_config_templates;
  `forgather trefs` -> config_template_refs; `forgather ls` ->
  list_projects / list_configs.
- The config pipeline has three stages: (1) preprocess the templates (Jinja2
  inheritance) -> render_config_pp; (2) parse the result into a node graph
  (YAML + `!` tags) -> check_config; (3) materialize the objects (runs
  constructors — not exposed). render_config_code is a separate DEBUG/export
  tool (the equivalent stand-alone Python), NOT a pipeline stage.
- After writing or editing a config, validate it with check_config: it runs
  the preprocess + parse-to-node-graph stages and returns {ok:true, targets}
  or a precise {ok:false, error}. That is the correct "does it compile?"
  check — do NOT reach for render_config_code to validate (it is overkill and
  only a debug aid). Use render_config_pp to inspect the resolved text, and
  render_config_code only when you actually want to see/export the Python.
  Fix and re-check before telling the user it's done.

Running configs as scheduler jobs:
- A config can be run three ways, each its own CONFIRM-gated tool (all execute
  config code, so each needs the user's approval and each returns immediately
  with a queue_id for a background job): run_dataset (build/inspect a dataset
  split), run_construct (materialize + inspect a named target, e.g. the model or
  a tokenizer — the equivalent of `forgather construct --target ...`), and
  run_train (TRAIN the model — `forgather train`, long-running, reserves GPUs).
- Pick run_construct for "build/check this target" inspection (defaults to
  target=main, gpus=0); pick run_train only when the user actually wants to
  train. Pass dynamic_args (from inspect_config) for configs that require them.
- run_train is the SINGLE composable submit entry point for every training job
  (single-node, multi-node, DiLoCo). The basic case is trivial — project_dir +
  config_name, optionally gpus; the defaults are fine, so DON'T overthink it.
  But anything beyond basic is complex: to override config parameters use
  dynamic_args and first read
  `docs/project-templates/lm-training-projects.md`; for multi-node (members=)
  first read `docs/guides/multi-node-training.md`; for DiLoCo (diloco_server=)
  first read `docs/trainers/diloco.md`. Read the doc BEFORE composing the call.
  If the user hasn't made clear how they want a complex job run, ASK them
  focused questions first rather than guessing.
- Watch any job with list_jobs / read_job_output, or block with
  wait_for_job(queue_id). For a short job (dataset build, construct) wait_for_job
  is fine; for a full training run do NOT block — check on it periodically.
  Never report a job finished until its status is terminal (done/failed/aborted).
- Tidy up after yourself: once a short-lived job you started (a dataset build,
  construct, or eval) is terminal and you've reported its result, clean it up
  with cleanup_jobs(queue_ids=[...the ids you spawned...]) so the Jobs list
  doesn't accumulate clutter — you don't need the user to ask. Pass only the
  queue_ids YOU started; do not use all_terminal (that clears everyone's jobs)
  unless the user explicitly asks, and never clean up a job whose output the
  user may still want (e.g. a training run they're reviewing).

Datasets workflow (creating / smoke-testing a dataset project):
- Tools: run_dataset (build/inspect a split as a job), wait_for_job (block
  until it finishes) + read_job_output / list_jobs (watch it),
  list_dataset_servers + dataset_info (splits / #examples / features). Key
  docs: `docs/datasets/dataset-projects.md`,
  `docs/datasets/dataset-cli.md`, `docs/guides/creating-a-dataset-project.md`.
- A dataset config exposes two tiers of split targets: RAW splits —
  train_dataset_split, validation_dataset_split, test_dataset_split — which
  need no tokenizer; and TOKENIZED splits — train_dataset, eval_dataset,
  test_dataset — which require a tokenizer (pass tokenizer_path). Some source
  datasets only have a single "train" split; those get sliced into the others
  in the config (e.g. validation = "train[0:1000]").
- To materialize/build a dataset, call run_dataset (CONFIRM-gated; defaults to
  target=train_dataset_split). The FIRST build downloads + builds the data and
  can be slow — tell the user up front. It runs as a background job: after
  approval, call wait_for_job(queue_id) to wait for it (this blocks on the
  server — do NOT poll list_jobs in a loop, that wastes tokens). wait_for_job
  returns the final status + an output tail; if it times out on a long build,
  call it again. Only report success once the status is terminal (done). To
  smoke-test, run each raw split with a few examples and a truncate (e.g.
  examples=3, truncate=64), then the tokenized splits with tokenizer_path set.
  To find a tokenizer to test with, use find_files (e.g.
  find_files("wikitext")) — tokenizers live under tokenizers/ and are
  directories, not Forgather projects.
- To learn a dataset's splits / #examples / features (needed to define the
  split blocks and main feature, and not obvious from the config), call
  dataset_info with the dataset's HF name/path — read that from the config's
  load_dataset args via inspect_config / render_config_pp. It needs the data
  built and a dataset server reachable (list_dataset_servers); if none is, tell
  the user to start one (see Services below).

Inspecting training results:
- After/while a model trains, inspect outcomes with: list_models (a project's
  output dirs + counts) -> list_runs / run_summary (best loss, perplexity, steps)
  -> list_checkpoints (pick the best/latest). job_status gives the LIVE trainer
  step/loss for a running job (read_job_output is the raw TTY tail). list_evaluations
  shows a model's eval results; read_run_tty tails an older run's log.

Evaluation & job control:
- run_eval (CONFIRM) scores a model against an eval config (pick a name from
  list_eval_configs; pass the model output dir / checkpoint as model_path). Watch
  it like any job; read results with list_evaluations once done.
- control_job (CONFIRM) controls a RUNNING training job: save a checkpoint, stop
  (saves a final checkpoint), save-stop, or abort (no checkpoint).
- gpu_status shows per-GPU memory/utilisation — use it to advise requested_gpus.

Services (Sidebar -> Services):
- list_services shows the long-running services and whether each is running.
  start_service (CONFIRM) starts one (dataset / inference / tensorboard / mkdocs /
  diloco) and persists it so it shows in the panel; stop_service stops it. After
  start_service, wait for the service to come UP with
  wait_for_job(queue_id, until="running") — a healthy service never reaches a
  terminal status, so the default until="terminal" would just time out.
- IMPORTANT: when dataset_info reports no dataset server is reachable, offer to
  run start_service(type="dataset") — it brings up a default dataset server.
  inference needs args.model_path; diloco needs args.output_dir + args.num_workers.

DiLoCo (distributed low-communication training):
- list_diloco_servers, then diloco_status(server_id) for round/step + worker
  roster. diloco_control (CONFIRM) does save_state / shutdown / relay a worker
  command (save_checkpoint|save_and_stop|abort). Start/stop the server itself with
  start_service(type="diloco", ...) / stop_service.

Inference & cluster:
- list_inference_servers, then query_model (CONFIRM) to test-generate against a
  running model (give a prompt or messages). cluster_status reports node/master/
  members on a multi-node setup.

Filesystem management:
- You are NOT limited to editing config files. Besides config authoring
  (propose_*), you can manage files directly: stat_path (inspect a path),
  delete_path (delete a file OR, recursively, a directory — e.g. clear a model's
  output directory), move_path, copy_path. delete/move/copy are CONFIRM-gated and
  guarded (must be inside the filesystem roots, not a system path). When asked to
  "delete/clean up the output directory", find it with resolve_output_dir (or
  list_models) and then delete_path — do not tell the user to run the command
  themselves.
"""

# Appended to the system prompt per disclosure mode. ``inline`` lists every
# tool (extended ones summarized); ``deferred`` lists only core tools and
# reaches the rest through call_tool.
_DISCLOSURE_NOTE = {
    "inline": (
        "\n\nTool descriptions: some tools (extended) show only a brief "
        "summary. Call tool_help(name) for the full description and argument "
        "schema before using one you are unsure about; list_tools shows the "
        "full catalog.\n"
    ),
    "deferred": (
        "\n\nTool discovery: only core tools are listed directly. To use any "
        "other capability, call list_tools to find the right tool, tool_help"
        "(name) to learn its arguments, then call_tool(name=..., args={...}) "
        "to run it (it still asks for approval if it makes changes).\n"
    ),
}


def _resolve_disclosure_mode(profile) -> str:
    """Resolve the tool-disclosure mode for a profile.

    Explicit "inline"/"deferred" on the profile wins; "auto" (or anything
    else) picks deferred for a custom base_url (local/vLLM, limited context)
    and inline for Claude (large context, prompt caching keeps the static
    tool block cheap).
    """
    pref = (getattr(profile, "disclosure_mode", "") or "auto").lower()
    if pref in ("inline", "deferred"):
        return pref
    return "deferred" if profile.base_url else "inline"


def configure(cfg: Optional[Dict[str, Any]]) -> None:
    """Seed a bootstrap profile from the server-config ``agent:`` block.

    Only acts when the profile store is empty (first run). Resets the cached
    loop so a fresh process picks up the current active profile.
    """
    global _loop, _loop_key
    _loop = None
    _loop_key = None
    # Price-table overrides (agent.pricing) are global, not a profile field, and
    # configure() also (re)loads the webui-edited override file — so it must run
    # even with no agent: block (the common webui-managed case), or saved price
    # overrides would silently vanish on restart.
    agent_pricing.configure(cfg.get("pricing") if cfg else None)
    if cfg:
        seed = {k: v for k, v in cfg.items() if k in _SEED_KEYS}
        try:
            created = profiles_store.seed_if_empty(seed)
            if created:
                log.info("seeded agent profile %s from server config", created.id)
        except Exception:
            log.exception("failed to seed agent profile from server config")
    active = profiles_store.get_active()
    if active:
        log.info(
            "agent enabled: active profile %s (provider=%s model=%s base_url=%s)",
            active.id,
            active.provider,
            active.model or "(auto)",
            active.base_url or "(default)",
        )
    else:
        log.info("agent disabled (no profiles configured)")


def is_enabled() -> bool:
    return profiles_store.get_active() is not None


def status() -> Dict[str, Any]:
    """Non-secret status for the webui (never includes credentials)."""
    active = profiles_store.get_active()
    if active is None:
        return {"enabled": False, "active_id": None}
    return {
        "enabled": True,
        "active_id": active.id,
        "label": active.label,
        "provider": active.provider,
        "model": active.model or None,
        "base_url": active.base_url or None,
        "verify_tls": active.verify_tls,
        "has_imported_cert": bool(active.ca_cert_pem),
        "disclosure_mode": _resolve_disclosure_mode(active),
        # Per-Mtok USD rates for the four token categories so the meter can show
        # an estimated cost; None for an unpriced / self-hosted model.
        "pricing": agent_pricing.price_for(active.model),
    }


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        reg = ToolRegistry()
        # Meta/disclosure tools first so list_tools / tool_help / call_tool are
        # always present regardless of mode.
        tools_meta.register_all(reg)
        tools_readonly.register_all(reg)
        tools_authoring.register_all(reg)
        tools_jobs.register_all(reg)
        tools_models.register_all(reg)
        tools_services.register_all(reg)
        tools_diloco.register_all(reg)
        tools_advanced.register_all(reg)
        tools_fs.register_all(reg)
        _registry = reg
    return _registry


def resolve_credential(
    api_key: Optional[str], api_key_env: Optional[str], base_url: Optional[str]
) -> Optional[str]:
    """Resolve a profile credential, guarding high-value model API keys.

    An explicit ``api_key`` wins. Otherwise read ``api_key_env`` from the
    environment — EXCEPT we refuse to auto-read the well-known
    ``ANTHROPIC_API_KEY`` when a custom ``base_url`` is set (i.e. a local or
    third-party server). A real Anthropic key has monetary value and must
    never be silently sent as a bearer to anything but Claude; a local
    server needs an explicit key in the profile or a deliberately-named env
    var. This prevents, e.g., a blank-key vLLM profile from forwarding the
    operator's Anthropic key to the local box.
    """
    if api_key:
        return api_key
    env_name = (api_key_env or "").strip()
    if not env_name:
        return None
    if base_url and env_name == profiles_store.DEFAULT_API_KEY_ENV:
        return None
    return os.environ.get(env_name) or None


def _resolve_api_key(profile) -> Optional[str]:
    return resolve_credential(profile.api_key, profile.api_key_env, profile.base_url)


# Auto output-token budget. 32K is a sensible default for a verbose
# reasoning model (the Qwen card recommends ~32K general) and is the cap
# even on huge-context models (Gemma 256K, NVIDIA 1M) — the rest of the
# window stays available for the prompt. Overridable per profile.
AUTO_MAX_TOKENS_CAP = 32768
# Used when the context window is unknown (e.g. Claude, which doesn't
# report max_model_len).
AUTO_MAX_TOKENS_FALLBACK = 8192


def _resolve_model_and_context(profile, credential):
    """Return ``(model_id, max_model_len)`` for the active profile.

    Queries the server's model list to (a) pick the model when the profile
    leaves it blank ("weak binding" — vLLM serves one model, so this
    auto-tracks a swap on the box) and (b) learn the context window so the
    output budget can be sized automatically. The probe honors the profile's
    TLS posture (verify / imported cert / off). Context length is best-effort
    (``None`` when the server/provider doesn't report it); only a total
    inability to determine a model raises.
    """
    try:
        models = agent_tls.list_models(
            provider=profile.provider,
            base_url=profile.base_url,
            api_key=credential or "",
            verify_tls=profile.verify_tls,
            ca_cert_pem=profile.ca_cert_pem,
        )
    except Exception:
        models = []
    if profile.model:
        for m in models:
            if m.get("id") == profile.model:
                return profile.model, m.get("max_model_len")
        return profile.model, None
    if not models:
        raise RuntimeError(
            "no model is set on this profile and the server returned no "
            "models — set a model on the agent profile"
        )
    first = models[0]
    return str(first["id"]), first.get("max_model_len")


def _auto_max_tokens(max_model_len: Optional[int]) -> int:
    if max_model_len:
        return min(int(max_model_len), AUTO_MAX_TOKENS_CAP)
    return AUTO_MAX_TOKENS_FALLBACK


def _build_loop(profile) -> AgentLoop:
    if profile.provider != "anthropic":
        raise RuntimeError(
            f"unsupported agent provider {profile.provider!r}; only 'anthropic' "
            "is implemented (Claude or a local vLLM model via base_url)"
        )
    from .providers.anthropic import AnthropicProvider

    credential = _resolve_api_key(profile)
    # Only query the server's model list when we actually need it: to pick
    # the model (profile leaves it blank) or to size auto max_tokens. When
    # both model and max_tokens are explicitly pinned, skip the network probe
    # entirely (avoids a per-activation round-trip that can also block).
    explicit_tokens = int(profile.max_tokens or 0)
    if profile.model and explicit_tokens > 0:
        model, max_model_len = profile.model, None
    else:
        model, max_model_len = _resolve_model_and_context(profile, credential)
    verify = agent_tls.build_verify(
        base_url=profile.base_url,
        verify_tls=profile.verify_tls,
        ca_cert_pem=profile.ca_cert_pem,
    )
    # A custom base_url is a local/self-hosted server (vLLM), which checks
    # ``Authorization: Bearer`` — send the credential as auth_token. Claude
    # (no base_url) uses the x-api-key header — send it as api_key.
    if profile.base_url:
        api_key, auth_token = None, credential
    else:
        api_key, auth_token = credential, None
    # max_tokens: explicit (>0) on the profile, else auto from the model's
    # context window. The provider further clamps per request so output
    # never collides with the (growing) prompt.
    base_max_tokens = explicit_tokens if explicit_tokens > 0 else _auto_max_tokens(max_model_len)
    # Prompt caching: "auto" -> on for Claude (no base_url), off for a custom
    # base_url (vLLM does its own prefix caching and may reject cache_control).
    caching_pref = (profile.prompt_caching or "auto").lower()
    prompt_caching = {"on": True, "off": False}.get(
        caching_pref, not bool(profile.base_url)
    )
    # Disclosure mode: "auto" -> deferred for a custom base_url (local/vLLM,
    # limited context), inline for Claude (large context + prompt caching).
    disclosure_mode = _resolve_disclosure_mode(profile)
    provider = AnthropicProvider(
        model=model,
        api_key=api_key,
        auth_token=auth_token,
        base_url=profile.base_url or None,
        max_tokens=base_max_tokens,
        max_model_len=max_model_len,
        verify=verify,
        prompt_caching=prompt_caching,
    )
    return AgentLoop(
        provider,
        get_registry(),
        system=SYSTEM_PROMPT + _DISCLOSURE_NOTE.get(disclosure_mode, _DISCLOSURE_NOTE["inline"]),
        max_iterations=int(profile.max_iterations or profiles_store.DEFAULT_MAX_ITERATIONS),
        disclosure_mode=disclosure_mode,
    )


def get_loop() -> AgentLoop:
    """Return the loop for the active profile, rebuilding on any change."""
    global _loop, _loop_key
    active = profiles_store.get_active()
    if active is None:
        raise RuntimeError("no agent profile is configured")
    key = (active.id, profiles_store.revision())
    if _loop is None or _loop_key != key:
        _loop = _build_loop(active)
        _loop_key = key
    return _loop
