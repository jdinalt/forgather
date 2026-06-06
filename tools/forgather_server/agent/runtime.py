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

from .. import agent_profiles_store as profiles_store
from .. import agent_tls
from .loop import AgentLoop
from .registry import ToolRegistry
from . import tools_authoring, tools_readonly

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
- Prefer read-only tools (list_projects, inspect_config, render_config_pp,
  read_file, search_docs, scheduler_status) to ground every answer in the
  actual project state. Do not guess project_dir / config_name — discover
  them with list_projects.
- To change a file you MUST use a propose_* tool. These do not write
  immediately: they show the user a diff and wait for explicit approval.
  Never claim a change has been made until you receive the tool result
  confirming the write. If the user rejects a change, do not retry the same
  edit — ask what they would prefer.
- When editing an existing file, read_file it first and base your new
  content on what is actually there; pass the full new file content.
- Be concise. When you cite documentation, give the file path.
"""


def configure(cfg: Optional[Dict[str, Any]]) -> None:
    """Seed a bootstrap profile from the server-config ``agent:`` block.

    Only acts when the profile store is empty (first run). Resets the cached
    loop so a fresh process picks up the current active profile.
    """
    global _loop, _loop_key
    _loop = None
    _loop_key = None
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
    }


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        reg = ToolRegistry()
        tools_readonly.register_all(reg)
        tools_authoring.register_all(reg)
        _registry = reg
    return _registry


def _resolve_api_key(profile) -> Optional[str]:
    if profile.api_key:
        return profile.api_key
    env_name = profile.api_key_env or profiles_store.DEFAULT_API_KEY_ENV
    return os.environ.get(env_name)


def _resolve_model(profile, api_key: Optional[str]) -> str:
    """Concrete model name for the provider.

    Honors the profile's stored model; if it is empty ("weak binding"),
    queries the server's model list and uses the first available — vLLM
    serves one model, so this auto-tracks a swap on the box.
    """
    if profile.model:
        return profile.model
    models = agent_tls.list_models(
        provider=profile.provider,
        base_url=profile.base_url,
        api_key=api_key or "",
        verify_tls=profile.verify_tls,
        ca_cert_pem=profile.ca_cert_pem,
    )
    if not models:
        raise RuntimeError(
            "no model is set on this profile and the server returned no "
            "models — set a model on the agent profile"
        )
    return models[0]


def _build_loop(profile) -> AgentLoop:
    if profile.provider != "anthropic":
        raise RuntimeError(
            f"unsupported agent provider {profile.provider!r}; only 'anthropic' "
            "is implemented (Claude or a local vLLM model via base_url)"
        )
    from .providers.anthropic import AnthropicProvider

    api_key = _resolve_api_key(profile)
    model = _resolve_model(profile, api_key)
    verify = agent_tls.build_verify(
        base_url=profile.base_url,
        verify_tls=profile.verify_tls,
        ca_cert_pem=profile.ca_cert_pem,
    )
    provider = AnthropicProvider(
        model=model,
        api_key=api_key,
        base_url=profile.base_url or None,
        max_tokens=int(profile.max_tokens or profiles_store.DEFAULT_MAX_TOKENS),
        verify=verify,
    )
    return AgentLoop(
        provider,
        get_registry(),
        system=SYSTEM_PROMPT,
        max_iterations=int(profile.max_iterations or profiles_store.DEFAULT_MAX_ITERATIONS),
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
