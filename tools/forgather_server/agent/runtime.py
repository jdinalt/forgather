"""Agent runtime: config injection, the global tool registry, and factories.

``server.py`` calls :func:`configure` at startup with the ``agent:`` block
from ``server_config`` (which ``args_defaults`` does not touch — it only
normalizes ``args:``). Everything downstream reads config through here, the
same module-global injection idiom ``server.py`` uses for
``DOCS_LANDING_OVERRIDE`` and ``meta_templates.configure_roots``.

The tool registry is built once (read-only + authoring tools) and is the
single source of truth a future ``forgather mcp`` server would re-export.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .loop import AgentLoop
from .registry import ToolRegistry
from . import tools_authoring, tools_readonly

log = logging.getLogger("forgather_server.agent.runtime")

# Populated by configure(). Empty dict => agent disabled (no provider config).
_config: Dict[str, Any] = {}
_registry: Optional[ToolRegistry] = None
_loop: Optional[AgentLoop] = None

DEFAULT_PROVIDER = "anthropic"
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"

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
    """Install the ``agent:`` config block. Idempotent; resets factories."""
    global _config, _registry, _loop
    _config = dict(cfg or {})
    _registry = None
    _loop = None
    if is_enabled():
        log.info(
            "agent enabled: provider=%s model=%s base_url=%s",
            _config.get("provider", DEFAULT_PROVIDER),
            _config.get("model"),
            _config.get("base_url") or "(default)",
        )
    else:
        log.info("agent disabled (no agent.model configured)")


def is_enabled() -> bool:
    """The agent is enabled once a model is configured."""
    return bool(_config.get("model"))


def status() -> Dict[str, Any]:
    """Non-secret status for the webui (never includes the API key)."""
    return {
        "enabled": is_enabled(),
        "provider": _config.get("provider", DEFAULT_PROVIDER) if is_enabled() else None,
        "model": _config.get("model") if is_enabled() else None,
        "base_url": _config.get("base_url") if is_enabled() else None,
    }


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        reg = ToolRegistry()
        tools_readonly.register_all(reg)
        tools_authoring.register_all(reg)
        _registry = reg
    return _registry


def _resolve_api_key() -> Optional[str]:
    # Explicit key wins (handy for local vLLM bearer); otherwise read the
    # named env var (defaults to ANTHROPIC_API_KEY for Claude).
    if _config.get("api_key"):
        return str(_config["api_key"])
    env_name = _config.get("api_key_env", DEFAULT_API_KEY_ENV)
    return os.environ.get(env_name)


def _build_provider():
    provider = _config.get("provider", DEFAULT_PROVIDER)
    if provider != "anthropic":
        raise RuntimeError(
            f"unsupported agent provider {provider!r}; only 'anthropic' is "
            "implemented (Claude or a local vLLM model via base_url)"
        )
    from .providers.anthropic import AnthropicProvider

    kwargs: Dict[str, Any] = {
        "model": _config["model"],
        "api_key": _resolve_api_key(),
        "base_url": _config.get("base_url"),
    }
    if _config.get("max_tokens"):
        kwargs["max_tokens"] = int(_config["max_tokens"])
    return AnthropicProvider(**kwargs)


def get_loop() -> AgentLoop:
    """Return the shared loop. Raises if the agent is not configured."""
    global _loop
    if not is_enabled():
        raise RuntimeError("agent is not configured (set agent.model in server config)")
    if _loop is None:
        _loop = AgentLoop(
            _build_provider(),
            get_registry(),
            system=_config.get("system_prompt") or SYSTEM_PROMPT,
            max_iterations=int(_config.get("max_iterations", 12)),
        )
    return _loop
