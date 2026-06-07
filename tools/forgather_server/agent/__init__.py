"""In-process AI agent for the Forgather server webui.

The agent runs a provider-agnostic tool-use loop (``loop.py``) over an
in-process tool registry (``registry.py``) that wraps the existing
``*_ops.py`` modules. Read-only tools run automatically; mutating tools
return a *preview* and require explicit user approval before the real
write happens (the approval gate is enforced server-side in ``loop.py``,
never in the browser).

Provider access goes through the ``ChatProvider`` seam
(``providers/base.py``) so the loop never imports a vendor SDK directly.
Today the only adapter is Anthropic (``providers/anthropic.py``), which
talks to Claude (``api.anthropic.com``) or to a local vLLM model via
vLLM's native Anthropic Messages API by switching ``base_url``.

Session state (conversation history + pending approvals) lives in an
in-memory singleton (``session.py``) modeled on ``scheduler._state`` —
single-user localhost, ephemeral by design.
"""
