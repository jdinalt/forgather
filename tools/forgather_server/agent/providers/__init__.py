"""Chat provider adapters behind the ``ChatProvider`` seam.

``base`` defines the provider-neutral interface and event types the loop
consumes. ``anthropic`` is the only concrete adapter today (Claude +
local vLLM via base_url). A future ``openai`` adapter drops in here for
non-Anthropic-API providers without touching ``loop.py``.
"""
