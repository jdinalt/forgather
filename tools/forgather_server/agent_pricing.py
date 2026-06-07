"""Per-model price estimates for the agent token meter.

These are **estimates** for an in-UI cost readout, NOT a billing source of
truth. The per-message API never returns a dollar figure (only token counts), so
a live per-session cost is necessarily tokens x a price table; Anthropic's
dashboard / Cost Admin API (``/v1/organizations/cost_report``, Admin key) is
authoritative.

Prices are USD per million tokens, ``(input, output)``. Cache prices derive from
Anthropic's uniform multipliers: a 5-minute cache *write* is 1.25x the input
rate, a cache *read* is 0.1x.

Three layers, highest priority first:

1. The user override file (``<config>/server/agent_pricing.json``), edited from
   the webui — hot-reloaded on save, no restart.
2. The server-config ``agent.pricing`` block (startup seed for headless setups).
3. The built-in defaults below.

Models are matched by LONGEST matching id prefix, checked within each layer
before falling through to the next — so a user override of ``claude-opus-4``
wins over the built-in ``claude-opus-4-8`` entry. A model in no layer (e.g. a
self-hosted vLLM model) reports no cost.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from ._atomic import atomic_write_text
from .paths import agent_pricing_file

log = logging.getLogger("forgather_server.agent_pricing")

# Anthropic cache rate multipliers (uniform across models).
CACHE_WRITE_MULT = 1.25  # 5-minute ephemeral cache write
CACHE_READ_MULT = 0.10

# USD per million tokens (input, output), matched by LONGEST id prefix so
# "claude-3-5-haiku" beats "claude-3-haiku". As of 2026-06-07
# (https://platform.claude.com/docs/en/about-claude/pricing) — override in the
# webui or server config when they drift. Notes:
#   - Opus 4.5+ dropped to $5/$25; Opus 4.0/4.1 (deprecated) stay $15/$75, so
#     the minor-version entries are distinct.
#   - Sonnet/Haiku 4.x are uniform within the family, so one prefix each.
#   - Fast mode (Opus) and inference_geo=us (1.1x) are not modeled.
_DEFAULT_PRICES: Dict[str, tuple] = {
    # Opus 4.5-4.8: $5 / $25
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    # Opus 4.0 / 4.1 (deprecated) and Claude 3 Opus: $15 / $75
    "claude-opus-4-1": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-3-opus": (15.0, 75.0),
    # Sonnet 4.x (and 3.7 / 3.5): $3 / $15
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    # Haiku 4.5: $1 / $5
    "claude-haiku-4": (1.0, 5.0),
    # Haiku 3.5: $0.80 / $4 (retired except Bedrock/Vertex)
    "claude-3-5-haiku": (0.80, 4.0),
    # Claude 3 Haiku: $0.25 / $1.25
    "claude-3-haiku": (0.25, 1.25),
}

# Override layers (model-prefix -> (input, output)).
_config_overrides: Dict[str, tuple] = {}  # from server config agent.pricing
_file_overrides: Dict[str, tuple] = {}  # from agent_pricing.json (webui-edited)


def _coerce_table(raw: Optional[Dict[str, Any]]) -> Dict[str, tuple]:
    """Validate a ``{prefix: [input, output]}`` mapping; skip malformed rows."""
    out: Dict[str, tuple] = {}
    if not raw:
        return out
    for key, val in raw.items():
        try:
            inp, outp = val
            out[str(key)] = (float(inp), float(outp))
        except (TypeError, ValueError):
            log.warning("ignoring malformed pricing entry %r: %r", key, val)
    return out


def _load_file() -> None:
    global _file_overrides
    try:
        path = agent_pricing_file()
        if path.exists():
            _file_overrides = _coerce_table(json.loads(path.read_text()))
        else:
            _file_overrides = {}
    except Exception:
        log.exception("failed to load agent pricing overrides; using defaults")
        _file_overrides = {}


def configure(pricing: Optional[Dict[str, Any]]) -> None:
    """Install server-config price overrides and (re)load the override file.

    Called at startup with the ``agent.pricing`` block.
    """
    global _config_overrides
    _config_overrides = _coerce_table(pricing)
    _load_file()


def _match(model: str, table: Dict[str, tuple]) -> Optional[tuple]:
    best: Optional[tuple] = None
    best_len = -1
    for prefix, price in table.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = price, len(prefix)
    return best


def _base_price(model: str) -> Optional[tuple]:
    """``(input, output)`` per Mtok, override layers first, then defaults."""
    if not model:
        return None
    for table in (_file_overrides, _config_overrides, _DEFAULT_PRICES):
        hit = _match(model, table)
        if hit is not None:
            return hit
    return None


def price_for(model: Optional[str]) -> Optional[Dict[str, float]]:
    """Per-Mtok USD rates for the four token categories, or ``None`` if unknown.

    ``None`` (no layer matches) means "don't show a cost" — the right behavior
    for a self-hosted / vLLM model where the tokens are free.
    """
    base = _base_price(model or "")
    if base is None:
        return None
    inp, out = base
    return {
        "input": inp,
        "output": out,
        "cache_read": round(inp * CACHE_READ_MULT, 6),
        "cache_write": round(inp * CACHE_WRITE_MULT, 6),
    }


def estimate_usd(model: Optional[str], totals: Dict[str, int]) -> Optional[float]:
    """Estimated USD for cumulative token ``totals``, or ``None`` if unpriced.

    ``totals`` keys: ``input_tokens``, ``output_tokens``,
    ``cache_read_input_tokens``, ``cache_creation_input_tokens`` (any missing
    treated as 0). Mirrors the frontend computation so a server-side report
    (logs, exports) matches the meter.
    """
    rates = price_for(model)
    if rates is None:
        return None
    per = 1_000_000.0
    usd = (
        (totals.get("input_tokens", 0) or 0) * rates["input"]
        + (totals.get("output_tokens", 0) or 0) * rates["output"]
        + (totals.get("cache_read_input_tokens", 0) or 0) * rates["cache_read"]
        + (totals.get("cache_creation_input_tokens", 0) or 0) * rates["cache_write"]
    ) / per
    return round(usd, 6)


# ---- webui-editable override file ------------------------------------------


def get_overrides() -> Dict[str, list]:
    """The user override table (file layer) as JSON-friendly lists."""
    return {k: [inp, out] for k, (inp, out) in _file_overrides.items()}


def get_defaults() -> Dict[str, list]:
    """The built-in default table (reference for the editor)."""
    return {k: [inp, out] for k, (inp, out) in _DEFAULT_PRICES.items()}


def set_overrides(raw: Optional[Dict[str, Any]]) -> Dict[str, list]:
    """Validate, persist, and hot-reload the user override table.

    Raises ``ValueError`` if any entry is malformed (the editor should surface
    it rather than silently dropping rows the user typed).
    """
    cleaned: Dict[str, tuple] = {}
    for key, val in (raw or {}).items():
        try:
            inp, outp = val
            fi, fo = float(inp), float(outp)
        except (TypeError, ValueError):
            raise ValueError(
                f"pricing entry {key!r} must be [input, output] numbers, got {val!r}"
            )
        if fi < 0 or fo < 0:
            raise ValueError(f"pricing entry {key!r} must be non-negative")
        cleaned[str(key)] = (fi, fo)
    payload = json.dumps(
        {k: [inp, out] for k, (inp, out) in cleaned.items()}, indent=2, sort_keys=True
    )
    atomic_write_text(agent_pricing_file(), payload)
    global _file_overrides
    _file_overrides = cleaned
    return get_overrides()
