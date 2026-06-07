"""Per-model price estimates for the agent token meter.

These are **estimates** for an in-UI cost readout, NOT a billing source of
truth. Anthropic prices change; the dashboard / Cost Admin API
(``/v1/organizations/cost_report``, Admin key) is authoritative. The per-message
API never returns a dollar figure — only token counts — so a live per-session
cost is necessarily computed from tokens x a price table.

Prices are USD per million tokens, ``(input, output)``. Cache prices derive from
uniform Anthropic multipliers: a 5-minute cache *write* is 1.25x the input rate,
a cache *read* is 0.1x. Override the table from the server config
``agent.pricing`` block (keyed by model-id prefix) when rates drift; local /
vLLM models (no table match) simply report no cost.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Anthropic cache rate multipliers (uniform across models).
CACHE_WRITE_MULT = 1.25  # 5-minute ephemeral cache write
CACHE_READ_MULT = 0.10

# USD per million tokens (input, output). Matched to a model id by LONGEST
# matching prefix, so "claude-3-5-haiku" wins over "claude-3". As of 2026-01 —
# verify against https://www.anthropic.com/pricing and override in server config
# if they have changed.
_DEFAULT_PRICES: Dict[str, tuple] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-haiku": (0.25, 1.25),
}

# Server-config overrides, merged over the defaults (config wins).
_overrides: Dict[str, tuple] = {}


def configure(pricing: Optional[Dict[str, Any]]) -> None:
    """Install server-config price overrides (``agent.pricing`` block).

    Each entry maps a model-id prefix to ``[input, output]`` USD per million
    tokens. Malformed entries are ignored so a typo can't break the agent.
    """
    _overrides.clear()
    if not pricing:
        return
    for key, val in pricing.items():
        try:
            inp, out = val  # list/tuple of two numbers
            _overrides[str(key)] = (float(inp), float(out))
        except (TypeError, ValueError):
            continue


def _base_price(model: str) -> Optional[tuple]:
    """``(input, output)`` per Mtok for ``model`` by longest-prefix match."""
    if not model:
        return None
    table = {**_DEFAULT_PRICES, **_overrides}
    best: Optional[tuple] = None
    best_len = -1
    for prefix, price in table.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = price, len(prefix)
    return best


def price_for(model: Optional[str]) -> Optional[Dict[str, float]]:
    """Per-Mtok USD rates for the four token categories, or ``None`` if unknown.

    ``None`` (no table match) means "don't show a cost" — the right behavior for
    a self-hosted / vLLM model where the tokens are free.
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
