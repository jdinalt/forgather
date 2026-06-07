"""Price-table estimate for the agent token meter (estimates, not billing)."""

from __future__ import annotations

import pytest

from forgather_server import agent_pricing


@pytest.fixture(autouse=True)
def _reset_overrides():
    agent_pricing.configure(None)  # clear any leaked overrides
    yield
    agent_pricing.configure(None)


def test_price_for_longest_prefix_wins():
    # "claude-3-5-haiku" must beat "claude-3-haiku" / "claude-3".
    p = agent_pricing.price_for("claude-3-5-haiku-20241022")
    assert p["input"] == 0.80 and p["output"] == 4.0


def test_price_for_derives_cache_rates():
    p = agent_pricing.price_for("claude-sonnet-4-6")
    assert p["input"] == 3.0
    assert p["cache_read"] == pytest.approx(0.30)  # 0.1x input
    assert p["cache_write"] == pytest.approx(3.75)  # 1.25x input


def test_price_for_unknown_model_is_none():
    # A self-hosted / vLLM model isn't in the table -> no cost shown.
    assert agent_pricing.price_for("qwen3-35b") is None
    assert agent_pricing.price_for(None) is None


def test_config_override_wins():
    agent_pricing.configure({"claude-opus-4": [20.0, 100.0]})
    p = agent_pricing.price_for("claude-opus-4-8")
    assert p["input"] == 20.0 and p["output"] == 100.0
    assert p["cache_write"] == pytest.approx(25.0)


def test_config_override_ignores_malformed():
    agent_pricing.configure({"claude-opus-4": "nonsense", "claude-haiku-4": [2, 8]})
    # Bad entry skipped (default used); good entry applied.
    assert agent_pricing.price_for("claude-opus-4-8")["input"] == 15.0
    assert agent_pricing.price_for("claude-haiku-4-5")["input"] == 2.0


def test_estimate_usd_weights_categories():
    # 1M fresh input + 1M cache read + 1M cache write + 1M output on Sonnet.
    totals = {
        "input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    usd = agent_pricing.estimate_usd("claude-sonnet-4-6", totals)
    # 3.0 + 0.30 + 3.75 + 15.0
    assert usd == pytest.approx(22.05)


def test_estimate_usd_unknown_model_none():
    assert agent_pricing.estimate_usd("qwen3-35b", {"input_tokens": 5}) is None
