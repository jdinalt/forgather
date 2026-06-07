"""Price-table estimate for the agent token meter (estimates, not billing)."""

from __future__ import annotations

import json

import pytest

from forgather_server import agent_pricing


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    # Reset both override layers and point the override file at a temp path so
    # tests never read or write the real ~/.config file.
    monkeypatch.setattr(agent_pricing, "_file_overrides", {})
    monkeypatch.setattr(agent_pricing, "_config_overrides", {})
    monkeypatch.setattr(
        agent_pricing, "agent_pricing_file", lambda: tmp_path / "agent_pricing.json"
    )
    yield


def test_price_for_longest_prefix_wins():
    # "claude-3-5-haiku" must beat "claude-3-haiku" / "claude-3".
    p = agent_pricing.price_for("claude-3-5-haiku-20241022")
    assert p["input"] == 0.80 and p["output"] == 4.0


def test_opus_minor_versions_priced_distinctly():
    # 4.5+ dropped to $5/$25; 4.0/4.1 (deprecated) stay $15/$75.
    assert agent_pricing.price_for("claude-opus-4-8")["input"] == 5.0
    assert agent_pricing.price_for("claude-opus-4-5")["output"] == 25.0
    assert agent_pricing.price_for("claude-opus-4-1-20250805")["input"] == 15.0
    assert agent_pricing.price_for("claude-opus-4-20250514")["input"] == 15.0


def test_price_for_derives_cache_rates():
    p = agent_pricing.price_for("claude-sonnet-4-6")
    assert p["input"] == 3.0
    assert p["cache_read"] == pytest.approx(0.30)  # 0.1x input
    assert p["cache_write"] == pytest.approx(3.75)  # 1.25x input


def test_price_for_unknown_model_is_none():
    # A self-hosted / vLLM model isn't in the table -> no cost shown.
    assert agent_pricing.price_for("qwen3-35b") is None
    assert agent_pricing.price_for(None) is None


def test_config_override_wins_over_default():
    agent_pricing.configure({"claude-opus-4": [20.0, 100.0]})
    p = agent_pricing.price_for("claude-opus-4-8")
    assert p["input"] == 20.0 and p["output"] == 100.0
    assert p["cache_write"] == pytest.approx(25.0)


def test_config_override_ignores_malformed():
    agent_pricing.configure({"claude-opus-4": "nonsense", "claude-haiku-4": [2, 8]})
    # Bad entry skipped (default used); good entry applied.
    assert agent_pricing.price_for("claude-opus-4-8")["input"] == 5.0  # default
    assert agent_pricing.price_for("claude-haiku-4-5")["input"] == 2.0  # override


def test_estimate_usd_weights_categories():
    # 1M of each category on Sonnet: 3.0 + 0.30 + 3.75 + 15.0.
    totals = {
        "input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    usd = agent_pricing.estimate_usd("claude-sonnet-4-6", totals)
    assert usd == pytest.approx(22.05)


def test_estimate_usd_unknown_model_none():
    assert agent_pricing.estimate_usd("qwen3-35b", {"input_tokens": 5}) is None


# ---- webui-editable override file ------------------------------------------


def test_set_overrides_persists_reloads_and_wins(tmp_path):
    out = agent_pricing.set_overrides({"claude-opus-4-8": [7.0, 30.0]})
    assert out == {"claude-opus-4-8": [7.0, 30.0]}
    # File layer beats the built-in default for the same model.
    assert agent_pricing.price_for("claude-opus-4-8")["input"] == 7.0
    assert agent_pricing.get_overrides() == {"claude-opus-4-8": [7.0, 30.0]}
    # Persisted to disk as JSON.
    on_disk = json.loads((tmp_path / "agent_pricing.json").read_text())
    assert on_disk == {"claude-opus-4-8": [7.0, 30.0]}


def test_file_overrides_win_over_config():
    agent_pricing.configure({"claude-haiku-4": [2.0, 8.0]})
    agent_pricing.set_overrides({"claude-haiku-4": [9.0, 40.0]})
    assert agent_pricing.price_for("claude-haiku-4-5")["input"] == 9.0


def test_set_overrides_rejects_malformed():
    for bad in ({"x": [1]}, {"x": ["a", "b"]}, {"x": [-1, 5]}):
        with pytest.raises(ValueError):
            agent_pricing.set_overrides(bad)


def test_configure_loads_existing_file(tmp_path):
    (tmp_path / "agent_pricing.json").write_text(json.dumps({"claude-opus-4-8": [6, 26]}))
    agent_pricing.configure(None)  # startup reload picks up the file
    assert agent_pricing.price_for("claude-opus-4-8")["input"] == 6.0
    assert agent_pricing.price_for("claude-opus-4-8")["output"] == 26.0


# ---- route wrappers --------------------------------------------------------


def test_pricing_endpoints_round_trip():
    from forgather_server.routes import agent as agent_routes

    got = agent_routes.get_agent_pricing()
    assert got["overrides"] == {}
    assert "claude-opus-4-8" in got["defaults"]  # reference table present

    saved = agent_routes.put_agent_pricing(
        agent_routes.PricingWrite(overrides={"claude-opus-4-8": [7.0, 30.0]})
    )
    assert saved["overrides"] == {"claude-opus-4-8": [7.0, 30.0]}
    assert agent_pricing.price_for("claude-opus-4-8")["input"] == 7.0


def test_pricing_endpoint_rejects_malformed():
    from fastapi import HTTPException

    from forgather_server.routes import agent as agent_routes

    with pytest.raises(HTTPException) as ei:
        agent_routes.put_agent_pricing(
            agent_routes.PricingWrite(overrides={"claude-opus-4-8": [1]})
        )
    assert ei.value.status_code == 400
