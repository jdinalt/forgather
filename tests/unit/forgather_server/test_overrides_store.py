"""Tests for tools/forgather_server/overrides_store.py."""

import forgather_server.overrides_store as overrides_store
import pytest


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # overrides_store imports overrides_dir from paths; patch the local binding.
    overrides = state_dir / "overrides"
    overrides.mkdir()
    monkeypatch.setattr(overrides_store, "overrides_dir", lambda: overrides)
    yield state_dir


class TestGetOverrides:
    def test_missing_returns_empty(self):
        result = overrides_store.get_overrides("/proj", "config.yaml")
        assert result == {}

    def test_returns_stored_values(self):
        overrides_store.set_overrides("/proj", "config.yaml", {"lr": 1e-3})
        result = overrides_store.get_overrides("/proj", "config.yaml")
        assert result == {"lr": 1e-3}


class TestSetOverrides:
    def test_sets_and_returns_payload(self):
        payload = overrides_store.set_overrides("/proj", "cfg.yaml", {"x": 42})
        assert payload["values"] == {"x": 42}
        assert payload["updated_at"] is not None

    def test_overwrites_previous(self):
        overrides_store.set_overrides("/proj", "cfg.yaml", {"a": 1})
        overrides_store.set_overrides("/proj", "cfg.yaml", {"b": 2})
        result = overrides_store.get_overrides("/proj", "cfg.yaml")
        assert result == {"b": 2}

    def test_different_configs_independent(self):
        overrides_store.set_overrides("/proj", "a.yaml", {"k": 1})
        overrides_store.set_overrides("/proj", "b.yaml", {"k": 99})
        assert overrides_store.get_overrides("/proj", "a.yaml") == {"k": 1}
        assert overrides_store.get_overrides("/proj", "b.yaml") == {"k": 99}


class TestClearOverrides:
    def test_clear_existing_returns_true(self):
        overrides_store.set_overrides("/proj", "cfg.yaml", {"x": 1})
        ok = overrides_store.clear_overrides("/proj", "cfg.yaml")
        assert ok is True

    def test_clear_missing_returns_false(self):
        ok = overrides_store.clear_overrides("/proj", "nonexistent.yaml")
        assert ok is False

    def test_clear_removes_data(self):
        overrides_store.set_overrides("/proj", "cfg.yaml", {"lr": 1e-3})
        overrides_store.clear_overrides("/proj", "cfg.yaml")
        result = overrides_store.get_overrides("/proj", "cfg.yaml")
        assert result == {}


class TestGetOverridesPayload:
    def test_missing_returns_null_stub(self):
        p = overrides_store.get_overrides_payload("/proj", "cfg.yaml")
        assert p["values"] == {}
        assert p["updated_at"] is None

    def test_returns_full_payload(self):
        overrides_store.set_overrides("/proj", "cfg.yaml", {"x": 5})
        p = overrides_store.get_overrides_payload("/proj", "cfg.yaml")
        assert p["values"] == {"x": 5}
        assert isinstance(p["updated_at"], float)
