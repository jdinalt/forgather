"""Tests for tools/forgather_server/gpu_policy.py."""

import forgather_server.gpu_policy as gpu_policy
import pytest
from forgather_server.gpu_policy import GpuPolicy


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # gpu_policy imports gpu_policy_file from paths; patch the local binding.
    monkeypatch.setattr(
        gpu_policy, "gpu_policy_file", lambda: state_dir / "gpu_policy.json"
    )
    yield state_dir


class TestGetPolicy:
    def test_default_policy_no_restrictions(self):
        p = gpu_policy.get_policy(0)
        assert p.disabled is False
        assert p.min_priority == 0

    def test_default_for_any_index(self):
        p = gpu_policy.get_policy(99)
        assert isinstance(p, GpuPolicy)


class TestSetPolicy:
    def test_set_disabled(self):
        p = gpu_policy.set_policy(0, disabled=True)
        assert p.disabled is True
        assert gpu_policy.get_policy(0).disabled is True

    def test_set_min_priority(self):
        p = gpu_policy.set_policy(1, min_priority=5)
        assert p.min_priority == 5
        assert gpu_policy.get_policy(1).min_priority == 5

    def test_partial_update_preserves_other_field(self):
        gpu_policy.set_policy(2, disabled=True, min_priority=10)
        # Update only disabled; min_priority must stay 10.
        gpu_policy.set_policy(2, disabled=False)
        p = gpu_policy.get_policy(2)
        assert p.disabled is False
        assert p.min_priority == 10

    def test_multiple_gpus_independent(self):
        gpu_policy.set_policy(0, disabled=True)
        gpu_policy.set_policy(1, disabled=False, min_priority=7)
        assert gpu_policy.get_policy(0).disabled is True
        assert gpu_policy.get_policy(1).min_priority == 7
        assert gpu_policy.get_policy(1).disabled is False


class TestAllPolicies:
    def test_empty_when_no_policies(self):
        assert gpu_policy.all_policies() == {}

    def test_returns_set_policies(self):
        gpu_policy.set_policy(3, disabled=True)
        gpu_policy.set_policy(4, min_priority=2)
        policies = gpu_policy.all_policies()
        assert 3 in policies
        assert 4 in policies
        assert policies[3].disabled is True
        assert policies[4].min_priority == 2

    def test_integer_keys(self):
        gpu_policy.set_policy(5, disabled=False)
        policies = gpu_policy.all_policies()
        for key in policies:
            assert isinstance(key, int)


class TestClearPolicy:
    def test_clear_existing(self):
        gpu_policy.set_policy(0, disabled=True)
        ok = gpu_policy.clear_policy(0)
        assert ok is True
        # Should revert to default after clear.
        p = gpu_policy.get_policy(0)
        assert p.disabled is False

    def test_clear_nonexistent_returns_false(self):
        ok = gpu_policy.clear_policy(99)
        assert ok is False

    def test_clear_removes_from_all_policies(self):
        gpu_policy.set_policy(0, disabled=True)
        gpu_policy.clear_policy(0)
        assert 0 not in gpu_policy.all_policies()
