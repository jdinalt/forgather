"""Tests for ``preprocess._resolve_partition_method``.

Covers the validity matrix that gates ``shard_dataset.method``
selection against whether DiLoCo is active in the environment.
"""

from __future__ import annotations

import pytest

from forgather.ml.datasets.preprocess import _resolve_partition_method

# ---------------------------------------------------------------------------
# Legacy shapes — bool / None / explicit dict
# ---------------------------------------------------------------------------


class TestLegacyShapes:
    def test_none_is_noop(self):
        assert _resolve_partition_method(None, has_diloco=False) == (None, None)
        assert _resolve_partition_method(None, has_diloco=True) == (None, None)

    def test_false_is_noop(self):
        assert _resolve_partition_method(False, has_diloco=False) == (None, None)
        # Under DiLoCo, False is fine — used by the eval path so every
        # worker runs the full eval pass.
        assert _resolve_partition_method(False, has_diloco=True) == (None, None)

    def test_bool_true_picks_conventional(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        method, kwargs = _resolve_partition_method(True, has_diloco=False)
        assert method == "conventional"
        assert kwargs == {"num_shards": 4, "index": 2}

    def test_explicit_dict_legacy_form(self):
        method, kwargs = _resolve_partition_method(
            {"num_shards": 8, "index": 3}, has_diloco=False
        )
        assert method == "conventional"
        assert kwargs == {"num_shards": 8, "index": 3}


# ---------------------------------------------------------------------------
# New method shape
# ---------------------------------------------------------------------------


class TestNewMethodShape:
    def test_explicit_conventional(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "1")
        method, kwargs = _resolve_partition_method(
            {"method": "conventional"}, has_diloco=False
        )
        assert method == "conventional"
        assert kwargs == {"num_shards": 4, "index": 1}

    def test_explicit_conventional_with_overrides(self):
        method, kwargs = _resolve_partition_method(
            {"method": "conventional", "num_shards": 16, "index": 7},
            has_diloco=False,
        )
        assert method == "conventional"
        assert kwargs == {"num_shards": 16, "index": 7}

    def test_work_units(self):
        method, kwargs = _resolve_partition_method(
            {"method": "work_units"}, has_diloco=True
        )
        assert method == "work_units"
        assert kwargs is None

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown shard_dataset.method"):
            _resolve_partition_method({"method": "bogus"}, has_diloco=False)


# ---------------------------------------------------------------------------
# Validity matrix — the two error cells
# ---------------------------------------------------------------------------


class TestValidityMatrix:
    def test_work_units_without_diloco_raises(self):
        with pytest.raises(ValueError, match="DILOCO_SERVER"):
            _resolve_partition_method({"method": "work_units"}, has_diloco=False)

    def test_conventional_dict_with_diloco_raises(self):
        with pytest.raises(ValueError, match="asymmetric-DDP"):
            _resolve_partition_method({"method": "conventional"}, has_diloco=True)

    def test_legacy_bool_true_with_diloco_raises(self):
        """``True`` (legacy form) under DiLoCo is still rejected — same
        asymmetric-DDP problem regardless of which shape the operator
        used."""
        with pytest.raises(ValueError, match="asymmetric-DDP"):
            _resolve_partition_method(True, has_diloco=True)

    def test_legacy_explicit_dict_with_diloco_raises(self):
        with pytest.raises(ValueError, match="asymmetric-DDP"):
            _resolve_partition_method({"num_shards": 4, "index": 0}, has_diloco=True)


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------


class TestTypeChecking:
    def test_non_dict_non_bool_raises(self):
        with pytest.raises(TypeError):
            _resolve_partition_method("conventional", has_diloco=False)
        with pytest.raises(TypeError):
            _resolve_partition_method(42, has_diloco=False)
