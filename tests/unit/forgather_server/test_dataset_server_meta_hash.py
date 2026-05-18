"""
Unit tests for `tools/dataset_server/hf_cache.py::_compute_meta_hash`.

The hash is the content-equivalence signal that the master uses to
detect collisions in ``local/<name>`` across cluster nodes — same
hash = "these two servers' local/X is the same dataset"; different
hash = "operators have named distinct datasets the same thing." The
exact bytes don't matter (it's never compared against external
state), but the *equivalence semantics* do, so this file locks them
in.
"""

from __future__ import annotations

import pytest

# tools/ isn't on sys.path by default for hf_cache; the conftest adds
# it (same pattern test_dataset_server_wire.py uses).
from dataset_server.hf_cache import _compute_meta_hash


class TestMetaHashEquivalence:
    """Same content → same hash. Different content → different hash.
    Things-that-don't-matter (path, size, config_name) should NOT
    affect the hash."""

    def test_identical_metadata_yields_same_hash(self):
        a = {
            "features": ["text", "label"],
            "splits": [
                {"name": "train", "num_examples": 1000},
                {"name": "test", "num_examples": 100},
            ],
        }
        b = dict(a)  # shallow copy is enough — _compute_meta_hash only reads
        assert _compute_meta_hash(a) == _compute_meta_hash(b)

    def test_feature_order_doesnt_matter(self):
        """Schema is set-equality, not list-equality. Two servers
        ordering their dataset_info.json features differently must
        still hash to the same value."""
        a = {
            "features": ["text", "label", "id"],
            "splits": [{"name": "train", "num_examples": 1000}],
        }
        b = {
            "features": ["id", "label", "text"],
            "splits": [{"name": "train", "num_examples": 1000}],
        }
        assert _compute_meta_hash(a) == _compute_meta_hash(b)

    def test_split_order_doesnt_matter(self):
        a = {
            "features": ["x"],
            "splits": [
                {"name": "train", "num_examples": 100},
                {"name": "test", "num_examples": 10},
            ],
        }
        b = {
            "features": ["x"],
            "splits": [
                {"name": "test", "num_examples": 10},
                {"name": "train", "num_examples": 100},
            ],
        }
        assert _compute_meta_hash(a) == _compute_meta_hash(b)

    def test_path_doesnt_affect_hash(self):
        """Two servers serving the same dataset from different
        filesystem paths must hash to the same value. ``local/<name>``
        is the whole point of the abstraction."""
        a = {
            "path": "/data/host-a/stories",
            "features": ["text"],
            "splits": [{"name": "train", "num_examples": 5}],
        }
        b = {
            "path": "/srv/datasets/stories",
            "features": ["text"],
            "splits": [{"name": "train", "num_examples": 5}],
        }
        assert _compute_meta_hash(a) == _compute_meta_hash(b)

    def test_size_bytes_doesnt_affect_hash(self):
        """A re-compressed copy of the same dataset has different
        on-disk size but is content-equivalent."""
        a = {
            "features": ["x"],
            "splits": [{"name": "train", "num_examples": 100}],
            "size_bytes": 1_000_000,
        }
        b = {
            "features": ["x"],
            "splits": [{"name": "train", "num_examples": 100}],
            "size_bytes": 500_000,
        }
        assert _compute_meta_hash(a) == _compute_meta_hash(b)

    def test_config_name_doesnt_affect_hash(self):
        """``config_name`` is often auto-generated from ``save_to_disk``
        and varies harmlessly. Equivalence shouldn't trip on it."""
        a = {
            "features": ["x"],
            "splits": [{"name": "train", "num_examples": 100}],
            "config_name": "default",
        }
        b = {
            "features": ["x"],
            "splits": [{"name": "train", "num_examples": 100}],
            "config_name": "default_a1b2",
        }
        assert _compute_meta_hash(a) == _compute_meta_hash(b)


class TestMetaHashDistinguishes:
    """Different content → different hash. These are the collision
    cases the master should surface as warnings."""

    def test_different_features_differ(self):
        a = {
            "features": ["text"],
            "splits": [{"name": "train", "num_examples": 100}],
        }
        b = {
            "features": ["text", "label"],
            "splits": [{"name": "train", "num_examples": 100}],
        }
        assert _compute_meta_hash(a) != _compute_meta_hash(b)

    def test_different_split_counts_differ(self):
        a = {
            "features": ["x"],
            "splits": [{"name": "train", "num_examples": 100}],
        }
        b = {
            "features": ["x"],
            "splits": [{"name": "train", "num_examples": 200}],
        }
        assert _compute_meta_hash(a) != _compute_meta_hash(b)

    def test_different_split_names_differ(self):
        a = {
            "features": ["x"],
            "splits": [{"name": "train", "num_examples": 100}],
        }
        b = {
            "features": ["x"],
            "splits": [{"name": "validation", "num_examples": 100}],
        }
        assert _compute_meta_hash(a) != _compute_meta_hash(b)


class TestMetaHashShape:
    def test_returns_stable_16_hex_chars(self):
        h = _compute_meta_hash(
            {"features": ["x"], "splits": [{"name": "train", "num_examples": 1}]}
        )
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_handles_missing_fields_gracefully(self):
        """Unknown-layout dirs come through ``inspect_local_path``
        with empty features + splits — the hash must still be
        computable, even if it's just the well-known "empty" value."""
        empty = _compute_meta_hash({})
        also_empty = _compute_meta_hash({"features": [], "splits": []})
        assert empty == also_empty

    def test_non_dict_splits_skipped_safely(self):
        """A malformed dataset_info.json should not crash the hash;
        it just yields a degenerate value that the operator can
        notice."""
        h = _compute_meta_hash(
            {
                "features": ["x"],
                "splits": [
                    {"name": "train", "num_examples": 100},
                    "not-a-dict",  # should be silently dropped
                ],
            }
        )
        # Same as the single-split-only result.
        baseline = _compute_meta_hash(
            {
                "features": ["x"],
                "splits": [{"name": "train", "num_examples": 100}],
            }
        )
        assert h == baseline
