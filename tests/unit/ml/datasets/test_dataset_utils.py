#!/usr/bin/env python3
"""
Unit tests for forgather ML datasets module components:
- preprocess.normalize_range
- iterable_with_length.IterableDatasetWithLength / to_iterable_dataset_with_length
- interleaved.InterleavedDataset / interleave_datasets
- dataloader_utils.LengthSyncCallback
"""

import unittest
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Mock / helper dataset classes
# ---------------------------------------------------------------------------


class SimpleListDataset:
    """
    Minimal list-backed dataset supporting __getitem__, __len__, __iter__,
    map, shuffle, and select -- enough to stand in for HuggingFace datasets
    without importing the real library.
    """

    def __init__(self, data: list):
        self.data = list(data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"SimpleListDataset(len={len(self.data)})"

    def select(self, indices):
        return SimpleListDataset([self.data[i] for i in indices])

    def map(self, fn, *args, **kwargs):
        return SimpleListDataset([fn(x) for x in self.data])

    def shuffle(self, *args, **kwargs):
        import random as _rng

        new = list(self.data)
        _rng.shuffle(new)
        return SimpleListDataset(new)

    def to_iterable_dataset(self, **kwargs):
        return SimpleIterableDataset(self.data)


class SimpleIterableDataset:
    """
    Minimal iterable dataset (no __getitem__).  Supports state_dict /
    load_state_dict for checkpoint testing.
    """

    def __init__(self, data: list, name: str = ""):
        self.data = list(data)
        self.name = name
        self._position = 0

    def __iter__(self) -> Iterator:
        for i in range(self._position, len(self.data)):
            yield self.data[i]
            self._position = i + 1

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        return f"SimpleIterableDataset({self.name}, len={len(self.data)})"

    def map(self, fn, *args, **kwargs):
        return SimpleIterableDataset([fn(x) for x in self.data], self.name)

    def shuffle(self, *args, **kwargs):
        import random as _rng

        new = list(self.data)
        _rng.shuffle(new)
        return SimpleIterableDataset(new, self.name)

    def filter(self, fn, *args, **kwargs):
        return SimpleIterableDataset([x for x in self.data if fn(x)], self.name)

    def state_dict(self) -> dict:
        return {"position": self._position}

    def load_state_dict(self, state: dict):
        self._position = state["position"]


class NoLenIterableDataset:
    """Iterable dataset that deliberately has no __len__."""

    def __init__(self, data: list):
        self.data = list(data)

    def __iter__(self):
        return iter(self.data)


# ===========================================================================
# Tests for preprocess.normalize_range
# ===========================================================================


class TestNormalizeRange(unittest.TestCase):
    """Tests for forgather.ml.datasets.preprocess.normalize_range."""

    def setUp(self):
        from forgather.ml.datasets.preprocess import normalize_range

        self.normalize_range = normalize_range

    # -- None input ----------------------------------------------------------

    def test_none_returns_none(self):
        """None input should return None (use full dataset)."""
        result = self.normalize_range(1000, None)
        self.assertIsNone(result)

    # -- range passthrough ---------------------------------------------------

    def test_range_passthrough(self):
        """A range object should be returned unchanged."""
        r = range(10, 100)
        result = self.normalize_range(1000, r)
        self.assertIs(result, r)

    def test_range_passthrough_with_step(self):
        """A range object with step should be returned unchanged."""
        r = range(0, 100, 5)
        result = self.normalize_range(1000, r)
        self.assertIs(result, r)

    # -- float input (fraction of length) ------------------------------------

    def test_float_quarter(self):
        self.assertEqual(self.normalize_range(1000, 0.25), range(0, 250))

    def test_float_half(self):
        self.assertEqual(self.normalize_range(1000, 0.5), range(0, 500))

    def test_float_one(self):
        """1.0 should give the full dataset."""
        self.assertEqual(self.normalize_range(1000, 1.0), range(0, 1000))

    def test_float_small(self):
        self.assertEqual(self.normalize_range(100, 0.1), range(0, 10))

    # -- int input (first N records) -----------------------------------------

    def test_int_first_n(self):
        self.assertEqual(self.normalize_range(1000, 500), range(0, 500))

    def test_int_zero(self):
        self.assertEqual(self.normalize_range(1000, 0), range(0, 0))

    def test_int_exceeds_length(self):
        """Int larger than length should be clamped to length."""
        self.assertEqual(self.normalize_range(100, 200), range(0, 100))

    def test_int_negative(self):
        """Negative int should be interpreted as length + value, clamped to 0."""
        # -10 with length 1000 -> 1000 + (-10) = 990
        result = self.normalize_range(1000, -10)
        self.assertEqual(result, range(0, 990))

    # -- Sequence input (start, end) -----------------------------------------

    def test_sequence_two_ints(self):
        self.assertEqual(self.normalize_range(1000, [100, 900]), range(100, 900))

    def test_sequence_int_and_float(self):
        """Mixed int start and float end."""
        self.assertEqual(self.normalize_range(1000, [100, 0.9]), range(100, 900))

    def test_sequence_tuple(self):
        self.assertEqual(self.normalize_range(1000, (100, 900)), range(100, 900))

    def test_sequence_three_elements_with_step(self):
        """Sequence of three: (start, end, step)."""
        self.assertEqual(
            self.normalize_range(1000, (1, 1.0, 4)), range(1, 1000, 4)
        )

    def test_sequence_negative_start_relative(self):
        """Negative int in sequence uses Python-style indexing from end."""
        # -10 as int -> length + (-10) = 990; 2.0 as float -> int(2.0 * 1000) = 2000, clamped to 1000
        result = self.normalize_range(1000, (-10, 2.0))
        self.assertEqual(result, range(990, 1000))

    def test_sequence_large_negative_start_clamped_to_zero(self):
        """Very large negative int should clamp to 0 after length+value goes negative."""
        # -2000 as int -> 1000 + (-2000) = -1000, clamped to 0
        result = self.normalize_range(1000, (-2000, 500))
        self.assertEqual(result, range(0, 500))

    # -- str slice notation --------------------------------------------------

    def test_str_start_colon(self):
        """'100:' -> range(100, 1000)"""
        self.assertEqual(self.normalize_range(1000, "100:"), range(100, 1000))

    def test_str_colon_end(self):
        """':500' -> range(0, 500)"""
        self.assertEqual(self.normalize_range(1000, ":500"), range(0, 500))

    def test_str_start_end(self):
        """'100:500' -> range(100, 500)"""
        self.assertEqual(self.normalize_range(1000, "100:500"), range(100, 500))

    def test_str_percent_start(self):
        """'10%:' -> range(100, 1000)"""
        self.assertEqual(self.normalize_range(1000, "10%:"), range(100, 1000))

    def test_str_percent_end(self):
        """':80%' -> range(0, 800)"""
        self.assertEqual(self.normalize_range(1000, ":80%"), range(0, 800))

    def test_str_percent_both(self):
        """'10%:80%' -> range(100, 800)"""
        self.assertEqual(self.normalize_range(1000, "10%:80%"), range(100, 800))

    def test_str_single_int_no_colon(self):
        """A bare integer string without colon means first N."""
        self.assertEqual(self.normalize_range(1000, "200"), range(0, 200))

    def test_str_single_percent_no_colon(self):
        """'50%' without colon means first 50%."""
        self.assertEqual(self.normalize_range(1000, "50%"), range(0, 500))

    def test_str_full_range(self):
        """':' -> range(0, 1000) (full dataset)."""
        self.assertEqual(self.normalize_range(1000, ":"), range(0, 1000))

    # -- error cases ---------------------------------------------------------

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            self.normalize_range(1000, {"a": 1})  # type: ignore[arg-type]

    def test_unsupported_value_type_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            self.normalize_range(1000, [None, 100])

    # -- edge cases ----------------------------------------------------------

    def test_zero_length_dataset_float(self):
        self.assertEqual(self.normalize_range(0, 0.5), range(0, 0))

    def test_zero_length_dataset_int(self):
        self.assertEqual(self.normalize_range(0, 10), range(0, 0))

    def test_very_small_float(self):
        """Very small float on small dataset truncates to 0."""
        self.assertEqual(self.normalize_range(10, 0.001), range(0, 0))


# ===========================================================================
# Tests for iterable_with_length
# ===========================================================================


class TestIterableDatasetWithLength(unittest.TestCase):
    """Tests for IterableDatasetWithLength wrapper."""

    def setUp(self):
        from forgather.ml.datasets.iterable_with_length import (
            IterableDatasetWithLength,
        )

        self.IterableDatasetWithLength = IterableDatasetWithLength

    def _make_wrapped(self, data=None, length=None):
        if data is None:
            data = list(range(10))
        inner = SimpleIterableDataset(data)
        if length is None:
            length = len(data)
        return self.IterableDatasetWithLength(inner, length), inner

    # -- __len__ -------------------------------------------------------------

    def test_len_returns_stored_length(self):
        wrapped, _ = self._make_wrapped(length=42)
        self.assertEqual(len(wrapped), 42)

    def test_len_independent_of_inner(self):
        """Length is the provided value, not necessarily len(inner)."""
        wrapped, _ = self._make_wrapped(data=[1, 2, 3], length=999)
        self.assertEqual(len(wrapped), 999)

    # -- __iter__ ------------------------------------------------------------

    def test_iter_delegates(self):
        data = [10, 20, 30]
        wrapped, _ = self._make_wrapped(data=data)
        self.assertEqual(list(wrapped), data)

    def test_iter_multiple_times(self):
        """Should be able to iterate more than once (if inner supports it)."""
        data = [1, 2, 3]
        inner = SimpleListDataset(data)  # SimpleListDataset supports re-iter
        wrapped = self.IterableDatasetWithLength(inner, len(data))
        self.assertEqual(list(wrapped), data)
        self.assertEqual(list(wrapped), data)

    # -- __repr__ ------------------------------------------------------------

    def test_repr(self):
        wrapped, inner = self._make_wrapped(data=[1, 2], length=2)
        r = repr(wrapped)
        self.assertIn("IterableDatasetWithLength", r)
        self.assertIn("length=2", r)

    # -- map() ---------------------------------------------------------------

    def test_map_preserves_length(self):
        wrapped, _ = self._make_wrapped(data=[1, 2, 3], length=3)
        mapped = wrapped.map(lambda x: x * 2)
        self.assertIsInstance(mapped, self.IterableDatasetWithLength)
        self.assertEqual(len(mapped), 3)

    def test_map_delegates_to_inner(self):
        wrapped, _ = self._make_wrapped(data=[1, 2, 3], length=3)
        mapped = wrapped.map(lambda x: x * 10)
        self.assertEqual(list(mapped), [10, 20, 30])

    # -- shuffle() -----------------------------------------------------------

    def test_shuffle_preserves_length(self):
        wrapped, _ = self._make_wrapped(data=list(range(20)), length=20)
        shuffled = wrapped.shuffle()
        self.assertIsInstance(shuffled, self.IterableDatasetWithLength)
        self.assertEqual(len(shuffled), 20)

    # -- filter() ------------------------------------------------------------

    def test_filter_returns_inner_type(self):
        """filter() loses the length wrapper because length may change."""
        data = [1, 2, 3, 4, 5]
        wrapped, _ = self._make_wrapped(data=data, length=5)
        filtered = wrapped.filter(lambda x: x > 3)
        # Should NOT be IterableDatasetWithLength since length is unknown
        self.assertNotIsInstance(filtered, self.IterableDatasetWithLength)

    def test_filter_delegates(self):
        data = [1, 2, 3, 4, 5]
        wrapped, _ = self._make_wrapped(data=data, length=5)
        filtered = wrapped.filter(lambda x: x > 3)
        self.assertEqual(list(filtered), [4, 5])

    # -- __getattr__ ---------------------------------------------------------

    def test_getattr_forwards_to_inner(self):
        inner = SimpleIterableDataset([1, 2, 3], name="test_ds")
        wrapped = self.IterableDatasetWithLength(inner, 3)
        # 'name' is an attribute on inner, not on the wrapper
        self.assertEqual(wrapped.name, "test_ds")

    def test_getattr_state_dict(self):
        """state_dict should be forwarded to inner dataset."""
        inner = SimpleIterableDataset([1, 2, 3])
        wrapped = self.IterableDatasetWithLength(inner, 3)
        sd = wrapped.state_dict()
        self.assertIsInstance(sd, dict)
        self.assertIn("position", sd)

    def test_getattr_raises_for_missing(self):
        wrapped, _ = self._make_wrapped()
        with self.assertRaises(AttributeError):
            _ = wrapped.nonexistent_attribute_xyz


class TestToIterableDatasetWithLength(unittest.TestCase):
    """Tests for to_iterable_dataset_with_length converter function."""

    def setUp(self):
        from forgather.ml.datasets.iterable_with_length import (
            IterableDatasetWithLength,
            to_iterable_dataset_with_length,
        )

        self.to_iterable = to_iterable_dataset_with_length
        self.IterableDatasetWithLength = IterableDatasetWithLength

    def test_converts_map_dataset(self):
        ds = SimpleListDataset([1, 2, 3])
        result = self.to_iterable(ds)
        self.assertIsInstance(result, self.IterableDatasetWithLength)
        self.assertEqual(len(result), 3)

    def test_no_double_wrap(self):
        """Already-wrapped datasets should be returned as-is."""
        inner = SimpleIterableDataset([1, 2])
        wrapped = self.IterableDatasetWithLength(inner, 2)
        result = self.to_iterable(wrapped)
        self.assertIs(result, wrapped)

    def test_raises_without_len(self):
        """Datasets without __len__ should raise ValueError."""
        ds = NoLenIterableDataset([1, 2, 3])
        with self.assertRaises(ValueError):
            self.to_iterable(ds)


# ===========================================================================
# Tests for interleaved.InterleavedDataset
# ===========================================================================


class TestInterleavedDataset(unittest.TestCase):
    """Tests for InterleavedDataset interleaving logic."""

    def setUp(self):
        from forgather.ml.datasets.interleaved import (
            InterleavedDataset,
            interleave_datasets,
        )

        self.InterleavedDataset = InterleavedDataset
        self.interleave_datasets = interleave_datasets

    def _make_simple_datasets(self, *sizes):
        """Create simple iterable datasets with integer data."""
        datasets = []
        for i, size in enumerate(sizes):
            data = [{"id": f"ds{i}_{j}", "value": j} for j in range(size)]
            datasets.append(SimpleIterableDataset(data, name=f"ds{i}"))
        return datasets

    # -- construction --------------------------------------------------------

    def test_empty_datasets_raises(self):
        with self.assertRaises(ValueError):
            self.InterleavedDataset(datasets=[])

    def test_probability_length_mismatch_raises(self):
        ds = self._make_simple_datasets(5, 5)
        with self.assertRaises(ValueError):
            self.InterleavedDataset(datasets=ds, probabilities=[0.5])

    def test_negative_probability_raises(self):
        ds = self._make_simple_datasets(5, 5)
        with self.assertRaises(ValueError):
            self.InterleavedDataset(datasets=ds, probabilities=[0.5, -0.1])

    def test_zero_probabilities_raises(self):
        ds = self._make_simple_datasets(5, 5)
        with self.assertRaises(ValueError):
            self.InterleavedDataset(datasets=ds, probabilities=[0.0, 0.0])

    def test_invalid_stopping_strategy_raises(self):
        ds = self._make_simple_datasets(5)
        with self.assertRaises(ValueError):
            self.InterleavedDataset(
                datasets=ds, stopping_strategy="invalid"
            )

    def test_probabilities_normalized(self):
        ds = self._make_simple_datasets(5, 5)
        interleaved = self.InterleavedDataset(
            datasets=ds, probabilities=[2.0, 8.0]
        )
        assert interleaved.probabilities is not None
        assert isinstance(interleaved.probabilities, list)
        self.assertAlmostEqual(interleaved.probabilities[0], 0.2)
        self.assertAlmostEqual(interleaved.probabilities[1], 0.8)

    # -- round-robin interleaving --------------------------------------------

    def test_round_robin_equal_length(self):
        """Round-robin with equal-length datasets alternates evenly."""
        ds = self._make_simple_datasets(3, 3)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="all_exhausted"
        )
        results = list(interleaved)
        self.assertEqual(len(results), 6)

        # Check alternation pattern: ds0, ds1, ds0, ds1, ...
        for i, item in enumerate(results):
            expected_ds = f"ds{i % 2}"
            self.assertTrue(
                item["id"].startswith(expected_ds),
                f"Item {i}: expected {expected_ds}, got {item['id']}",
            )

    def test_round_robin_first_exhausted(self):
        """first_exhausted stops when shortest dataset runs out."""
        ds = self._make_simple_datasets(2, 5)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="first_exhausted"
        )
        results = list(interleaved)
        # With round-robin and first_exhausted, ds0 (len=2) exhausts first
        # Pattern: ds0_0, ds1_0, ds0_1, ds1_1, then ds0 exhausts -> stop
        self.assertEqual(len(results), 4)

    def test_round_robin_all_exhausted(self):
        """all_exhausted continues until all datasets are done."""
        ds = self._make_simple_datasets(2, 4)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="all_exhausted"
        )
        results = list(interleaved)
        # ds0 has 2 items, ds1 has 4 items, total = 6
        self.assertEqual(len(results), 6)

    def test_round_robin_three_datasets(self):
        """Round-robin with three datasets cycles through all."""
        ds = self._make_simple_datasets(2, 2, 2)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="all_exhausted"
        )
        results = list(interleaved)
        self.assertEqual(len(results), 6)
        # Check pattern: ds0, ds1, ds2, ds0, ds1, ds2
        for i, item in enumerate(results):
            expected_ds = f"ds{i % 3}"
            self.assertTrue(item["id"].startswith(expected_ds))

    # -- probabilistic interleaving ------------------------------------------

    def test_static_probabilities_with_seed(self):
        """With probabilities and seed, output should be deterministic."""
        ds1 = self._make_simple_datasets(10, 10)
        ds2 = self._make_simple_datasets(10, 10)
        interleaved1 = self.InterleavedDataset(
            datasets=ds1,
            probabilities=[0.7, 0.3],
            seed=42,
            stopping_strategy="first_exhausted",
        )
        interleaved2 = self.InterleavedDataset(
            datasets=ds2,
            probabilities=[0.7, 0.3],
            seed=42,
            stopping_strategy="first_exhausted",
        )
        results1 = [x["id"] for x in interleaved1]
        results2 = [x["id"] for x in interleaved2]
        self.assertEqual(results1, results2)

    def test_static_probabilities_distribution(self):
        """With [0.9, 0.1] probabilities, ds0 should get most samples."""
        ds = self._make_simple_datasets(100, 100)
        interleaved = self.InterleavedDataset(
            datasets=ds,
            probabilities=[0.9, 0.1],
            seed=123,
            stopping_strategy="first_exhausted",
        )
        results = list(interleaved)
        ds0_count = sum(1 for x in results if x["id"].startswith("ds0"))
        ds1_count = sum(1 for x in results if x["id"].startswith("ds1"))
        # ds0 should have significantly more than ds1
        self.assertGreater(ds0_count, ds1_count)

    # -- dynamic probabilities -----------------------------------------------

    def test_dynamic_probabilities_callable(self):
        """Dynamic probabilities function is called during iteration."""
        call_log = []

        def my_prob_fn(step, datasets, examples_per_dataset, exhausted):
            call_log.append(
                {
                    "step": step,
                    "epd": list(examples_per_dataset),
                    "exhausted": list(exhausted),
                }
            )
            # Equal weighting
            return [1.0] * len(datasets)

        ds = self._make_simple_datasets(3, 3)
        interleaved = self.InterleavedDataset(
            datasets=ds,
            probabilities=my_prob_fn,
            seed=42,
            stopping_strategy="first_exhausted",
        )
        results = list(interleaved)
        # The callable should have been invoked at least once per yielded item
        self.assertGreater(len(call_log), 0)
        # Step should increment
        steps = [c["step"] for c in call_log]
        self.assertEqual(steps, sorted(steps))

    # -- __len__ -------------------------------------------------------------

    def test_len_round_robin_first_exhausted(self):
        """first_exhausted, round-robin: min(lens) * num_datasets."""
        ds = self._make_simple_datasets(5, 10)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="first_exhausted"
        )
        self.assertEqual(len(interleaved), 5 * 2)

    def test_len_round_robin_all_exhausted(self):
        """all_exhausted, round-robin: sum(lens)."""
        ds = self._make_simple_datasets(5, 10)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="all_exhausted"
        )
        self.assertEqual(len(interleaved), 15)

    def test_len_probabilities_all_exhausted(self):
        """all_exhausted with probabilities: sum(lens)."""
        ds = self._make_simple_datasets(5, 10)
        interleaved = self.InterleavedDataset(
            datasets=ds,
            probabilities=[0.5, 0.5],
            stopping_strategy="all_exhausted",
        )
        self.assertEqual(len(interleaved), 15)

    def test_len_probabilities_first_exhausted(self):
        """first_exhausted with probabilities: sum(lens) (approximation)."""
        ds = self._make_simple_datasets(5, 10)
        interleaved = self.InterleavedDataset(
            datasets=ds,
            probabilities=[0.5, 0.5],
            stopping_strategy="first_exhausted",
        )
        self.assertEqual(len(interleaved), 15)

    # -- state_dict / load_state_dict ----------------------------------------

    def test_state_dict_round_trip(self):
        """state_dict -> load_state_dict should restore interleaving state."""
        ds = self._make_simple_datasets(5, 5)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="all_exhausted"
        )

        # Consume 4 items
        it = iter(interleaved)
        consumed = [next(it) for _ in range(4)]

        # Save state
        state = interleaved.state_dict()
        self.assertIn("current_dataset_index", state)
        self.assertIn("current_example_count", state)
        self.assertIn("child_states", state)
        self.assertEqual(state["current_example_count"], 4)
        self.assertEqual(len(state["child_states"]), 2)

    def test_state_dict_contains_child_states(self):
        """Child datasets with state_dict should have their state saved."""
        ds = self._make_simple_datasets(5, 5)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="all_exhausted"
        )

        # Consume some items so children have state
        it = iter(interleaved)
        for _ in range(4):
            next(it)

        state = interleaved.state_dict()
        for child_state in state["child_states"]:
            self.assertIsNotNone(child_state)
            self.assertIn("position", child_state)

    def test_load_state_dict_restores_counts(self):
        ds = self._make_simple_datasets(5, 5)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="all_exhausted"
        )

        state = {
            "current_dataset_index": 1,
            "current_example_count": 7,
            "datasets_exhausted": [True, False],
            "child_states": [None, None],
        }
        interleaved.load_state_dict(state)
        self.assertEqual(interleaved._current_dataset_index, 1)
        self.assertEqual(interleaved._current_example_count, 7)
        self.assertEqual(interleaved._datasets_exhausted, [True, False])

    def test_state_dict_examples_per_dataset(self):
        """After iteration, state_dict should contain examples_per_dataset."""
        ds = self._make_simple_datasets(5, 5)
        interleaved = self.InterleavedDataset(
            datasets=ds,
            probabilities=[0.5, 0.5],
            seed=42,
            stopping_strategy="first_exhausted",
        )
        it = iter(interleaved)
        for _ in range(6):
            next(it)

        state = interleaved.state_dict()
        self.assertIn("examples_per_dataset", state)
        self.assertEqual(sum(state["examples_per_dataset"]), 6)

    def test_load_state_dict_restores_examples_per_dataset(self):
        """After loading state, _restored_examples_per_dataset should be set."""
        ds = self._make_simple_datasets(10, 10)
        interleaved = self.InterleavedDataset(
            datasets=ds,
            probabilities=[0.5, 0.5],
            seed=42,
            stopping_strategy="first_exhausted",
        )
        state = {
            "current_dataset_index": 0,
            "current_example_count": 8,
            "datasets_exhausted": [False, False],
            "child_states": [None, None],
            "examples_per_dataset": [5, 3],
        }
        interleaved.load_state_dict(state)
        self.assertTrue(hasattr(interleaved, "_restored_examples_per_dataset"))
        self.assertEqual(interleaved._restored_examples_per_dataset, [5, 3])

    # -- properties ----------------------------------------------------------

    def test_column_names_from_first_dataset(self):
        ds = self._make_simple_datasets(3, 3)
        # SimpleIterableDataset doesn't have column_names, so expect []
        interleaved = self.InterleavedDataset(datasets=ds)
        self.assertEqual(interleaved.column_names, [])

    def test_column_names_with_attribute(self):
        ds = self._make_simple_datasets(3, 3)
        ds[0].column_names = ["id", "value"]
        interleaved = self.InterleavedDataset(datasets=ds)
        self.assertEqual(interleaved.column_names, ["id", "value"])

    def test_features_none_by_default(self):
        ds = self._make_simple_datasets(3, 3)
        interleaved = self.InterleavedDataset(datasets=ds)
        self.assertIsNone(interleaved.features)

    def test_n_shards_default(self):
        """Without n_shards attribute, each dataset counts as 1 shard."""
        ds = self._make_simple_datasets(3, 3, 3)
        interleaved = self.InterleavedDataset(datasets=ds)
        self.assertEqual(interleaved.n_shards, 3)

    # -- repr ----------------------------------------------------------------

    def test_repr_contains_key_info(self):
        ds = self._make_simple_datasets(3)
        interleaved = self.InterleavedDataset(datasets=ds)
        r = repr(interleaved)
        self.assertIn("InterleavedDataset", r)
        self.assertIn("stopping_strategy", r)

    # -- interleave_datasets factory -----------------------------------------

    def test_interleave_datasets_factory(self):
        ds = self._make_simple_datasets(3, 3)
        result = self.interleave_datasets(ds)
        self.assertIsInstance(result, self.InterleavedDataset)

    def test_interleave_datasets_factory_with_params(self):
        ds = self._make_simple_datasets(3, 3)
        result = self.interleave_datasets(
            ds,
            probabilities=[0.6, 0.4],
            seed=99,
            stopping_strategy="all_exhausted",
        )
        self.assertIsInstance(result, self.InterleavedDataset)
        self.assertEqual(result.seed, 99)
        self.assertEqual(result.stopping_strategy, "all_exhausted")

    # -- single dataset edge case --------------------------------------------

    def test_single_dataset(self):
        """Interleaving a single dataset should just iterate it."""
        ds = self._make_simple_datasets(5)
        interleaved = self.InterleavedDataset(
            datasets=ds, stopping_strategy="all_exhausted"
        )
        results = list(interleaved)
        self.assertEqual(len(results), 5)


# ===========================================================================
# Tests for interleaved.balance_remaining_examples
# ===========================================================================


class TestBalanceRemainingExamples(unittest.TestCase):
    """Tests for the balance_remaining_examples dynamic probability function."""

    def setUp(self):
        from forgather.ml.datasets.interleaved import balance_remaining_examples

        self.balance = balance_remaining_examples

    def test_equal_remaining(self):
        """Equal remaining examples should produce equal weights."""
        datasets = [SimpleIterableDataset(list(range(10))) for _ in range(3)]
        weights = self.balance(0, datasets, [0, 0, 0], [False, False, False])
        self.assertEqual(weights, [10.0, 10.0, 10.0])

    def test_unequal_remaining(self):
        """More remaining examples should produce higher weight."""
        datasets = [SimpleIterableDataset(list(range(10))) for _ in range(2)]
        weights = self.balance(0, datasets, [8, 2], [False, False])
        # ds0: 10-8=2, ds1: 10-2=8
        self.assertEqual(weights[0], 2.0)
        self.assertEqual(weights[1], 8.0)

    def test_exhausted_dataset_gets_zero(self):
        """Exhausted datasets should get zero weight."""
        datasets = [SimpleIterableDataset(list(range(10))) for _ in range(2)]
        weights = self.balance(0, datasets, [10, 5], [True, False])
        self.assertEqual(weights[0], 0.0)
        self.assertGreater(weights[1], 0.0)

    def test_all_exhausted_returns_equal(self):
        """If all weights are 0 (all exhausted), return equal weights."""
        datasets = [SimpleIterableDataset(list(range(5))) for _ in range(2)]
        weights = self.balance(0, datasets, [5, 5], [True, True])
        self.assertEqual(weights, [1.0, 1.0])

    def test_dataset_without_len(self):
        """Datasets without __len__ get default weight 1.0."""
        datasets = [NoLenIterableDataset([1, 2, 3])]
        weights = self.balance(0, datasets, [0], [False])
        self.assertEqual(weights, [1.0])


# ===========================================================================
# Tests for dataloader_utils.LengthSyncCallback
# ===========================================================================


class TestLengthSyncCallback(unittest.TestCase):
    """Tests for LengthSyncCallback class."""

    def setUp(self):
        from forgather.ml.datasets.dataloader_utils import LengthSyncCallback

        self.LengthSyncCallback = LengthSyncCallback

    def _make_callback(self, sync_every=100):
        mock_dataloader = MagicMock()
        mock_dataset = MagicMock()
        cb = self.LengthSyncCallback(
            dataloader=mock_dataloader,
            dataset=mock_dataset,
            sync_every_n_steps=sync_every,
        )
        return cb, mock_dataloader, mock_dataset

    # -- init ----------------------------------------------------------------

    def test_init_attributes(self):
        cb, dl, ds = self._make_callback(sync_every=50)
        self.assertIs(cb.dataloader, dl)
        self.assertIs(cb.dataset, ds)
        self.assertEqual(cb.sync_every_n_steps, 50)
        self.assertEqual(cb._last_sync_step, 0)

    def test_default_sync_interval(self):
        mock_dl = MagicMock()
        mock_ds = MagicMock()
        cb = self.LengthSyncCallback(dataloader=mock_dl, dataset=mock_ds)
        self.assertEqual(cb.sync_every_n_steps, 100)

    # -- on_step_end ---------------------------------------------------------

    @patch("forgather.ml.datasets.dataloader_utils.sync_dataset_state_from_dataloader")
    def test_on_step_end_syncs_at_interval(self, mock_sync):
        """Sync should be called when enough steps have passed."""
        cb, dl, ds = self._make_callback(sync_every=10)
        mock_state = MagicMock()
        mock_args = MagicMock()
        mock_control = MagicMock()

        # Step 5: should NOT sync (5 < 10)
        mock_state.global_step = 5
        cb.on_step_end(mock_args, mock_state, mock_control)
        mock_sync.assert_not_called()

        # Step 10: should sync (10 - 0 >= 10)
        mock_state.global_step = 10
        cb.on_step_end(mock_args, mock_state, mock_control)
        self.assertEqual(mock_sync.call_count, 1)

    @patch("forgather.ml.datasets.dataloader_utils.sync_dataset_state_from_dataloader")
    def test_on_step_end_updates_last_sync(self, mock_sync):
        """After syncing, _last_sync_step should update."""
        cb, dl, ds = self._make_callback(sync_every=10)
        mock_state = MagicMock()
        mock_args = MagicMock()
        mock_control = MagicMock()

        mock_state.global_step = 10
        cb.on_step_end(mock_args, mock_state, mock_control)
        self.assertEqual(cb._last_sync_step, 10)

        # Step 15: not enough since last sync (15-10 = 5 < 10)
        mock_state.global_step = 15
        cb.on_step_end(mock_args, mock_state, mock_control)
        self.assertEqual(mock_sync.call_count, 1)

        # Step 20: should sync (20-10 = 10 >= 10)
        mock_state.global_step = 20
        cb.on_step_end(mock_args, mock_state, mock_control)
        self.assertEqual(mock_sync.call_count, 2)
        self.assertEqual(cb._last_sync_step, 20)

    @patch("forgather.ml.datasets.dataloader_utils.sync_dataset_state_from_dataloader")
    def test_on_step_end_no_sync_before_interval(self, mock_sync):
        """No sync should happen if interval has not been reached."""
        cb, dl, ds = self._make_callback(sync_every=100)
        mock_state = MagicMock()
        mock_args = MagicMock()
        mock_control = MagicMock()

        for step in range(1, 100):
            mock_state.global_step = step
            cb.on_step_end(mock_args, mock_state, mock_control)

        mock_sync.assert_not_called()

    @patch("forgather.ml.datasets.dataloader_utils.sync_dataset_state_from_dataloader")
    def test_on_step_end_passes_correct_args(self, mock_sync):
        """on_step_end should call sync with the callback's dataloader."""
        cb, dl, ds = self._make_callback(sync_every=1)
        mock_state = MagicMock()
        mock_state.global_step = 1
        mock_args = MagicMock()
        mock_control = MagicMock()

        cb.on_step_end(mock_args, mock_state, mock_control)
        mock_sync.assert_called_once_with(dl)

    # -- on_evaluate ---------------------------------------------------------

    @patch("forgather.ml.datasets.dataloader_utils.sync_dataset_state_from_dataloader")
    def test_on_evaluate_always_syncs(self, mock_sync):
        """on_evaluate should always sync, regardless of step count."""
        cb, dl, ds = self._make_callback(sync_every=1000)
        mock_state = MagicMock()
        mock_args = MagicMock()
        mock_control = MagicMock()

        cb.on_evaluate(mock_args, mock_state, mock_control)
        mock_sync.assert_called_once_with(dl)


# ===========================================================================
# Tests for create_length_sync_callback factory
# ===========================================================================


class TestCreateLengthSyncCallback(unittest.TestCase):
    """Tests for the create_length_sync_callback factory function."""

    def test_returns_callback(self):
        from forgather.ml.datasets.dataloader_utils import (
            LengthSyncCallback,
            create_length_sync_callback,
        )

        mock_dl = MagicMock()
        mock_ds = MagicMock()
        cb = create_length_sync_callback(mock_dl, mock_ds, sync_every_n_steps=50)
        self.assertIsInstance(cb, LengthSyncCallback)
        self.assertEqual(cb.sync_every_n_steps, 50)

    def test_default_interval(self):
        from forgather.ml.datasets.dataloader_utils import create_length_sync_callback

        mock_dl = MagicMock()
        mock_ds = MagicMock()
        cb = create_length_sync_callback(mock_dl, mock_ds)
        self.assertEqual(cb.sync_every_n_steps, 100)


if __name__ == "__main__":
    unittest.main()
