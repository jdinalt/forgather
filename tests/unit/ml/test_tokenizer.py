#!/usr/bin/env python3
"""
Unit tests for normalize_range in forgather.ml.tokenizer.

Tests the normalize_range function which converts various input types
(None, int, float, Sequence, range) into a range object. Documents
known bugs where actual behavior diverges from the docstring.
"""

import unittest

from forgather.ml.tokenizer import normalize_range


class TestNormalizeRangeNoneInput(unittest.TestCase):
    """Test normalize_range with None input."""

    def test_none_returns_full_range(self):
        """normalize_range(1000, None) returns range(0, 1000) -- full dataset."""
        result = normalize_range(1000, None)
        self.assertEqual(result, range(0, 1000))

    def test_none_with_zero_length(self):
        """None input with length=0 returns range(0, 0)."""
        result = normalize_range(0, None)
        self.assertEqual(result, range(0, 0))


class TestNormalizeRangeRangeInput(unittest.TestCase):
    """Test normalize_range with range input."""

    def test_range_returned_as_is(self):
        """A range input should be returned unchanged."""
        r = range(10, 100)
        result = normalize_range(1000, r)
        self.assertIs(result, r)
        self.assertEqual(result, range(10, 100))

    def test_range_with_step(self):
        """A range with step should be returned unchanged."""
        r = range(0, 100, 5)
        result = normalize_range(1000, r)
        self.assertEqual(result, range(0, 100, 5))

    def test_range_exceeding_length_returned_as_is(self):
        """A range exceeding the length is not clamped -- it is returned as-is."""
        r = range(0, 5000)
        result = normalize_range(1000, r)
        self.assertEqual(result, range(0, 5000))

    def test_empty_range_returned_as_is(self):
        """An empty range is returned unchanged."""
        r = range(0, 0)
        result = normalize_range(1000, r)
        self.assertEqual(result, range(0, 0))


class TestNormalizeRangeFloatInput(unittest.TestCase):
    """Test normalize_range with float input (percentage of length)."""

    def test_float_quarter(self):
        """0.25 of 1000 = 250, returns range(0, 250)."""
        result = normalize_range(1000, 0.25)
        self.assertEqual(result, range(0, 250))

    def test_float_half(self):
        """0.5 of 1000 = 500, returns range(0, 500)."""
        result = normalize_range(1000, 0.5)
        self.assertEqual(result, range(0, 500))

    def test_float_full(self):
        """1.0 of 1000 = 1000, returns range(0, 1000)."""
        result = normalize_range(1000, 1.0)
        self.assertEqual(result, range(0, 1000))

    def test_float_zero(self):
        """0.0 of 1000 = 0, returns range(0, 0)."""
        result = normalize_range(1000, 0.0)
        self.assertEqual(result, range(0, 0))

    def test_float_exceeding_one(self):
        """2.0 of 1000 = 2000, clamped to 1000, returns range(0, 1000)."""
        result = normalize_range(1000, 2.0)
        self.assertEqual(result, range(0, 1000))

    def test_float_small_fraction(self):
        """0.001 of 1000 = 1, returns range(0, 1)."""
        result = normalize_range(1000, 0.001)
        self.assertEqual(result, range(0, 1))

    def test_float_negative(self):
        """Negative float: -0.1 * 1000 = -100, clamped to 0, returns range(0, 0)."""
        result = normalize_range(1000, -0.1)
        self.assertEqual(result, range(0, 0))


class TestNormalizeRangeIntInput(unittest.TestCase):
    """Test normalize_range with int input (first n records)."""

    def test_int_positive(self):
        """Positive int returns range(0, n)."""
        result = normalize_range(1000, 500)
        self.assertEqual(result, range(0, 500))

    def test_int_zero(self):
        """Zero returns range(0, 0)."""
        result = normalize_range(1000, 0)
        self.assertEqual(result, range(0, 0))

    def test_int_exceeding_length(self):
        """Int exceeding length is clamped to length."""
        result = normalize_range(1000, 1500)
        self.assertEqual(result, range(0, 1000))

    def test_int_equal_to_length(self):
        """Int equal to length returns range(0, length)."""
        result = normalize_range(1000, 1000)
        self.assertEqual(result, range(0, 1000))

    def test_negative_int_behavior(self):
        """Negative int uses Python-style indexing: length + value.

        For value=-10, length=1000: length + (-10) = 990.
        """
        result = normalize_range(1000, -10)
        self.assertEqual(result, range(0, 990))

    def test_negative_int_large(self):
        """Large negative int: length + (-500) = 500."""
        result = normalize_range(1000, -500)
        self.assertEqual(result, range(0, 500))

    def test_negative_int_minus_one(self):
        """value=-1: length + (-1) = 999."""
        result = normalize_range(1000, -1)
        self.assertEqual(result, range(0, 999))


class TestNormalizeRangeSequenceInput(unittest.TestCase):
    """Test normalize_range with Sequence input (list, tuple)."""

    def test_list_two_ints(self):
        """List of two ints: [start, end]."""
        result = normalize_range(1000, [100, 900])
        self.assertEqual(result, range(100, 900))

    def test_tuple_two_ints(self):
        """Tuple of two ints: (start, end)."""
        result = normalize_range(1000, (100, 900))
        self.assertEqual(result, range(100, 900))

    def test_mixed_int_and_float(self):
        """Sequence with int start and float end: [100, 0.9] -> range(100, 900)."""
        result = normalize_range(1000, [100, 0.9])
        self.assertEqual(result, range(100, 900))

    def test_mixed_float_and_int(self):
        """Sequence with float start and int end: [0.1, 900] -> range(100, 900)."""
        result = normalize_range(1000, [0.1, 900])
        self.assertEqual(result, range(100, 900))

    def test_two_floats(self):
        """Sequence with two floats: [0.1, 0.5] -> range(100, 500)."""
        result = normalize_range(1000, [0.1, 0.5])
        self.assertEqual(result, range(100, 500))

    def test_three_values_with_step(self):
        """Sequence of three values: (start, end, step).

        The docstring shows: (1, 1.0, 4) -> range(1, 1000, 4).
        Here 1.0 as float becomes int(1.0 * 1000) = 1000, and 4 is the step.
        """
        result = normalize_range(1000, (1, 1.0, 4))
        self.assertEqual(result, range(1, 1000, 4))

    def test_sequence_with_clamping(self):
        """Values exceeding length are clamped."""
        result = normalize_range(1000, [0, 2000])
        self.assertEqual(result, range(0, 1000))

    def test_sequence_start_clamped_to_zero(self):
        """Negative float start clamped to 0."""
        result = normalize_range(1000, [-0.1, 0.5])
        self.assertEqual(result, range(0, 500))

    def test_sequence_negative_int_in_sequence(self):
        """Negative int in sequence uses Python-style indexing: length + value.

        normalize_value(-10) with length=1000: 1000 + (-10) = 990, clamped to [0, 1000].
        The docstring shows: normalize_range(1000, (-10, 2.0)) -> range(0, 1000).
        With the fix, start = max(0, 1000 + (-10)) = 990 (not 0 as docstring says,
        but docstring used a different example value). end = min(2000, 1000) = 1000.
        """
        result = normalize_range(1000, (-10, 2.0))
        # start: -10 -> length + (-10) = 990
        # end: 2.0 -> int(2.0 * 1000) = 2000, clamped to 1000
        self.assertEqual(result, range(990, 1000))

    def test_empty_sequence(self):
        """Empty sequence returns range(length) -- treated as 'use full dataset'."""
        result = normalize_range(1000, [])
        self.assertEqual(result, range(0, 1000))


class TestNormalizeRangeUnsupportedTypes(unittest.TestCase):
    """Test normalize_range with unsupported types."""

    def test_string_raises_value_error(self):
        """String input should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            normalize_range(1000, "half")
        self.assertIn("Unsupported data-type for dataset range", str(ctx.exception))

    def test_dict_raises_value_error(self):
        """Dict input should raise ValueError."""
        with self.assertRaises(ValueError):
            normalize_range(1000, {"start": 0, "end": 500})

    def test_set_raises_value_error(self):
        """Set input should raise ValueError (set is not a Sequence)."""
        with self.assertRaises(ValueError):
            normalize_range(1000, {100, 500})

    def test_bool_treated_as_int(self):
        """bool is a subclass of int, so True=1 and False=0.

        isinstance(True, int) is True, so this goes through the int path.
        """
        result = normalize_range(1000, True)
        self.assertEqual(result, range(0, 1))

        result = normalize_range(1000, False)
        self.assertEqual(result, range(0, 0))

    def test_string_in_sequence_raises(self):
        """A string element inside a sequence should raise ValueError from normalize_value."""
        with self.assertRaises(ValueError) as ctx:
            normalize_range(1000, [100, "half"])
        self.assertIn("Unsupported data-type for dataset range value", str(ctx.exception))


class TestNormalizeRangeEdgeCases(unittest.TestCase):
    """Test normalize_range edge cases."""

    def test_zero_length(self):
        """With length=0, float value results in 0."""
        result = normalize_range(0, 0.5)
        self.assertEqual(result, range(0, 0))

    def test_zero_length_with_int(self):
        """With length=0, int value is clamped to 0."""
        result = normalize_range(0, 100)
        self.assertEqual(result, range(0, 0))

    def test_length_one(self):
        """With length=1, 0.5 becomes int(0.5 * 1) = 0."""
        result = normalize_range(1, 0.5)
        self.assertEqual(result, range(0, 0))

    def test_length_one_full(self):
        """With length=1, 1.0 becomes int(1.0 * 1) = 1."""
        result = normalize_range(1, 1.0)
        self.assertEqual(result, range(0, 1))

    def test_very_large_length(self):
        """With very large length, float fraction works correctly."""
        result = normalize_range(10_000_000, 0.001)
        self.assertEqual(result, range(0, 10_000))


if __name__ == "__main__":
    unittest.main()
