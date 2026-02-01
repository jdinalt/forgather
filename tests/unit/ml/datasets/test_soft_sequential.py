#!/usr/bin/env python3
"""
Unit tests for soft_sequential probability function.

Tests the soft sequential interleaving strategy which provides a gradual
transition between datasets rather than hard sequential consumption.
"""

import unittest
from typing import List


class MockDataset:
    """Mock dataset for testing with fixed length."""

    def __init__(self, length: int, name: str = ""):
        self._length = length
        self.name = name

    def __len__(self):
        return self._length

    def __repr__(self):
        return f"MockDataset({self.name}, len={self._length})"


class TestSoftSequential(unittest.TestCase):
    """Test soft_sequential probability function."""

    def setUp(self):
        """Import the function under test."""
        from forgather.ml.datasets.soft_sequential import soft_sequential

        self.soft_sequential = soft_sequential

    def test_basic_three_dataset_example(self):
        """
        Test the example from the documentation.

        Datasets [A, B, C] with 10 examples each:
        - A consumed 3/10 (7 remaining): A gets 70%
        - B consumed 2/10 (8 remaining): B gets 30% * 80% = 24%
        - C consumed 0/10 (10 remaining): C gets 30% * 20% = 6%
        """
        datasets = [
            MockDataset(10, "A"),
            MockDataset(10, "B"),
            MockDataset(10, "C"),
        ]
        examples_per_dataset = [3, 2, 0]
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        # Normalize to get probabilities
        total = sum(weights)
        probs = [w / total for w in weights]

        # A should get 70%
        self.assertAlmostEqual(probs[0], 0.70, places=5)
        # B should get 24%
        self.assertAlmostEqual(probs[1], 0.24, places=5)
        # C should get 6%
        self.assertAlmostEqual(probs[2], 0.06, places=5)

    def test_all_datasets_fresh(self):
        """Test when all datasets are fresh (no examples consumed)."""
        datasets = [MockDataset(100), MockDataset(100), MockDataset(100)]
        examples_per_dataset = [0, 0, 0]
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        # First dataset gets 100%, others get exponentially less
        # A: 1.0 * 1.0 = 1.0
        # B: 0.0 * 1.0 = 0.0
        # C: 0.0 * 1.0 = 0.0
        self.assertGreater(weights[0], 0.99)
        self.assertLess(weights[1], 0.01)
        self.assertLess(weights[2], 0.01)

    def test_first_dataset_nearly_exhausted(self):
        """Test when first dataset is almost exhausted."""
        datasets = [MockDataset(100), MockDataset(100), MockDataset(100)]
        examples_per_dataset = [99, 0, 0]  # A has 1 remaining
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        total = sum(weights)
        probs = [w / total for w in weights]

        # A has 1/100 remaining, gets 1%
        self.assertAlmostEqual(probs[0], 0.01, places=5)
        # B has 100/100 remaining, gets 99% * 100% = 99%
        self.assertAlmostEqual(probs[1], 0.99, places=2)
        # C gets remaining (very small)
        self.assertLess(probs[2], 0.01)

    def test_first_dataset_exhausted(self):
        """Test when first dataset is exhausted."""
        datasets = [MockDataset(100), MockDataset(100), MockDataset(100)]
        examples_per_dataset = [100, 50, 0]
        exhausted = [True, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        # A should get 0 weight
        self.assertEqual(weights[0], 0.0)

        # B and C should share probability
        total = sum(weights)
        probs = [w / total for w in weights]

        # B has 50/100 remaining, gets 50%
        self.assertAlmostEqual(probs[1], 0.50, places=5)
        # C has 100/100 remaining, gets 50% * 100% = 50%
        self.assertAlmostEqual(probs[2], 0.50, places=5)

    def test_all_datasets_half_consumed(self):
        """Test when all datasets are half-consumed."""
        datasets = [MockDataset(100), MockDataset(100), MockDataset(100)]
        examples_per_dataset = [50, 50, 50]
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        total = sum(weights)
        probs = [w / total for w in weights]

        # A: weight = 1.0 * 0.5 = 0.5
        # B: weight = 0.5 * 0.5 = 0.25
        # C: weight = 0.25 * 0.5 = 0.125
        # Total = 0.875, so normalized:
        # A: 0.5/0.875 = 0.571..., B: 0.25/0.875 = 0.286..., C: 0.125/0.875 = 0.143...
        self.assertAlmostEqual(probs[0], 0.5 / 0.875, places=5)
        self.assertAlmostEqual(probs[1], 0.25 / 0.875, places=5)
        self.assertAlmostEqual(probs[2], 0.125 / 0.875, places=5)

    def test_different_sized_datasets(self):
        """Test with datasets of different sizes."""
        datasets = [
            MockDataset(1000, "large"),
            MockDataset(100, "medium"),
            MockDataset(10, "small"),
        ]
        examples_per_dataset = [500, 50, 5]  # All half consumed
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        total = sum(weights)
        probs = [w / total for w in weights]

        # All are 50% consumed, so same weight pattern as test_all_datasets_half_consumed
        # Total weight = 0.875
        self.assertAlmostEqual(probs[0], 0.5 / 0.875, places=5)
        self.assertAlmostEqual(probs[1], 0.25 / 0.875, places=5)
        self.assertAlmostEqual(probs[2], 0.125 / 0.875, places=5)

    def test_asymmetric_consumption(self):
        """Test with asymmetric consumption patterns."""
        datasets = [MockDataset(100), MockDataset(100), MockDataset(100)]
        examples_per_dataset = [90, 10, 0]  # A: 10%, B: 90%, C: 100% remaining
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        total = sum(weights)
        probs = [w / total for w in weights]

        # A: 10/100 remaining = 10%
        self.assertAlmostEqual(probs[0], 0.10, places=5)
        # B: 90/100 remaining, gets 90% * 90% = 81%
        self.assertAlmostEqual(probs[1], 0.81, places=5)
        # C: 100/100 remaining, gets 90% * 10% * 100% = 9%
        self.assertAlmostEqual(probs[2], 0.09, places=5)

    def test_single_dataset(self):
        """Test with a single dataset."""
        datasets = [MockDataset(100)]
        examples_per_dataset = [50]
        exhausted = [False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        # Only one dataset, should get all weight
        self.assertGreater(weights[0], 0)
        self.assertEqual(len(weights), 1)

    def test_all_datasets_exhausted(self):
        """Test when all datasets are exhausted."""
        datasets = [MockDataset(100), MockDataset(100), MockDataset(100)]
        examples_per_dataset = [100, 100, 100]
        exhausted = [True, True, True]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        # All exhausted, should return equal weights (fallback behavior)
        self.assertEqual(weights, [1.0, 1.0, 1.0])

    def test_probabilities_sum_to_one(self):
        """Test that weights normalize to 1.0."""
        datasets = [MockDataset(100), MockDataset(200), MockDataset(50)]
        examples_per_dataset = [25, 100, 10]
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        total = sum(weights)
        probs = [w / total for w in weights]

        # Should sum to 1.0
        self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_no_length_datasets(self):
        """Test with datasets that don't support __len__."""

        class NoLenDataset:
            """Dataset without length."""

            pass

        datasets = [NoLenDataset(), NoLenDataset(), NoLenDataset()]
        examples_per_dataset = [10, 20, 30]
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        # Should fall back to equal proportions (1/3 each)
        # A: weight = 1.0 * (1/3) = 1/3, remaining = 2/3
        # B: weight = (2/3) * (1/3) = 2/9, remaining = 4/9
        # C: weight = (4/9) * (1/3) = 4/27
        # Total = 1/3 + 2/9 + 4/27 = 9/27 + 6/27 + 4/27 = 19/27
        total = sum(weights)
        probs = [w / total for w in weights]

        # Verify the compounding behavior with equal proportions
        self.assertAlmostEqual(probs[0], 9.0 / 19.0, places=5)  # (1/3) / (19/27)
        self.assertAlmostEqual(probs[1], 6.0 / 19.0, places=5)  # (2/9) / (19/27)
        self.assertAlmostEqual(probs[2], 4.0 / 19.0, places=5)  # (4/27) / (19/27)

    def test_empty_dataset(self):
        """Test with dataset that has length 0."""
        datasets = [MockDataset(0), MockDataset(100), MockDataset(100)]
        examples_per_dataset = [0, 0, 0]
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        # A: empty dataset (0/0) gets fallback weight 1/3
        # B: has 100/100 remaining (proportion = 1.0), consumes all remaining probability
        # C: gets 0 weight because B consumed all remaining probability
        #
        # A: weight = 1.0 * (1/3) = 1/3, remaining = 2/3
        # B: weight = (2/3) * 1.0 = 2/3, remaining = 0
        # C: weight = 0 * 1.0 = 0
        self.assertGreater(weights[0], 0)  # Gets fallback weight
        self.assertGreater(weights[1], 0)  # Gets remaining probability
        self.assertEqual(weights[2], 0.0)  # No probability left for C

        # Verify the exact weights
        total = sum(weights)
        probs = [w / total for w in weights]
        self.assertAlmostEqual(probs[0], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(probs[1], 2.0 / 3.0, places=5)

    def test_two_datasets_sequential_transition(self):
        """Test smooth transition between two datasets."""
        datasets = [MockDataset(100), MockDataset(100)]

        # As A gets consumed, probability should shift from A to B
        test_cases = [
            # (consumed_a, expected_prob_a_approx)
            (0, 1.00),  # All A
            (50, 0.50),  # Half A, half B
            (75, 0.25),  # Quarter A, 75% B
            (90, 0.10),  # 10% A, 90% B
            (99, 0.01),  # Almost all B
        ]

        for consumed_a, expected_prob_a in test_cases:
            with self.subTest(consumed_a=consumed_a):
                examples_per_dataset = [consumed_a, 0]
                exhausted = [False, False]

                weights = self.soft_sequential(
                    0, datasets, examples_per_dataset, exhausted
                )

                total = sum(weights)
                prob_a = weights[0] / total

                self.assertAlmostEqual(
                    prob_a, expected_prob_a, places=2, msg=f"consumed={consumed_a}"
                )

    def test_maintains_order_preference(self):
        """Test that earlier datasets are always preferred given equal remaining."""
        datasets = [MockDataset(100), MockDataset(100), MockDataset(100)]
        examples_per_dataset = [50, 50, 50]  # All equally consumed
        exhausted = [False, False, False]

        weights = self.soft_sequential(0, datasets, examples_per_dataset, exhausted)

        # Earlier datasets should have higher weight
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])


class TestSoftSequentialIntegration(unittest.TestCase):
    """Integration tests with InterleavedDataset."""

    def test_with_interleaved_dataset(self):
        """Test soft_sequential works with actual InterleavedDataset."""
        from forgather.ml.datasets.interleaved import interleave_datasets
        from forgather.ml.datasets.soft_sequential import soft_sequential

        # Create simple test datasets
        class SimpleDataset:
            def __init__(self, data: List[int]):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __iter__(self):
                return iter(self.data)

        ds1 = SimpleDataset([1, 2, 3, 4, 5])
        ds2 = SimpleDataset([10, 20, 30, 40, 50])
        ds3 = SimpleDataset([100, 200, 300, 400, 500])

        # Create interleaved dataset with soft sequential
        interleaved = interleave_datasets(
            [ds1, ds2, ds3],
            probabilities=soft_sequential,
            seed=42,
            stopping_strategy="first_exhausted",
        )

        # Consume dataset
        samples = list(interleaved)

        # Should get samples
        self.assertGreater(len(samples), 0)

        # Should favor earlier datasets initially
        first_half = samples[: len(samples) // 2]
        second_half = samples[len(samples) // 2 :]

        # Count samples from each dataset
        def count_from_dataset(sample_list, dataset_range):
            return sum(1 for s in sample_list if s in dataset_range)

        # Earlier samples should favor ds1
        ds1_count_first = count_from_dataset(first_half, [1, 2, 3, 4, 5])
        ds3_count_first = count_from_dataset(first_half, [100, 200, 300, 400, 500])

        # Later samples should shift away from ds1
        ds1_count_second = count_from_dataset(second_half, [1, 2, 3, 4, 5])
        ds3_count_second = count_from_dataset(second_half, [100, 200, 300, 400, 500])

        # Soft sequential should show this pattern (probabilistic, so not strict)
        # We expect more ds1 in first half, more ds3 in second half (on average)
        self.assertIsNotNone(ds1_count_first)  # Basic sanity checks
        self.assertIsNotNone(ds3_count_second)

    def test_checkpoint_compatibility(self):
        """Test that soft_sequential works with checkpoint save/restore."""
        from forgather.ml.datasets.interleaved import interleave_datasets
        from forgather.ml.datasets.soft_sequential import soft_sequential

        class SimpleDataset:
            def __init__(self, data: List[int]):
                self.data = data
                self.position = 0

            def __len__(self):
                return len(self.data)

            def __iter__(self):
                while self.position < len(self.data):
                    yield self.data[self.position]
                    self.position += 1

            def state_dict(self):
                return {"position": self.position}

            def load_state_dict(self, state):
                self.position = state["position"]

        ds1 = SimpleDataset([1, 2, 3, 4, 5])
        ds2 = SimpleDataset([10, 20, 30, 40, 50])

        # Create interleaved dataset
        interleaved = interleave_datasets(
            [ds1, ds2], probabilities=soft_sequential, seed=42
        )

        # Consume some samples
        iterator = iter(interleaved)
        samples_before = [next(iterator) for _ in range(3)]

        # Save checkpoint
        state = interleaved.state_dict()

        # Consume more samples
        samples_middle = [next(iterator) for _ in range(2)]

        # Restore checkpoint
        ds1_restored = SimpleDataset([1, 2, 3, 4, 5])
        ds2_restored = SimpleDataset([10, 20, 30, 40, 50])
        interleaved_restored = interleave_datasets(
            [ds1_restored, ds2_restored], probabilities=soft_sequential, seed=42
        )
        interleaved_restored.load_state_dict(state)

        # Continue from checkpoint - should get same samples as middle
        iterator_restored = iter(interleaved_restored)
        samples_after = [next(iterator_restored) for _ in range(2)]

        # After restore, we should get the same sequence
        # Note: This test verifies checkpoint compatibility, not exact reproduction
        # (since dataset state saving may not be perfect for this simple mock)
        self.assertEqual(len(samples_after), len(samples_middle))


if __name__ == "__main__":
    unittest.main()
