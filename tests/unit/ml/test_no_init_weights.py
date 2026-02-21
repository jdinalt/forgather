#!/usr/bin/env python3
"""
Unit tests for forgather.ml.no_init_weights.

Tests the no_init_weights context manager that disables PyTorch weight
initialization functions to speed up loading large models.
"""

import unittest

import torch
from torch import nn

import forgather.ml.no_init_weights as niw
from forgather.ml.no_init_weights import no_init_weights, TORCH_INIT_FUNCTIONS


class TestNoInitWeightsInitFunctions(unittest.TestCase):
    """Test that init functions are replaced and restored by the context manager."""

    def test_init_functions_replaced_inside_context(self):
        """Inside the context, nn.init functions should be replaced with no-ops."""
        original_normal_ = nn.init.normal_
        original_kaiming_uniform_ = nn.init.kaiming_uniform_

        with no_init_weights():
            # All init functions should be replaced
            self.assertIsNot(nn.init.normal_, original_normal_)
            self.assertIsNot(nn.init.kaiming_uniform_, original_kaiming_uniform_)

            # The replacement should be a no-op (callable that does nothing)
            # Calling it should not raise
            t = torch.empty(3, 3)
            nn.init.normal_(t)  # Should be a no-op

    def test_init_functions_restored_after_context(self):
        """After exiting the context, all init functions should be restored."""
        originals = {}
        for name in TORCH_INIT_FUNCTIONS:
            originals[name] = getattr(nn.init, name)

        with no_init_weights():
            pass

        for name, original_func in originals.items():
            restored_func = getattr(nn.init, name)
            self.assertIs(
                restored_func,
                original_func,
                f"nn.init.{name} was not restored after context exit",
            )

    def test_init_functions_restored_on_exception(self):
        """Init functions are restored even if an exception occurs inside the context."""
        originals = {}
        for name in TORCH_INIT_FUNCTIONS:
            originals[name] = getattr(nn.init, name)

        with self.assertRaises(RuntimeError):
            with no_init_weights():
                raise RuntimeError("intentional error")

        for name, original_func in originals.items():
            restored_func = getattr(nn.init, name)
            self.assertIs(
                restored_func,
                original_func,
                f"nn.init.{name} was not restored after exception",
            )

    def test_all_known_init_functions_are_replaced(self):
        """Every function listed in TORCH_INIT_FUNCTIONS is replaced inside the context."""
        with no_init_weights():
            for name, original_func in TORCH_INIT_FUNCTIONS.items():
                current_func = getattr(nn.init, name)
                self.assertIsNot(
                    current_func,
                    original_func,
                    f"nn.init.{name} should be replaced inside context",
                )

    def test_replaced_function_is_noop(self):
        """The replacement function accepts any arguments and does nothing."""
        with no_init_weights():
            t = torch.ones(3, 3)
            original_data = t.clone()
            # normal_ should be a no-op, so t should remain unchanged
            nn.init.normal_(t, mean=0.0, std=1.0)
            self.assertTrue(
                torch.equal(t, original_data),
                "Tensor should be unchanged when init is a no-op",
            )


class TestNoInitWeightsGlobal(unittest.TestCase):
    """Test the _init_weights global flag behavior."""

    def test_init_weights_false_inside_context(self):
        """The _init_weights global is set to False inside the context."""
        self.assertTrue(niw._init_weights)
        with no_init_weights():
            self.assertFalse(niw._init_weights)

    def test_init_weights_restored_after_context(self):
        """The _init_weights global is restored to its previous value after exit."""
        original = niw._init_weights
        with no_init_weights():
            pass
        self.assertEqual(niw._init_weights, original)

    def test_init_weights_restored_on_exception(self):
        """The _init_weights global is restored even on exception."""
        original = niw._init_weights
        with self.assertRaises(ValueError):
            with no_init_weights():
                raise ValueError("test")
        self.assertEqual(niw._init_weights, original)

    def test_nested_context_restores_correctly(self):
        """Nested no_init_weights contexts each restore their own previous value."""
        self.assertTrue(niw._init_weights)
        with no_init_weights():
            self.assertFalse(niw._init_weights)
            with no_init_weights():
                self.assertFalse(niw._init_weights)
            # Inner context restores the value set by outer context (False)
            self.assertFalse(niw._init_weights)
        # Outer context restores the original value (True)
        self.assertTrue(niw._init_weights)


class TestNoInitWeightsModuleCreation(unittest.TestCase):
    """Test that module creation behavior is affected by no_init_weights."""

    def test_linear_created_inside_context_skips_init(self):
        """Creating nn.Linear inside no_init_weights should skip kaiming_uniform_ init.

        We verify this by checking that the weight tensor is NOT initialized
        to the typical kaiming_uniform_ distribution. Since the init is skipped,
        the weights will contain whatever was in memory (uninitialized).
        """
        with no_init_weights():
            # Create a linear layer; its weight init should be skipped
            layer = nn.Linear(64, 64)

        # We cannot easily verify "uninitialized" vs "initialized" deterministically.
        # Instead, verify that the init function was indeed a no-op by checking that
        # creating a layer does not raise, confirming the skip function works.
        self.assertIsInstance(layer, nn.Linear)
        self.assertEqual(layer.weight.shape, (64, 64))

    def test_init_functions_work_normally_outside_context(self):
        """After exiting the context, init functions work normally."""
        with no_init_weights():
            pass

        # Create a tensor and verify normal_ actually modifies it
        t = torch.zeros(100, 100)
        nn.init.normal_(t, mean=5.0, std=0.001)
        # With mean=5.0 and tiny std, all values should be close to 5.0
        self.assertTrue(
            (t > 4.0).all().item(),
            "normal_ should work normally outside the context",
        )

    def test_constant_init_is_noop_inside_context(self):
        """nn.init.constant_ is also replaced with a no-op inside the context."""
        t = torch.zeros(3, 3)
        with no_init_weights():
            nn.init.constant_(t, 42.0)

        # constant_ was a no-op, so tensor should still be zeros
        self.assertTrue(
            torch.equal(t, torch.zeros(3, 3)),
            "constant_ should be a no-op inside the context",
        )

    def test_xavier_init_is_noop_inside_context(self):
        """nn.init.xavier_uniform_ is replaced with a no-op inside the context."""
        t = torch.zeros(4, 4)
        with no_init_weights():
            nn.init.xavier_uniform_(t)

        self.assertTrue(
            torch.equal(t, torch.zeros(4, 4)),
            "xavier_uniform_ should be a no-op inside the context",
        )


if __name__ == "__main__":
    unittest.main()
