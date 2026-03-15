#!/usr/bin/env python3
"""
Unit tests for forgather.ml.utils.

Tests utility functions and classes used across the ML training system,
including alt_repr, ConversionDescriptor, DiagnosticEnum, count_parameters,
and the default_dtype context manager.
"""

import unittest
from enum import Enum

import torch
import torch.nn as nn

from forgather.ml.utils import (
    ConversionDescriptor,
    DiagnosticEnum,
    ModelParameterStats,
    alt_repr,
    count_parameters,
    default_dtype,
)


class TestAltRepr(unittest.TestCase):
    """Test alt_repr fallback representation."""

    def test_object_without_custom_repr(self):
        """An object without a custom __repr__ should show its public attributes."""

        class Plain:
            def __init__(self):
                self.x = 10
                self.name = "test"

        obj = Plain()
        result = alt_repr(obj)
        self.assertIn("x", result)
        self.assertIn("10", result)
        self.assertIn("name", result)
        self.assertIn("test", result)

    def test_object_without_custom_repr_excludes_private(self):
        """Private and protected attributes (starting with _) are excluded."""

        class Plain:
            def __init__(self):
                self.public = 1
                self._private = 2
                self.__dunder = 3

        obj = Plain()
        result = alt_repr(obj)
        self.assertIn("public", result)
        self.assertNotIn("_private", result)

    def test_object_without_custom_repr_excludes_methods(self):
        """Callable attributes (methods) are excluded from the representation."""

        class Plain:
            def __init__(self):
                self.value = 42

            def some_method(self):
                pass

        obj = Plain()
        result = alt_repr(obj)
        self.assertIn("value", result)
        self.assertNotIn("some_method", result)

    def test_object_with_custom_repr(self):
        """An object with a custom __repr__ uses its own repr directly."""

        class Custom:
            def __repr__(self):
                return "CustomRepr()"

        obj = Custom()
        result = alt_repr(obj)
        self.assertEqual(result, "CustomRepr()")

    def test_list_uses_attr_repr(self):
        """list.__repr__ is also a MethodWrapperType, so alt_repr shows attrs."""
        # list inherits a C-slot __repr__, which is MethodWrapperType.
        # alt_repr therefore enumerates public non-callable attributes.
        # An empty list has no such attributes, so it returns '{}'.
        result = alt_repr([1, 2, 3])
        self.assertEqual(result, "{}")

    def test_int_uses_attr_repr(self):
        """int's __repr__ is a MethodWrapperType, so alt_repr shows attributes."""
        # int.__repr__ is a slot wrapper (MethodWrapperType), so alt_repr
        # treats it as not having a custom __repr__ and shows attributes.
        result = alt_repr(42)
        self.assertIn("numerator", result)


class TestConversionDescriptor(unittest.TestCase):
    """Test ConversionDescriptor type coercion."""

    def test_converts_value_on_set(self):
        """Setting a value converts it via the provided class/callable."""

        class MyClass:
            value = ConversionDescriptor(int, default=0)

        obj = MyClass()
        obj.value = "42"
        self.assertEqual(obj.value, 42)
        self.assertIsInstance(obj.value, int)

    def test_returns_default_when_not_set(self):
        """Accessing a descriptor that was never set returns the default."""

        class MyClass:
            value = ConversionDescriptor(int, default=99)

        obj = MyClass()
        self.assertEqual(obj.value, 99)

    def test_class_level_access_returns_default(self):
        """Accessing the descriptor on the class (not instance) returns the default."""

        class MyClass:
            value = ConversionDescriptor(int, default=0)

        self.assertEqual(MyClass.value, 0)

    def test_float_conversion(self):
        """Float conversion descriptor works correctly."""

        class MyClass:
            ratio = ConversionDescriptor(float, default=0.0)

        obj = MyClass()
        obj.ratio = "3.14"
        self.assertAlmostEqual(obj.ratio, 3.14)
        self.assertIsInstance(obj.ratio, float)

    def test_enum_conversion(self):
        """ConversionDescriptor can convert string values to enum members."""

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        class MyClass:
            color = ConversionDescriptor(Color, default=Color.RED)

        obj = MyClass()
        obj.color = "blue"
        self.assertEqual(obj.color, Color.BLUE)

    def test_different_instances_independent(self):
        """Each instance maintains its own value independently."""

        class MyClass:
            value = ConversionDescriptor(int, default=0)

        a = MyClass()
        b = MyClass()
        a.value = "10"
        b.value = "20"
        self.assertEqual(a.value, 10)
        self.assertEqual(b.value, 20)

    def test_set_already_correct_type(self):
        """Setting a value that already has the correct type still works."""

        class MyClass:
            value = ConversionDescriptor(int, default=0)

        obj = MyClass()
        obj.value = 42
        self.assertEqual(obj.value, 42)


class TestDiagnosticEnum(unittest.TestCase):
    """Test DiagnosticEnum error messages."""

    def setUp(self):
        """Define a test enum."""

        class Mode(DiagnosticEnum):
            TRAIN = "train"
            EVAL = "eval"
            TEST = "test"

        self.Mode = Mode

    def test_valid_value_works(self):
        """Creating an enum from a valid value works normally."""
        self.assertEqual(self.Mode("train"), self.Mode.TRAIN)
        self.assertEqual(self.Mode("eval"), self.Mode.EVAL)
        self.assertEqual(self.Mode("test"), self.Mode.TEST)

    def test_invalid_value_raises_value_error(self):
        """An invalid value raises a ValueError."""
        with self.assertRaises(ValueError):
            self.Mode("invalid")

    def test_error_message_shows_valid_options(self):
        """The error message includes available choices."""
        with self.assertRaises(ValueError) as ctx:
            self.Mode("nonexistent")
        error_msg = str(ctx.exception)
        self.assertIn("choose one of", error_msg)
        self.assertIn("train", error_msg)

    def test_error_message_includes_class_name(self):
        """The error message includes the enum class name."""
        with self.assertRaises(ValueError) as ctx:
            self.Mode("bad")
        self.assertIn("Mode", str(ctx.exception))

    def test_error_message_includes_bad_value(self):
        """The error message includes the invalid value that was provided."""
        with self.assertRaises(ValueError) as ctx:
            self.Mode("purple")
        self.assertIn("purple", str(ctx.exception))


class _MockCausalLM(nn.Module):
    """Minimal causal LM with HF-style embedding accessors for testing."""

    def __init__(self, vocab_size, hidden, *, tie_weights=False, bias=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, hidden, bias=False)
        if tie_weights:
            self.lm_head = nn.Linear(hidden, vocab_size, bias=bias)
            self.lm_head.weight = self.embed.weight  # tie
        else:
            self.lm_head = nn.Linear(hidden, vocab_size, bias=bias)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


class TestCountParameters(unittest.TestCase):
    """Test count_parameters model parameter counting."""

    def test_return_type(self):
        """Returns a ModelParameterStats dataclass."""
        model = nn.Linear(10, 10, bias=False)
        result = count_parameters(model)
        self.assertIsInstance(result, ModelParameterStats)

    def test_simple_model_no_embeddings(self):
        """A plain nn.Linear has no embedding methods; embedding=0."""
        model = nn.Linear(10, 10, bias=False)  # 100 params
        result = count_parameters(model)
        self.assertEqual(result.total, 100)
        self.assertEqual(result.trainable, 100)
        self.assertEqual(result.embedding, 0)
        self.assertEqual(result.non_embedding, 100)
        self.assertEqual(result.trainable_non_embedding, 100)
        self.assertFalse(result.tied_embeddings)

    def test_frozen_params(self):
        """Frozen params are counted in total but not trainable."""
        model = nn.Sequential(
            nn.Linear(100, 100, bias=False),  # 10000 params
            nn.Linear(100, 100, bias=False),  # 10000 params
        )
        for param in model[0].parameters():
            param.requires_grad = False
        result = count_parameters(model)
        self.assertEqual(result.total, 20000)
        self.assertEqual(result.trainable, 10000)

    def test_all_frozen(self):
        """A fully frozen model has trainable=0."""
        model = nn.Linear(100, 100, bias=False)
        for p in model.parameters():
            p.requires_grad = False
        result = count_parameters(model)
        self.assertEqual(result.total, 10000)
        self.assertEqual(result.trainable, 0)

    def test_tied_embeddings(self):
        """Tied input/output embeddings: detected and counted once."""
        model = _MockCausalLM(100, 32, tie_weights=True)
        result = count_parameters(model)
        self.assertTrue(result.tied_embeddings)
        # embed: 100*32=3200 (shared), linear: 32*32=1024, lm_head bias=0
        # total = 3200 + 1024 = 4224  (tied weight counted once)
        self.assertEqual(result.total, 3200 + 1024)
        self.assertEqual(result.embedding, 3200)
        self.assertEqual(result.non_embedding, 1024)

    def test_untied_embeddings(self):
        """Untied input/output embeddings: both counted, not tied."""
        model = _MockCausalLM(100, 32, tie_weights=False)
        result = count_parameters(model)
        self.assertFalse(result.tied_embeddings)
        # embed: 3200, lm_head: 3200, linear: 1024
        self.assertEqual(result.total, 3200 + 3200 + 1024)
        self.assertEqual(result.embedding, 3200 + 3200)
        self.assertEqual(result.non_embedding, 1024)

    def test_untied_with_bias(self):
        """Output embedding bias is counted as embedding parameter."""
        model = _MockCausalLM(100, 32, tie_weights=False, bias=True)
        result = count_parameters(model)
        # embed: 3200, lm_head weight: 3200 + bias: 100, linear: 1024
        self.assertEqual(result.total, 3200 + 3200 + 100 + 1024)
        self.assertEqual(result.embedding, 3200 + 3200 + 100)
        self.assertEqual(result.non_embedding, 1024)

    def test_frozen_embedding(self):
        """Frozen embedding params excluded from trainable_non_embedding."""
        model = _MockCausalLM(100, 32, tie_weights=True)
        for p in model.embed.parameters():
            p.requires_grad = False
        result = count_parameters(model)
        # embed weight is frozen (3200), linear (1024) trainable
        self.assertEqual(result.trainable, 1024)
        self.assertEqual(result.trainable_non_embedding, 1024)

    def test_output_embeddings_none(self):
        """Model whose get_output_embeddings returns None."""

        class EmbedOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(50, 16)
                self.fc = nn.Linear(16, 16, bias=False)

            def get_input_embeddings(self):
                return self.embed

            def get_output_embeddings(self):
                return None

        model = EmbedOnly()
        result = count_parameters(model)
        self.assertEqual(result.embedding, 50 * 16)
        self.assertEqual(result.non_embedding, 16 * 16)
        self.assertFalse(result.tied_embeddings)

    def test_chinchilla_optimal_tokens(self):
        """Chinchilla optimal = 20 * non_embedding."""
        model = _MockCausalLM(100, 32, tie_weights=True)
        result = count_parameters(model)
        self.assertEqual(result.chinchilla_optimal_tokens, 20 * result.non_embedding)

    def test_flops_per_token(self):
        """FLOPs per token = 6 * non_embedding."""
        model = _MockCausalLM(100, 32, tie_weights=True)
        result = count_parameters(model)
        self.assertAlmostEqual(result.flops_per_token, 6.0 * result.non_embedding)


class TestDefaultDtype(unittest.TestCase):
    """Test the default_dtype context manager."""

    def test_changes_dtype_inside_context(self):
        """The default dtype is changed inside the context."""
        original = torch.get_default_dtype()
        with default_dtype(torch.float64):
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        # Ensure we restore
        self.assertEqual(torch.get_default_dtype(), original)

    def test_restores_dtype_on_exit(self):
        """The previous default dtype is restored after the context exits."""
        original = torch.get_default_dtype()
        with default_dtype(torch.float16):
            pass
        self.assertEqual(torch.get_default_dtype(), original)

    def test_restores_dtype_on_exception(self):
        """The dtype is restored even if an exception occurs inside the context."""
        original = torch.get_default_dtype()
        with self.assertRaises(RuntimeError):
            with default_dtype(torch.float64):
                self.assertEqual(torch.get_default_dtype(), torch.float64)
                raise RuntimeError("test error")
        self.assertEqual(torch.get_default_dtype(), original)

    def test_tensor_created_uses_context_dtype(self):
        """Tensors created inside the context use the overridden default dtype."""
        with default_dtype(torch.float64):
            t = torch.zeros(3)
            self.assertEqual(t.dtype, torch.float64)

    def test_nested_contexts(self):
        """Nested default_dtype contexts each restore their own previous dtype."""
        original = torch.get_default_dtype()
        with default_dtype(torch.float64):
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            with default_dtype(torch.float16):
                self.assertEqual(torch.get_default_dtype(), torch.float16)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        self.assertEqual(torch.get_default_dtype(), original)


if __name__ == "__main__":
    unittest.main()
