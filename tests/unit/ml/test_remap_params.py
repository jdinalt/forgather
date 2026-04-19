#!/usr/bin/env python3
"""
Unit tests for forgather.ml.remap_params.

Tests the parameter name remapping utilities used for converting
state dict keys between different model architectures.
"""

import unittest

import torch

from forgather.ml.remap_params import (
    sub_param_name,
    remap_parameter_fqns,
    remap_state_dict,
)


class TestSubParamName(unittest.TestCase):
    """Test sub_param_name recursive regex substitution."""

    def test_no_match_returns_original(self):
        """When no pattern matches, the original string is returned unchanged."""
        psub_list = [
            (r"foo", "bar", []),
        ]
        result = sub_param_name("baz.weight", psub_list)
        self.assertEqual(result, "baz.weight")

    def test_simple_replacement(self):
        """A simple pattern with no groups replaces the matched prefix."""
        psub_list = [
            (r"old_prefix\.", "new_prefix.", []),
        ]
        result = sub_param_name("old_prefix.weight", psub_list)
        self.assertEqual(result, "new_prefix.weight")

    def test_regex_group_capture(self):
        """Captured groups in the pattern can be used in the replacement."""
        psub_list = [
            (r"layers\.(\d+)\.", r"blocks.\1.", []),
        ]
        result = sub_param_name("layers.3.weight", psub_list)
        self.assertEqual(result, "blocks.3.weight")

    def test_recursive_child_patterns(self):
        """Child substitution lists are applied to the remaining tail."""
        child_list = [
            (r"weight", "W", []),
        ]
        psub_list = [
            (r"model\.", "net.", child_list),
        ]
        result = sub_param_name("model.weight", psub_list)
        self.assertEqual(result, "net.W")

    def test_deeply_nested_child_patterns(self):
        """Multiple levels of child nesting work correctly."""
        grandchild_list = [
            (r"weight", "W", []),
            (r"bias", "b", []),
        ]
        child_list = [
            (r"attention\.", "attn.", grandchild_list),
        ]
        psub_list = [
            (r"layers\.(\d+)\.", r"blocks.\1.", child_list),
        ]
        result = sub_param_name("layers.0.attention.weight", psub_list)
        self.assertEqual(result, "blocks.0.attn.W")

        result_bias = sub_param_name("layers.2.attention.bias", psub_list)
        self.assertEqual(result_bias, "blocks.2.attn.b")

    def test_multiple_patterns_in_list(self):
        """Multiple patterns in the same substitution list are tried in order."""
        psub_list = [
            (r"encoder\.", "enc.", []),
            (r"decoder\.", "dec.", []),
        ]
        self.assertEqual(sub_param_name("encoder.weight", psub_list), "enc.weight")
        self.assertEqual(sub_param_name("decoder.weight", psub_list), "dec.weight")

    def test_first_matching_pattern_wins(self):
        """Only the first matching pattern in the list is applied at each level."""
        psub_list = [
            (r"layer\.", "FIRST.", []),
            (r"layer\.", "SECOND.", []),
        ]
        # The first pattern should match, and then the second pattern is also
        # checked against the remaining string. Since "FIRST.weight" does not
        # start with "layer.", only the first match matters.
        result = sub_param_name("layer.weight", psub_list)
        self.assertEqual(result, "FIRST.weight")

    def test_empty_psub_list_returns_original(self):
        """An empty substitution list returns the original string."""
        result = sub_param_name("some.param.name", [])
        self.assertEqual(result, "some.param.name")

    def test_partial_match_replaces_prefix(self):
        """A pattern that matches only the prefix leaves the rest for children."""
        psub_list = [
            (r"model\.", "m.", []),
        ]
        result = sub_param_name("model.layers.0.weight", psub_list)
        self.assertEqual(result, "m.layers.0.weight")

    def test_full_string_match_with_no_tail(self):
        """When the pattern matches the entire string, the tail is empty."""
        psub_list = [
            (r"weight", "W", []),
        ]
        result = sub_param_name("weight", psub_list)
        self.assertEqual(result, "W")


class TestRemapParameterFqns(unittest.TestCase):
    """Test remap_parameter_fqns mapping generation."""

    def test_empty_parameter_list(self):
        """An empty parameter list returns an empty mapping."""
        psub_list = [
            (r"foo", "bar", []),
        ]
        result = remap_parameter_fqns([], psub_list)
        self.assertEqual(result, [])

    def test_single_mapping(self):
        """A single parameter is mapped correctly."""
        psub_list = [
            (r"old\.", "new.", []),
        ]
        result = remap_parameter_fqns(["old.weight"], psub_list)
        self.assertEqual(result, [("old.weight", "new.weight")])

    def test_multiple_mappings(self):
        """Multiple parameters are each mapped independently."""
        psub_list = [
            (r"layers\.(\d+)\.", r"blocks.\1.", []),
        ]
        plist = [
            "layers.0.weight",
            "layers.1.weight",
            "layers.2.bias",
        ]
        result = remap_parameter_fqns(plist, psub_list)
        expected = [
            ("layers.0.weight", "blocks.0.weight"),
            ("layers.1.weight", "blocks.1.weight"),
            ("layers.2.bias", "blocks.2.bias"),
        ]
        self.assertEqual(result, expected)

    def test_no_match_preserves_original(self):
        """Parameters that do not match any pattern are preserved unchanged."""
        psub_list = [
            (r"encoder\.", "enc.", []),
        ]
        result = remap_parameter_fqns(["decoder.weight"], psub_list)
        self.assertEqual(result, [("decoder.weight", "decoder.weight")])

    def test_returns_list_of_tuples(self):
        """The return type is a list of (input_name, output_name) tuples."""
        result = remap_parameter_fqns(["a", "b"], [])
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_mixed_matching_and_non_matching(self):
        """Some parameters match and some do not."""
        psub_list = [
            (r"prefix\.", "p.", []),
        ]
        plist = ["prefix.weight", "other.bias", "prefix.bias"]
        result = remap_parameter_fqns(plist, psub_list)
        expected = [
            ("prefix.weight", "p.weight"),
            ("other.bias", "other.bias"),
            ("prefix.bias", "p.bias"),
        ]
        self.assertEqual(result, expected)


class TestRemapStateDict(unittest.TestCase):
    """Test remap_state_dict with actual tensor values."""

    def test_preserves_tensor_values(self):
        """Tensor values are preserved (same object) after remapping."""
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([4.0, 5.0])
        state_dict = {
            "old.weight": t1,
            "old.bias": t2,
        }
        psub_list = [
            (r"old\.", "new.", []),
        ]
        result = remap_state_dict(state_dict, psub_list)

        self.assertIn("new.weight", result)
        self.assertIn("new.bias", result)
        self.assertNotIn("old.weight", result)
        self.assertNotIn("old.bias", result)
        # The tensor objects should be the exact same references
        self.assertIs(result["new.weight"], t1)
        self.assertIs(result["new.bias"], t2)

    def test_applies_name_mapping(self):
        """Keys in the returned dict reflect the substitution rules."""
        state_dict = {
            "layers.0.attention.weight": torch.zeros(2, 2),
            "layers.0.attention.bias": torch.ones(2),
            "layers.1.attention.weight": torch.zeros(3, 3),
        }
        child_list = [
            (r"attention\.", "attn.", []),
        ]
        psub_list = [
            (r"layers\.(\d+)\.", r"blocks.\1.", child_list),
        ]
        result = remap_state_dict(state_dict, psub_list)

        expected_keys = {
            "blocks.0.attn.weight",
            "blocks.0.attn.bias",
            "blocks.1.attn.weight",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_empty_state_dict(self):
        """An empty state dict returns an empty dict."""
        result = remap_state_dict({}, [(r"foo", "bar", [])])
        self.assertEqual(result, {})

    def test_no_substitutions(self):
        """An empty substitution list returns keys unchanged."""
        t = torch.tensor([1.0])
        state_dict = {"param.weight": t}
        result = remap_state_dict(state_dict, [])
        self.assertIn("param.weight", result)
        self.assertIs(result["param.weight"], t)

    def test_handles_nested_patterns(self):
        """Deeply nested substitution patterns are applied correctly."""
        grandchild = [
            (r"query\.", "q.", []),
            (r"key\.", "k.", []),
            (r"value\.", "v.", []),
        ]
        child = [
            (r"self_attn\.", "attn.", grandchild),
            (r"mlp\.", "ffn.", []),
        ]
        psub_list = [
            (r"model\.layers\.(\d+)\.", r"transformer.h.\1.", child),
        ]

        state_dict = {
            "model.layers.0.self_attn.query.weight": torch.zeros(1),
            "model.layers.0.self_attn.key.weight": torch.zeros(1),
            "model.layers.0.self_attn.value.weight": torch.zeros(1),
            "model.layers.0.mlp.weight": torch.zeros(1),
            "model.layers.1.self_attn.query.bias": torch.zeros(1),
        }

        result = remap_state_dict(state_dict, psub_list)
        expected_keys = {
            "transformer.h.0.attn.q.weight",
            "transformer.h.0.attn.k.weight",
            "transformer.h.0.attn.v.weight",
            "transformer.h.0.ffn.weight",
            "transformer.h.1.attn.q.bias",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_tensor_data_integrity(self):
        """Verify tensor data is not corrupted during remapping."""
        original = torch.randn(4, 4)
        state_dict = {"source.weight": original}
        psub_list = [(r"source\.", "target.", [])]

        result = remap_state_dict(state_dict, psub_list)
        self.assertTrue(torch.equal(result["target.weight"], original))


if __name__ == "__main__":
    unittest.main()
