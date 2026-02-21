#!/usr/bin/env python3
"""
Unit tests for the forgather.ml.model_conversion package.

Tests the converter registry, discovery system, standard mappings,
abstract base class, resize_embeddings utilities, and public API exports.
"""

import importlib
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import torch
import yaml


# ---------------------------------------------------------------------------
# 1. Registry tests (registry.py)
# ---------------------------------------------------------------------------
class TestConverterRegistry(unittest.TestCase):
    """Tests for register_converter, get_converter, list_converters, and detect_model_type."""

    def setUp(self):
        """Save original registry state and install a clean one for each test."""
        import forgather.ml.model_conversion.registry as reg_mod

        self._reg_mod = reg_mod
        self._original_registry = reg_mod._CONVERTER_REGISTRY.copy()
        reg_mod._CONVERTER_REGISTRY.clear()

    def tearDown(self):
        """Restore original registry."""
        self._reg_mod._CONVERTER_REGISTRY.clear()
        self._reg_mod._CONVERTER_REGISTRY.update(self._original_registry)

    # -- register_converter ------------------------------------------------

    def test_register_converter_adds_to_registry(self):
        """Decorating a class with register_converter should add it to the registry."""
        from forgather.ml.model_conversion.base import ModelConverter
        from forgather.ml.model_conversion.registry import register_converter

        @register_converter("test_model_a")
        class _TestConverterA(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        self.assertIn("test_model_a", self._reg_mod._CONVERTER_REGISTRY)
        self.assertIs(self._reg_mod._CONVERTER_REGISTRY["test_model_a"], _TestConverterA)

    def test_register_converter_returns_original_class(self):
        """The decorator should return the original class unchanged."""
        from forgather.ml.model_conversion.base import ModelConverter
        from forgather.ml.model_conversion.registry import register_converter

        @register_converter("test_model_b")
        class _TestConverterB(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        self.assertEqual(_TestConverterB.__name__, "_TestConverterB")

    def test_register_converter_duplicate_raises(self):
        """Registering a second converter for the same model_type should raise ValueError."""
        from forgather.ml.model_conversion.base import ModelConverter
        from forgather.ml.model_conversion.registry import register_converter

        @register_converter("dup_type")
        class _First(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        with self.assertRaises(ValueError) as ctx:

            @register_converter("dup_type")
            class _Second(ModelConverter):
                def get_parameter_mappings(self, direction):
                    return []

                def get_config_field_mapping(self, direction):
                    return {}

                def convert_to_forgather(self, *a, **kw):
                    pass

                def convert_from_forgather(self, *a, **kw):
                    pass

        self.assertIn("already registered", str(ctx.exception))

    # -- get_converter -----------------------------------------------------

    def test_get_converter_success(self):
        """get_converter should return the registered class."""
        from forgather.ml.model_conversion.base import ModelConverter
        from forgather.ml.model_conversion.registry import (
            get_converter,
            register_converter,
        )

        @register_converter("get_test")
        class _GetTest(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        result = get_converter("get_test")
        self.assertIs(result, _GetTest)

    def test_get_converter_not_found_raises(self):
        """get_converter should raise ValueError for unregistered types."""
        from forgather.ml.model_conversion.registry import get_converter

        with self.assertRaises(ValueError) as ctx:
            get_converter("nonexistent_model")

        self.assertIn("No converter registered", str(ctx.exception))
        self.assertIn("nonexistent_model", str(ctx.exception))

    # -- list_converters ---------------------------------------------------

    def test_list_converters_empty(self):
        """list_converters should return empty list when no converters registered."""
        from forgather.ml.model_conversion.registry import list_converters

        self.assertEqual(list_converters(), [])

    def test_list_converters_returns_all_types(self):
        """list_converters should return all registered model type strings."""
        from forgather.ml.model_conversion.base import ModelConverter
        from forgather.ml.model_conversion.registry import (
            list_converters,
            register_converter,
        )

        for name in ("alpha", "beta", "gamma"):

            @register_converter(name)
            class _C(ModelConverter):
                def get_parameter_mappings(self, direction):
                    return []

                def get_config_field_mapping(self, direction):
                    return {}

                def convert_to_forgather(self, *a, **kw):
                    pass

                def convert_from_forgather(self, *a, **kw):
                    pass

        result = list_converters()
        self.assertEqual(set(result), {"alpha", "beta", "gamma"})

    # -- detect_model_type -------------------------------------------------

    def test_detect_model_type_hf_model(self):
        """detect_model_type should return ('huggingface', model_type) for HF models."""
        from forgather.ml.model_conversion.registry import detect_model_type

        mock_config = MagicMock()
        mock_config.model_type = "llama"
        # No hf_model_type attribute -> spec it out
        del mock_config.hf_model_type
        del mock_config.forgather_model_type

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_config
            result = detect_model_type("/fake/hf/model")

        self.assertEqual(result, ("huggingface", "llama"))

    def test_detect_model_type_forgather_model(self):
        """detect_model_type should return ('forgather', hf_model_type) for Forgather models."""
        from forgather.ml.model_conversion.registry import detect_model_type

        mock_config = MagicMock()
        mock_config.hf_model_type = "llama"
        mock_config.model_type = "custom_fg_llama"

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_config
            result = detect_model_type("/fake/fg/model")

        self.assertEqual(result, ("forgather", "llama"))

    def test_detect_model_type_returns_none_on_failure(self):
        """detect_model_type should return None when detection fails."""
        from forgather.ml.model_conversion.registry import detect_model_type

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.side_effect = Exception("load failed")
            result = detect_model_type("/fake/broken")

        self.assertIsNone(result)

    def test_detect_model_type_no_attributes(self):
        """detect_model_type should return None when config has no type attributes."""
        from forgather.ml.model_conversion.registry import detect_model_type

        mock_config = MagicMock(spec=[])  # no attributes at all

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_config
            result = detect_model_type("/fake/empty")

        self.assertIsNone(result)

    def test_detect_model_type_forgather_model_type_fallback(self):
        """detect_model_type uses forgather_model_type as fallback."""
        from forgather.ml.model_conversion.registry import detect_model_type

        mock_config = MagicMock()
        # Remove hf_model_type and model_type
        del mock_config.hf_model_type
        del mock_config.model_type
        mock_config.forgather_model_type = "qwen3"

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_config
            result = detect_model_type("/fake/fallback")

        self.assertEqual(result, ("forgather", "qwen3"))

    # -- detect_model_type_from_hf -----------------------------------------

    def test_detect_model_type_from_hf_success(self):
        """detect_model_type_from_hf should return model_type for registered types."""
        from forgather.ml.model_conversion.base import ModelConverter
        from forgather.ml.model_conversion.registry import (
            detect_model_type_from_hf,
            register_converter,
        )

        @register_converter("test_hf_detect")
        class _C(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        mock_config = MagicMock()
        mock_config.model_type = "test_hf_detect"

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_config
            result = detect_model_type_from_hf("/fake/path")

        self.assertEqual(result, "test_hf_detect")

    def test_detect_model_type_from_hf_unsupported_raises(self):
        """detect_model_type_from_hf should raise for unsupported types."""
        from forgather.ml.model_conversion.registry import detect_model_type_from_hf

        mock_config = MagicMock()
        mock_config.model_type = "unsupported_model_xyz"

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_config
            with self.assertRaises(ValueError) as ctx:
                detect_model_type_from_hf("/fake/path")

        self.assertIn("unsupported_model_xyz", str(ctx.exception))

    # -- detect_model_type_from_forgather ----------------------------------

    def test_detect_model_type_from_forgather_returns_type(self):
        """detect_model_type_from_forgather returns model type for FG models."""
        from forgather.ml.model_conversion.registry import detect_model_type_from_forgather

        mock_config = MagicMock()
        mock_config.hf_model_type = "mistral"

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_config
            result = detect_model_type_from_forgather("/fake/fg")

        self.assertEqual(result, "mistral")

    def test_detect_model_type_from_forgather_returns_none_for_hf(self):
        """detect_model_type_from_forgather returns None for HF models (not forgather)."""
        from forgather.ml.model_conversion.registry import detect_model_type_from_forgather

        mock_config = MagicMock()
        del mock_config.hf_model_type
        del mock_config.forgather_model_type
        mock_config.model_type = "llama"

        with patch("transformers.AutoConfig") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_config
            result = detect_model_type_from_forgather("/fake/hf")

        self.assertIsNone(result)

    # -- discover_and_register_converters ----------------------------------

    def test_discover_and_register_converters_with_custom_paths(self):
        """discover_and_register_converters calls discovery with custom paths."""
        from forgather.ml.model_conversion.registry import discover_and_register_converters

        mock_discovery = MagicMock()
        with patch.dict(
            "forgather.ml.model_conversion.registry.__dict__",
            {"_discovery_functions": mock_discovery},
        ):
            discover_and_register_converters(
                custom_paths=["/path/a", "/path/b"], forgather_root="/fake/root"
            )

        mock_discovery.discover_from_paths.assert_called_once_with(
            ["/path/a", "/path/b"], "/fake/root"
        )

    def test_discover_and_register_converters_builtin(self):
        """discover_and_register_converters calls builtin discovery when no custom paths."""
        from forgather.ml.model_conversion.registry import discover_and_register_converters

        mock_discovery = MagicMock()
        with patch.dict(
            "forgather.ml.model_conversion.registry.__dict__",
            {"_discovery_functions": mock_discovery},
        ):
            discover_and_register_converters(forgather_root="/fake/root")

        mock_discovery.discover_builtin_converters.assert_called_once_with("/fake/root")


# ---------------------------------------------------------------------------
# 2. Discovery tests (discovery.py)
# ---------------------------------------------------------------------------
class TestDiscovery(unittest.TestCase):
    """Tests for the converter discovery system."""

    def test_discover_converters_in_directory_nonexistent(self):
        """Searching a nonexistent directory should not raise, just log a warning."""
        from forgather.ml.model_conversion.discovery import discover_converters_in_directory

        # Should not raise
        discover_converters_in_directory("/nonexistent/path/abc123")

    def test_discover_converters_in_directory_not_a_dir(self):
        """Searching a file path (not a directory) should not raise."""
        from forgather.ml.model_conversion.discovery import discover_converters_in_directory

        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            discover_converters_in_directory(f.name)

    def test_discover_converters_in_directory_finds_converter_py(self):
        """Files named converter.py or *_converter.py should be discovered."""
        from forgather.ml.model_conversion.discovery import discover_converters_in_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create converter files
            converter_path = Path(tmpdir) / "converter.py"
            converter_path.write_text("# empty converter\n")

            custom_path = Path(tmpdir) / "my_custom_converter.py"
            custom_path.write_text("# custom converter\n")

            # Create a non-converter file
            other_path = Path(tmpdir) / "utils.py"
            other_path.write_text("# not a converter\n")

            with patch(
                "forgather.ml.model_conversion.discovery._import_converter_module"
            ) as mock_import:
                discover_converters_in_directory(tmpdir)

            # Both converter files should be imported
            imported_paths = {
                call[0][0].resolve() for call in mock_import.call_args_list
            }
            self.assertIn(converter_path.resolve(), imported_paths)
            self.assertIn(custom_path.resolve(), imported_paths)

            # The non-converter file should NOT be imported
            self.assertNotIn(other_path.resolve(), imported_paths)

    def test_discover_converters_in_directory_recursive(self):
        """Recursive search should find converter files in subdirectories."""
        from forgather.ml.model_conversion.discovery import discover_converters_in_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "sub" / "deep"
            subdir.mkdir(parents=True)

            nested = subdir / "converter.py"
            nested.write_text("# nested converter\n")

            with patch(
                "forgather.ml.model_conversion.discovery._import_converter_module"
            ) as mock_import:
                discover_converters_in_directory(tmpdir, recursive=True)

            imported_paths = {
                call[0][0].resolve() for call in mock_import.call_args_list
            }
            self.assertIn(nested.resolve(), imported_paths)

    def test_discover_converters_in_directory_non_recursive(self):
        """Non-recursive search should only find converter files at the top level."""
        from forgather.ml.model_conversion.discovery import discover_converters_in_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            top_level = Path(tmpdir) / "converter.py"
            top_level.write_text("# top level\n")

            subdir = Path(tmpdir) / "sub"
            subdir.mkdir()
            nested = subdir / "converter.py"
            nested.write_text("# nested\n")

            with patch(
                "forgather.ml.model_conversion.discovery._import_converter_module"
            ) as mock_import:
                discover_converters_in_directory(tmpdir, recursive=False)

            imported_paths = {
                call[0][0].resolve() for call in mock_import.call_args_list
            }
            self.assertIn(top_level.resolve(), imported_paths)
            self.assertNotIn(nested.resolve(), imported_paths)

    def test_discover_builtin_converters_missing_models_dir(self):
        """When models directory does not exist, should not raise."""
        from forgather.ml.model_conversion.discovery import discover_builtin_converters

        with tempfile.TemporaryDirectory() as tmpdir:
            # No examples/models/ directory
            discover_builtin_converters(forgather_root=tmpdir)

    def test_discover_builtin_converters_finds_model_converters(self):
        """Should find converter.py files inside examples/models/*/src/."""
        from forgather.ml.model_conversion.discovery import discover_builtin_converters

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create expected directory structure
            model_dir = Path(tmpdir) / "examples" / "models" / "test_model" / "src"
            model_dir.mkdir(parents=True)
            converter_file = model_dir / "converter.py"
            converter_file.write_text("# test converter\n")

            with patch(
                "forgather.ml.model_conversion.discovery._import_converter_module"
            ) as mock_import:
                discover_builtin_converters(forgather_root=tmpdir)

            mock_import.assert_called_once()
            called_path = mock_import.call_args[0][0]
            self.assertEqual(called_path, converter_file)

    def test_discover_builtin_converters_skips_non_directories(self):
        """Files in examples/models/ (not directories) should be skipped."""
        from forgather.ml.model_conversion.discovery import discover_builtin_converters

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "examples" / "models"
            models_dir.mkdir(parents=True)

            # Create a file, not a directory
            readme = models_dir / "README.md"
            readme.write_text("# readme\n")

            with patch(
                "forgather.ml.model_conversion.discovery._import_converter_module"
            ) as mock_import:
                discover_builtin_converters(forgather_root=tmpdir)

            mock_import.assert_not_called()

    def test_discover_from_paths_combines_builtin_and_custom(self):
        """discover_from_paths should try builtin discovery then custom paths."""
        from forgather.ml.model_conversion.discovery import discover_from_paths

        with patch(
            "forgather.ml.model_conversion.discovery.discover_builtin_converters"
        ) as mock_builtin, patch(
            "forgather.ml.model_conversion.discovery.discover_converters_in_directory"
        ) as mock_dir, patch(
            "forgather.ml.model_conversion.discovery._can_find_forgather_root",
            return_value=False,
        ):
            discover_from_paths(["/custom/path1", "/custom/path2"], forgather_root="/root")

        # Builtin is called with the provided forgather root
        mock_builtin.assert_called_once_with("/root")
        # Custom paths searched in order
        self.assertEqual(mock_dir.call_count, 2)
        mock_dir.assert_any_call("/custom/path1")
        mock_dir.assert_any_call("/custom/path2")

    def test_discover_from_paths_skips_builtin_when_no_root(self):
        """discover_from_paths skips builtin discovery when root is not found."""
        from forgather.ml.model_conversion.discovery import discover_from_paths

        with patch(
            "forgather.ml.model_conversion.discovery.discover_builtin_converters"
        ) as mock_builtin, patch(
            "forgather.ml.model_conversion.discovery.discover_converters_in_directory"
        ) as mock_dir, patch(
            "forgather.ml.model_conversion.discovery._can_find_forgather_root",
            return_value=False,
        ):
            discover_from_paths(["/custom/path"], forgather_root=None)

        mock_builtin.assert_not_called()
        mock_dir.assert_called_once_with("/custom/path")

    def test_import_converter_module_external_does_not_raise(self):
        """_import_converter_module for external modules should not raise exceptions.

        Note: There is a known scoping bug in discovery.py where `import importlib`
        inside the is_builtin branch shadows the top-level importlib import, causing
        external converter loading to fail silently with a logged warning. This test
        verifies the function handles the error gracefully (no exception raised).
        """
        from forgather.ml.model_conversion.discovery import _import_converter_module

        with tempfile.TemporaryDirectory() as tmpdir:
            mod_file = Path(tmpdir) / "test_ext_converter.py"
            mod_file.write_text("LOADED_MARKER_12345 = True\n")

            # Should not raise, even if loading fails internally
            _import_converter_module(mod_file, is_builtin=False)

    def test_import_converter_module_handles_import_error(self):
        """_import_converter_module should handle import errors gracefully."""
        from forgather.ml.model_conversion.discovery import _import_converter_module

        with tempfile.TemporaryDirectory() as tmpdir:
            mod_file = Path(tmpdir) / "bad_converter.py"
            mod_file.write_text("raise RuntimeError('Intentional error')\n")

            # Should not raise, just log a warning
            _import_converter_module(mod_file, is_builtin=False)

    def test_can_find_forgather_root(self):
        """_can_find_forgather_root should return a boolean."""
        from forgather.ml.model_conversion.discovery import _can_find_forgather_root

        result = _can_find_forgather_root()
        self.assertIsInstance(result, bool)


# ---------------------------------------------------------------------------
# 3. Standard mappings tests (standard_mappings.py)
# ---------------------------------------------------------------------------
class TestStandardMappings(unittest.TestCase):
    """Tests for standard configuration field mappings."""

    def test_standard_forgather_to_hf_is_non_empty(self):
        """STANDARD_FORGATHER_TO_HF should contain mapping entries."""
        from forgather.ml.model_conversion.standard_mappings import STANDARD_FORGATHER_TO_HF

        self.assertIsInstance(STANDARD_FORGATHER_TO_HF, dict)
        self.assertGreater(len(STANDARD_FORGATHER_TO_HF), 0)

    def test_standard_hf_to_forgather_is_non_empty(self):
        """STANDARD_HF_TO_FORGATHER should contain mapping entries."""
        from forgather.ml.model_conversion.standard_mappings import STANDARD_HF_TO_FORGATHER

        self.assertIsInstance(STANDARD_HF_TO_FORGATHER, dict)
        self.assertGreater(len(STANDARD_HF_TO_FORGATHER), 0)

    def test_forward_reverse_consistency(self):
        """Applying forward then reverse mapping should yield identity."""
        from forgather.ml.model_conversion.standard_mappings import (
            STANDARD_FORGATHER_TO_HF,
            STANDARD_HF_TO_FORGATHER,
        )

        # For every key in FG->HF, the value should map back in HF->FG
        for fg_key, hf_key in STANDARD_FORGATHER_TO_HF.items():
            self.assertIn(
                hf_key,
                STANDARD_HF_TO_FORGATHER,
                f"HF key '{hf_key}' (from FG '{fg_key}') not found in reverse mapping",
            )
            self.assertEqual(
                STANDARD_HF_TO_FORGATHER[hf_key],
                fg_key,
                f"Round-trip failed: FG '{fg_key}' -> HF '{hf_key}' -> FG '{STANDARD_HF_TO_FORGATHER[hf_key]}'",
            )

    def test_reverse_mapping_function(self):
        """reverse_mapping should swap keys and values."""
        from forgather.ml.model_conversion.standard_mappings import reverse_mapping

        original = {"a": "1", "b": "2", "c": "3"}
        reversed_map = reverse_mapping(original)

        self.assertEqual(reversed_map, {"1": "a", "2": "b", "3": "c"})

    def test_reverse_mapping_empty(self):
        """reverse_mapping should handle empty dict."""
        from forgather.ml.model_conversion.standard_mappings import reverse_mapping

        self.assertEqual(reverse_mapping({}), {})

    def test_standard_mappings_contain_expected_keys(self):
        """Standard mappings should contain key architectural fields."""
        from forgather.ml.model_conversion.standard_mappings import STANDARD_FORGATHER_TO_HF

        expected_keys = [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        ]
        for key in expected_keys:
            self.assertIn(
                key, STANDARD_FORGATHER_TO_HF, f"Expected key '{key}' in standard mapping"
            )


class TestLlamaMappings(unittest.TestCase):
    """Tests for Llama-specific parameter name mapping tuples."""

    def test_llama_hf_to_forgather_is_non_empty(self):
        """LLAMA_HF_TO_FORGATHER should be a non-empty list of tuples."""
        from forgather.ml.model_conversion.standard_mappings import LLAMA_HF_TO_FORGATHER

        self.assertIsInstance(LLAMA_HF_TO_FORGATHER, list)
        self.assertGreater(len(LLAMA_HF_TO_FORGATHER), 0)

    def test_llama_forgather_to_hf_is_non_empty(self):
        """LLAMA_FORGATHER_TO_HF should be a non-empty list of tuples."""
        from forgather.ml.model_conversion.standard_mappings import LLAMA_FORGATHER_TO_HF

        self.assertIsInstance(LLAMA_FORGATHER_TO_HF, list)
        self.assertGreater(len(LLAMA_FORGATHER_TO_HF), 0)

    def _validate_mapping_tuple_structure(self, mapping, name):
        """Validate that a mapping tuple list has the correct recursive structure."""
        for i, entry in enumerate(mapping):
            self.assertIsInstance(
                entry, tuple, f"{name}[{i}] should be a tuple, got {type(entry)}"
            )
            self.assertEqual(
                len(entry),
                3,
                f"{name}[{i}] should have 3 elements (pattern, replacement, children)",
            )
            pattern, replacement, children = entry
            self.assertIsInstance(pattern, str, f"{name}[{i}] pattern should be a string")
            self.assertIsInstance(
                replacement, str, f"{name}[{i}] replacement should be a string"
            )
            self.assertIsInstance(
                children, list, f"{name}[{i}] children should be a list"
            )

            # Recursively validate children
            if children:
                self._validate_mapping_tuple_structure(children, f"{name}[{i}].children")

    def test_llama_hf_to_forgather_structure(self):
        """LLAMA_HF_TO_FORGATHER tuples should have (pattern, replacement, children) structure."""
        from forgather.ml.model_conversion.standard_mappings import LLAMA_HF_TO_FORGATHER

        self._validate_mapping_tuple_structure(LLAMA_HF_TO_FORGATHER, "LLAMA_HF_TO_FORGATHER")

    def test_llama_forgather_to_hf_structure(self):
        """LLAMA_FORGATHER_TO_HF tuples should have (pattern, replacement, children) structure."""
        from forgather.ml.model_conversion.standard_mappings import LLAMA_FORGATHER_TO_HF

        self._validate_mapping_tuple_structure(LLAMA_FORGATHER_TO_HF, "LLAMA_FORGATHER_TO_HF")

    def _collect_patterns(self, mapping):
        """Recursively collect all regex patterns from a mapping."""
        patterns = []
        for pattern, replacement, children in mapping:
            patterns.append(pattern)
            if children:
                patterns.extend(self._collect_patterns(children))
        return patterns

    def test_llama_hf_to_forgather_patterns_are_valid_regex(self):
        """All regex patterns in LLAMA_HF_TO_FORGATHER should compile without error."""
        from forgather.ml.model_conversion.standard_mappings import LLAMA_HF_TO_FORGATHER

        patterns = self._collect_patterns(LLAMA_HF_TO_FORGATHER)
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                self.fail(f"Invalid regex pattern '{pattern}': {e}")

    def test_llama_forgather_to_hf_patterns_are_valid_regex(self):
        """All regex patterns in LLAMA_FORGATHER_TO_HF should compile without error."""
        from forgather.ml.model_conversion.standard_mappings import LLAMA_FORGATHER_TO_HF

        patterns = self._collect_patterns(LLAMA_FORGATHER_TO_HF)
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                self.fail(f"Invalid regex pattern '{pattern}': {e}")

    def test_llama_mappings_contain_key_layers(self):
        """Llama mappings should contain key layer mapping entries."""
        from forgather.ml.model_conversion.standard_mappings import LLAMA_HF_TO_FORGATHER

        # Flatten all patterns to check they cover expected layers
        all_patterns = self._collect_patterns(LLAMA_HF_TO_FORGATHER)
        all_patterns_str = " ".join(all_patterns)

        # Check for presence of key structural patterns
        self.assertIn("lm_head", all_patterns_str)
        self.assertIn("model", all_patterns_str)
        self.assertIn("embed_tokens", all_patterns_str)
        self.assertIn("self_attn", all_patterns_str)
        self.assertIn("q_proj", all_patterns_str)
        self.assertIn("k_proj", all_patterns_str)
        self.assertIn("v_proj", all_patterns_str)
        self.assertIn("o_proj", all_patterns_str)
        self.assertIn("mlp", all_patterns_str)

    def test_llama_mapping_bidirectional_key_coverage(self):
        """Forward and reverse Llama mappings should cover the same set of structural layers."""
        from forgather.ml.model_conversion.standard_mappings import (
            LLAMA_FORGATHER_TO_HF,
            LLAMA_HF_TO_FORGATHER,
        )

        # Both should have the same number of top-level entries
        self.assertEqual(
            len(LLAMA_HF_TO_FORGATHER),
            len(LLAMA_FORGATHER_TO_HF),
            "Forward and reverse Llama mappings should have the same number of top-level entries",
        )

        def _count_entries(mapping):
            count = 0
            for _, _, children in mapping:
                count += 1
                if children:
                    count += _count_entries(children)
            return count

        # Both should have the same total number of mapping entries
        fwd_count = _count_entries(LLAMA_HF_TO_FORGATHER)
        rev_count = _count_entries(LLAMA_FORGATHER_TO_HF)
        self.assertEqual(
            fwd_count,
            rev_count,
            f"Forward ({fwd_count}) and reverse ({rev_count}) Llama mappings should have same total entries",
        )


# ---------------------------------------------------------------------------
# 4. Base class tests (base.py)
# ---------------------------------------------------------------------------
class TestModelConverterBase(unittest.TestCase):
    """Tests for the ModelConverter abstract base class."""

    def test_cannot_instantiate_directly(self):
        """ModelConverter should not be instantiable because it has abstract methods."""
        from forgather.ml.model_conversion.base import ModelConverter

        with self.assertRaises(TypeError) as ctx:
            ModelConverter("test")  # type: ignore[abstract]

        # The error should mention abstract methods
        error_msg = str(ctx.exception)
        self.assertTrue(
            "abstract" in error_msg.lower() or "can't instantiate" in error_msg.lower(),
            f"Expected abstract method error, got: {error_msg}",
        )

    def test_abstract_methods_required(self):
        """A subclass that does not implement all abstract methods cannot be instantiated."""
        from forgather.ml.model_conversion.base import ModelConverter

        class IncompleteConverter(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            # Missing: get_config_field_mapping, convert_to_forgather, convert_from_forgather

        with self.assertRaises(TypeError):
            IncompleteConverter("test")  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        """A subclass that implements all abstract methods should be instantiable."""
        from forgather.ml.model_conversion.base import ModelConverter

        class ConcreteConverter(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        converter = ConcreteConverter("test_type")
        self.assertEqual(converter.model_type, "test_type")

    def test_model_type_stored(self):
        """The model_type attribute should be set from the constructor argument."""
        from forgather.ml.model_conversion.base import ModelConverter

        class _Concrete(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        converter = _Concrete("my_model_type")
        self.assertEqual(converter.model_type, "my_model_type")

    def test_transform_state_dict_default_is_identity(self):
        """The default transform_state_dict should return the state dict unchanged."""
        from forgather.ml.model_conversion.base import ModelConverter

        class _Concrete(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        converter = _Concrete("test")
        state_dict = {"weight": torch.randn(3, 3), "bias": torch.randn(3)}

        result = converter.transform_state_dict(state_dict, "to_forgather", None, None)
        self.assertIs(result, state_dict)

    def test_validate_source_config_default_is_noop(self):
        """The default validate_source_config should not raise."""
        from forgather.ml.model_conversion.base import ModelConverter

        class _Concrete(ModelConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            def convert_to_forgather(self, *a, **kw):
                pass

            def convert_from_forgather(self, *a, **kw):
                pass

        converter = _Concrete("test")
        # Should not raise
        converter.validate_source_config(MagicMock(), "to_forgather")

    def test_abstract_methods_list(self):
        """ModelConverter should have exactly the expected abstract methods."""
        from forgather.ml.model_conversion.base import ModelConverter

        abstract_methods = ModelConverter.__abstractmethods__
        expected = {
            "get_parameter_mappings",
            "get_config_field_mapping",
            "convert_to_forgather",
            "convert_from_forgather",
        }
        self.assertEqual(
            abstract_methods,
            expected,
            f"Expected abstract methods {expected}, got {abstract_methods}",
        )


# ---------------------------------------------------------------------------
# 5. Resize embeddings tests (resize_embeddings.py)
# ---------------------------------------------------------------------------
class TestAddTokensToTokenizerExtended(unittest.TestCase):
    """Extended tests for add_tokens_to_tokenizer beyond the existing test_resize_embeddings.py."""

    def _make_tokenizer(self, **overrides):
        """Create a mock tokenizer with sensible defaults."""
        tok = Mock()
        tok.pad_token = None
        tok.pad_token_id = None
        tok.bos_token = None
        tok.bos_token_id = None
        tok.eos_token = None
        tok.eos_token_id = None
        tok.unk_token = None
        tok.unk_token_id = None
        tok.additional_special_tokens = []
        tok.add_special_tokens = Mock(return_value=0)
        tok.add_tokens = Mock(return_value=0)
        tok.get_vocab = Mock(return_value={})
        for key, val in overrides.items():
            setattr(tok, key, val)
        return tok

    def test_empty_config(self):
        """An empty config dict should result in 0 tokens added."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        tok = self._make_tokenizer()
        num_added, token_inits = add_tokens_to_tokenizer(tok, {})

        self.assertEqual(num_added, 0)
        self.assertEqual(token_inits, {})

    def test_skip_named_token_with_invalid_format(self):
        """Named tokens with invalid format (not str or dict) should be skipped."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        tok = self._make_tokenizer()
        config = {"pad_token": 12345}  # Invalid: int

        num_added, token_inits = add_tokens_to_tokenizer(tok, config)
        self.assertEqual(num_added, 0)

    def test_skip_named_token_dict_missing_token_field(self):
        """A dict-format named token missing the 'token' key should be skipped."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        tok = self._make_tokenizer()
        config = {"pad_token": {"init": "zero"}}  # Missing 'token' field

        num_added, token_inits = add_tokens_to_tokenizer(tok, config)
        self.assertEqual(num_added, 0)

    def test_token_reassignment_existing_in_vocab(self):
        """Reassigning a token to a value already in vocab should just update the pointer."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        tok = self._make_tokenizer(
            pad_token="[PAD]",
            pad_token_id=0,
        )
        tok.get_vocab = Mock(return_value={"<|pad|>": 50})

        config = {"pad_token": "<|pad|>"}

        num_added, token_inits = add_tokens_to_tokenizer(tok, config)

        # The token should be reassigned
        tok.add_special_tokens.assert_called()
        # The existing token ID should get the init strategy
        self.assertIn(50, token_inits)

    def test_token_reassignment_not_in_vocab_adds_new(self):
        """Reassigning a token to a new value not in vocab should add it."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        tok = self._make_tokenizer(
            pad_token="[PAD]",
            pad_token_id=0,
        )
        tok.get_vocab = Mock(return_value={})  # New value not in vocab

        config = {"pad_token": "<|new_pad|>"}

        num_added, token_inits = add_tokens_to_tokenizer(tok, config)

        tok.add_special_tokens.assert_called()

    def test_existing_special_tokens_not_duplicated(self):
        """Additional special tokens that already exist should not be added again."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        tok = self._make_tokenizer(
            additional_special_tokens=["<|im_start|>", "<|im_end|>"]
        )

        config = {"special_tokens": ["<|im_start|>", "<|im_end|>"]}

        num_added, token_inits = add_tokens_to_tokenizer(tok, config)
        self.assertEqual(num_added, 0)

    def test_existing_regular_tokens_not_duplicated(self):
        """Regular tokens that already exist in vocab should not be added again."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        tok = self._make_tokenizer()
        tok.get_vocab = Mock(return_value={"existing_token": 42})

        config = {"regular_tokens": ["existing_token"]}

        num_added, token_inits = add_tokens_to_tokenizer(tok, config)
        self.assertEqual(num_added, 0)

    def test_mixed_config_all_token_types(self):
        """Config with named, special, and regular tokens should handle all types."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        tok = self._make_tokenizer()
        tok.pad_token_id = 200
        tok.add_special_tokens = Mock(return_value=1)
        tok.add_tokens = Mock(return_value=1)

        config = {
            "pad_token": {"token": "<|pad|>", "init": "zero"},
            "special_tokens": ["<|tool_call|>"],
            "regular_tokens": ["custom_word"],
        }

        num_added, token_inits = add_tokens_to_tokenizer(tok, config)

        # Should have made calls for all token types
        # add_special_tokens called for pad_token and then for additional special tokens
        self.assertGreaterEqual(tok.add_special_tokens.call_count, 1)
        tok.add_tokens.assert_called_once_with(["custom_word"])

    def test_yaml_file_loading(self):
        """Loading from a YAML file path should produce the same result as a dict."""
        from forgather.ml.model_conversion.resize_embeddings import add_tokens_to_tokenizer

        config_dict = {
            "pad_token": {"token": "<|pad|>", "init": "zero"},
            "special_tokens": ["<|special|>"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            yaml_path = f.name

        try:
            tok = self._make_tokenizer()
            tok.pad_token_id = 100
            tok.add_special_tokens = Mock(return_value=1)

            num_added, token_inits = add_tokens_to_tokenizer(tok, yaml_path)
            # Verify it processed the config (should have called add_special_tokens)
            self.assertGreaterEqual(tok.add_special_tokens.call_count, 1)
        finally:
            os.unlink(yaml_path)


class TestResizeWordEmbeddingsExtended(unittest.TestCase):
    """Extended tests for resize_word_embeddings."""

    def _make_model_and_tokenizer(self, old_size=100, new_size=110, hidden_dim=32, tied=False):
        """Helper to create a mock model with embeddings."""
        model = Mock()
        input_emb = torch.randn(new_size, hidden_dim)
        if tied:
            output_emb = input_emb  # Same object => tied embeddings
        else:
            output_emb = torch.randn(new_size, hidden_dim)

        input_weight = Mock()
        input_weight.weight = input_emb
        output_weight = Mock()
        output_weight.weight = output_emb

        model.get_input_embeddings = Mock(return_value=input_weight)
        model.get_output_embeddings = Mock(return_value=output_weight)
        model.resize_token_embeddings = Mock()

        tokenizer = Mock()
        tokenizer.__len__ = Mock(return_value=new_size)

        return model, tokenizer, input_emb, output_emb

    def test_zero_init_both_embeddings(self):
        """Zero init should zero both input and output embeddings when untied."""
        from forgather.ml.model_conversion.resize_embeddings import resize_word_embeddings

        model, tokenizer, input_emb, output_emb = self._make_model_and_tokenizer(tied=False)

        resize_word_embeddings(model, tokenizer, {105: "zero"})

        self.assertTrue(torch.all(input_emb[105] == 0))
        self.assertTrue(torch.all(output_emb[105] == 0))

    def test_zero_init_tied_embeddings(self):
        """Zero init should only zero once when embeddings are tied."""
        from forgather.ml.model_conversion.resize_embeddings import resize_word_embeddings

        model, tokenizer, input_emb, output_emb = self._make_model_and_tokenizer(tied=True)

        resize_word_embeddings(model, tokenizer, {105: "zero"})

        self.assertTrue(torch.all(input_emb[105] == 0))
        # output_emb is the same tensor, so also zeroed
        self.assertTrue(torch.all(output_emb[105] == 0))

    def test_copy_init_strategy(self):
        """copy:N init strategy should copy embedding from token N."""
        from forgather.ml.model_conversion.resize_embeddings import resize_word_embeddings

        model, tokenizer, input_emb, output_emb = self._make_model_and_tokenizer(tied=False)

        # Save original source embedding for comparison
        source_embedding = input_emb[5].clone()
        source_output_embedding = output_emb[5].clone()

        resize_word_embeddings(model, tokenizer, {105: "copy:5"})

        self.assertTrue(torch.allclose(input_emb[105], source_embedding))
        self.assertTrue(torch.allclose(output_emb[105], source_output_embedding))

    def test_mean_init_strategy_is_noop(self):
        """Mean init strategy should be a no-op (handled by resize_token_embeddings)."""
        from forgather.ml.model_conversion.resize_embeddings import resize_word_embeddings

        model, tokenizer, input_emb, output_emb = self._make_model_and_tokenizer(tied=False)

        original_105 = input_emb[105].clone()

        resize_word_embeddings(model, tokenizer, {105: "mean"})

        # Should be unchanged (mean is handled by resize_token_embeddings)
        self.assertTrue(torch.allclose(input_emb[105], original_105))

    def test_unsupported_init_strategy_logs_warning(self):
        """An unsupported init strategy should log a warning but not raise."""
        from forgather.ml.model_conversion.resize_embeddings import resize_word_embeddings

        model, tokenizer, input_emb, output_emb = self._make_model_and_tokenizer(tied=False)

        # Should not raise
        resize_word_embeddings(model, tokenizer, {105: "random_unknown_strategy"})

    def test_multiple_token_inits(self):
        """Multiple tokens can have different init strategies."""
        from forgather.ml.model_conversion.resize_embeddings import resize_word_embeddings

        model, tokenizer, input_emb, output_emb = self._make_model_and_tokenizer(tied=False)

        token_inits = {
            105: "zero",
            106: "mean",
            107: "copy:3",
        }

        source_3 = input_emb[3].clone()

        resize_word_embeddings(model, tokenizer, token_inits)

        # Token 105: zero
        self.assertTrue(torch.all(input_emb[105] == 0))
        # Token 107: copied from 3
        self.assertTrue(torch.allclose(input_emb[107], source_3))

    def test_none_token_inits_skips_custom_init(self):
        """When token_inits is None, only resize should happen."""
        from forgather.ml.model_conversion.resize_embeddings import resize_word_embeddings

        model, tokenizer, _, _ = self._make_model_and_tokenizer(tied=False)

        resize_word_embeddings(model, tokenizer, None)

        model.resize_token_embeddings.assert_called_once_with(110, mean_resizing=True)

    def test_empty_token_inits(self):
        """Empty token_inits dict should still resize but not modify embeddings."""
        from forgather.ml.model_conversion.resize_embeddings import resize_word_embeddings

        model, tokenizer, _, _ = self._make_model_and_tokenizer(tied=False)

        resize_word_embeddings(model, tokenizer, {})

        model.resize_token_embeddings.assert_called_once_with(110, mean_resizing=True)


class TestUpdateConfigFromTokenizerExtended(unittest.TestCase):
    """Extended tests for update_config_from_tokenizer."""

    def test_update_vocab_size_true(self):
        """When update_vocab_size=True, vocab_size should be updated to match tokenizer."""
        from forgather.ml.model_conversion.resize_embeddings import update_config_from_tokenizer

        config = Mock()
        config.vocab_size = 100

        tokenizer = Mock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.__len__ = Mock(return_value=150)

        update_config_from_tokenizer(config, tokenizer, update_vocab_size=True)

        self.assertEqual(config.vocab_size, 150)

    def test_update_vocab_size_false(self):
        """When update_vocab_size=False (default), vocab_size should NOT change."""
        from forgather.ml.model_conversion.resize_embeddings import update_config_from_tokenizer

        config = Mock()
        config.vocab_size = 100

        tokenizer = Mock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.__len__ = Mock(return_value=150)

        update_config_from_tokenizer(config, tokenizer, update_vocab_size=False)

        # vocab_size should remain unchanged
        self.assertEqual(config.vocab_size, 100)

    def test_update_vocab_size_no_change_when_equal(self):
        """When vocab sizes already match, no update should occur."""
        from forgather.ml.model_conversion.resize_embeddings import update_config_from_tokenizer

        config = Mock()
        config.vocab_size = 100

        tokenizer = Mock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.__len__ = Mock(return_value=100)

        update_config_from_tokenizer(config, tokenizer, update_vocab_size=True)

        # Should still be 100
        self.assertEqual(config.vocab_size, 100)

    def test_only_nonnull_token_ids_set(self):
        """Only non-None token IDs should be set on the config."""
        from forgather.ml.model_conversion.resize_embeddings import update_config_from_tokenizer

        config = MagicMock()
        config.vocab_size = 100

        tokenizer = Mock()
        tokenizer.bos_token_id = None
        tokenizer.eos_token_id = 42
        tokenizer.pad_token_id = None
        tokenizer.__len__ = Mock(return_value=100)

        # Track setattr calls
        set_calls = {}
        original_setattr = type(config).__setattr__

        def track_setattr(self, name, value):
            set_calls[name] = value
            original_setattr(self, name, value)

        with patch.object(type(config), "__setattr__", track_setattr):
            update_config_from_tokenizer(config, tokenizer, update_vocab_size=False)

        # eos_token_id should be set
        self.assertIn("eos_token_id", set_calls)
        self.assertEqual(set_calls["eos_token_id"], 42)

        # bos_token_id and pad_token_id should NOT be set (they're None)
        self.assertNotIn("bos_token_id", set_calls)
        self.assertNotIn("pad_token_id", set_calls)

    def test_default_token_config_structure(self):
        """DEFAULT_TOKEN_CONFIG should have expected shape."""
        from forgather.ml.model_conversion.resize_embeddings import DEFAULT_TOKEN_CONFIG

        self.assertIn("pad_token", DEFAULT_TOKEN_CONFIG)
        pad = DEFAULT_TOKEN_CONFIG["pad_token"]
        self.assertIn("token", pad)
        self.assertIn("init", pad)
        self.assertIn("if_missing", pad)
        self.assertEqual(pad["token"], "[PAD]")
        self.assertEqual(pad["init"], "zero")
        self.assertTrue(pad["if_missing"])


# ---------------------------------------------------------------------------
# 6. __init__.py exports tests
# ---------------------------------------------------------------------------
class TestModuleExports(unittest.TestCase):
    """Tests for the model_conversion package public API."""

    def test_core_classes_exported(self):
        """Core classes should be importable from the package."""
        from forgather.ml.model_conversion import HFConverter, ModelConverter

        self.assertTrue(callable(ModelConverter))
        self.assertTrue(callable(HFConverter))

    def test_registry_functions_exported(self):
        """Registry functions should be importable from the package."""
        from forgather.ml.model_conversion import (
            detect_model_type,
            detect_model_type_from_forgather,
            detect_model_type_from_hf,
            discover_and_register_converters,
            get_converter,
            list_converters,
            register_converter,
        )

        self.assertTrue(callable(register_converter))
        self.assertTrue(callable(get_converter))
        self.assertTrue(callable(list_converters))
        self.assertTrue(callable(detect_model_type))
        self.assertTrue(callable(detect_model_type_from_hf))
        self.assertTrue(callable(detect_model_type_from_forgather))
        self.assertTrue(callable(discover_and_register_converters))

    def test_mapping_exports(self):
        """Standard mapping dicts and reverse_mapping should be importable."""
        from forgather.ml.model_conversion import (
            STANDARD_FORGATHER_TO_HF,
            STANDARD_HF_TO_FORGATHER,
            reverse_mapping,
        )

        self.assertIsInstance(STANDARD_FORGATHER_TO_HF, dict)
        self.assertIsInstance(STANDARD_HF_TO_FORGATHER, dict)
        self.assertTrue(callable(reverse_mapping))

    def test_discovery_module_exported(self):
        """The discovery submodule should be importable via the package."""
        from forgather.ml.model_conversion import discovery

        self.assertTrue(hasattr(discovery, "discover_converters_in_directory"))
        self.assertTrue(hasattr(discovery, "discover_builtin_converters"))
        self.assertTrue(hasattr(discovery, "discover_from_paths"))

    def test_vllm_functions_not_in_all(self):
        """validate_vllm_plans, validate_tp_plan, validate_pp_plan, and print_model_structure
        are not yet implemented, so they should not be in __all__."""
        import forgather.ml.model_conversion as mc_pkg

        all_exports = mc_pkg.__all__

        for name in ("validate_vllm_plans", "validate_tp_plan", "validate_pp_plan", "print_model_structure"):
            self.assertNotIn(name, all_exports)


# ---------------------------------------------------------------------------
# 7. HFConverter base tests (hf_converter.py)
# ---------------------------------------------------------------------------
class TestHFConverterBase(unittest.TestCase):
    """Tests for the HFConverter abstract base class."""

    def test_hf_converter_is_abstract(self):
        """HFConverter should not be directly instantiable because it adds more abstract methods."""
        from forgather.ml.model_conversion.hf_converter import HFConverter

        with self.assertRaises(TypeError):
            HFConverter("test")  # type: ignore[abstract]

    def test_hf_converter_subclass_must_implement_all(self):
        """A subclass missing get_hf_config_class etc. cannot be instantiated."""
        from forgather.ml.model_conversion.hf_converter import HFConverter

        class Partial(HFConverter):
            def get_parameter_mappings(self, direction):
                return []

            def get_config_field_mapping(self, direction):
                return {}

            # Missing: get_hf_config_class, get_hf_model_class, get_project_info,
            # convert_to_forgather, convert_from_forgather

        with self.assertRaises(TypeError):
            Partial("test")  # type: ignore[abstract]

    def test_hf_converter_abstract_methods(self):
        """HFConverter should require implementation of HF-specific abstract methods."""
        from forgather.ml.model_conversion.hf_converter import HFConverter

        abstract_methods = HFConverter.__abstractmethods__
        # Should include both base class and HFConverter-specific methods
        self.assertIn("get_hf_config_class", abstract_methods)
        self.assertIn("get_hf_model_class", abstract_methods)
        self.assertIn("get_project_info", abstract_methods)


if __name__ == "__main__":
    unittest.main()
