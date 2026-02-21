#!/usr/bin/env python3
"""
Unit tests for uncovered functions in forgather.ml.construct.

Covers: register_for_auto_class, add_special_tokens, _check_needs_build,
torch_dtype, module_to_dtype, _should_write_file, dependency_list,
_compare_file_to_str.

Functions already covered in test_file_locking.py (file_lock_build, build_sync,
build_rule, copy_package_files, write_file) are NOT duplicated here.
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from forgather.ml.construct import (
    _check_needs_build,
    _compare_file_to_str,
    _should_write_file,
    add_special_tokens,
    dependency_list,
    module_to_dtype,
    register_for_auto_class,
    torch_dtype,
)


class TestRegisterForAutoClass(unittest.TestCase):
    """Test register_for_auto_class function."""

    def test_calls_register_and_returns_object(self):
        """Should call register_for_auto_class on the object and return it."""
        mock_obj = MagicMock()
        result = register_for_auto_class(mock_obj)
        mock_obj.register_for_auto_class.assert_called_once_with()
        self.assertIs(result, mock_obj)

    def test_passes_args_through(self):
        """Should forward positional arguments to the method."""
        mock_obj = MagicMock()
        result = register_for_auto_class(mock_obj, "AutoModel")
        mock_obj.register_for_auto_class.assert_called_once_with("AutoModel")
        self.assertIs(result, mock_obj)

    def test_passes_kwargs_through(self):
        """Should forward keyword arguments to the method."""
        mock_obj = MagicMock()
        result = register_for_auto_class(mock_obj, cls_name="AutoModelForCausalLM")
        mock_obj.register_for_auto_class.assert_called_once_with(
            cls_name="AutoModelForCausalLM"
        )
        self.assertIs(result, mock_obj)

    def test_passes_mixed_args_kwargs(self):
        """Should forward both positional and keyword arguments."""
        mock_obj = MagicMock()
        result = register_for_auto_class(mock_obj, "AutoModel", trust_remote_code=True)
        mock_obj.register_for_auto_class.assert_called_once_with(
            "AutoModel", trust_remote_code=True
        )
        self.assertIs(result, mock_obj)


class TestAddSpecialTokens(unittest.TestCase):
    """Test add_special_tokens function."""

    def test_calls_add_special_tokens_and_returns_tokenizer(self):
        """Should call add_special_tokens on the tokenizer and return it."""
        mock_tokenizer = MagicMock()
        token_map = {"pad_token": "<pad>", "eos_token": "<eos>"}
        result = add_special_tokens(mock_tokenizer, token_map)
        mock_tokenizer.add_special_tokens.assert_called_once_with(token_map)
        self.assertIs(result, mock_tokenizer)

    def test_empty_token_map(self):
        """Should handle an empty token map."""
        mock_tokenizer = MagicMock()
        result = add_special_tokens(mock_tokenizer, {})
        mock_tokenizer.add_special_tokens.assert_called_once_with({})
        self.assertIs(result, mock_tokenizer)


class TestCheckNeedsBuild(unittest.TestCase):
    """Test _check_needs_build function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_target_does_not_exist(self):
        """When target does not exist, should return True."""
        target = os.path.join(self.tmpdir, "nonexistent")
        result = _check_needs_build(target, [])
        self.assertTrue(result)

    def test_target_exists_no_prerequisites(self):
        """When target exists and there are no prerequisites, should return False."""
        target = os.path.join(self.tmpdir, "target")
        with open(target, "w") as f:
            f.write("target")
        result = _check_needs_build(target, [])
        self.assertFalse(result)

    def test_target_newer_than_prerequisites(self):
        """When target is newer than all prerequisites, should return False."""
        prereq = os.path.join(self.tmpdir, "prereq")
        with open(prereq, "w") as f:
            f.write("prereq")

        time.sleep(0.05)

        target = os.path.join(self.tmpdir, "target")
        with open(target, "w") as f:
            f.write("target")

        result = _check_needs_build(target, [prereq])
        self.assertFalse(result)

    def test_prerequisite_newer_than_target(self):
        """When a prerequisite is newer than the target, should return True."""
        target = os.path.join(self.tmpdir, "target")
        with open(target, "w") as f:
            f.write("target")

        time.sleep(0.05)

        prereq = os.path.join(self.tmpdir, "prereq")
        with open(prereq, "w") as f:
            f.write("prereq")

        result = _check_needs_build(target, [prereq])
        self.assertTrue(result)

    def test_nonexistent_prerequisite_ignored(self):
        """Non-existent prerequisites are skipped (not treated as newer)."""
        target = os.path.join(self.tmpdir, "target")
        with open(target, "w") as f:
            f.write("target")

        nonexistent = os.path.join(self.tmpdir, "nonexistent_prereq")
        result = _check_needs_build(target, [nonexistent])
        self.assertFalse(result)

    def test_mixed_prerequisites(self):
        """When one prereq is newer and one is older, should return True."""
        old_prereq = os.path.join(self.tmpdir, "old_prereq")
        with open(old_prereq, "w") as f:
            f.write("old")

        time.sleep(0.05)

        target = os.path.join(self.tmpdir, "target")
        with open(target, "w") as f:
            f.write("target")

        time.sleep(0.05)

        new_prereq = os.path.join(self.tmpdir, "new_prereq")
        with open(new_prereq, "w") as f:
            f.write("new")

        result = _check_needs_build(target, [old_prereq, new_prereq])
        self.assertTrue(result)


class TestTorchDtype(unittest.TestCase):
    """Test torch_dtype function."""

    def test_float32(self):
        """Should return torch.float32."""
        self.assertEqual(torch_dtype("float32"), torch.float32)

    def test_bfloat16(self):
        """Should return torch.bfloat16."""
        self.assertEqual(torch_dtype("bfloat16"), torch.bfloat16)

    def test_float16(self):
        """Should return torch.float16."""
        self.assertEqual(torch_dtype("float16"), torch.float16)

    def test_int64(self):
        """Should return torch.int64."""
        self.assertEqual(torch_dtype("int64"), torch.int64)

    def test_int8(self):
        """Should return torch.int8."""
        self.assertEqual(torch_dtype("int8"), torch.int8)

    def test_bool(self):
        """Should return torch.bool."""
        self.assertEqual(torch_dtype("bool"), torch.bool)

    def test_float_alias(self):
        """'float' should return torch.float (same as torch.float32)."""
        self.assertEqual(torch_dtype("float"), torch.float)

    def test_half_alias(self):
        """'half' should return torch.half (same as torch.float16)."""
        self.assertEqual(torch_dtype("half"), torch.half)

    def test_double_alias(self):
        """'double' should return torch.double (same as torch.float64)."""
        self.assertEqual(torch_dtype("double"), torch.double)

    def test_invalid_key_raises_key_error(self):
        """An invalid dtype string should raise KeyError."""
        with self.assertRaises(KeyError):
            torch_dtype("invalid_dtype")

    def test_uint8(self):
        """Should return torch.uint8."""
        self.assertEqual(torch_dtype("uint8"), torch.uint8)

    def test_complex64(self):
        """Should return torch.complex64."""
        self.assertEqual(torch_dtype("complex64"), torch.complex64)


class TestModuleToDtype(unittest.TestCase):
    """Test module_to_dtype function."""

    def test_constructs_and_converts(self):
        """Should construct a module and convert it to the specified dtype."""
        result = module_to_dtype(
            lambda **kwargs: nn.Linear(**kwargs),
            "bfloat16",
            in_features=4,
            out_features=2,
        )
        self.assertIsInstance(result, nn.Linear)
        self.assertEqual(result.weight.dtype, torch.bfloat16)
        self.assertEqual(result.bias.dtype, torch.bfloat16)

    def test_float16_conversion(self):
        """Should convert to float16."""
        result = module_to_dtype(
            lambda **kwargs: nn.Linear(**kwargs),
            "float16",
            in_features=3,
            out_features=1,
        )
        self.assertEqual(result.weight.dtype, torch.float16)

    def test_preserves_module_structure(self):
        """Converted module should have the same structure."""
        result = module_to_dtype(
            lambda **kwargs: nn.Linear(**kwargs),
            "float32",
            in_features=10,
            out_features=5,
        )
        self.assertEqual(result.in_features, 10)
        self.assertEqual(result.out_features, 5)

    def test_no_kwargs_constructor(self):
        """Should work with a constructor that takes no kwargs."""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3))

        result = module_to_dtype(SimpleModule, "bfloat16")
        self.assertEqual(result.param.dtype, torch.bfloat16)


class TestShouldWriteFile(unittest.TestCase):
    """Test _should_write_file function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.existing_file = os.path.join(self.tmpdir, "existing.txt")
        with open(self.existing_file, "w") as f:
            f.write("content")
        self.nonexistent_file = os.path.join(self.tmpdir, "nonexistent.txt")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_file_does_not_exist_returns_true(self):
        """When file does not exist, should return True regardless of policy."""
        for policy in ("ok", "warn", "skip", "raise"):
            result = _should_write_file(self.nonexistent_file, policy)
            self.assertTrue(result, f"Expected True for policy '{policy}' when file does not exist")

    def test_existing_file_ok(self):
        """With 'ok' policy and existing file, should return True."""
        result = _should_write_file(self.existing_file, "ok")
        self.assertTrue(result)

    def test_existing_file_warn(self):
        """With 'warn' policy and existing file, should return True (and log a warning)."""
        result = _should_write_file(self.existing_file, "warn")
        self.assertTrue(result)

    def test_existing_file_skip(self):
        """With 'skip' policy and existing file, should return False."""
        result = _should_write_file(self.existing_file, "skip")
        self.assertFalse(result)

    def test_existing_file_raise(self):
        """With 'raise' policy and existing file, should raise RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            _should_write_file(self.existing_file, "raise")
        self.assertIn("overwrite is prohibited", str(ctx.exception))

    def test_existing_file_invalid_policy(self):
        """With an invalid policy and existing file, should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _should_write_file(self.existing_file, "invalid_policy")
        self.assertIn("File overwrite policy must be one of", str(ctx.exception))

    def test_directory_not_treated_as_file(self):
        """os.path.isfile returns False for directories, so should return True."""
        dir_path = os.path.join(self.tmpdir, "some_dir")
        os.makedirs(dir_path)
        result = _should_write_file(dir_path, "raise")
        self.assertTrue(result)


class TestDependencyList(unittest.TestCase):
    """Test dependency_list function."""

    def test_returns_first_argument(self):
        """Should return the first argument unchanged."""
        obj = {"key": "value"}
        result = dependency_list(obj, "dep1", "dep2")
        self.assertIs(result, obj)

    def test_single_argument(self):
        """Should work with just one argument."""
        obj = [1, 2, 3]
        result = dependency_list(obj)
        self.assertIs(result, obj)

    def test_none_first_argument(self):
        """Should return None if first argument is None."""
        result = dependency_list(None, "dep1")
        self.assertIsNone(result)

    def test_ignores_subsequent_arguments(self):
        """Only the first argument matters; others are dependencies resolved for side effects."""
        sentinel = object()
        result = dependency_list(sentinel, "a", "b", "c", "d")
        self.assertIs(result, sentinel)


class TestCompareFileToStr(unittest.TestCase):
    """Test _compare_file_to_str function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_file_matches_string(self):
        """Should return True when file content matches the string."""
        file_path = os.path.join(self.tmpdir, "test.txt")
        content = "hello world"
        with open(file_path, "w") as f:
            f.write(content)
        self.assertTrue(_compare_file_to_str(file_path, content))

    def test_file_differs_from_string(self):
        """Should return False when file content differs from the string."""
        file_path = os.path.join(self.tmpdir, "test.txt")
        with open(file_path, "w") as f:
            f.write("hello world")
        self.assertFalse(_compare_file_to_str(file_path, "different content"))

    def test_file_does_not_exist(self):
        """Should return False when file does not exist."""
        file_path = os.path.join(self.tmpdir, "nonexistent.txt")
        self.assertFalse(_compare_file_to_str(file_path, "anything"))

    def test_empty_file_matches_empty_string(self):
        """Should return True for empty file vs empty string."""
        file_path = os.path.join(self.tmpdir, "empty.txt")
        with open(file_path, "w") as f:
            pass
        self.assertTrue(_compare_file_to_str(file_path, ""))

    def test_empty_file_does_not_match_nonempty_string(self):
        """Should return False for empty file vs non-empty string."""
        file_path = os.path.join(self.tmpdir, "empty.txt")
        with open(file_path, "w") as f:
            pass
        self.assertFalse(_compare_file_to_str(file_path, "content"))

    def test_multiline_content(self):
        """Should correctly compare multiline content."""
        file_path = os.path.join(self.tmpdir, "multiline.txt")
        content = "line1\nline2\nline3\n"
        with open(file_path, "w") as f:
            f.write(content)
        self.assertTrue(_compare_file_to_str(file_path, content))

    def test_directory_returns_false(self):
        """Should return False if path is a directory (os.path.isfile returns False)."""
        dir_path = os.path.join(self.tmpdir, "some_dir")
        os.makedirs(dir_path)
        self.assertFalse(_compare_file_to_str(dir_path, "anything"))


if __name__ == "__main__":
    unittest.main()
