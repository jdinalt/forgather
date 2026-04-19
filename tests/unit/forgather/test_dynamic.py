"""
Unit tests for forgather.dynamic
"""

import os
import sys
import tempfile

import pytest

from forgather.dynamic import (
    DynamicImportParseError,
    dynamic_import,
    encode_import_spec,
    from_dynamic_module_import,
    get_builtin,
    import_dynamic_module,
    normalize_import_spec,
    parse_dynamic_import_spec,
    parse_module_name_or_path,
)


class TestParseModuleNameOrPath:
    def test_module_name_no_path(self):
        name, path = parse_module_name_or_path("os")
        assert name == "os"
        assert path is None

    def test_dotted_module_name(self):
        name, path = parse_module_name_or_path("os.path")
        assert name == "os.path"
        assert path is None

    def test_file_path(self):
        name, path = parse_module_name_or_path("/some/path/mymodule.py")
        assert name == "mymodule"
        assert path == "/some/path/mymodule.py"

    def test_relative_file_path(self):
        name, path = parse_module_name_or_path("./stuff/my_module.py")
        assert name == "my_module"
        assert path == "./stuff/my_module.py"

    def test_simple_filename(self):
        name, path = parse_module_name_or_path("mymodule.py")
        assert name == "mymodule"
        assert path == "mymodule.py"


class TestParseDynamicImportSpec:
    def test_simple_module_and_symbol(self):
        module, symbol = parse_dynamic_import_spec("os:getcwd")
        assert module == "os"
        assert symbol == "getcwd"

    def test_dotted_module(self):
        module, symbol = parse_dynamic_import_spec("os.path:join")
        assert module == "os.path"
        assert symbol == "join"

    def test_nested_symbol(self):
        module, symbol = parse_dynamic_import_spec("os.path:join")
        assert symbol == "join"

    def test_nested_attribute_symbol(self):
        module, symbol = parse_dynamic_import_spec("my_module:MyClass.my_method")
        assert module == "my_module"
        assert symbol == "MyClass.my_method"

    def test_file_path_module(self):
        module, symbol = parse_dynamic_import_spec("./stuff/my_module.py:MyClass")
        assert module == "./stuff/my_module.py"
        assert symbol == "MyClass"

    def test_type_error_for_non_string(self):
        with pytest.raises(TypeError, match="import_spec must be of type 'str'"):
            parse_dynamic_import_spec(123)

    def test_parse_error_no_colon(self):
        with pytest.raises(DynamicImportParseError):
            parse_dynamic_import_spec("invalid_no_colon")

    def test_parse_error_multiple_colons(self):
        """Multiple colons produce an error since split(':', maxsplit) gives len > 2."""
        # Actually this test depends on implementation - two colons gives 3 parts
        # Let's verify what actually happens
        with pytest.raises(DynamicImportParseError):
            parse_dynamic_import_spec("a:b:c")

    def test_parse_error_empty_string(self):
        with pytest.raises(DynamicImportParseError):
            parse_dynamic_import_spec("")


class TestEncodeImportSpec:
    def test_basic_encoding(self):
        result = encode_import_spec("os.path", "join")
        assert result == "os.path:join"

    def test_roundtrip(self):
        original = "torch.nn:Linear"
        module, symbol = parse_dynamic_import_spec(original)
        reconstructed = encode_import_spec(module, symbol)
        assert reconstructed == original


class TestNormalizeImportSpec:
    def test_module_name_unchanged(self):
        spec = "os.path:join"
        assert normalize_import_spec(spec) == spec

    def test_path_normalized(self):
        spec = "/foo/../foo/module.py:Class"
        normalized = normalize_import_spec(spec)
        assert "/../" not in normalized

    def test_path_becomes_absolute(self):
        spec = "./module.py:Foo"
        normalized = normalize_import_spec(spec)
        # Result should be absolute path
        module, _ = parse_dynamic_import_spec(normalized)
        assert os.path.isabs(module)


class TestGetBuiltin:
    def test_get_int(self):
        assert get_builtin("int") is int

    def test_get_str(self):
        assert get_builtin("str") is str

    def test_get_list(self):
        assert get_builtin("list") is list

    def test_get_dict(self):
        assert get_builtin("dict") is dict

    def test_get_nonexistent(self):
        assert get_builtin("nonexistent_builtin_xyz_abc") is None

    def test_get_none_for_empty_subattr(self):
        # Accessing a missing sub-attribute returns None
        assert get_builtin("int.nonexistent") is None


class TestImportDynamicModule:
    def test_stdlib_module(self):
        mod = import_dynamic_module("os")
        import os as real_os

        assert mod is real_os

    def test_dotted_stdlib_module(self):
        mod = import_dynamic_module("os.path")
        import os.path as real_os_path

        assert mod is real_os_path

    def test_from_file(self, tmp_path):
        module_file = tmp_path / "test_dynamic_module.py"
        module_file.write_text("MY_VALUE = 42\n")
        mod = import_dynamic_module(str(module_file))
        assert mod.MY_VALUE == 42


class TestFromDynamicModuleImport:
    def test_simple_symbol(self):
        result = from_dynamic_module_import("os", "getcwd")
        assert result is os.getcwd

    def test_nested_symbol(self):
        result = from_dynamic_module_import("os", "path.join")
        assert result is os.path.join

    def test_from_file(self, tmp_path):
        module_file = tmp_path / "test_from_dynamic.py"
        module_file.write_text("class Foo:\n    pass\n")
        Foo = from_dynamic_module_import(str(module_file), "Foo")
        assert Foo.__name__ == "Foo"


class TestDynamicImport:
    def test_stdlib(self):
        join = dynamic_import("os.path:join")
        assert join is os.path.join

    def test_pathlib(self):
        from pathlib import Path

        PathCls = dynamic_import("pathlib:Path")
        assert PathCls is Path

    def test_builtins(self):
        list_cls = dynamic_import("builtins:list")
        assert list_cls is list

    def test_from_file(self, tmp_path):
        module_file = tmp_path / "test_dynimport.py"
        module_file.write_text("MY_CONST = 'hello'\n")
        result = dynamic_import(f"{module_file}:MY_CONST")
        assert result == "hello"

    def test_invalid_module_raises(self):
        with pytest.raises(Exception):
            dynamic_import("nonexistent_module_xyz:SomeClass")
