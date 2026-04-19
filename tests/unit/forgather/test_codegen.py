"""
Unit tests for forgather.codegen
"""

import pytest

from forgather.codegen import PyEncoder, generate_code
from forgather.latent import (
    FactoryNode,
    PartialNode,
    SingletonNode,
    Undefined,
    VarNode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode(obj):
    """Return the full PyEncoder result dict for obj."""
    return PyEncoder()(obj)


def main_body(obj):
    """Return just the main_body string from encoding obj."""
    return encode(obj)["main_body"]


# ---------------------------------------------------------------------------
# Basic type encoding
# ---------------------------------------------------------------------------


class TestPyEncoderBasicTypes:
    def test_encodes_none(self):
        assert main_body(None) == "None"

    def test_encodes_int(self):
        assert main_body(42) == "42"
        assert main_body(0) == "0"
        assert main_body(-7) == "-7"

    def test_encodes_float(self):
        assert main_body(3.14) == "3.14"

    def test_encodes_bool_true(self):
        assert main_body(True) == "True"

    def test_encodes_bool_false(self):
        assert main_body(False) == "False"

    def test_encodes_simple_str(self):
        assert main_body("hello") == repr("hello")

    def test_encodes_multiline_str_with_parentheses(self):
        result = main_body("line1\nline2")
        # Multiline strings are wrapped in parentheses
        assert result.startswith("(")
        assert result.endswith(")")
        assert "line1" in result
        assert "line2" in result

    def test_encodes_list(self):
        result = main_body([1, 2, 3])
        assert result.startswith("[")
        assert result.endswith("]")
        assert "1" in result
        assert "3" in result

    def test_encodes_empty_list(self):
        assert main_body([]) == "[]"

    def test_encodes_dict(self):
        result = main_body({"key": "val"})
        assert result.startswith("{")
        assert result.endswith("}")
        assert "'key'" in result
        assert "'val'" in result

    def test_encodes_empty_dict(self):
        assert main_body({}) == "{}"

    def test_encodes_tuple(self):
        result = main_body((1, 2))
        assert result.startswith("(")
        assert result.endswith(")")
        assert "1" in result
        assert "2" in result

    def test_encodes_empty_tuple(self):
        assert main_body(()) == "tuple()"


# ---------------------------------------------------------------------------
# VarNode encoding
# ---------------------------------------------------------------------------


class TestPyEncoderVarNode:
    def test_varnode_adds_to_vars_set(self):
        v = VarNode("myvar")
        result = encode(v)
        # The variable should be tracked in variables
        assert len(result["variables"]) == 1
        var_name = result["variables"][0][0]
        assert var_name == "myvar"

    def test_varnode_returns_var_name_as_main_body(self):
        v = VarNode("my_param")
        assert main_body(v) == "my_param"

    def test_varnode_with_default_adds_has_default_true(self):
        v = VarNode("lr", default=0.001)
        result = encode(v)
        name, has_default, default_value = result["variables"][0]
        assert name == "lr"
        assert has_default is True
        assert default_value == 0.001

    def test_varnode_without_default_adds_has_default_false(self):
        v = VarNode("required")
        result = encode(v)
        name, has_default, default_value = result["variables"][0]
        assert name == "required"
        assert has_default is False
        assert default_value is Undefined


# ---------------------------------------------------------------------------
# CallableNode encoding – imports
# ---------------------------------------------------------------------------


class TestPyEncoderImports:
    def test_singleton_with_module_tracks_import(self):
        s = SingletonNode("mymodule:MyClass")
        result = encode(s)
        assert ("mymodule", "MyClass") in result["imports"]

    def test_dynamic_import_for_py_module(self):
        s = SingletonNode("mymodule.py:MyClass", submodule_searchpath=["/some/path"])
        result = encode(s)
        assert len(result["dynamic_imports"]) == 1
        module, symbol, searchpath = result["dynamic_imports"][0]
        assert module == "mymodule.py"
        assert symbol == "MyClass"
        assert "/some/path" in searchpath

    def test_singleton_generates_call(self):
        s = SingletonNode("mymodule:MyClass")
        result = main_body(s)
        # SingletonNode becomes a call: MyClass()
        assert "MyClass()" in result

    def test_factory_node_with_args_generates_call(self):
        f = FactoryNode("mymodule:MyFactory", "arg1", key="val")
        result = main_body(f)
        assert "MyFactory(" in result
        assert "'arg1'" in result
        assert "key='val'" in result

    def test_partial_node_without_args_returns_symbol(self):
        p = PartialNode("mymodule:MyPartial")
        result = main_body(p)
        # Empty PartialNode just returns the callable name (no call)
        assert "MyPartial" in result

    def test_partial_node_nested_with_kwargs_generates_partial_call(self):
        p = PartialNode("mymodule:MyPartial", lr=0.01)
        outer = SingletonNode("mymodule:Outer", partial_arg=p)
        result = main_body(outer)
        # When nested at depth > 1, partial() call wraps the node
        assert "partial(MyPartial" in result
        assert "lr=0.01" in result


# ---------------------------------------------------------------------------
# generate_code
# ---------------------------------------------------------------------------


class TestGenerateCode:
    def test_produces_valid_python_string(self):
        result = generate_code(42)
        assert isinstance(result, str)
        # Must contain the return statement
        assert "return 42" in result

    def test_includes_construct_factory_function_by_default(self):
        result = generate_code(42)
        assert "def construct(" in result

    def test_factory_name_kwarg_overrides_function_name(self):
        result = generate_code(42, factory_name="build")
        assert "def build(" in result
        assert "def construct(" not in result

    def test_variables_generate_function_arguments(self):
        v = VarNode("lr", default=0.001)
        result = generate_code(v)
        assert "lr=0.001" in result

    def test_required_variable_generates_argument_without_default(self):
        v = VarNode("required_param")
        result = generate_code(v)
        # No default value for required vars
        assert "required_param" in result
        # The argument should appear without "=" assignment in the function signature
        lines = result.splitlines()
        sig_lines = [l for l in lines if "required_param" in l]
        assert any(
            "required_param," in l.strip() or "required_param" == l.strip().rstrip(",")
            for l in sig_lines
        )

    def test_imports_included_in_generated_code(self):
        s = SingletonNode("mymodule:MyClass")
        result = generate_code(s)
        assert "from mymodule import MyClass" in result

    def test_functools_partial_always_imported(self):
        result = generate_code(42)
        assert "from functools import partial" in result

    def test_generate_code_with_dict_graph(self):
        graph = {"x": 1, "y": 2}
        result = generate_code(graph)
        assert "return" in result
        assert "'x'" in result
        assert "'y'" in result

    def test_generate_code_with_singleton_node(self):
        s = SingletonNode("mymodule:MyClass", lr=0.01)
        result = generate_code(s)
        assert "MyClass(" in result
        assert "lr=0.01" in result


# ---------------------------------------------------------------------------
# Known bug: duplicate _getitem method
# ---------------------------------------------------------------------------


class TestGetItemBug:
    def test_single_getitem_definition_uses_encode(self):
        # The duplicate first _getitem() definition (which used repr(key)) has been removed.
        # The active _getitem uses self._encode(key) for recursive encoding.
        import inspect

        import forgather.codegen as codegen_module

        src = inspect.getsource(codegen_module.PyEncoder._getitem)
        assert (
            "self._encode(key)" in src
        ), "Active _getitem should use self._encode(key)"
        assert (
            "repr(key)" not in src
        ), "Dead-code _getitem using repr(key) should be gone"
