"""
Unit tests for forgather.yaml_encoder
"""

import pytest

from forgather.latent import (
    FactoryNode,
    PartialNode,
    SingletonNode,
    Undefined,
    VarNode,
)
from forgather.yaml_encoder import YamlEncoder, to_yaml


class TestYamlEncoderBasicTypes:
    def setup_method(self):
        self.enc = YamlEncoder()

    def test_encodes_none_as_null(self):
        result = self.enc(None)
        assert result == "null"

    def test_encodes_plain_string(self):
        result = self.enc("hello")
        assert "'hello'" == result

    def test_encodes_multiline_string_with_pipe(self):
        result = self.enc("line1\nline2")
        assert result.startswith("|")
        assert "line1" in result
        assert "line2" in result

    def test_encodes_int(self):
        result = self.enc(42)
        assert result == "42"

    def test_encodes_float(self):
        result = self.enc(3.14)
        assert "3.14" in result

    def test_encodes_bool_true(self):
        result = self.enc(True)
        assert result == "True"

    def test_encodes_bool_false(self):
        result = self.enc(False)
        assert result == "False"

    def test_encodes_list(self):
        result = self.enc([1, 2, 3])
        assert "- 1" in result
        assert "- 2" in result
        assert "- 3" in result

    def test_encodes_dict(self):
        result = self.enc({"key": "value"})
        assert "key:" in result
        assert "'value'" in result

    def test_encodes_tuple(self):
        result = self.enc((1, 2))
        # Tuples get the !tuple tag
        assert "!tuple" in result
        assert "- 1" in result
        assert "- 2" in result


class TestYamlEncoderVarNode:
    def setup_method(self):
        self.enc = YamlEncoder()

    def test_encodes_varnode_without_default(self):
        v = VarNode("myvar")
        result = self.enc(v)
        assert "!var" in result
        assert "myvar" in result
        # No default info
        assert "default" not in result

    def test_encodes_varnode_with_default(self):
        v = VarNode("myvar", default=42)
        result = self.enc(v)
        assert "!var" in result
        assert "myvar" in result
        assert "default" in result
        assert "42" in result


class TestYamlEncoderCallableNodes:
    def setup_method(self):
        self.enc = YamlEncoder()

    def test_encodes_factory_node(self):
        f = FactoryNode("mymodule:MyClass")
        result = self.enc(f)
        assert "!factory:mymodule:MyClass" in result

    def test_encodes_singleton_node(self):
        s = SingletonNode("mymodule:MySingleton")
        result = self.enc(s)
        assert "!singleton:mymodule:MySingleton" in result

    def test_encodes_partial_node_as_lambda(self):
        p = PartialNode("mymodule:MyPartial")
        result = self.enc(p)
        assert "!lambda:mymodule:MyPartial" in result

    def test_named_node_gets_anchor(self):
        s = SingletonNode("mymodule:MySingleton", _identity="my_name")
        result = self.enc(s)
        assert "&my_name" in result

    def test_repeated_node_uses_alias(self):
        s = SingletonNode("mymodule:MySingleton", _identity="shared")
        # Referencing the same object twice forces it into the name_map
        graph = {"a": s, "b": s}
        result = self.enc(graph)
        # The anchor should appear once in a .define: section
        assert "&shared" in result
        # The alias should appear for references
        assert "*shared" in result

    def test_encodes_singleton_with_kwargs(self):
        s = SingletonNode("mymodule:MySingleton", lr=0.01, batch_size=32)
        result = self.enc(s)
        assert "!singleton:mymodule:MySingleton" in result
        assert "lr:" in result
        assert "0.01" in result
        assert "batch_size:" in result
        assert "32" in result

    def test_encodes_singleton_with_positional_args(self):
        s = SingletonNode("mymodule:MySingleton", "arg1", "arg2")
        result = self.enc(s)
        assert "!singleton:mymodule:MySingleton" in result
        assert "- 'arg1'" in result
        assert "- 'arg2'" in result


class TestToYamlConvenienceWrapper:
    def test_to_yaml_returns_string(self):
        result = to_yaml({"key": 1})
        assert isinstance(result, str)

    def test_to_yaml_encodes_dict(self):
        result = to_yaml({"x": 10})
        assert "x:" in result
        assert "10" in result

    def test_to_yaml_encodes_singleton_node(self):
        s = SingletonNode("mod:Cls")
        result = to_yaml(s)
        assert "!singleton:mod:Cls" in result

    def test_to_yaml_encodes_none(self):
        result = to_yaml(None)
        assert result == "null"
