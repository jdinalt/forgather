"""
Unit tests for forgather.graph_encoder module.

Tests cover:
- NamePolicy enum
- NodeNameDict (node counting and naming under each policy)
- GraphEncoder base class (via a concrete subclass)
  - _indent()
  - split_text()
  - _encode() type dispatcher
"""

import pytest

from forgather.graph_encoder import (
    GraphEncoder,
    NamePolicy,
    NodeNameDict,
)
from forgather.latent import (
    CallableNode,
    FactoryNode,
    MetaNode,
    SingletonNode,
    VarNode,
)

# ---------------------------------------------------------------------------
# Concrete subclass required to instantiate the abstract GraphEncoder
# ---------------------------------------------------------------------------


class ConcreteEncoder(GraphEncoder):
    """
    Minimal concrete subclass of GraphEncoder.

    Every abstract method returns a simple string representation of the type
    and value so we can verify dispatch.
    """

    def __call__(self, obj, name_policy=None):
        self.init(obj, name_policy)
        return self._encode(obj)

    def _list(self, obj: list):
        return f"list({obj})"

    def _dict(self, obj: dict):
        return f"dict({obj})"

    def _tuple(self, obj: tuple):
        return f"tuple({obj})"

    def _var(self, obj: VarNode):
        return f"var({obj.constructor})"

    def _callable(self, obj: CallableNode):
        return f"callable({obj.constructor})"

    def _none(self):
        return "none"

    def _str(self, obj: str):
        return f"str({obj})"

    def _int(self, obj: int):
        return f"int({obj})"

    def _float(self, obj: float):
        return f"float({obj})"

    def _bool(self, obj: bool):
        return f"bool({obj})"


# ---------------------------------------------------------------------------
# NamePolicy enum
# ---------------------------------------------------------------------------


class TestNamePolicy:
    def test_required_exists(self):
        assert hasattr(NamePolicy, "REQUIRED")

    def test_named_exists(self):
        assert hasattr(NamePolicy, "NAMED")

    def test_all_exists(self):
        assert hasattr(NamePolicy, "ALL")

    def test_values_are_distinct(self):
        assert NamePolicy.REQUIRED != NamePolicy.NAMED
        assert NamePolicy.NAMED != NamePolicy.ALL
        assert NamePolicy.REQUIRED != NamePolicy.ALL

    def test_is_enum(self):
        from enum import Enum

        assert issubclass(NamePolicy, Enum)


# ---------------------------------------------------------------------------
# NodeNameDict – REQUIRED policy
# ---------------------------------------------------------------------------


class TestNodeNameDictRequired:
    def test_single_occurrence_not_named(self):
        """REQUIRED: a node that appears only once should NOT get a name."""
        node = FactoryNode("list")
        graph = {"a": node}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.REQUIRED)
        assert node.identity not in name_dict

    def test_duplicate_occurrence_named(self):
        """REQUIRED: a node that appears more than once SHOULD get a name."""
        node = SingletonNode("list")
        graph = {"a": node, "b": node}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.REQUIRED)
        assert node.identity in name_dict

    def test_two_different_single_nodes_not_named(self):
        node1 = FactoryNode("list")
        node2 = FactoryNode("dict")
        graph = {"a": node1, "b": node2}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.REQUIRED)
        assert node1.identity not in name_dict
        assert node2.identity not in name_dict


# ---------------------------------------------------------------------------
# NodeNameDict – NAMED policy
# ---------------------------------------------------------------------------


class TestNodeNameDictNamed:
    def test_string_identity_node_named(self):
        """NAMED: a node with a string identity should always get a name."""
        node = FactoryNode("list", _identity="my_model")
        graph = {"a": node}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.NAMED)
        assert "my_model" in name_dict

    def test_string_identity_name_equals_identity(self):
        """String-identity nodes use their identity string directly as the name."""
        node = FactoryNode("list", _identity="my_list")
        graph = {"a": node}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.NAMED)
        assert name_dict["my_list"] == "my_list"

    def test_int_identity_single_occurrence_not_named(self):
        """NAMED: a node with default (int) identity appearing once is NOT named."""
        node = FactoryNode("list")
        graph = {"a": node}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.NAMED)
        assert node.identity not in name_dict

    def test_duplicate_int_identity_named(self):
        """NAMED: duplicate int-identity node should get an AutoName."""
        node = SingletonNode("list")
        graph = {"a": node, "b": node}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.NAMED)
        assert node.identity in name_dict

    def test_is_default_policy(self):
        """NodeNameDict without explicit policy uses NAMED."""
        node = FactoryNode("list", _identity="explicit")
        graph = {"a": node}
        name_dict = NodeNameDict(graph)  # no name_policy arg
        assert "explicit" in name_dict


# ---------------------------------------------------------------------------
# NodeNameDict – ALL policy
# ---------------------------------------------------------------------------


class TestNodeNameDictAll:
    def test_single_occurrence_named(self):
        """ALL: every callable node gets a name, even unique ones."""
        node = FactoryNode("list")
        graph = {"a": node}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.ALL)
        assert node.identity in name_dict

    def test_multiple_unique_nodes_all_named(self):
        node1 = FactoryNode("list")
        node2 = FactoryNode("dict")
        node3 = SingletonNode("list")
        graph = {"a": node1, "b": node2, "c": node3}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.ALL)
        assert node1.identity in name_dict
        assert node2.identity in name_dict
        assert node3.identity in name_dict


# ---------------------------------------------------------------------------
# NodeNameDict – name assignment details
# ---------------------------------------------------------------------------


class TestNodeNameDictNameAssignment:
    def test_string_identity_uses_identity_as_name(self):
        """Nodes with a string identity use that string as their human-readable name."""
        node = SingletonNode("list", _identity="optimizer")
        graph = {"a": node, "b": node}  # duplicate to ensure it appears in REQUIRED
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.REQUIRED)
        assert name_dict["optimizer"] == "optimizer"

    def test_integer_identity_gets_autoname(self):
        """
        Nodes with an integer (default) identity that require a name receive an
        AutoName-generated string (e.g., 'alpha_', 'beta_', ...).
        """
        node = SingletonNode("list")
        graph = {"a": node, "b": node}
        name_dict = NodeNameDict(graph, name_policy=NamePolicy.REQUIRED)
        name = name_dict[node.identity]
        # AutoName generates names ending with '_'
        assert isinstance(name, str)
        assert name.endswith("_")

    def test_string_policy_conversion_required(self):
        """String 'required' is converted to NamePolicy.REQUIRED."""
        node = SingletonNode("list")
        graph = {"a": node, "b": node}
        name_dict = NodeNameDict(graph, name_policy="required")
        assert node.identity in name_dict

    def test_string_policy_conversion_named(self):
        """String 'named' is converted to NamePolicy.NAMED."""
        node = FactoryNode("list", _identity="explicit")
        graph = {"a": node}
        name_dict = NodeNameDict(graph, name_policy="named")
        assert "explicit" in name_dict

    def test_string_policy_conversion_all(self):
        """String 'all' is converted to NamePolicy.ALL."""
        node = FactoryNode("list")
        graph = {"a": node}
        name_dict = NodeNameDict(graph, name_policy="all")
        assert node.identity in name_dict

    def test_invalid_string_policy_raises_value_error(self):
        """An unrecognized string policy should raise ValueError."""
        node = FactoryNode("list")
        graph = {"a": node}
        with pytest.raises(ValueError):
            NodeNameDict(graph, name_policy="bogus_policy")


# ---------------------------------------------------------------------------
# GraphEncoder._indent()
# ---------------------------------------------------------------------------


class TestGraphEncoderIndent:
    def setup_method(self):
        self.encoder = ConcreteEncoder()
        # Manually initialise so we can set level directly
        self.encoder.level = -1  # default before init
        import re

        self.encoder.split_text_re = re.compile(r"(\n)")

    def test_indent_level_0(self):
        self.encoder.level = 0
        assert self.encoder._indent() == " " * 4 * 0

    def test_indent_level_1(self):
        self.encoder.level = 1
        assert self.encoder._indent() == " " * 4

    def test_indent_level_2(self):
        self.encoder.level = 2
        assert self.encoder._indent() == " " * 8

    def test_indent_with_positive_offset(self):
        self.encoder.level = 1
        assert self.encoder._indent(offset=1) == " " * 8

    def test_indent_with_negative_offset(self):
        self.encoder.level = 2
        assert self.encoder._indent(offset=-1) == " " * 4

    def test_indent_level_minus_one(self):
        """Level -1 (initial state) with offset=0 gives empty string."""
        self.encoder.level = -1
        assert self.encoder._indent() == ""


# ---------------------------------------------------------------------------
# GraphEncoder.split_text()
# ---------------------------------------------------------------------------


class TestGraphEncoderSplitText:
    def setup_method(self):
        self.encoder = ConcreteEncoder()
        import re

        self.encoder.split_text_re = re.compile(r"(\n)")
        self.encoder.level = 0

    def test_empty_string_yields_one_chunk(self):
        """An empty input yields exactly one empty string chunk."""
        chunks = list(self.encoder.split_text(""))
        assert chunks == [""]

    def test_no_newline_yields_one_chunk(self):
        chunks = list(self.encoder.split_text("hello world"))
        assert chunks == ["hello world"]

    def test_single_newline(self):
        """'foo\nbar' -> ['foo\n', 'bar']"""
        chunks = list(self.encoder.split_text("foo\nbar"))
        assert chunks == ["foo\n", "bar"]

    def test_multiple_newlines(self):
        chunks = list(self.encoder.split_text("a\nb\nc"))
        assert chunks == ["a\n", "b\n", "c"]

    def test_trailing_newline(self):
        """'hello\n' should yield one chunk 'hello\n' with no empty trailing chunk."""
        chunks = list(self.encoder.split_text("hello\n"))
        assert chunks == ["hello\n"]

    def test_chunks_include_newlines(self):
        """Each chunk that was followed by a newline should end with '\n'."""
        chunks = list(self.encoder.split_text("line1\nline2\nline3"))
        for chunk in chunks[:-1]:
            assert chunk.endswith(
                "\n"
            ), f"Expected chunk to end with newline: {chunk!r}"


# ---------------------------------------------------------------------------
# GraphEncoder._encode() type dispatcher
# ---------------------------------------------------------------------------


class TestGraphEncoderDispatch:
    def setup_method(self):
        self.encoder = ConcreteEncoder()
        # Initialise encoder state using a dummy object
        self.encoder.init({"dummy": 1}, name_policy=NamePolicy.ALL)

    def test_dispatches_str(self):
        result = self.encoder._encode("hello")
        assert result == "str(hello)"

    def test_dispatches_int(self):
        result = self.encoder._encode(42)
        assert result == "int(42)"

    def test_dispatches_float(self):
        result = self.encoder._encode(3.14)
        assert result == "float(3.14)"

    @pytest.mark.xfail(
        reason=(
            "Known bug: _encode checks isinstance(obj, int) before isinstance(obj, bool). "
            "Since bool is a subclass of int, True/False are dispatched to _int, not _bool."
        )
    )
    def test_dispatches_bool(self):
        """bool values should dispatch to _bool, but currently dispatch to _int (see xfail reason)."""
        result = self.encoder._encode(True)
        assert result == "bool(True)"

    def test_dispatches_none(self):
        result = self.encoder._encode(None)
        assert result == "none"

    def test_dispatches_list(self):
        result = self.encoder._encode([1, 2])
        assert result == "list([1, 2])"

    def test_dispatches_dict(self):
        result = self.encoder._encode({"a": 1})
        assert result == "dict({'a': 1})"

    def test_dispatches_tuple(self):
        result = self.encoder._encode((1, 2))
        assert result == "tuple((1, 2))"

    def test_dispatches_var_node(self):
        node = VarNode("my_var")
        result = self.encoder._encode(node)
        assert result == "var(my_var)"

    def test_dispatches_callable_node(self):
        node = FactoryNode("list")
        result = self.encoder._encode(node)
        assert result == "callable(list)"

    def test_dispatches_singleton_callable(self):
        """SingletonNode is a CallableNode; should dispatch to _callable."""
        node = SingletonNode("dict")
        result = self.encoder._encode(node)
        assert result == "callable(dict)"

    @pytest.mark.xfail(
        reason=(
            "Known bug: _encode checks isinstance(obj, int) BEFORE isinstance(obj, bool). "
            "Because bool is a subclass of int, True/False are incorrectly dispatched to _int. "
            "The fix would be to swap the order: check bool before int."
        )
    )
    def test_bool_before_int(self):
        """
        bool is a subclass of int in Python. The dispatcher should check bool BEFORE int
        to ensure True/False route to _bool, not _int.

        Currently the dispatcher has the checks in the wrong order, so this test is
        marked xfail to document the bug.
        """
        result_bool = self.encoder._encode(True)
        result_int = self.encoder._encode(1)
        assert result_bool == "bool(True)"
        assert result_int == "int(1)"
