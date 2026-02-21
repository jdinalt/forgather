"""Unit tests for forgather.yaml_utils"""

import pytest
import yaml

from forgather.latent import (
    FactoryNode,
    PartialNode,
    SingletonNode,
    VarNode,
)
from forgather.yaml_utils import (
    CallableConstructor,
    dict_constructor,
    dlist_constructor,
    list_constructor,
    load_depth_first,
    split_args,
    split_tag_idenity,
    tuple_constructor,
    var_constructor,
)

# ---------------------------------------------------------------------------
# split_tag_idenity
# ---------------------------------------------------------------------------


class TestSplitTagIdentity:
    def test_empty_string_returns_none_none(self):
        """Empty suffix means no tag was specified; both parts are None."""
        constructor, identity = split_tag_idenity("")
        assert constructor is None
        assert identity is None

    def test_constructor_only_no_identity(self):
        """:torch.nn:Linear has no '@identity' suffix."""
        constructor, identity = split_tag_idenity(":torch.nn:Linear")
        assert constructor == "torch.nn:Linear"
        assert identity is None

    def test_constructor_with_identity(self):
        """:dict@my_dict splits at '@' into constructor and identity."""
        constructor, identity = split_tag_idenity(":dict@my_dict")
        assert constructor == "dict"
        assert identity == "my_dict"

    def test_constructor_with_module_and_identity(self):
        """:builtins:list@my_id splits correctly at the first '@'."""
        constructor, identity = split_tag_idenity(":builtins:list@my_id")
        assert constructor == "builtins:list"
        assert identity == "my_id"

    def test_at_sign_only_splits_once(self):
        """Only the first '@' is used as a split point (maxsplit=1)."""
        constructor, identity = split_tag_idenity(":mod:cls@id@extra")
        assert constructor == "mod:cls"
        assert identity == "id@extra"


# ---------------------------------------------------------------------------
# split_args
# ---------------------------------------------------------------------------


class TestSplitArgs:
    def test_empty_dict_returns_empty_args_and_kwargs(self):
        """An empty dict yields an empty generator and an empty dict."""
        args, kwargs = split_args({})
        assert list(args) == []
        assert kwargs == {}

    def test_only_regular_kwargs_returned_unchanged(self):
        """Non-positional keys are left intact in kwargs."""
        d = {"alpha": "a", "beta": "b"}
        args, kwargs = split_args(d)
        assert list(args) == []
        assert kwargs == {"alpha": "a", "beta": "b"}

    def test_positional_args_extracted_and_sorted(self):
        """arg0, arg1, arg2 are extracted in numeric order."""
        d = {"arg2": "c", "arg0": "a", "arg1": "b"}
        args, kwargs = split_args(d)
        assert list(args) == ["a", "b", "c"]
        assert kwargs == {}

    def test_mixed_positional_and_keyword_args(self):
        """Positional args are separated from regular kwargs."""
        d = {"arg0": 0, "arg1": 1, "key": "value"}
        args, kwargs = split_args(d)
        assert list(args) == [0, 1]
        assert kwargs == {"key": "value"}

    def test_sparse_positional_indices_sorted_correctly(self):
        """arg5 and arg0 come back in ascending index order [0, 5]."""
        d = {"arg5": 5, "arg0": 0}
        args, kwargs = split_args(d)
        assert list(args) == [0, 5]
        assert kwargs == {}

    def test_positional_keys_removed_from_original_dict(self):
        """split_args mutates the input dict, removing positional keys."""
        d = {"arg0": 10, "name": "hello"}
        args, kwargs = split_args(d)
        # Consume the generator so mutation is complete
        list(args)
        assert "arg0" not in d
        assert "name" in d


# ---------------------------------------------------------------------------
# var_constructor
# ---------------------------------------------------------------------------


class TestVarConstructor:
    """Tests using a raw yaml.Loader to exercise var_constructor directly."""

    def _load(self, yaml_str):
        loader = yaml.SafeLoader(yaml_str)
        loader.add_constructor("!var", var_constructor)
        node = loader.get_single_node()
        return loader.construct_object(node, deep=True)

    def test_scalar_node_creates_varnode_with_name(self):
        """A scalar !var node uses the scalar value as the variable name."""
        result = self._load("!var my_variable")
        assert isinstance(result, VarNode)
        assert result.constructor == "my_variable"

    def test_mapping_node_creates_varnode_with_name_and_default(self):
        """A mapping !var node with 'name' and 'default' sets both."""
        result = self._load("!var\n  name: my_var\n  default: 42")
        assert isinstance(result, VarNode)
        assert result.constructor == "my_var"
        assert result.value == 42


# ---------------------------------------------------------------------------
# load_depth_first
# ---------------------------------------------------------------------------


class TestLoadDepthFirst:
    def test_loads_simple_yaml_correctly(self):
        """A basic key-value YAML document is loaded as a plain dict."""
        result = load_depth_first("key: value\nnum: 123")
        assert result == {"key": "value", "num": 123}

    def test_returns_none_for_empty_document(self):
        """An empty YAML stream returns None."""
        result = load_depth_first("")
        assert result is None

    def test_returns_none_for_null_document(self):
        """A YAML null (~) returns None."""
        result = load_depth_first("~")
        assert result is None

    def test_loads_scalar_value(self):
        """A bare scalar is returned directly."""
        result = load_depth_first("42")
        assert result == 42


# ---------------------------------------------------------------------------
# CallableConstructor
# ---------------------------------------------------------------------------


class TestCallableConstructor:
    """Tests for CallableConstructor using ConfigLoader (which has all tags)."""

    def _load(self, yaml_str):
        from forgather.config import ConfigLoader

        return load_depth_first(yaml_str, Loader=ConfigLoader)

    def test_sequence_node_creates_factory_with_args(self):
        """A sequence under !factory becomes positional args."""
        result = self._load("!factory:builtins:list\n  - 1\n  - 2\n  - 3")
        assert isinstance(result, FactoryNode)
        assert result.constructor == "builtins:list"
        assert result.args == (1, 2, 3)
        assert result.kwargs == {}

    def test_mapping_node_creates_factory_with_kwargs(self):
        """A plain mapping under !factory becomes keyword args."""
        result = self._load("!factory:builtins:dict\n  key: value\n  num: 42")
        assert isinstance(result, FactoryNode)
        assert result.kwargs == {"key": "value", "num": 42}
        assert result.args == ()

    def test_explicit_args_and_kwargs_keys(self):
        """A mapping with exactly 'args' and 'kwargs' keys is treated specially."""
        result = self._load(
            "!factory:builtins:dict\n"
            "  args:\n"
            "    - 1\n"
            "    - 2\n"
            "  kwargs:\n"
            "    key: value\n"
        )
        assert isinstance(result, FactoryNode)
        assert result.args == (1, 2)
        assert result.kwargs == {"key": "value"}

    def test_singleton_constructor_creates_singleton_node(self):
        """!singleton tag produces a SingletonNode."""
        result = self._load("!singleton:builtins:list")
        assert isinstance(result, SingletonNode)
        assert result.constructor == "builtins:list"

    def test_identity_suffix_is_captured(self):
        """The @identity suffix of a tag is stored on the node."""
        result = self._load("!factory:builtins:dict@my_node\n  x: 1")
        assert isinstance(result, FactoryNode)
        assert result.identity == "my_node"


# ---------------------------------------------------------------------------
# list_constructor
# ---------------------------------------------------------------------------


class TestListConstructor:
    def _load(self, yaml_str):
        from forgather.config import ConfigLoader

        return load_depth_first(yaml_str, Loader=ConfigLoader)

    def test_sequence_creates_singleton_named_list(self):
        """A sequence under !list becomes a SingletonNode with constructor 'named_list'."""
        result = self._load("!list\n  - 1\n  - 2\n  - 3")
        assert isinstance(result, SingletonNode)
        assert result.constructor == "named_list"
        assert result.args == ([1, 2, 3],)

    def test_empty_scalar_creates_empty_named_list(self):
        """An empty !list scalar tag creates a named_list with an empty list."""
        result = self._load("!list")
        assert isinstance(result, SingletonNode)
        assert result.constructor == "named_list"
        # The empty list is stored as the first arg
        assert result.args == ([],)


# ---------------------------------------------------------------------------
# tuple_constructor
# ---------------------------------------------------------------------------


class TestTupleConstructor:
    def _load(self, yaml_str):
        from forgather.config import ConfigLoader

        return load_depth_first(yaml_str, Loader=ConfigLoader)

    def test_sequence_creates_singleton_named_tuple(self):
        """A sequence under !tuple becomes a SingletonNode with constructor 'named_tuple'."""
        result = self._load("!tuple\n  - 1\n  - 2")
        assert isinstance(result, SingletonNode)
        assert result.constructor == "named_tuple"

    @pytest.mark.xfail(
        reason="Known bug: empty !tuple scalar creates 'named_list' instead of 'named_tuple'"
    )
    def test_empty_scalar_creates_named_tuple_not_named_list(self):
        """An empty !tuple scalar should create a 'named_tuple', but currently creates 'named_list'."""
        result = self._load("!tuple")
        assert isinstance(result, SingletonNode)
        assert result.constructor == "named_tuple"


# ---------------------------------------------------------------------------
# dict_constructor
# ---------------------------------------------------------------------------


class TestDictConstructor:
    def _load(self, yaml_str):
        from forgather.config import ConfigLoader

        return load_depth_first(yaml_str, Loader=ConfigLoader)

    def test_mapping_creates_singleton_named_dict(self):
        """A mapping under !dict becomes a SingletonNode with constructor 'named_dict'."""
        result = self._load("!dict\n  key: value\n  num: 42")
        assert isinstance(result, SingletonNode)
        assert result.constructor == "named_dict"
        assert result.kwargs == {"key": "value", "num": 42}

    def test_empty_dict_scalar(self):
        """An empty !dict scalar creates a named_dict with no kwargs."""
        result = self._load("!dict")
        assert isinstance(result, SingletonNode)
        assert result.constructor == "named_dict"


# ---------------------------------------------------------------------------
# dlist_constructor
# ---------------------------------------------------------------------------


class TestDlistConstructor:
    def _load(self, yaml_str):
        from forgather.config import ConfigLoader

        return load_depth_first(yaml_str, Loader=ConfigLoader)

    def test_mapping_values_become_list(self):
        """A mapping under !dlist collects values into a SingletonNode('named_list')."""
        result = self._load("!dlist\n  first: 1\n  second: 2\n  third: 3")
        assert isinstance(result, SingletonNode)
        assert result.constructor == "named_list"
        # The list of values is passed as the first positional arg
        assert set(result.args[0]) == {1, 2, 3}

    def test_none_values_are_filtered_out(self):
        """Null entries in a !dlist mapping are filtered before building the list."""
        result = self._load("!dlist\n  first: 1\n  second: ~\n  third: 3")
        assert isinstance(result, SingletonNode)
        assert None not in result.args[0]
        assert set(result.args[0]) == {1, 3}
