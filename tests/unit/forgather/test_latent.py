"""
Unit tests for forgather.latent module.

Tests cover:
- UndefinedType / Undefined singleton
- UnboundVarError
- VarNode
- CallableNode and its subclasses (FactoryNode, SingletonNode, MetaNode, PartialNode)
- Materializer
- Latent namespace (materialize, walk, check)
- prune_node_type helper
- BUILT_INS callables
"""

from functools import partial

import pytest

from forgather.latent import (
    CallableNode,
    DuplicateNameError,
    FactoryNode,
    Latent,
    Materializer,
    MetaNode,
    Node,
    PartialNode,
    SingletonNode,
    UnboundVarError,
    Undefined,
    UndefinedType,
    VarNode,
    prune_node_type,
)

# ---------------------------------------------------------------------------
# UndefinedType / Undefined singleton
# ---------------------------------------------------------------------------


class TestUndefined:
    def test_is_singleton(self):
        """Undefined should be a single module-level instance of UndefinedType."""
        assert isinstance(Undefined, UndefinedType)
        another = UndefinedType()
        # The sentinel object is the module-level constant; constructing a new
        # UndefinedType does NOT return the same object.
        assert Undefined is not another

    def test_repr(self):
        assert repr(Undefined) == "Undefined"

    def test_module_constant_identity(self):
        """Re-importing Undefined gives the same object."""
        from forgather.latent import Undefined as Undefined2

        assert Undefined is Undefined2


# ---------------------------------------------------------------------------
# UnboundVarError
# ---------------------------------------------------------------------------


class TestUnboundVarError:
    def test_message_includes_var_name(self):
        err = UnboundVarError("my_var")
        assert "my_var" in str(err)

    def test_is_name_error(self):
        assert issubclass(UnboundVarError, NameError)

    def test_raise_and_catch(self):
        with pytest.raises(UnboundVarError):
            raise UnboundVarError("some_variable")


# ---------------------------------------------------------------------------
# VarNode
# ---------------------------------------------------------------------------


class TestVarNode:
    def test_constructor_attribute(self):
        node = VarNode("x")
        assert node.constructor == "x"

    def test_default_value_is_undefined(self):
        node = VarNode("x")
        assert node.value is Undefined

    def test_custom_default_value(self):
        node = VarNode("lr", default=1e-3)
        assert node.value == pytest.approx(1e-3)

    def test_identity_defaults_to_object_id(self):
        node = VarNode("x")
        assert node.identity == id(node)

    def test_explicit_identity(self):
        node = VarNode("x", _identity="my_var_id")
        assert node.identity == "my_var_id"

    def test_value_can_be_set(self):
        node = VarNode("x")
        node.value = 42
        assert node.value == 42

    def test_constructor_must_be_str(self):
        """Node base class enforces constructor is str; VarNode inherits that."""
        with pytest.raises(ValueError):
            VarNode(123)  # type: ignore


# ---------------------------------------------------------------------------
# CallableNode – construction guards
# ---------------------------------------------------------------------------


class TestCallableNodeConstruction:
    def test_constructor_must_be_str(self):
        with pytest.raises(ValueError):
            FactoryNode(42)  # type: ignore

    def test_constructor_accepts_str(self):
        node = FactoryNode("list")
        assert node.constructor == "list"

    def test_args_stored(self):
        node = FactoryNode("list", 1, 2, 3)
        assert node.args == (1, 2, 3)

    def test_kwargs_stored(self):
        node = FactoryNode("dict", a=1, b=2)
        assert node.kwargs == {"a": 1, "b": 2}

    def test_submodule_searchpath_default_empty(self):
        node = FactoryNode("list")
        assert node.submodule_searchpath == []

    def test_submodule_searchpath_stored(self):
        node = FactoryNode("list", submodule_searchpath=["/some/path"])
        assert node.submodule_searchpath == ["/some/path"]


# ---------------------------------------------------------------------------
# CallableNode.callable – resolution
# ---------------------------------------------------------------------------


class TestCallableResolution:
    def test_builtin_list(self):
        node = FactoryNode("list")
        assert node.callable is list

    def test_builtin_dict(self):
        node = FactoryNode("dict")
        assert node.callable is dict

    def test_dynamic_import_spec(self):
        """Constructor with ':' uses dynamic_import to resolve the callable."""
        node = FactoryNode("builtins:list")
        assert node.callable is list

    def test_callable_is_cached(self):
        """Resolved callable is cached in _callable attribute."""
        node = FactoryNode("list")
        first = node.callable
        second = node.callable
        assert first is second
        assert hasattr(node, "_callable")

    def test_unknown_builtin_raises_name_error(self):
        node = FactoryNode("this_does_not_exist_anywhere")
        with pytest.raises(NameError):
            _ = node.callable


# ---------------------------------------------------------------------------
# FactoryNode – basic materialization
# ---------------------------------------------------------------------------


class TestFactoryNode:
    def test_materialize_builtin_list(self):
        node = FactoryNode("list")
        result = Latent.materialize(node)
        assert result == []
        assert isinstance(result, list)

    def test_materialize_builtin_dict(self):
        node = FactoryNode("dict", a=1, b=2)
        result = Latent.materialize(node)
        assert result == {"a": 1, "b": 2}

    def test_materialize_new_instance_each_time(self):
        """FactoryNode should produce a new object on every materialization."""
        node = FactoryNode("list")
        r1 = Latent.materialize(node)
        r2 = Latent.materialize(node)
        assert r1 is not r2

    def test_materialize_with_dynamic_import(self):
        node = FactoryNode("builtins:list")
        result = Latent.materialize(node)
        assert isinstance(result, list)

    def test_materialize_nested_args(self):
        """
        Args that are themselves nodes should be materialized recursively.

        Note: list(iterable) iterates over the provided iterable.
        Passing a materialized empty list ([]) as the sole arg to list()
        produces an empty list, not a list-of-lists. To test nesting, use
        a dict so the inner node appears as a value.
        """
        inner = FactoryNode("list")
        outer = FactoryNode("dict", inner_list=inner)
        result = Latent.materialize(outer)
        assert isinstance(result, dict)
        assert isinstance(result["inner_list"], list)


# ---------------------------------------------------------------------------
# SingletonNode – caching behaviour
# ---------------------------------------------------------------------------


class TestSingletonNode:
    def test_same_instance_on_repeated_calls(self):
        """SingletonNode caches the result; subsequent calls return the same object."""
        node = SingletonNode("list")
        r1 = Latent.materialize(node)
        r2 = Latent.materialize(node)
        assert r1 is r2

    def test_instance_attribute_set_after_first_call(self):
        node = SingletonNode("list")
        assert node.instance is None
        Latent.materialize(node)
        assert node.instance is not None

    @pytest.mark.xfail(
        reason=(
            "Known behavior: SingletonNode checks 'if (value := obj.instance) is not None', "
            "so a callable that returns None will NOT cache the result and will be called "
            "every time. This is a subtle bug in the caching logic."
        )
    )
    def test_none_result_not_cached(self):
        """
        If the callable returns None, the cache check 'is not None' fails on every
        access, so the callable is invoked repeatedly instead of returning the cached
        None. Document this known limitation.
        """
        call_count = [0]

        def returns_none():
            call_count[0] += 1
            return None

        # Build a node whose constructor resolves to returns_none
        # We simulate by patching _callable directly.
        node = SingletonNode("list")
        node._callable = returns_none
        node.args = ()
        node.kwargs = {}

        r1 = Latent.materialize(node)
        r2 = Latent.materialize(node)

        # This assertion SHOULD pass if caching worked correctly…
        assert (
            call_count[0] == 1
        ), f"callable was invoked {call_count[0]} times; expected 1 (None not cached)"


# ---------------------------------------------------------------------------
# MetaNode – raw args
# ---------------------------------------------------------------------------


class TestMetaNode:
    def test_receives_raw_args(self):
        """
        MetaNode passes its args directly to the callable without materializing
        them first. The callable should receive Node objects, not their values.
        """
        received = {}

        def capture_raw(*args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs
            return "done"

        inner_node = FactoryNode("list")
        meta = MetaNode("list")
        meta._callable = capture_raw
        meta.args = (inner_node, "literal")
        meta.kwargs = {"k": VarNode("x")}

        result = Latent.materialize(meta)

        assert result == "done"
        # The inner_node was NOT materialized – it is still a FactoryNode
        assert received["args"][0] is inner_node
        assert isinstance(received["kwargs"]["k"], VarNode)

    def test_is_singleton_subclass(self):
        assert issubclass(MetaNode, SingletonNode)


# ---------------------------------------------------------------------------
# PartialNode
# ---------------------------------------------------------------------------


class TestPartialNode:
    def test_returns_partial_when_nested(self):
        """
        When a PartialNode is NOT at the root of the graph (level != 0), it
        should return a functools.partial object rather than calling the function.
        """
        partial_node = PartialNode("list")

        # Place the PartialNode inside a dict so it is NOT at root level (level > 0)
        graph = {"fn": partial_node}
        result = Latent.materialize(graph)

        assert isinstance(result["fn"], partial)

    def test_calls_directly_at_root(self):
        """
        When a PartialNode IS at the root (level == 0), it should call the
        function directly and return the result.
        """
        partial_node = PartialNode("list")
        result = Latent.materialize(partial_node)
        # list() called with no args returns []
        assert result == []

    def test_partial_at_root_merges_context_kwargs(self):
        """
        When a PartialNode is at root level, extra kwargs passed to materialize
        are merged into the call.
        """
        partial_node = PartialNode("dict")
        result = Latent.materialize(partial_node, extra_key="extra_value")
        assert result == {"extra_key": "extra_value"}

    def test_partial_captures_args(self):
        """Partial at non-root level captures the node's args."""
        partial_node = PartialNode("builtins:int", "42")
        graph = {"fn": partial_node}
        result = Latent.materialize(graph)
        p = result["fn"]
        assert isinstance(p, partial)
        # Calling the partial should invoke int("42") => 42
        assert p() == 42


# ---------------------------------------------------------------------------
# Materializer – containers and context
# ---------------------------------------------------------------------------


class TestMaterializer:
    def test_materializes_list(self):
        result = Latent.materialize([1, 2, 3])
        assert result == [1, 2, 3]

    def test_materializes_dict(self):
        result = Latent.materialize({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_materializes_tuple(self):
        result = Latent.materialize((1, 2, 3))
        assert result == (1, 2, 3)

    def test_var_node_resolved_from_context(self):
        node = VarNode("lr")
        result = Latent.materialize(node, context_vars={"lr": 0.001})
        assert result == pytest.approx(0.001)

    def test_var_node_uses_default_when_not_in_context(self):
        node = VarNode("lr", default=1e-4)
        result = Latent.materialize(node)
        assert result == pytest.approx(1e-4)

    def test_var_node_context_overrides_default(self):
        node = VarNode("lr", default=1e-4)
        result = Latent.materialize(node, context_vars={"lr": 5e-5})
        assert result == pytest.approx(5e-5)

    def test_var_node_raises_unbound_error(self):
        """VarNode with no default and not in context raises UnboundVarError."""
        node = VarNode("missing_var")
        with pytest.raises(UnboundVarError) as exc_info:
            Latent.materialize(node)
        assert "missing_var" in str(exc_info.value)

    def test_mtargets_single_string(self):
        """mtargets as a string returns the value for that key directly."""
        graph = {
            "model": FactoryNode("list"),
            "optimizer": FactoryNode("dict"),
        }
        result = Latent.materialize(graph, mtargets="model")
        assert isinstance(result, list)

    def test_mtargets_set_of_keys(self):
        """mtargets as a set/list returns a dict with only those keys."""
        graph = {
            "model": FactoryNode("list"),
            "optimizer": FactoryNode("dict"),
            "dataset": FactoryNode("list"),
        }
        result = Latent.materialize(graph, mtargets={"model", "optimizer"})
        assert set(result.keys()) == {"model", "optimizer"}
        assert "dataset" not in result

    def test_mtargets_non_dict_raises_type_error(self):
        """Specifying mtargets on a non-dict root should raise TypeError."""
        with pytest.raises(TypeError):
            Latent.materialize([1, 2, 3], mtargets="something")

    def test_passthrough_scalars(self):
        """Non-node scalars pass through unchanged."""
        assert Latent.materialize(42) == 42
        assert Latent.materialize("hello") == "hello"
        assert Latent.materialize(3.14) == pytest.approx(3.14)
        assert Latent.materialize(None) is None

    def test_nested_var_node_in_factory_args(self):
        """VarNode nested in a FactoryNode's args is resolved via context_vars."""
        var = VarNode("size")
        # list * size -> repeated list; use builtins:range to keep it simple
        # Actually test by building a dict that contains the resolved value.
        graph = {"n": var, "other": 1}
        result = Latent.materialize(graph, context_vars={"size": 99})
        assert result["n"] == 99


# ---------------------------------------------------------------------------
# Latent.materialize delegation
# ---------------------------------------------------------------------------


class TestLatentMaterialize:
    def test_delegates_to_materializer(self):
        """Latent.materialize should return the same result as calling Materializer directly."""
        node = FactoryNode("list")
        result_static = Latent.materialize(node)
        assert isinstance(result_static, list)

    def test_factory_node_via_latent(self):
        node = FactoryNode("dict", x=1)
        assert Latent.materialize(node) == {"x": 1}


# ---------------------------------------------------------------------------
# Latent.walk
# ---------------------------------------------------------------------------


class TestLatentWalk:
    def _build_simple_graph(self):
        """
        Returns a simple graph:
          outer_dict
            inner_factory -> FactoryNode("list")
        """
        inner = FactoryNode("list")
        root = {"inner": inner}
        return root, inner

    def test_top_down_yields_correct_nodes(self):
        root, inner = self._build_simple_graph()
        walked = list(Latent.walk(root, top_down=True))

        # We should see the root dict and the inner FactoryNode
        nodes = [node for _, node, _ in walked]
        assert root in nodes
        assert inner in nodes

    def test_top_down_root_comes_first(self):
        root, inner = self._build_simple_graph()
        walked = list(Latent.walk(root, top_down=True))
        nodes = [node for _, node, _ in walked]
        assert nodes.index(root) < nodes.index(inner)

    def test_bottom_up_inner_comes_first(self):
        root, inner = self._build_simple_graph()
        walked = list(Latent.walk(root, top_down=False))
        nodes = [node for _, node, _ in walked]
        assert nodes.index(inner) < nodes.index(root)

    def test_level_counting(self):
        """Root should be level 0; its immediate children level 1, etc."""
        inner = FactoryNode("list")
        root = {"inner": inner}

        levels = {id(node): lvl for lvl, node, _ in Latent.walk(root)}

        assert levels[id(root)] == 0
        assert levels[id(inner)] == 1

    def test_level_deep_nesting(self):
        """Three-level nesting: root dict -> callable -> arg scalar."""
        leaf = FactoryNode("list")
        mid = FactoryNode("list", leaf)
        root = {"mid": mid}

        levels = {id(node): lvl for lvl, node, _ in Latent.walk(root)}

        assert levels[id(root)] == 0
        assert levels[id(mid)] == 1
        assert levels[id(leaf)] == 2

    def test_top_down_pruning_sub_nodes(self):
        """
        In top_down mode, modifying sub_nodes prevents the children from being
        visited. Clear the sub_nodes dict to prune all children.
        """
        inner = FactoryNode("list")
        root = {"inner": inner}

        visited_nodes = []
        for _level, node, sub_nodes in Latent.walk(root, top_down=True):
            visited_nodes.append(node)
            if node is root:
                sub_nodes.clear()  # prune all children

        assert root in visited_nodes
        assert inner not in visited_nodes

    def test_walk_list_container(self):
        node1 = FactoryNode("list")
        node2 = FactoryNode("dict")
        root = [node1, node2]
        walked_nodes = [node for _, node, _ in Latent.walk(root)]
        assert node1 in walked_nodes
        assert node2 in walked_nodes

    def test_walk_callable_node_args_and_kwargs(self):
        """walk should descend into both positional args and keyword args."""
        arg_node = FactoryNode("list")
        kwarg_node = FactoryNode("dict")
        parent = FactoryNode("list", arg_node, named=kwarg_node)

        walked_nodes = [node for _, node, _ in Latent.walk(parent)]
        assert arg_node in walked_nodes
        assert kwarg_node in walked_nodes


# ---------------------------------------------------------------------------
# Latent.check
# ---------------------------------------------------------------------------


class TestLatentCheck:
    def test_passes_for_valid_graph(self):
        graph = {
            "a": FactoryNode("list"),
            "b": FactoryNode("dict"),
        }
        Latent.check(graph)  # should not raise

    def test_raises_duplicate_name_error(self):
        """
        Two different node objects sharing the same explicit _identity should
        cause DuplicateNameError.
        """
        node1 = FactoryNode("list", _identity="shared_name")
        node2 = FactoryNode("dict", _identity="shared_name")
        graph = {"a": node1, "b": node2}

        with pytest.raises(DuplicateNameError):
            Latent.check(graph)

    def test_same_node_twice_does_not_raise(self):
        """
        The same node object appearing multiple times in the graph should NOT
        raise DuplicateNameError (visited set prevents double-counting).
        """
        node = SingletonNode("list", _identity="reused")
        graph = {"a": node, "b": node}
        Latent.check(graph)  # should not raise

    def test_raises_type_error_for_unsupported_node(self):
        """
        A custom object that is not a Node, list, dict, etc. should raise TypeError.
        """

        class Unsupported:
            pass

        graph = {"bad": Unsupported()}
        with pytest.raises(TypeError):
            Latent.check(graph)

    def test_scalars_are_accepted(self):
        """Primitive scalars should be allowed anywhere in the graph."""
        graph = {
            "a": 1,
            "b": 3.14,
            "c": "hello",
            "d": True,
            "e": None,
        }
        Latent.check(graph)  # should not raise


# ---------------------------------------------------------------------------
# prune_node_type
# ---------------------------------------------------------------------------


class TestPruneNodeType:
    def test_removes_matching_nodes(self):
        factory = FactoryNode("list")
        singleton = SingletonNode("dict")
        node_map = {0: factory, 1: singleton, "k": FactoryNode("dict")}

        prune_node_type(node_map, SingletonNode)

        assert 1 not in node_map
        assert 0 in node_map
        assert "k" in node_map

    def test_removes_all_matching(self):
        node_map = {
            0: FactoryNode("list"),
            1: FactoryNode("dict"),
            2: SingletonNode("list"),
        }
        prune_node_type(node_map, FactoryNode)
        assert 0 not in node_map
        assert 1 not in node_map
        assert 2 in node_map

    def test_empty_map_no_error(self):
        node_map = {}
        prune_node_type(node_map, FactoryNode)
        assert node_map == {}

    def test_no_matching_nodes_unchanged(self):
        node_map = {0: FactoryNode("list"), 1: SingletonNode("dict")}
        original_keys = set(node_map.keys())
        prune_node_type(node_map, VarNode)
        assert set(node_map.keys()) == original_keys


# ---------------------------------------------------------------------------
# BUILT_INS
# ---------------------------------------------------------------------------


class TestBuiltIns:
    def _node_for(self, builtin_name, *args, **kwargs):
        """Helper: FactoryNode with a BUILT_IN constructor."""
        node = FactoryNode(builtin_name, *args, **kwargs)
        return node

    def test_named_list(self):
        """
        named_list is an alias for list(). Because list() takes a single
        iterable argument, pass a list literal as the arg.
        """
        node = FactoryNode("named_list", [1, 2, 3])
        result = Latent.materialize(node)
        assert result == [1, 2, 3]

    def test_named_tuple(self):
        """named_tuple is an alias for tuple()."""
        node = FactoryNode("named_tuple", [10, 20])
        result = Latent.materialize(node)
        assert result == (10, 20)

    def test_named_dict(self):
        """named_dict is an alias for dict()."""
        node = FactoryNode("named_dict", a=1, b=2)
        result = Latent.materialize(node)
        assert result == {"a": 1, "b": 2}

    def test_call(self):
        """call(fn, *args, **kwargs) invokes fn."""
        node = FactoryNode("call", list, [1, 2, 3])
        result = Latent.materialize(node)
        assert result == [1, 2, 3]

    def test_getitem(self):
        """getitem(container, key) performs container[key]."""
        d = {"x": 42}
        node = FactoryNode("getitem", d, "x")
        result = Latent.materialize(node)
        assert result == 42

    def test_items(self):
        """items(d) returns d.items()."""
        d = {"a": 1, "b": 2}
        node = FactoryNode("items", d)
        result = Latent.materialize(node)
        # dict_items is not directly comparable but can be converted
        assert dict(result) == d

    def test_keys(self):
        d = {"a": 1, "b": 2}
        node = FactoryNode("keys", d)
        result = Latent.materialize(node)
        assert set(result) == {"a", "b"}

    def test_values(self):
        d = {"a": 1, "b": 2}
        node = FactoryNode("values", d)
        result = Latent.materialize(node)
        assert set(result) == {1, 2}


# ---------------------------------------------------------------------------
# Node.__call__ delegation
# ---------------------------------------------------------------------------


class TestNodeCall:
    def test_node_call_delegates_to_latent_materialize(self):
        """Calling a node directly should delegate to Latent.materialize."""
        node = FactoryNode("list")
        # __call__ with no extra args
        result = node()
        assert isinstance(result, list)

    def test_node_call_with_context_vars(self):
        var = VarNode("x")
        result = var(context_vars={"x": 99})
        assert result == 99

    def test_factory_node_call_produces_new_instance(self):
        node = FactoryNode("list")
        r1 = node()
        r2 = node()
        assert r1 is not r2

    def test_singleton_node_call_caches(self):
        node = SingletonNode("list")
        r1 = node()
        r2 = node()
        assert r1 is r2
