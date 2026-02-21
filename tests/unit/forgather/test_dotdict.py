"""
Unit tests for forgather.dotdict
"""

import pytest

from forgather.dotdict import DotDict


class TestDotDictBasicAccess:
    def test_attribute_access(self):
        d = DotDict({"a": 1, "b": 2})
        assert d.a == 1
        assert d.b == 2

    def test_item_access_still_works(self):
        d = DotDict({"x": 10})
        assert d["x"] == 10

    def test_missing_key_raises_key_error(self):
        d = DotDict({"a": 1})
        with pytest.raises(KeyError):
            _ = d.missing

    def test_setattr_sets_dict_item(self):
        d = DotDict({"a": 1})
        d.b = 2
        assert d["b"] == 2
        assert d.b == 2

    def test_delattr_removes_dict_item(self):
        d = DotDict({"a": 1, "b": 2})
        del d.a
        assert "a" not in d
        assert "b" in d

    def test_empty_dict(self):
        d = DotDict({})
        assert len(d) == 0


class TestDotDictNested:
    def test_nested_dict_converted_to_dotdict(self):
        d = DotDict({"outer": {"inner": 42}})
        assert isinstance(d.outer, DotDict)
        assert d.outer.inner == 42

    def test_deeply_nested(self):
        d = DotDict({"a": {"b": {"c": 3}}})
        assert isinstance(d.a, DotDict)
        assert isinstance(d.a.b, DotDict)
        assert d.a.b.c == 3

    def test_non_dict_values_not_converted(self):
        d = DotDict({"nums": [1, 2, 3], "val": 42})
        assert d.nums == [1, 2, 3]
        assert isinstance(d.nums, list)

    def test_object_with_keys_method_converted(self):
        # Any object with a 'keys' method is treated as dict-like
        # OrderedDict has keys(), so should be converted
        from collections import OrderedDict

        od = OrderedDict([("x", 1), ("y", 2)])
        d = DotDict({"data": od})
        assert isinstance(d.data, DotDict)
        assert d.data.x == 1


class TestDotDictConstructor:
    def test_kwargs_constructor_when_no_positional(self):
        # When dct is None, kwargs are used
        d = DotDict(a=1, b=2)
        assert d.a == 1
        assert d.b == 2

    def test_positional_dict_constructor(self):
        d = DotDict({"key": "value"})
        assert d.key == "value"

    def test_none_dct_uses_kwargs(self):
        d = DotDict(None, x=10, y=20)
        assert d.x == 10
        assert d.y == 20


class TestDotDictInheritance:
    def test_is_dict_subclass(self):
        d = DotDict({"a": 1})
        assert isinstance(d, dict)

    def test_dict_methods_work(self):
        d = DotDict({"a": 1, "b": 2})
        assert set(d.keys()) == {"a", "b"}
        assert set(d.values()) == {1, 2}
        assert len(d) == 2

    def test_iteration(self):
        d = DotDict({"x": 10, "y": 20})
        keys = list(d)
        assert "x" in keys
        assert "y" in keys
