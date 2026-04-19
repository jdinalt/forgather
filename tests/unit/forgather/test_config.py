"""Unit tests for forgather.config"""

import pytest

from forgather.config import (
    Config,
    ConfigDict,
    ConfigEnvironment,
    ConfigText,
    fconfig,
    pconfig,
)
from forgather.latent import (
    FactoryNode,
    Latent,
    PartialNode,
    SingletonNode,
    VarNode,
)

# ---------------------------------------------------------------------------
# ConfigText
# ---------------------------------------------------------------------------


class TestConfigText:
    def test_is_str_subclass(self):
        """ConfigText must be a str subclass so it can be used wherever str is."""
        ct = ConfigText("hello")
        assert isinstance(ct, str)

    def test_with_line_numbers_adds_line_numbers(self):
        """with_line_numbers() prepends a right-justified line number to each line."""
        ct = ConfigText("first\nsecond\nthird")
        numbered = ct.with_line_numbers()
        lines = numbered.split("\n")
        # Each line should start with a number
        assert lines[0].lstrip().startswith("1:")
        assert lines[1].lstrip().startswith("2:")
        assert lines[2].lstrip().startswith("3:")

    def test_with_line_numbers_false_returns_self(self):
        """Passing False to with_line_numbers() returns the original object unchanged."""
        ct = ConfigText("hello")
        result = ct.with_line_numbers(False)
        assert result is ct

    def test_with_line_numbers_preserves_content(self):
        """The original content appears in the output of with_line_numbers()."""
        content = "key: value"
        ct = ConfigText(content)
        numbered = ct.with_line_numbers()
        assert content in numbered


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_get_returns_config_and_pp_config_tuple(self):
        """Config.get() returns exactly (config, pp_config)."""
        cfg = {"key": "value"}
        pp = "key: value"
        c = Config(cfg, pp)
        result = c.get()
        assert result == (cfg, pp)

    def test_get_is_a_two_tuple(self):
        """The returned tuple has exactly two elements."""
        c = Config(42, "42")
        assert len(c.get()) == 2

    def test_repr_includes_type_name(self):
        """__repr__ should include the class name 'Config'."""
        c = Config({"a": 1}, "a: 1")
        assert "Config" in repr(c)

    def test_repr_includes_config_content(self):
        """__repr__ should show both the config and pp_config."""
        c = Config({"a": 1}, "a: 1")
        r = repr(c)
        assert "config=" in r
        assert "pp_config=" in r


# ---------------------------------------------------------------------------
# ConfigDict
# ---------------------------------------------------------------------------


class TestConfigDict:
    def test_basic_attribute_access(self):
        """Keys are accessible as attributes."""
        cd = ConfigDict({"name": "Alice", "age": 30})
        assert cd.name == "Alice"
        assert cd.age == 30

    def test_filters_dot_prefixed_keys(self):
        """Keys starting with '.' are silently dropped during construction."""
        cd = ConfigDict({"key": "value", ".hidden": "secret"})
        assert "key" in cd
        assert ".hidden" not in cd

    def test_multiple_dot_prefixed_keys_filtered(self):
        """All dot-prefixed keys are removed, not just the first one."""
        cd = ConfigDict({".a": 1, ".b": 2, "c": 3})
        assert list(cd.keys()) == ["c"]

    def test_raises_key_error_for_missing_attribute(self):
        """Accessing a non-existent attribute raises KeyError (not AttributeError)."""
        cd = ConfigDict({"key": "value"})
        with pytest.raises(KeyError):
            _ = cd.missing_key

    def test_regular_dict_operations_work(self):
        """ConfigDict behaves as a normal dict for iteration and membership."""
        cd = ConfigDict({"x": 1, "y": 2})
        assert set(cd.keys()) == {"x", "y"}
        assert cd["x"] == 1


# ---------------------------------------------------------------------------
# ConfigEnvironment
# ---------------------------------------------------------------------------


class TestConfigEnvironmentConstruction:
    def test_basic_construction_with_default_searchpath(self):
        """ConfigEnvironment can be constructed without arguments."""
        env = ConfigEnvironment()
        assert env is not None

    def test_construction_with_string_searchpath(self):
        """A single string searchpath is accepted."""
        env = ConfigEnvironment(searchpath=".")
        assert env is not None

    def test_construction_with_list_searchpath(self):
        """A list of paths is accepted."""
        env = ConfigEnvironment(searchpath=["."])
        assert env is not None


class TestLoadFromString:
    def setup_method(self):
        self.env = ConfigEnvironment()

    def test_simple_yaml_dict(self):
        """A simple YAML dict is loaded into a ConfigDict."""
        result = self.env.load_from_string("key: value\nnum: 42")
        assert isinstance(result, Config)
        cfg, _ = result.get()
        assert isinstance(cfg, ConfigDict)
        assert cfg["key"] == "value"
        assert cfg["num"] == 42

    def test_simple_scalar_value(self):
        """A bare scalar YAML value is loaded as a Python scalar."""
        result = self.env.load_from_string("42")
        cfg, _ = result.get()
        assert cfg == 42

    def test_singleton_tag_creates_singleton_node(self):
        """!singleton tag creates a SingletonNode in the config."""
        result = self.env.load_from_string("!singleton:builtins:list")
        cfg, _ = result.get()
        assert isinstance(cfg, SingletonNode)
        assert cfg.constructor == "builtins:list"

    def test_factory_tag_creates_factory_node(self):
        """!factory tag creates a FactoryNode in the config."""
        result = self.env.load_from_string("model: !factory:builtins:dict\n  x: 1")
        cfg, _ = result.get()
        assert isinstance(cfg["model"], FactoryNode)
        assert cfg["model"].constructor == "builtins:dict"

    def test_partial_tag_creates_partial_node(self):
        """!partial tag creates a PartialNode in the config."""
        result = self.env.load_from_string("fn: !partial:builtins:list")
        cfg, _ = result.get()
        assert isinstance(cfg["fn"], PartialNode)

    def test_var_tag_creates_var_node(self):
        """!var tag creates a VarNode with the given name."""
        result = self.env.load_from_string("v: !var my_variable")
        cfg, _ = result.get()
        assert isinstance(cfg["v"], VarNode)
        assert cfg["v"].constructor == "my_variable"

    def test_list_tag(self):
        """!list tag creates a SingletonNode with constructor 'named_list'."""
        result = self.env.load_from_string("lst: !list\n  - 1\n  - 2")
        cfg, _ = result.get()
        assert isinstance(cfg["lst"], SingletonNode)
        assert cfg["lst"].constructor == "named_list"

    def test_tuple_tag(self):
        """!tuple tag creates a SingletonNode with constructor 'named_tuple'."""
        result = self.env.load_from_string("tup: !tuple\n  - 1\n  - 2")
        cfg, _ = result.get()
        assert isinstance(cfg["tup"], SingletonNode)
        assert cfg["tup"].constructor == "named_tuple"

    def test_dict_tag(self):
        """!dict tag creates a SingletonNode with constructor 'named_dict'."""
        result = self.env.load_from_string("dct: !dict\n  key: value")
        cfg, _ = result.get()
        assert isinstance(cfg["dct"], SingletonNode)
        assert cfg["dct"].constructor == "named_dict"

    def test_singleton_materializes_correctly(self):
        """A !singleton:builtins:list node materializes to an empty list."""
        result = self.env.load_from_string("!singleton:builtins:list")
        cfg, _ = result.get()
        materialized = Latent.materialize(cfg)
        assert materialized == []

    def test_list_tag_materializes_correctly(self):
        """A !list node materializes to a Python list."""
        result = self.env.load_from_string("lst: !list\n  - 10\n  - 20")
        cfg, _ = result.get()
        materialized = Latent.materialize(cfg)
        assert materialized["lst"] == [10, 20]

    def test_yaml_anchors_and_aliases(self):
        """YAML anchors and aliases are resolved correctly."""
        yaml_str = "anchor: &anchor\n  key: value\nref: *anchor"
        result = self.env.load_from_string(yaml_str)
        cfg, _ = result.get()
        assert cfg["anchor"] == {"key": "value"}
        assert cfg["ref"] == {"key": "value"}

    def test_dot_prefixed_keys_filtered_in_config_dict(self):
        """Dot-prefixed keys are filtered out when loading into ConfigDict."""
        result = self.env.load_from_string(".hidden: secret\nvisible: yes")
        cfg, _ = result.get()
        assert "visible" in cfg
        assert ".hidden" not in cfg

    def test_returns_config_object(self):
        """load_from_string always returns a Config object."""
        result = self.env.load_from_string("x: 1")
        assert isinstance(result, Config)

    def test_pp_config_stored_on_config_object(self):
        """The preprocessed YAML string is stored in the Config object."""
        yaml_str = "key: value"
        result = self.env.load_from_string(yaml_str)
        _, pp = result.get()
        # The preprocessed output should contain the original content
        assert "key" in pp
        assert "value" in pp


class TestLoadFromPpstring:
    def test_parses_already_preprocessed_yaml(self):
        """load_from_ppstring parses a plain YAML string (no Jinja preprocessing)."""
        env = ConfigEnvironment()
        result = env.load_from_ppstring("key: value\nnum: 99")
        assert isinstance(result, Config)
        cfg, _ = result.get()
        assert cfg["key"] == "value"
        assert cfg["num"] == 99

    def test_returns_config_object(self):
        """load_from_ppstring returns a Config instance."""
        env = ConfigEnvironment()
        result = env.load_from_ppstring("x: 1")
        assert isinstance(result, Config)

    def test_stores_pp_config(self):
        """The pp_config field of Config is the string that was passed in."""
        env = ConfigEnvironment()
        pp = "x: 42"
        result = env.load_from_ppstring(pp)
        _, stored_pp = result.get()
        assert stored_pp == pp


class TestPreprocessFromString:
    def test_returns_config_text(self):
        """preprocess_from_string returns a ConfigText instance."""
        env = ConfigEnvironment()
        result = env.preprocess_from_string("key: value")
        assert isinstance(result, ConfigText)

    def test_content_is_preserved(self):
        """The preprocessed output contains the original YAML content."""
        env = ConfigEnvironment()
        result = env.preprocess_from_string("key: value")
        assert "key" in result
        assert "value" in result

    def test_jinja_variable_substitution(self):
        """Jinja2 variables passed as kwargs are substituted during preprocessing."""
        env = ConfigEnvironment()
        result = env.preprocess_from_string("name: {{ my_name }}", my_name="Alice")
        assert "Alice" in result


# ---------------------------------------------------------------------------
# fconfig
# ---------------------------------------------------------------------------


class TestFconfig:
    def test_formats_dict_correctly(self):
        """A plain dict is formatted as key: value lines."""
        result = fconfig({"key": "value", "num": 42})
        assert "key" in result
        assert "value" in result
        assert "num" in result
        assert "42" in result

    def test_formats_node_correctly(self):
        """A SingletonNode is formatted with 'singleton' label and constructor."""
        node = SingletonNode("builtins:list", _identity="my_list")
        result = fconfig(node)
        assert "singleton" in result
        assert "builtins:list" in result
        assert "my_list" in result

    def test_formats_var_node(self):
        """A VarNode is formatted with 'var' prefix, name, and value."""
        node = VarNode("my_var", 42)
        result = fconfig(node)
        assert "var" in result
        assert "my_var" in result
        assert "42" in result

    def test_visited_set_prevents_recursion(self):
        """When a node's identity is in the 'visited' set, it is elided."""
        node = SingletonNode("builtins:list", _identity="my_list")
        result = fconfig(node, visited={"my_list"})
        assert "elided" in result

    def test_formats_string_with_quotes(self):
        """Plain strings are wrapped in single quotes in the output."""
        result = fconfig("hello")
        assert result == "'hello'"

    def test_formats_list(self):
        """A list is formatted with '- ' prefix per item."""
        result = fconfig(["a", "b", "c"])
        assert "- " in result

    def test_formats_nested_dict(self):
        """Nested dicts are indented in the output."""
        result = fconfig({"outer": {"inner": "val"}})
        assert "outer" in result
        assert "inner" in result
        assert "val" in result

    def test_formats_config_object(self):
        """A Config object is formatted by delegating to its config and pp_config."""
        cfg = Config({"x": 1}, "x: 1")
        result = fconfig(cfg)
        assert "x" in result

    def test_formats_config_text_with_line_numbers(self):
        """A ConfigText object is rendered with line numbers by fconfig."""
        ct = ConfigText("line one\nline two")
        result = fconfig(ct)
        # Should have line numbers
        assert "1:" in result
        assert "2:" in result

    def test_returns_string(self):
        """fconfig always returns a string."""
        assert isinstance(fconfig({"a": 1}), str)
        assert isinstance(fconfig(42), str)
        assert isinstance(fconfig(None), str)
