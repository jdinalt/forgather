"""
Unit tests for forgather.preprocess
"""

import os
import tempfile

import pytest

from forgather.preprocess import (
    LineStatementProcessor,
    PPEnvironment,
    preprocess,
    preprocess_toml_blocks,
    split_templates,
    toyaml,
)

# ---------------------------------------------------------------------------
# split_templates
# ---------------------------------------------------------------------------


class TestSplitTemplates:
    def test_no_markers_yields_single_tuple(self):
        result = list(split_templates("hello world"))
        assert len(result) == 1
        assert result[0] == (None, "hello world")

    def test_one_marker_splits_into_two_parts(self):
        template = "before\n#--- foo ---\nafter"
        result = list(split_templates(template))
        assert len(result) == 2
        assert result[0] == (None, "before")
        assert result[1] == ("foo", "after")

    def test_multiple_markers_yield_correct_count(self):
        template = "a\n#--- b ---\nc\n#--- d ---\ne"
        result = list(split_templates(template))
        assert len(result) == 3
        names = [r[0] for r in result]
        assert names == [None, "b", "d"]

    def test_initial_name_passed_as_first_tuple_name(self):
        result = list(split_templates("content", name="initial"))
        assert len(result) == 1
        assert result[0][0] == "initial"

    def test_marker_with_dots_and_slashes_in_name(self):
        template = "header\n#--- some.sub/name ---\nbody"
        result = list(split_templates(template))
        assert result[1][0] == "some.sub/name"

    def test_content_after_marker_is_correct(self):
        template = "part1\n#--- section ---\npart2"
        result = list(split_templates(template))
        assert "part2" in result[1][1]
        assert "part1" in result[0][1]


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Tests for the preprocess() function.

    Several tests temporarily override LineStatementProcessor.preserve_line_numbers
    to False so that comment lines are removed rather than turned into Jinja
    comments (the default).
    """

    def test_double_dash_becomes_jinja_statement(self):
        result = preprocess("-- if True")
        assert result == "{% if True %}"

    def test_double_equals_becomes_jinja_expression(self):
        result = preprocess("== myvar")
        assert result == "{{ myvar }}"

    def test_left_angle_becomes_minus_jinja_statement(self):
        result = preprocess("<< block foo")
        assert result == "{%- block foo %}"

    def test_right_angle_becomes_jinja_statement_minus(self):
        result = preprocess(">> endblock")
        assert result == "{% endblock -%}"

    def test_fat_arrow_becomes_trim_expression(self):
        result = preprocess("=> expr")
        assert result == "{{ expr|trim('\\n')}}"

    def test_full_comment_line_removed_when_not_preserving(self):
        prev = LineStatementProcessor.preserve_line_numbers
        try:
            LineStatementProcessor.preserve_line_numbers = False
            result = preprocess("## this is a comment")
            assert result == ""
        finally:
            LineStatementProcessor.preserve_line_numbers = prev

    def test_inline_comment_stripped_from_line(self):
        prev = LineStatementProcessor.preserve_line_numbers
        try:
            LineStatementProcessor.preserve_line_numbers = False
            result = preprocess("text ## inline comment")
            # The trailing comment is stripped; only the code remains
            assert result.strip() == "text"
            assert "inline comment" not in result
        finally:
            LineStatementProcessor.preserve_line_numbers = prev

    def test_full_comment_becomes_jinja_comment_when_preserving(self):
        prev = LineStatementProcessor.preserve_line_numbers
        try:
            LineStatementProcessor.preserve_line_numbers = True
            result = preprocess("## my comment")
            # Becomes a Jinja comment rather than being deleted
            assert result.startswith("{#")
            assert result.endswith("#}")
        finally:
            LineStatementProcessor.preserve_line_numbers = prev

    def test_multiple_lines_processed_correctly(self):
        prev = LineStatementProcessor.preserve_line_numbers
        try:
            LineStatementProcessor.preserve_line_numbers = False
            source = "-- if x\n== x\n-- endif"
            result = preprocess(source)
            assert "{% if x %}" in result
            assert "{{ x }}" in result
            assert "{% endif %}" in result
        finally:
            LineStatementProcessor.preserve_line_numbers = prev


# ---------------------------------------------------------------------------
# preprocess_toml_blocks
# ---------------------------------------------------------------------------


class TestPreprocessTomlBlocks:
    def test_single_block_becomes_jinja2_block(self):
        source = "[myblock]\ncontent"
        result = preprocess_toml_blocks(source)
        assert "{% block myblock %}" in result
        assert "content" in result

    def test_single_block_is_closed(self):
        source = "[myblock]\ncontent"
        result = preprocess_toml_blocks(source)
        assert "{% endblock myblock" in result

    def test_nested_blocks_at_lower_indent_closes_outer(self):
        # An outer block followed by an inner block at the same indent
        # should close the inner before opening the next sibling.
        source = "[outer]\n[inner]\ncontent"
        result = preprocess_toml_blocks(source)
        assert "{% block outer %}" in result
        assert "{% block inner %}" in result
        assert "{% endblock inner" in result

    def test_open_block_at_end_of_file_gets_closed(self):
        source = "[block_open]\ncontent"
        result = preprocess_toml_blocks(source)
        assert "{% endblock block_open" in result

    def test_non_block_lines_pass_through(self):
        source = "plain line\n[myblock]\nmore content"
        result = preprocess_toml_blocks(source)
        assert "plain line" in result
        assert "more content" in result


# ---------------------------------------------------------------------------
# toyaml
# ---------------------------------------------------------------------------


class TestToYaml:
    def test_none_returns_null(self):
        assert toyaml(None) == "null"

    def test_int_returns_string(self):
        assert toyaml(42) == "42"
        assert toyaml(0) == "0"
        assert toyaml(-1) == "-1"

    def test_bool_true_returns_string(self):
        assert toyaml(True) == "True"

    def test_bool_false_returns_string(self):
        assert toyaml(False) == "False"

    def test_dict_returns_inline_yaml_mapping(self):
        result = toyaml({"a": 1, "b": 2})
        # PyYAML flow style produces {a: 1, b: 2}
        assert "{" in result
        assert "a" in result
        assert "1" in result

    def test_list_returns_inline_yaml_sequence(self):
        result = toyaml([1, 2, 3])
        assert "[" in result
        assert "1" in result
        assert "3" in result

    def test_plain_string_is_not_quoted(self):
        result = toyaml("hello")
        assert result == "hello"

    def test_string_with_colon_is_quoted(self):
        result = toyaml("hello: world")
        # PyYAML must quote the string because it contains ': '
        assert "'" in result or '"' in result

    def test_empty_dict(self):
        result = toyaml({})
        assert "{}" in result

    def test_empty_list(self):
        result = toyaml([])
        assert "[]" in result


# ---------------------------------------------------------------------------
# PPEnvironment
# ---------------------------------------------------------------------------


class TestPPEnvironment:
    def _make_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            return PPEnvironment(searchpath=tmpdir), tmpdir

    def test_can_be_instantiated_with_searchpath(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = PPEnvironment(searchpath=tmpdir)
            assert env is not None

    def test_requires_loader_or_searchpath(self):
        with pytest.raises(AssertionError):
            PPEnvironment()  # neither loader nor searchpath

    def test_has_isotime_global(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = PPEnvironment(searchpath=tmpdir)
            assert "isotime" in env.globals
            # Calling it returns a string
            assert isinstance(env.globals["isotime"](), str)

    def test_has_joinpath_global(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = PPEnvironment(searchpath=tmpdir)
            assert "joinpath" in env.globals

    def test_has_toyaml_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = PPEnvironment(searchpath=tmpdir)
            assert "toyaml" in env.filters

    def test_from_string_renders_simple_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = PPEnvironment(searchpath=tmpdir)
            t = env.from_string("hello world")
            assert t.render() == "hello world"

    def test_from_string_renders_set_and_echo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = PPEnvironment(searchpath=tmpdir)
            # '-- set' preprocesses to {% set %} and '==' to {{ }}
            t = env.from_string("-- set x = 42\n== x")
            result = t.render()
            assert "42" in result

    def test_joinpath_global_works_in_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = PPEnvironment(searchpath=tmpdir)
            t = env.from_string('{{ joinpath("foo", "bar") }}')
            result = t.render()
            assert result == os.path.join("foo", "bar")

    def test_toyaml_filter_converts_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = PPEnvironment(searchpath=tmpdir)
            t = env.from_string('{{ {"a": 1} | toyaml }}')
            result = t.render()
            # Should produce YAML flow-style mapping
            assert "a" in result
            assert "1" in result
