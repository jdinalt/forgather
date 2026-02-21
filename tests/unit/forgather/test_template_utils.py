"""
Unit tests for forgather.template_utils
"""

import os
import tempfile

import pytest

from forgather.template_utils import (
    _build_extends_graph,
    extends_graph_iter,
    get_extends_graph,
    template_data_iter,
    template_extends_iter,
)


def write_template(directory, filename, content):
    """Helper: write a template file and return (name, path) tuple."""
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        f.write(content)
    return (filename, path)


class TestTemplateDataIter:
    def test_single_template_no_split(self, tmp_path):
        content = "-- block foo\nhello\n-- endblock foo\n"
        path = tmp_path / "simple.yaml"
        path.write_text(content)
        items = list(template_data_iter([("simple.yaml", str(path))]))
        assert len(items) == 1
        name, filepath, data = items[0]
        assert name == "simple.yaml"
        assert "hello" in data

    def test_template_with_split_marker(self, tmp_path):
        content = "main content\n\n# --- sub_template ---\nsub content\n"
        path = tmp_path / "multi.yaml"
        path.write_text(content)
        items = list(template_data_iter([("multi.yaml", str(path))]))
        assert len(items) == 2
        # First item has the initial name
        assert items[0][0] == "multi.yaml"
        assert "main content" in items[0][2]
        # Second item has the marker name
        assert items[1][0] == "sub_template"
        assert "sub content" in items[1][2]

    def test_multiple_templates(self, tmp_path):
        for i in range(3):
            (tmp_path / f"t{i}.yaml").write_text(f"content {i}")

        template_iter = [(f"t{i}.yaml", str(tmp_path / f"t{i}.yaml")) for i in range(3)]
        items = list(template_data_iter(template_iter))
        assert len(items) == 3


class TestTemplateExtendsIter:
    def test_no_extends(self, tmp_path):
        path = tmp_path / "base.yaml"
        path.write_text("key: value\n")
        items = list(template_extends_iter([("base.yaml", str(path), "key: value\n")]))
        assert len(items) == 1
        name, filepath, extends = items[0]
        assert name == "base.yaml"
        assert extends is None

    def test_with_extends_line_statement(self, tmp_path):
        content = "-- extends 'parent.yaml'\nkey: value\n"
        path = tmp_path / "child.yaml"
        path.write_text(content)
        items = list(template_extends_iter([("child.yaml", str(path), content)]))
        name, filepath, extends = items[0]
        assert extends == "parent.yaml"

    def test_with_jinja_extends(self, tmp_path):
        content = "{% extends 'parent.yaml' %}\nkey: value\n"
        path = tmp_path / "child.yaml"
        path.write_text(content)
        items = list(template_extends_iter([("child.yaml", str(path), content)]))
        name, filepath, extends = items[0]
        assert extends == "parent.yaml"

    def test_mixed_extends_and_no_extends(self, tmp_path):
        base_content = "base content"
        child_content = "-- extends 'base.yaml'\nchild content"

        base_path = tmp_path / "base.yaml"
        child_path = tmp_path / "child.yaml"
        base_path.write_text(base_content)
        child_path.write_text(child_content)

        items = list(
            template_extends_iter(
                [
                    ("base.yaml", str(base_path), base_content),
                    ("child.yaml", str(child_path), child_content),
                ]
            )
        )
        assert items[0][2] is None  # base has no extends
        assert items[1][2] == "base.yaml"  # child extends base


class TestBuildExtendsGraph:
    def test_empty_list(self):
        result = _build_extends_graph([], {})
        assert result == []

    def test_no_children(self):
        templates = [("root.yaml", "/path/root.yaml")]
        extends_map = {}
        result = _build_extends_graph(templates, extends_map)
        assert len(result) == 1
        name, path, children = result[0]
        assert name == "root.yaml"
        assert children is None

    def test_with_children(self):
        templates = [("root.yaml", "/path/root.yaml")]
        extends_map = {"root.yaml": [("child.yaml", "/path/child.yaml")]}
        result = _build_extends_graph(templates, extends_map)
        assert len(result) == 1
        name, path, children = result[0]
        assert name == "root.yaml"
        assert children is not None
        assert len(children) == 1
        assert children[0][0] == "child.yaml"

    def test_deeply_nested(self):
        templates = [("a.yaml", "/a")]
        extends_map = {
            "a.yaml": [("b.yaml", "/b")],
            "b.yaml": [("c.yaml", "/c")],
        }
        result = _build_extends_graph(templates, extends_map)
        a = result[0]
        b = a[2][0]
        c = b[2][0]
        assert a[0] == "a.yaml"
        assert b[0] == "b.yaml"
        assert c[0] == "c.yaml"
        assert c[2] is None


class TestGetExtendsGraph:
    def test_single_file_no_extends(self, tmp_path):
        path = tmp_path / "only.yaml"
        path.write_text("key: value")
        extends_iter = [("only.yaml", str(path), None)]
        graph = get_extends_graph(extends_iter)
        assert len(graph) == 1
        assert graph[0][0] == "only.yaml"

    def test_parent_child_relationship(self, tmp_path):
        parent_path = tmp_path / "parent.yaml"
        child_path = tmp_path / "child.yaml"
        parent_path.write_text("parent content")
        child_path.write_text("child content")

        extends_iter = [
            ("parent.yaml", str(parent_path), None),
            ("child.yaml", str(child_path), "parent.yaml"),
        ]
        graph = get_extends_graph(extends_iter)
        # parent should be a root node (not extended by anyone above it)
        assert any(node[0] == "parent.yaml" for node in graph)
        parent_node = next(n for n in graph if n[0] == "parent.yaml")
        # child should appear as a child of parent
        assert parent_node[2] is not None
        assert any(n[0] == "child.yaml" for n in parent_node[2])


class TestExtendsGraphIter:
    def test_empty_graph(self):
        items = list(extends_graph_iter([]))
        assert items == []

    def test_single_node(self):
        graph = [("root.yaml", "/root", None)]
        items = list(extends_graph_iter(graph))
        assert len(items) == 1
        level, name, path = items[0]
        assert level == 0
        assert name == "root.yaml"

    def test_levels_increment(self):
        graph = [
            (
                "root.yaml",
                "/root",
                [
                    (
                        "child.yaml",
                        "/child",
                        [("grandchild.yaml", "/grandchild", None)],
                    )
                ],
            )
        ]
        items = list(extends_graph_iter(graph))
        assert len(items) == 3
        levels = [level for level, _, _ in items]
        assert levels == [0, 1, 2]

    def test_multiple_roots(self):
        graph = [
            ("a.yaml", "/a", None),
            ("b.yaml", "/b", None),
        ]
        items = list(extends_graph_iter(graph))
        assert len(items) == 2
        assert all(level == 0 for level, _, _ in items)

    def test_custom_start_level(self):
        graph = [("node.yaml", "/node", None)]
        items = list(extends_graph_iter(graph, level=5))
        assert items[0][0] == 5
