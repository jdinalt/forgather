"""Read-tool registration + delegation for the config-debug tools."""

from __future__ import annotations

import pytest

from forgather_server import config_ops, discovery, paths
from forgather_server.agent import tools_readonly
from forgather_server.agent.registry import ToolRegistry, UiDirective


def test_expected_read_tools_registered():
    reg = ToolRegistry()
    tools_readonly.register_all(reg)
    names = {s.name for s in reg.specs()}
    assert {
        "list_workspaces",
        "list_projects",
        "list_configs",
        "inspect_config",
        "render_config_pp",
        "render_config_code",
        "check_config",
        "list_config_templates",
        "config_template_refs",
        "reveal_in_ui",
        "list_directory",
        "find_files",
        "search_docs",
    } <= names


def test_reveal_in_ui_validates(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)
    f = tmp_path / "x.yaml"
    f.write_text("a: 1")

    # A files reveal of an existing path returns a UiDirective.
    d = tools_readonly._reveal_in_ui({"path": str(f), "where": "files"})
    assert isinstance(d, UiDirective)
    assert d.action == "reveal"
    assert d.payload == {"path": str(f), "where": "files"}

    # A projects reveal of a path that isn't a known ws/project/config errors.
    monkeypatch.setattr(discovery, "discover_projects", lambda: [])
    with pytest.raises(ValueError, match="not a known"):
        tools_readonly._reveal_in_ui({"path": str(f), "where": "projects"})

    # An unknown 'where' errors; a non-existent path errors.
    with pytest.raises(ValueError, match="where must be"):
        tools_readonly._reveal_in_ui({"path": str(f), "where": "nope"})
    with pytest.raises(ValueError, match="does not exist"):
        tools_readonly._reveal_in_ui({"path": str(tmp_path / "nope.yaml")})


def test_render_config_code_delegates(monkeypatch):
    seen = {}

    def fake(pd, cn, target="main"):
        seen.update(project_dir=pd, config_name=cn, target=target)
        return "CODE"

    monkeypatch.setattr(config_ops, "render_code", fake)
    out = tools_readonly._render_config_code(
        {"project_dir": "/p", "config_name": "c.yaml"}
    )
    assert out == "CODE"
    assert seen["target"] == "main"  # defaults when omitted/blank


def test_check_config_delegates(monkeypatch):
    seen = {}

    def fake(pd, cn, target=None):
        seen.update(project_dir=pd, config_name=cn, target=target)
        return {"ok": True, "targets": ["main"]}

    monkeypatch.setattr(config_ops, "check_config", fake)
    out = tools_readonly._check_config(
        {"project_dir": "/p", "config_name": "c.yaml"}
    )
    assert out == {"ok": True, "targets": ["main"]}
    assert seen["target"] is None  # optional; None when omitted/blank
    # A blank target string is normalized to None (not passed through as "").
    tools_readonly._check_config(
        {"project_dir": "/p", "config_name": "c.yaml", "target": ""}
    )
    assert seen["target"] is None


def test_read_file_default_returns_whole_file(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)
    f = tmp_path / "doc.md"
    f.write_text("abcdefghij")
    # No offset/limit -> whole file unchanged (the loop budget clips if huge).
    assert tools_readonly._read_file({"path": str(f)}) == "abcdefghij"


def test_read_file_offset_and_limit_window(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)
    f = tmp_path / "doc.md"
    f.write_text("0123456789")

    # A bounded window that reaches EOF: no continuation footer.
    out = tools_readonly._read_file({"path": str(f), "offset": 5, "limit": 5})
    assert out == "56789"

    # A bounded window with more remaining: footer reports the resume offset.
    out = tools_readonly._read_file({"path": str(f), "offset": 0, "limit": 4})
    assert out.startswith("0123")
    assert "offset=4" in out
    assert "of 10" in out

    # Offset alone reads to EOF.
    assert tools_readonly._read_file({"path": str(f), "offset": 8}) == "89"

    # Offset past EOF is clamped (empty, no crash).
    assert tools_readonly._read_file({"path": str(f), "offset": 99}) == ""


def test_read_file_rejects_negative_offset(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)
    f = tmp_path / "doc.md"
    f.write_text("x")
    with pytest.raises(ValueError, match="offset must be"):
        tools_readonly._read_file({"path": str(f), "offset": -1})


def test_read_file_schema_documents_pagination():
    reg = ToolRegistry()
    tools_readonly.register_all(reg)
    spec = {s.name: s for s in reg.specs()}["read_file"]
    props = spec.json_schema["properties"]
    assert "offset" in props and "limit" in props
    assert spec.json_schema["required"] == ["path"]


def test_list_directory_lists_and_starts_from_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("hi")
    (tmp_path / ".hidden").write_text("x")

    # No path -> the starting roots.
    monkeypatch.setattr(
        tools_readonly, "_starting_roots", lambda: [str(tmp_path)]
    )
    roots_out = tools_readonly._list_directory({})
    assert roots_out["roots"] == [str(tmp_path)]

    out = tools_readonly._list_directory({"path": str(tmp_path)})
    names = [e["name"] for e in out["entries"]]
    assert ".hidden" not in names  # hidden skipped
    assert names == ["sub", "a.txt"]  # dirs first, then files
    assert out["entries"][0]["is_dir"] is True
    assert out["entries"][1]["size"] == 2


def test_list_directory_fs_root_and_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: False)
    with pytest.raises(PermissionError, match="outside"):
        tools_readonly._list_directory({"path": str(tmp_path)})
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)
    with pytest.raises(ValueError, match="does not exist"):
        tools_readonly._list_directory({"path": str(tmp_path / "nope")})
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        tools_readonly._list_directory({"path": str(f)})


def test_find_files_substring_and_glob(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)
    (tmp_path / "tokenizers" / "wikitext_32k").mkdir(parents=True)
    (tmp_path / "tokenizers" / "wikitext_32k" / "tokenizer.json").write_text("{}")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "wikitext_junk").write_text("x")  # pruned dir, not matched
    monkeypatch.setattr(tools_readonly, "_starting_roots", lambda: [str(tmp_path)])

    # Bare word -> substring match; finds both the dir and the file.
    out = tools_readonly._find_files({"pattern": "wikitext"})
    found = {m["path"]: m["is_dir"] for m in out["matches"]}
    assert any(p.endswith("wikitext_32k") and is_dir for p, is_dir in found.items())
    # The .git tree is pruned, so its wikitext_junk is not returned.
    assert not any(".git" in p for p in found)

    # Glob is honored verbatim.
    out2 = tools_readonly._find_files({"pattern": "*.json"})
    assert any(p["path"].endswith("tokenizer.json") for p in out2["matches"])


def test_find_files_respects_root_and_fs_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: False)
    with pytest.raises(PermissionError, match="outside"):
        tools_readonly._find_files({"pattern": "x", "root": str(tmp_path)})
    with pytest.raises(ValueError, match="pattern is required"):
        tools_readonly._find_files({"pattern": "  "})


def test_config_template_refs_delegates(monkeypatch):
    monkeypatch.setattr(
        config_ops, "render_trefs_tree", lambda pd, cn: f"tree:{pd}:{cn}"
    )
    assert (
        tools_readonly._config_template_refs({"project_dir": "/p", "config_name": "c"})
        == "tree:/p:c"
    )
