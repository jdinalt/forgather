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


def test_config_template_refs_delegates(monkeypatch):
    monkeypatch.setattr(
        config_ops, "render_trefs_tree", lambda pd, cn: f"tree:{pd}:{cn}"
    )
    assert (
        tools_readonly._config_template_refs({"project_dir": "/p", "config_name": "c"})
        == "tree:/p:c"
    )
