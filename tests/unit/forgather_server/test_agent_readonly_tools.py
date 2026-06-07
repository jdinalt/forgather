"""Read-tool registration + delegation for the config-debug tools."""

from __future__ import annotations

from forgather_server import config_ops
from forgather_server.agent import tools_readonly
from forgather_server.agent.registry import ToolRegistry


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
        "list_config_templates",
        "config_template_refs",
        "search_docs",
    } <= names


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


def test_config_template_refs_delegates(monkeypatch):
    monkeypatch.setattr(
        config_ops, "render_trefs_tree", lambda pd, cn: f"tree:{pd}:{cn}"
    )
    assert (
        tools_readonly._config_template_refs({"project_dir": "/p", "config_name": "c"})
        == "tree:/p:c"
    )
