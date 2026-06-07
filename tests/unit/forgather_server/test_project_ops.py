"""Validation tests for project_ops target resolvers (no CLI invocation).

The create_* happy paths drive the CLI's project/ws_create_cmd (which write
files); they're exercised via the route tests. Here we pin the pure
validation that both the route and the agent tools rely on.
"""

from __future__ import annotations

import pytest

from forgather_server import project_ops


def test_resolve_project_target_slugifies_name(tmp_path):
    target, parts = project_ops.resolve_new_project_target(str(tmp_path), "My Proj")
    assert target == str(tmp_path / "my_proj")
    assert parts == ["my_proj"]


def test_resolve_project_target_requires_name(tmp_path):
    with pytest.raises(ValueError):
        project_ops.resolve_new_project_target(str(tmp_path), "   ")


def test_resolve_project_target_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        project_ops.resolve_new_project_target(str(tmp_path), "x", "../escape")


def test_resolve_project_target_refuses_existing(tmp_path):
    (tmp_path / "p").mkdir()
    with pytest.raises(FileExistsError):
        project_ops.resolve_new_project_target(str(tmp_path), "p")


def test_resolve_workspace_target_slugifies(tmp_path):
    t = project_ops.resolve_new_workspace_target(str(tmp_path), "My WS")
    assert t == str(tmp_path / "my_ws")


def test_resolve_workspace_target_refuses_existing(tmp_path):
    (tmp_path / "my_ws").mkdir()
    with pytest.raises(FileExistsError):
        project_ops.resolve_new_workspace_target(str(tmp_path), "My WS")


def test_resolvers_enforce_fs_root(tmp_path, monkeypatch):
    from forgather_server import paths

    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: False)
    with pytest.raises(PermissionError):
        project_ops.resolve_new_project_target(str(tmp_path), "p")
    with pytest.raises(PermissionError):
        project_ops.resolve_new_workspace_target(str(tmp_path), "w")
