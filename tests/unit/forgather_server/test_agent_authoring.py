"""Tests for the agent authoring commit mechanics (real file writes, no LLM).

Exercises the preview/commit split in ``config_ops`` and the
``propose_edit_config`` tool end-to-end: a Proposal previews without
writing, and only its ``commit`` closure mutates the file. The
new-config-from-template path goes through ``new_template_file`` (now a
thin wrapper) which the existing ``test_routes_new_project`` suite covers.
"""

from __future__ import annotations

import os

import pytest

from forgather_server import config_ops
from forgather_server.agent import tools_authoring
from forgather_server.agent.registry import Proposal


def test_write_existing_file_overwrites_atomically(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("old: 1\n")
    info = config_ops.write_existing_file(str(f), "new: 2\n")
    assert f.read_text() == "new: 2\n"
    assert info["path"] == str(f)
    assert info["bytes_written"] == len("new: 2\n".encode("utf-8"))


def test_write_existing_file_stale_mtime_raises(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("old\n")
    stale = os.path.getmtime(str(f)) - 100  # pretend we read it long ago
    # Bump on-disk mtime so the guard trips.
    os.utime(str(f), None)
    with pytest.raises(config_ops.StaleEditError):
        config_ops.write_existing_file(str(f), "x\n", expected_mtime=stale)
    assert f.read_text() == "old\n"  # unchanged


def test_write_existing_file_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        config_ops.write_existing_file(str(tmp_path / "nope.yaml"), "x")


def test_write_existing_file_enforces_fs_root(tmp_path, monkeypatch):
    # Activate an fs-root allowlist that does NOT include tmp_path.
    from forgather_server import paths

    other = tmp_path / "allowed"
    other.mkdir()
    target = tmp_path / "outside.yaml"
    target.write_text("x\n")
    monkeypatch.setattr(paths, "fs_roots", lambda: (other,))
    monkeypatch.setattr(paths, "fs_roots_active", lambda: True)
    monkeypatch.setattr(
        paths,
        "is_path_in_fs_root",
        lambda p: paths._is_descendant(__import__("pathlib").Path(p).resolve(), other),
    )
    with pytest.raises(PermissionError):
        config_ops.write_existing_file(str(target), "y\n")
    assert target.read_text() == "x\n"


def test_propose_edit_config_previews_then_commits(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("before: yes\n")

    proposal = tools_authoring._propose_edit_config(
        {
            "project_dir": str(tmp_path),
            "config_name": "c.yaml",
            "path": str(f),
            "new_content": "after: yes\n",
        }
    )
    assert isinstance(proposal, Proposal)
    assert proposal.before == "before: yes\n"
    assert proposal.after == "after: yes\n"
    # Preview must not have written anything yet.
    assert f.read_text() == "before: yes\n"

    result = proposal.commit()
    assert f.read_text() == "after: yes\n"
    assert "wrote" in result.lower()


def test_propose_new_project_previews_and_commits(tmp_path, monkeypatch):
    from forgather_server import project_ops

    captured = {}

    def fake_create(**kw):
        captured.update(kw)
        return str(tmp_path / "demo")

    monkeypatch.setattr(project_ops, "create_project", fake_create)
    p = tools_authoring._propose_new_project(
        {"workspace_dir": str(tmp_path), "name": "Demo", "description": "d"}
    )
    assert isinstance(p, Proposal)
    assert p.path == str(tmp_path / "demo")  # previewed target (slug of name)
    assert p.before is None and p.after is None  # no diff for a scaffold
    res = p.commit()
    assert captured["name"] == "Demo" and captured["workspace_dir"] == str(tmp_path)
    assert "created project" in res.lower()


def test_propose_new_workspace_previews_and_commits(tmp_path, monkeypatch):
    from forgather_server import project_ops

    captured = {}

    def fake_create_ws(**kw):
        captured.update(kw)
        return str(tmp_path / "my_ws")

    monkeypatch.setattr(project_ops, "create_workspace", fake_create_ws)
    p = tools_authoring._propose_new_workspace(
        {"parent_dir": str(tmp_path), "name": "My WS", "description": "d"}
    )
    assert p.path == str(tmp_path / "my_ws")
    res = p.commit()
    assert captured["parent_dir"] == str(tmp_path)
    assert "created workspace" in res.lower()


def test_propose_new_config_refuses_existing(tmp_path):
    # resolve_new_template_target needs a real project; here we only assert
    # the propose handler surfaces validation by hitting a missing project,
    # which raises before any write.
    with pytest.raises(Exception):
        tools_authoring._propose_new_config(
            {"project_dir": str(tmp_path), "name": "x.yaml"}
        )
