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


def test_propose_edit_config_guards_on_external_change(tmp_path):
    # The mtime baseline is captured server-side at propose time (the model
    # cannot supply it). If the file changes before approval, commit refuses.
    import time

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
    # Simulate an external change after the proposal was made (bump mtime).
    future = time.time() + 10
    os.utime(f, (future, future))
    with pytest.raises(config_ops.StaleEditError):
        proposal.commit()
    assert f.read_text() == "before: yes\n"  # untouched


def test_propose_edit_config_ignores_model_supplied_mtime(tmp_path):
    # A stale/garbage expected_mtime from the model must NOT cause a spurious
    # failure — the param is gone; the tool uses its own captured baseline.
    f = tmp_path / "c.yaml"
    f.write_text("before: yes\n")
    proposal = tools_authoring._propose_edit_config(
        {
            "project_dir": str(tmp_path),
            "config_name": "c.yaml",
            "path": str(f),
            "new_content": "after: yes\n",
            "expected_mtime": 1,  # bogus; from before the file existed
        }
    )
    proposal.commit()  # would have raised StaleEditError under the old design
    assert f.read_text() == "after: yes\n"


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


def test_propose_new_project_copy_from_passthrough(tmp_path, monkeypatch):
    from forgather_server import project_ops

    captured = {}
    monkeypatch.setattr(
        project_ops, "create_project", lambda **kw: (captured.update(kw), str(tmp_path / "demo"))[1]
    )
    p = tools_authoring._propose_new_project(
        {
            "workspace_dir": str(tmp_path),
            "name": "Demo",
            "description": "d",
            "copy_from": "/abs/example/configs/default.yaml",
        }
    )
    assert "copy:" in p.extra["starting_point"]
    p.commit()
    assert captured["copy_from"] == "/abs/example/configs/default.yaml"


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


def _stub_config_ops(monkeypatch, tmp_path, *, rendered=""):
    """Stub resolve/write/validate so the three starting-point paths can be
    exercised without a real project on disk."""
    from types import SimpleNamespace

    target = str(tmp_path / "c.yaml")
    monkeypatch.setattr(
        config_ops, "resolve_new_template_target",
        lambda pd, kind, name, meta_template=None, values=None: (
            target, rendered if meta_template else ""
        ),
    )
    written = {}
    monkeypatch.setattr(
        config_ops, "write_template_file",
        lambda t, c: written.update(target=t, content=c),
    )
    monkeypatch.setattr(
        config_ops, "load_config_meta",
        lambda pd, n: SimpleNamespace(parse_error=None),
    )
    return written


def test_propose_new_config_meta_template(tmp_path, monkeypatch):
    written = _stub_config_ops(monkeypatch, tmp_path, rendered="scaffolded: yes\n")
    p = tools_authoring._propose_new_config(
        {"project_dir": str(tmp_path), "name": "c.yaml",
         "meta_template": "tiny", "values": {"a": 1}}
    )
    assert p.after == "scaffolded: yes\n" and "scaffold:" in p.extra["starting_point"]
    assert written == {}  # preview did not write
    p.commit()
    assert written["content"] == "scaffolded: yes\n"


def test_propose_new_config_copy_from(tmp_path, monkeypatch):
    from forgather_server import paths

    written = _stub_config_ops(monkeypatch, tmp_path)
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)
    monkeypatch.setattr(config_ops, "read_raw", lambda path: "copied: yes\n")
    p = tools_authoring._propose_new_config(
        {"project_dir": str(tmp_path), "name": "c.yaml",
         "copy_from": "/abs/example/configs/base.yaml"}
    )
    assert p.after == "copied: yes\n" and "copy:" in p.extra["starting_point"]
    p.commit()
    assert written["content"] == "copied: yes\n"


def test_propose_new_config_inline_content(tmp_path, monkeypatch):
    _stub_config_ops(monkeypatch, tmp_path)
    p = tools_authoring._propose_new_config(
        {"project_dir": str(tmp_path), "name": "c.yaml", "content": "x: 1\n"}
    )
    assert p.after == "x: 1\n" and p.extra["starting_point"] == "inline content"


def test_propose_new_config_starting_points_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="at most one"):
        tools_authoring._propose_new_config(
            {"project_dir": str(tmp_path), "name": "c.yaml",
             "meta_template": "tiny", "copy_from": "/abs/x.yaml"}
        )
