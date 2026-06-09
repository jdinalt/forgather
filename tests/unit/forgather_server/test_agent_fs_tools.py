"""Filesystem tools: stat / delete / move / copy (real ops under tmp_path).

No fs-root is configured in the test env, so is_path_in_fs_root is a no-op
(True); tmp_path is deep enough to clear the route guards' depth floor.
"""

from __future__ import annotations

import pytest

from forgather_server import config_ops
from forgather_server import paths as fs_paths
from forgather_server.agent import tools_fs
from forgather_server.agent.registry import (
    CONFIRM,
    EXTENDED,
    PROPOSE,
    READ,
    Proposal,
    ToolRegistry,
)


def _reg():
    reg = ToolRegistry()
    tools_fs.register_all(reg)
    return reg


def test_registration_risk_and_tier():
    by = {s.name: s for s in _reg().specs()}
    # File management is core (always in the array, incl. deferred mode).
    assert by["stat_path"].risk == READ and by["stat_path"].tier != EXTENDED
    for name in ("delete_path", "move_path", "copy_path", "create_file"):
        assert by[name].risk == CONFIRM and by[name].tier != EXTENDED
    # edit_file produces a diff preview, so it is PROPOSE (not CONFIRM).
    assert by["edit_file"].risk == PROPOSE and by["edit_file"].tier != EXTENDED


def test_playbook_covers_filesystem_tools():
    # The fs procedure moved to the playbook; the prompt points at it.
    from forgather_server.agent import playbook

    fs = playbook.read("filesystem")
    for token in (
        "delete_path",
        "move_path",
        "copy_path",
        "stat_path",
        "create_file",
        "edit_file",
    ):
        assert token in fs


# ---- stat ------------------------------------------------------------------


def test_stat_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    out = tools_fs._stat_path({"path": str(f)})
    assert out["exists"] and out["is_file"] and out["size_bytes"] == 5


def test_stat_dir_entry_count(tmp_path):
    (tmp_path / "x").write_text("1")
    (tmp_path / "y").write_text("2")
    out = tools_fs._stat_path({"path": str(tmp_path)})
    assert out["is_dir"] and out["entry_count"] == 2


def test_stat_missing(tmp_path):
    out = tools_fs._stat_path({"path": str(tmp_path / "nope")})
    assert out["exists"] is False


def test_stat_fs_root_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(fs_paths, "is_path_in_fs_root", lambda p: False)
    with pytest.raises(ValueError):
        tools_fs._stat_path({"path": str(tmp_path / "a.txt")})


# ---- delete ----------------------------------------------------------------


def test_delete_file_preview_then_commit(tmp_path):
    f = tmp_path / "stale.txt"
    f.write_text("junk")
    prop = tools_fs._delete_path({"path": str(f)})
    assert isinstance(prop, Proposal) and prop.extra["kind"] == "file"
    assert f.exists()  # preview did not delete
    msg = prop.commit()
    assert not f.exists() and "deleted file" in msg


def test_delete_dir_recursive(tmp_path):
    d = tmp_path / "run_dir"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "ckpt.bin").write_text("x" * 10)
    prop = tools_fs._delete_path({"path": str(d)})
    assert prop.extra["kind"] == "directory" and prop.extra["entry_count"] == 1
    assert d.exists()  # preview did not delete
    prop.commit()
    assert not d.exists()


def test_delete_missing_raises(tmp_path):
    with pytest.raises(ValueError):
        tools_fs._delete_path({"path": str(tmp_path / "ghost")})


def test_delete_fs_root_refusal(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("x")
    monkeypatch.setattr(fs_paths, "is_path_in_fs_root", lambda p: False)
    with pytest.raises(ValueError):
        tools_fs._delete_path({"path": str(f)})


# ---- move / copy -----------------------------------------------------------


def test_move_preview_then_commit(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("data")
    dest = tmp_path / "dest"
    dest.mkdir()
    prop = tools_fs._move_path({"src": str(src), "dest_dir": str(dest)})
    assert src.exists()  # preview did not move
    msg = prop.commit()
    assert not src.exists() and (dest / "src.txt").exists() and "moved to" in msg


def test_copy_then_overwrite_refused_then_auto_rename(tmp_path):
    src = tmp_path / "c.txt"
    src.write_text("data")
    dest = tmp_path / "dest"
    dest.mkdir()
    tools_fs._copy_path({"src": str(src), "dest_dir": str(dest)}).commit()
    assert (dest / "c.txt").read_text() == "data"
    # Second copy without auto_rename collides -> route 409 -> ValueError.
    with pytest.raises(ValueError):
        tools_fs._copy_path({"src": str(src), "dest_dir": str(dest)}).commit()
    # With auto_rename it lands beside the original.
    msg = tools_fs._copy_path(
        {"src": str(src), "dest_dir": str(dest), "auto_rename": True}
    ).commit()
    assert "(copy)" in msg


# ---- create_file -----------------------------------------------------------


def test_create_file_preview_then_commit(tmp_path):
    f = tmp_path / "notes.md"
    prop = tools_fs._create_file({"path": str(f)})
    assert isinstance(prop, Proposal) and prop.extra["bytes"] == 0
    assert not f.exists()  # preview did not create
    msg = prop.commit()
    assert f.exists() and f.read_text() == "" and "created empty file" in msg


def test_create_file_refuses_existing(tmp_path):
    f = tmp_path / "there.md"
    f.write_text("x")
    with pytest.raises(ValueError):
        tools_fs._create_file({"path": str(f)})


def test_create_file_refuses_missing_parent(tmp_path):
    # touch semantics: do not materialize a missing directory tree.
    f = tmp_path / "no_such_dir" / "child.md"
    with pytest.raises(ValueError):
        tools_fs._create_file({"path": str(f)})


def test_create_file_fs_root_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(fs_paths, "is_path_in_fs_root", lambda p: False)
    with pytest.raises(ValueError):
        tools_fs._create_file({"path": str(tmp_path / "a.md")})


# ---- edit_file -------------------------------------------------------------


def test_edit_file_preview_diff_then_commit(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# old\n")
    prop = tools_fs._edit_file({"path": str(f), "new_content": "# new\nbody\n"})
    assert isinstance(prop, Proposal)
    # The preview carries the diff fields that drive the webui Monaco editor.
    assert prop.path == str(f) and prop.before == "# old\n"
    assert prop.after == "# new\nbody\n"
    assert f.read_text() == "# old\n"  # preview did not write
    msg = prop.commit()
    assert f.read_text() == "# new\nbody\n" and "wrote" in msg


def test_edit_file_refuses_missing(tmp_path):
    with pytest.raises(ValueError):
        tools_fs._edit_file({"path": str(tmp_path / "ghost.md"), "new_content": "x"})


def test_edit_file_stale_mtime_refused(tmp_path):
    f = tmp_path / "race.md"
    f.write_text("v1\n")
    prop = tools_fs._edit_file({"path": str(f), "new_content": "v2\n"})
    # Someone else writes after the preview captured the baseline mtime.
    import os
    import time

    later = os.path.getmtime(f) + 10
    os.utime(f, (later, later))
    time.sleep(0)
    with pytest.raises(config_ops.StaleEditError):
        prop.commit()
    assert f.read_text() == "v1\n"  # the clobber was refused
