"""Tests for the _reject_symlink_in_chain helper in routes/fs.py."""

from pathlib import Path

import pytest
from fastapi import HTTPException
from forgather_server.routes.fs import _reject_symlink_in_chain


class TestRejectSymlinkInChain:
    def test_clean_path_does_not_raise(self, tmp_path):
        real_dir = tmp_path / "a" / "b" / "c"
        real_dir.mkdir(parents=True)
        # Should complete without raising.
        _reject_symlink_in_chain(str(real_dir))

    def test_nonexistent_path_does_not_raise(self, tmp_path):
        # A path that doesn't exist has no symlink components — just not there.
        _reject_symlink_in_chain(str(tmp_path / "no" / "such" / "path"))

    def test_symlink_leaf_raises(self, tmp_path):
        real = tmp_path / "real_dir"
        real.mkdir()
        link = tmp_path / "link_to_real"
        link.symlink_to(real)
        with pytest.raises(HTTPException) as exc_info:
            _reject_symlink_in_chain(str(link))
        assert exc_info.value.status_code == 400
        assert "symlink" in exc_info.value.detail.lower()

    def test_symlink_in_ancestor_raises(self, tmp_path):
        real = tmp_path / "real_dir"
        real.mkdir()
        link = tmp_path / "linked"
        link.symlink_to(real)
        # Path that descends through the symlink.
        deep = link / "sub" / "target"
        with pytest.raises(HTTPException) as exc_info:
            _reject_symlink_in_chain(str(deep))
        assert exc_info.value.status_code == 400

    def test_home_dir_not_a_symlink(self):
        """Home directory is a real directory on a normal system."""
        home = str(Path.home())
        # If home itself is a real dir this should not raise.
        # (On some CI systems ~ may be a symlink; skip rather than fail.)
        if Path(home).is_symlink():
            pytest.skip("Home directory is a symlink on this system")
        _reject_symlink_in_chain(home)
