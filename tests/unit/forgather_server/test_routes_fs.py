"""Tests for routes/fs.py helpers and endpoints."""

import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from forgather_server.routes.fs import _reject_symlink_in_chain, router


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


class TestDownloadFile:
    """Tests for the GET /fs/download endpoint."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """A TestClient with the fs-root gate satisfied.

        Stubs ``fs_roots_active`` True (so the endpoint isn't refused
        outright) and ``is_path_in_fs_root`` True (so any tmp_path
        passes the per-request allowlist check).
        """
        import forgather_server.paths as fp
        from fastapi import FastAPI

        monkeypatch.setattr(fp, "fs_roots_active", lambda: True)
        monkeypatch.setattr(fp, "is_path_in_fs_root", lambda p: True)
        app = FastAPI()
        app.include_router(router, prefix="/api")
        yield TestClient(app)

    def test_download_text_file(self, client, tmp_path):
        text_file = tmp_path / "hello.txt"
        text_file.write_text("hello world", encoding="utf-8")
        r = client.get(f"/api/fs/download?path={text_file}")
        assert r.status_code == 200
        assert r.text == "hello world"
        assert "attachment" in r.headers.get("content-disposition", "")

    def test_download_binary_file(self, client, tmp_path):
        bin_file = tmp_path / "image.png"
        bin_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        r = client.get(f"/api/fs/download?path={bin_file}")
        assert r.status_code == 200
        assert r.content == b"\x89PNG\r\n\x1a\n"

    def test_download_nonexistent_raises_404(self, client, tmp_path):
        r = client.get("/api/fs/download?path=/tmp/forgather_test_nonexistent")
        assert r.status_code == 404

    def test_download_directory_raises_400(self, client, tmp_path):
        r = client.get(f"/api/fs/download?path={tmp_path}")
        assert r.status_code == 400

    def test_download_symlink_raises_400(self, client, tmp_path):
        real = tmp_path / "real_file.txt"
        real.write_text("content")
        link = tmp_path / "link_to_real"
        link.symlink_to(real)
        r = client.get(f"/api/fs/download?path={link}")
        assert r.status_code == 400

    def test_download_disabled_when_no_fs_root_configured(self, tmp_path, monkeypatch):
        """With no fs-root allowlist, /fs/download must refuse outright.

        Otherwise the default (no-allowlist) prototype config silently
        becomes an arbitrary-file-read endpoint.
        """
        import forgather_server.paths as fp
        from fastapi import FastAPI

        f = tmp_path / "readable.txt"
        f.write_text("anything")

        monkeypatch.setattr(fp, "fs_roots_active", lambda: False)
        app = FastAPI()
        app.include_router(router, prefix="/api")
        client = TestClient(app)

        r = client.get(f"/api/fs/download?path={f}")
        assert r.status_code == 403

    def test_download_outside_fs_root_raises_403(self, tmp_path, monkeypatch):
        """Path-allowlist gate must reject reads outside the configured roots."""
        import forgather_server.paths as fp
        from fastapi import FastAPI

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")

        # Allowlist IS active (so the no-allowlist gate doesn't fire
        # first), but ``outside`` is not under ``allowed`` so the
        # per-request check must reject it.
        monkeypatch.setattr(fp, "fs_roots_active", lambda: True)
        monkeypatch.setattr(
            fp,
            "is_path_in_fs_root",
            lambda p: str(Path(p).resolve()).startswith(str(allowed.resolve())),
        )
        app = FastAPI()
        app.include_router(router, prefix="/api")
        client = TestClient(app)

        r = client.get(f"/api/fs/download?path={outside}")
        assert r.status_code == 403
