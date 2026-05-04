"""Tests for tools/forgather_server/_atomic.py."""

import os
import stat

import pytest
from forgather_server._atomic import atomic_write_bytes, atomic_write_text


class TestAtomicWriteText:
    def test_writes_correct_content(self, tmp_path):
        target = tmp_path / "out.json"
        atomic_write_text(target, '{"hello": "world"}')
        assert target.read_text() == '{"hello": "world"}'

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        atomic_write_text(target, "hello")
        assert target.read_text() == "hello"

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("old content")
        atomic_write_text(target, "new content")
        assert target.read_text() == "new content"

    def test_no_tmp_file_left_behind(self, tmp_path):
        target = tmp_path / "data.json"
        atomic_write_text(target, "data")
        tmp = target.with_suffix(target.suffix + ".tmp")
        assert not tmp.exists()

    def test_empty_string(self, tmp_path):
        target = tmp_path / "empty.txt"
        atomic_write_text(target, "")
        assert target.read_text() == ""

    def test_unicode_content(self, tmp_path):
        target = tmp_path / "unicode.txt"
        content = "hello ☃ world"
        atomic_write_text(target, content)
        assert target.read_text() == content


class TestAtomicWriteBytes:
    def test_writes_correct_bytes(self, tmp_path):
        target = tmp_path / "data.bin"
        atomic_write_bytes(target, b"\x00\x01\x02\x03")
        assert target.read_bytes() == b"\x00\x01\x02\x03"

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "nested" / "dir" / "data.bin"
        atomic_write_bytes(target, b"abc")
        assert target.read_bytes() == b"abc"

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "data.bin"
        target.write_bytes(b"old")
        atomic_write_bytes(target, b"new")
        assert target.read_bytes() == b"new"

    def test_no_tmp_file_left_behind(self, tmp_path):
        target = tmp_path / "data.bin"
        atomic_write_bytes(target, b"payload")
        tmp = target.with_suffix(target.suffix + ".tmp")
        assert not tmp.exists()

    def test_empty_bytes(self, tmp_path):
        target = tmp_path / "empty.bin"
        atomic_write_bytes(target, b"")
        assert target.read_bytes() == b""


class TestAtomicWriteMode:
    def test_text_mode_0600(self, tmp_path):
        target = tmp_path / "secret.txt"
        atomic_write_text(target, "shh", mode=0o600)
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o600

    def test_bytes_mode_0600(self, tmp_path):
        target = tmp_path / "secret.bin"
        atomic_write_bytes(target, b"shh", mode=0o600)
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o600

    def test_text_mode_overwrite_loose(self, tmp_path):
        # Even if the destination already exists at 0644, the new content
        # should land at 0600 (rename inherits the tmp file's mode).
        target = tmp_path / "secret.txt"
        target.write_text("old")
        os.chmod(target, 0o644)
        atomic_write_text(target, "new", mode=0o600)
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o600
        assert target.read_text() == "new"
