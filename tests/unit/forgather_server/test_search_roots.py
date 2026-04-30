"""Tests for tools/forgather_server/search_roots.py."""

import json
from pathlib import Path

import forgather_server.search_roots as search_roots
import pytest


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # search_roots imports search_roots_file from paths; patch the local binding.
    roots_file = state_dir / "search_roots.json"
    monkeypatch.setattr(search_roots, "search_roots_file", lambda: roots_file)
    yield state_dir


def _write_roots(state_dir, roots):
    """Directly write the search_roots.json to bypass the seed-on-first-boot path."""
    p = state_dir / "search_roots.json"
    p.write_text(json.dumps(roots))


class TestSeedOnFirstBoot:
    def test_first_read_seeds_file(self, isolated_state):
        """Reading search_roots when no file exists should seed defaults and persist."""
        roots_file = isolated_state / "search_roots.json"
        assert not roots_file.exists()
        roots = search_roots.list_roots()
        # File should now exist.
        assert roots_file.exists()
        # Seeded defaults should be non-empty.
        assert len(roots) > 0

    def test_seeded_roots_survive_second_read(self, isolated_state):
        # Trigger seed.
        first = search_roots.list_roots()
        # Second call should return the same data (from persisted file, not re-seeding).
        second = search_roots.list_roots()
        assert [r.path for r in first] == [r.path for r in second]


class TestListRoots:
    def test_returns_search_root_objects(self, isolated_state):
        _write_roots(isolated_state, ["/some/path"])
        roots = search_roots.list_roots()
        assert len(roots) == 1
        assert roots[0].path == "/some/path"
        # exists is just a filesystem check; don't assert its value for /some/path.
        assert isinstance(roots[0].exists, bool)

    def test_empty_file_returns_empty(self, isolated_state):
        _write_roots(isolated_state, [])
        assert search_roots.list_roots() == []


class TestAddRoot:
    def test_add_new_root(self, isolated_state, tmp_path):
        _write_roots(isolated_state, [])
        new_root = str(tmp_path / "new_root")
        sr = search_roots.add_root(new_root)
        assert sr.path == str(Path(new_root).resolve())
        paths = [r.path for r in search_roots.list_roots()]
        assert sr.path in paths

    def test_add_duplicate_is_idempotent(self, isolated_state, tmp_path):
        root = str(tmp_path / "root")
        _write_roots(isolated_state, [root])
        search_roots.add_root(root)
        paths = [r.path for r in search_roots.list_roots()]
        assert paths.count(root) == 1

    def test_tilde_expanded(self, isolated_state):
        _write_roots(isolated_state, [])
        sr = search_roots.add_root("~/some_test_dir_xyz")
        assert not sr.path.startswith("~")


class TestRemoveRoot:
    def test_remove_existing(self, isolated_state, tmp_path):
        root = str(tmp_path / "to_remove")
        _write_roots(isolated_state, [root])
        ok = search_roots.remove_root(root)
        assert ok is True
        paths = [r.path for r in search_roots.list_roots()]
        assert root not in paths

    def test_remove_nonexistent_returns_false(self, isolated_state):
        _write_roots(isolated_state, [])
        ok = search_roots.remove_root("/no/such/root")
        assert ok is False

    def test_remove_leaves_others(self, isolated_state, tmp_path):
        a = str(tmp_path / "a")
        b = str(tmp_path / "b")
        _write_roots(isolated_state, [a, b])
        search_roots.remove_root(a)
        paths = [r.path for r in search_roots.list_roots()]
        assert b in paths
        assert a not in paths
