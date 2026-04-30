"""Tests for tools/forgather_server/discovery.py — pruning behavior."""

from pathlib import Path

import pytest
from forgather_server.discovery import (
    _PRUNED_DIR_NAMES,
    _iter_project_dirs,
    _iter_workspace_dirs,
)

from forgather.meta_config import PROJECT_META_NAME, WORKSPACE_CONFIG_DIR_NAME


def _plant_meta(directory: Path) -> None:
    """Create a meta.yaml in the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PROJECT_META_NAME).write_text("name: test\n")


def _plant_workspace(directory: Path) -> None:
    """Create a forgather_workspace/ subdir inside directory."""
    ws = directory / WORKSPACE_CONFIG_DIR_NAME
    ws.mkdir(parents=True, exist_ok=True)


class TestIterProjectDirs:
    def test_finds_project_in_root(self, tmp_path):
        project = tmp_path / "myproject"
        _plant_meta(project)
        found = list(_iter_project_dirs(str(tmp_path)))
        assert str(project) in found

    def test_prunes_output_models(self, tmp_path):
        """meta.yaml inside output_models/ must not be yielded."""
        bad = tmp_path / "output_models" / "checkpoint_dir"
        _plant_meta(bad)
        found = list(_iter_project_dirs(str(tmp_path)))
        assert str(bad) not in found

    def test_prunes_node_modules(self, tmp_path):
        bad = tmp_path / "node_modules" / "some_pkg"
        _plant_meta(bad)
        found = list(_iter_project_dirs(str(tmp_path)))
        assert str(bad) not in found

    def test_prunes_pycache(self, tmp_path):
        bad = tmp_path / "__pycache__" / "subdir"
        _plant_meta(bad)
        found = list(_iter_project_dirs(str(tmp_path)))
        assert str(bad) not in found

    def test_prunes_git(self, tmp_path):
        bad = tmp_path / ".git" / "refs"
        _plant_meta(bad)
        found = list(_iter_project_dirs(str(tmp_path)))
        assert str(bad) not in found

    def test_hidden_dirs_pruned(self, tmp_path):
        hidden = tmp_path / ".hidden_dir"
        _plant_meta(hidden)
        found = list(_iter_project_dirs(str(tmp_path)))
        assert str(hidden) not in found

    def test_nested_valid_project_found(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "project"
        _plant_meta(nested)
        found = list(_iter_project_dirs(str(tmp_path)))
        assert str(nested) in found

    def test_all_pruned_names_are_blocked(self, tmp_path):
        """Verify every name in _PRUNED_DIR_NAMES is actually pruned."""
        for pruned_name in _PRUNED_DIR_NAMES:
            bad = tmp_path / pruned_name / "inner_project"
            _plant_meta(bad)
        found = set(_iter_project_dirs(str(tmp_path)))
        for pruned_name in _PRUNED_DIR_NAMES:
            bad = str(tmp_path / pruned_name / "inner_project")
            assert bad not in found, f"Expected {pruned_name!r} to be pruned"


class TestIterWorkspaceDirs:
    def test_finds_workspace(self, tmp_path):
        _plant_workspace(tmp_path / "ws1")
        found = list(_iter_workspace_dirs(str(tmp_path)))
        assert str(tmp_path / "ws1") in found

    def test_prunes_output_models(self, tmp_path):
        bad = tmp_path / "output_models"
        _plant_workspace(bad)
        found = list(_iter_workspace_dirs(str(tmp_path)))
        assert str(bad) not in found

    def test_prunes_pycache(self, tmp_path):
        bad = tmp_path / "__pycache__"
        _plant_workspace(bad)
        found = list(_iter_workspace_dirs(str(tmp_path)))
        assert str(bad) not in found

    def test_hidden_dirs_pruned(self, tmp_path):
        hidden = tmp_path / ".hidden"
        _plant_workspace(hidden)
        found = list(_iter_workspace_dirs(str(tmp_path)))
        assert str(hidden) not in found
