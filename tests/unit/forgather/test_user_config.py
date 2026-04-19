"""Unit tests for forgather.user_config.

Covers both ``load_user_config`` and ``eval_search_paths`` with a mocked
``Path.home()`` so the tests do not touch the real ``~/.forgather``.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from forgather import user_config


@pytest.fixture
def fake_home(tmp_path):
    """Patch Path.home() so user_config_path() points at tmp_path."""
    with patch("forgather.user_config.Path.home", return_value=tmp_path):
        yield tmp_path


def _write_user_config(home, body: str):
    cfg_dir = home / ".forgather"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(body)


class TestLoadUserConfig:
    def test_missing_file_returns_empty_dict(self, fake_home):
        assert user_config.load_user_config() == {}

    def test_valid_yaml_is_parsed(self, fake_home):
        _write_user_config(fake_home, "eval:\n  search_paths:\n    - /a\n    - /b\n")
        cfg = user_config.load_user_config()
        assert cfg == {"eval": {"search_paths": ["/a", "/b"]}}

    def test_yaml_parse_error_returns_empty_dict(self, fake_home):
        _write_user_config(fake_home, "eval: [unterminated")
        assert user_config.load_user_config() == {}

    def test_empty_file_returns_empty_dict(self, fake_home):
        _write_user_config(fake_home, "")
        assert user_config.load_user_config() == {}


class TestEvalSearchPaths:
    def _default_path(self, forgather_dir):
        return os.path.abspath(os.path.join(forgather_dir, "examples", "evaluation"))

    def test_default_only_when_no_user_config(self, fake_home, tmp_path):
        paths = user_config.eval_search_paths(str(tmp_path))
        assert paths == [self._default_path(str(tmp_path))]

    def test_user_paths_extend_default(self, fake_home, tmp_path):
        extra_a = tmp_path / "extra_a"
        extra_b = tmp_path / "extra_b"
        _write_user_config(
            fake_home,
            f"eval:\n  search_paths:\n    - {extra_a}\n    - {extra_b}\n",
        )
        paths = user_config.eval_search_paths(str(tmp_path))
        assert paths == [
            self._default_path(str(tmp_path)),
            os.path.abspath(str(extra_a)),
            os.path.abspath(str(extra_b)),
        ]

    def test_replace_default_drops_builtin(self, fake_home, tmp_path):
        extra = tmp_path / "extra"
        _write_user_config(
            fake_home,
            f"eval:\n  replace_default: true\n  search_paths:\n    - {extra}\n",
        )
        paths = user_config.eval_search_paths(str(tmp_path))
        assert paths == [os.path.abspath(str(extra))]

    def test_scalar_search_path_is_coerced_to_list(self, fake_home, tmp_path):
        extra = tmp_path / "extra"
        _write_user_config(
            fake_home,
            f"eval:\n  search_paths: {extra}\n",
        )
        paths = user_config.eval_search_paths(str(tmp_path))
        assert paths == [
            self._default_path(str(tmp_path)),
            os.path.abspath(str(extra)),
        ]

    def test_duplicates_are_removed(self, fake_home, tmp_path):
        default_path = self._default_path(str(tmp_path))
        _write_user_config(
            fake_home,
            f"eval:\n  search_paths:\n    - {default_path}\n",
        )
        paths = user_config.eval_search_paths(str(tmp_path))
        assert paths == [default_path]

    def test_tilde_in_user_path_is_expanded(self, fake_home, tmp_path):
        # Point HOME at tmp_path; "~/foo" should resolve under tmp_path.
        with patch.dict(os.environ, {"HOME": str(tmp_path)}):
            _write_user_config(
                fake_home,
                "eval:\n  search_paths:\n    - ~/foo\n",
            )
            paths = user_config.eval_search_paths(str(tmp_path))
        assert os.path.abspath(str(tmp_path / "foo")) in paths
