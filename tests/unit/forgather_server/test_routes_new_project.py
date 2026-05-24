"""Tests for the meta-template plumbing added to POST /workspace/new-project.

Covers the request-validation behavior added when a project's default
config is seeded from a meta-template scaffold — mutual exclusion with
``copy_from``, missing required fields, and unknown ids. The happy path
(scaffold actually rendered into a real new project) goes through
``project_create_cmd`` which needs a workspace on disk; the unit tests
in ``test_meta_templates.py`` already cover ``meta_templates.render``
end-to-end so we don't duplicate that here.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from forgather_server import meta_templates, paths
from forgather_server.routes import projects as projects_routes


@pytest.fixture
def meta_root(tmp_path: Path, monkeypatch) -> Path:
    """A fixture catalog with one scaffold; point meta_templates at it."""
    root = tmp_path / "meta"
    (root / "datasets").mkdir(parents=True)
    (root / "datasets" / "tiny.yaml").write_text("x: $NAME\n")
    (root / "datasets" / "tiny.meta.yaml").write_text(textwrap.dedent("""
            title: "Tiny"
            target_kind: "config"
            fields:
              - name: NAME
                required: true
            """).lstrip())
    monkeypatch.setattr(meta_templates, "META_ROOT", str(root))
    return root


@pytest.fixture
def client(tmp_path: Path, monkeypatch, meta_root: Path) -> Iterator[TestClient]:
    """A TestClient with the workspace dir registered under fs-root.

    The route's ``_enforce_fs_root`` rejects paths outside the allowlist;
    point it at ``tmp_path`` so a fake workspace under there is accepted.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda p: True)

    app = FastAPI()
    app.include_router(projects_routes.router, prefix="/api")
    yield TestClient(app)


def _payload(**overrides):
    base = {
        "workspace_dir": "/will/be/overridden",
        "name": "Test Project",
        "description": "Test",
    }
    base.update(overrides)
    return base


class TestNewProjectMetaTemplate:
    def test_mutual_exclusion(self, client: TestClient, tmp_path: Path):
        # When both copy_from and meta_template are supplied, the server
        # must reject the request at the validation layer — picking one
        # winner silently would surprise the user.
        src = tmp_path / "src.yaml"
        src.write_text("x: 1\n")
        r = client.post(
            "/api/workspace/new-project",
            json=_payload(
                workspace_dir=str(tmp_path / "ws"),
                copy_from=str(src),
                meta_template="datasets/tiny",
                values={"NAME": "x"},
            ),
        )
        assert r.status_code == 400
        assert "mutually exclusive" in r.json()["detail"]

    def test_missing_required_field(self, client: TestClient, tmp_path: Path):
        r = client.post(
            "/api/workspace/new-project",
            json=_payload(
                workspace_dir=str(tmp_path / "ws"),
                meta_template="datasets/tiny",
                values={},  # NAME is required but absent
            ),
        )
        assert r.status_code == 400
        assert "NAME" in r.json()["detail"]

    def test_unknown_meta_template(self, client: TestClient, tmp_path: Path):
        r = client.post(
            "/api/workspace/new-project",
            json=_payload(
                workspace_dir=str(tmp_path / "ws"),
                meta_template="does/not/exist",
                values={},
            ),
        )
        assert r.status_code == 404
        assert "meta-template not found" in r.json()["detail"]


class TestCreateConfigSeedText:
    """Direct unit test for the create_config / _resolve_seed refactor."""

    def test_create_config_with_seed_text_writes_verbatim(self, tmp_path):
        from forgather.cli.project import create_config

        target = tmp_path / "out.yaml"
        create_config(str(target), "hello: world\n")
        assert target.read_text() == "hello: world\n"

    def test_create_config_none_writes_stub(self, tmp_path):
        from forgather.cli.project import create_config, default_config_template

        target = tmp_path / "out.yaml"
        create_config(str(target), None)
        assert target.read_text() == default_config_template

    def test_create_config_empty_string_writes_empty(self, tmp_path):
        # Empty string is distinct from None — must not fall back to the stub.
        from forgather.cli.project import create_config

        target = tmp_path / "out.yaml"
        create_config(str(target), "")
        assert target.read_text() == ""

    def test_resolve_seed_prefers_seed_text(self, tmp_path):
        from types import SimpleNamespace

        from forgather.cli.project import _resolve_seed

        src = tmp_path / "copy.yaml"
        src.write_text("from copy\n")
        # When both are set, seed_text wins — that's how the webui server
        # short-circuits the file read when it has already rendered a
        # meta-template.
        args = SimpleNamespace(seed_text="from render\n", copy_from=str(src))
        assert _resolve_seed(args) == "from render\n"

    def test_resolve_seed_falls_back_to_copy_from(self, tmp_path):
        from types import SimpleNamespace

        from forgather.cli.project import _resolve_seed

        src = tmp_path / "copy.yaml"
        src.write_text("from copy\n")
        args = SimpleNamespace(seed_text=None, copy_from=str(src))
        assert _resolve_seed(args) == "from copy\n"

    def test_resolve_seed_returns_none_when_neither(self):
        from types import SimpleNamespace

        from forgather.cli.project import _resolve_seed

        args = SimpleNamespace(seed_text=None, copy_from=None)
        assert _resolve_seed(args) is None
