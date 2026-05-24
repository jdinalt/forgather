"""Tests for tools/forgather_server/meta_templates.py.

Discovery walks a real on-disk tree of meta-templates, so each test
builds its own fixture tree under tmp_path and points the module at it
via the ``meta_root`` argument — keeps tests hermetic and unaffected by
whatever scaffolds happen to be checked in under templatelib/meta/.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from forgather_server import meta_templates as mt

# ---------------------------------------------------------------------------
# Fixture builders


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip())


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small but representative meta-template tree."""
    root = tmp_path / "meta"

    # datasets/_category.yaml + huggingface/_category.yaml
    _write(
        root / "datasets" / "_category.yaml",
        """
        title: "Datasets"
        description: "Configs for loading data."
        """,
    )
    _write(
        root / "datasets" / "huggingface" / "_category.yaml",
        """
        title: "Hugging Face Hub"
        """,
    )

    # A leaf with required + defaulted + optional fields.
    _write(
        root / "datasets" / "huggingface" / "with_config.yaml",
        """
        -- extends "datasets/load_packed_dataset.yaml"
        path: "$DATASET_ID"
        name: "$DATASET_CONFIG"
        split: "$VALIDATION_SPLIT"
        note: "$NOTE"
        """,
    )
    _write(
        root / "datasets" / "huggingface" / "with_config.meta.yaml",
        """
        title: "HF with config"
        description: "C4-style."
        target_kind: "config"
        fields:
          - name: DATASET_ID
            label: "Dataset id"
            required: true
          - name: DATASET_CONFIG
            required: true
          - name: VALIDATION_SPLIT
            default: "validation[:1000]"
          - name: NOTE
            description: "Optional note."
        """,
    )

    # A sibling meta-template directly under datasets/ (templates and
    # subcategories coexisting at the same level).
    _write(
        root / "datasets" / "raw.yaml",
        """
        path: "$PATH"
        """,
    )
    _write(
        root / "datasets" / "raw.meta.yaml",
        """
        title: "Raw"
        fields:
          - name: PATH
            required: true
        """,
    )

    return root


# ---------------------------------------------------------------------------
# Discovery


def test_discover_returns_top_level_categories(tree: Path) -> None:
    cats = mt.discover(str(tree))
    assert [c.name for c in cats] == ["datasets"]
    datasets = cats[0]
    assert datasets.title == "Datasets"
    assert datasets.description == "Configs for loading data."


def test_discover_includes_templates_and_children(tree: Path) -> None:
    [datasets] = mt.discover(str(tree))
    # Both a leaf and a sub-category live under datasets/.
    template_ids = [t.id for t in datasets.templates]
    child_names = [c.name for c in datasets.children]
    assert "datasets/raw" in template_ids
    assert "huggingface" in child_names


def test_category_falls_back_to_title_case(tmp_path: Path) -> None:
    # No _category.yaml -> title is title-cased directory name.
    _write(tmp_path / "meta" / "my_group" / "x.yaml", "x: $X\n")
    _write(
        tmp_path / "meta" / "my_group" / "x.meta.yaml",
        """
        title: "X"
        fields:
          - name: X
            required: true
        """,
    )
    [grp] = mt.discover(str(tmp_path / "meta"))
    assert grp.name == "my_group"
    assert grp.title == "My Group"


def test_manifest_without_body_is_skipped(tmp_path: Path) -> None:
    # A stray .meta.yaml with no matching body file should be ignored,
    # not crash discovery.
    _write(
        tmp_path / "meta" / "datasets" / "ghost.meta.yaml",
        """
        title: "Ghost"
        fields: []
        """,
    )
    # Plus one real template so the category isn't pruned for being empty.
    _write(tmp_path / "meta" / "datasets" / "real.yaml", "x: $X\n")
    _write(
        tmp_path / "meta" / "datasets" / "real.meta.yaml",
        """
        title: "Real"
        fields:
          - name: X
            required: true
        """,
    )
    [datasets] = mt.discover(str(tmp_path / "meta"))
    template_ids = {t.id for t in datasets.templates}
    assert template_ids == {"datasets/real"}


def test_empty_categories_are_pruned(tmp_path: Path) -> None:
    # A directory with no templates and no populated subdirs shouldn't
    # appear as a dead branch in the picker.
    (tmp_path / "meta" / "datasets" / "empty").mkdir(parents=True)
    _write(tmp_path / "meta" / "datasets" / "real.yaml", "x: $X\n")
    _write(
        tmp_path / "meta" / "datasets" / "real.meta.yaml",
        """
        title: "Real"
        fields:
          - name: X
            required: true
        """,
    )
    [datasets] = mt.discover(str(tmp_path / "meta"))
    assert datasets.children == []


def test_missing_meta_root_returns_empty(tmp_path: Path) -> None:
    assert mt.discover(str(tmp_path / "does_not_exist")) == []


# ---------------------------------------------------------------------------
# Rendering


def _values():
    return {"DATASET_ID": "allenai/c4", "DATASET_CONFIG": "en"}


def test_render_substitutes_values(tree: Path) -> None:
    text = mt.render("datasets/huggingface/with_config", _values(), meta_root=str(tree))
    assert 'path: "allenai/c4"' in text
    assert 'name: "en"' in text


def test_render_uses_default_when_value_missing(tree: Path) -> None:
    text = mt.render("datasets/huggingface/with_config", _values(), meta_root=str(tree))
    assert 'split: "validation[:1000]"' in text


def test_render_substitutes_empty_for_optional_with_no_default(tree: Path) -> None:
    text = mt.render("datasets/huggingface/with_config", _values(), meta_root=str(tree))
    assert 'note: ""' in text


def test_render_value_overrides_default(tree: Path) -> None:
    values = {**_values(), "VALIDATION_SPLIT": "validation[:50]"}
    text = mt.render("datasets/huggingface/with_config", values, meta_root=str(tree))
    assert 'split: "validation[:50]"' in text


def test_render_empty_string_value_falls_back_to_default(tree: Path) -> None:
    # The UI may submit empty strings for fields the user left blank;
    # the server should treat that as "use the default" rather than as
    # an explicit empty value, otherwise required defaults are useless.
    values = {**_values(), "VALIDATION_SPLIT": ""}
    text = mt.render("datasets/huggingface/with_config", values, meta_root=str(tree))
    assert 'split: "validation[:1000]"' in text


def test_render_missing_required_raises(tree: Path) -> None:
    with pytest.raises(mt.MissingFieldsError) as ei:
        mt.render(
            "datasets/huggingface/with_config",
            {"DATASET_ID": "allenai/c4"},
            meta_root=str(tree),
        )
    assert "DATASET_CONFIG" in ei.value.missing


def test_render_ignores_unknown_keys(tree: Path) -> None:
    values = {**_values(), "NOT_A_FIELD": "ignored"}
    text = mt.render("datasets/huggingface/with_config", values, meta_root=str(tree))
    # No exception, and the unknown key doesn't appear anywhere.
    assert "NOT_A_FIELD" not in text
    assert "ignored" not in text


def test_render_unknown_meta_id_raises(tree: Path) -> None:
    with pytest.raises(KeyError):
        mt.render("does/not/exist", {}, meta_root=str(tree))


def test_field_picker_roundtrips(tmp_path: Path) -> None:
    # ``picker:`` is opt-in metadata that drives the webui's picker
    # selection. Discovery should pass it through verbatim; missing
    # ``picker:`` should default to the empty string so unknown picker
    # kinds (or no picker at all) render as plain inputs.
    _write(
        tmp_path / "meta" / "datasets" / "tpl.yaml",
        "id: $ID\nname: $NAME\n",
    )
    _write(
        tmp_path / "meta" / "datasets" / "tpl.meta.yaml",
        """
        title: "T"
        fields:
          - name: ID
            picker: "dataset"
            required: true
          - name: NAME
            required: true
        """,
    )
    [datasets] = mt.discover(str(tmp_path / "meta"))
    [tpl] = datasets.templates
    fields_by_name = {f.name: f for f in tpl.fields}
    assert fields_by_name["ID"].picker == "dataset"
    assert fields_by_name["NAME"].picker == ""


class TestConfiguredRoots:
    """Multi-root scan + first-wins merge.

    These tests mutate module-level state via ``configure_roots`` so each
    one resets to defaults afterwards to avoid leaking into the rest of
    the suite. Each fixture tree is created under tmp_path; the
    ``META_ROOT`` patch makes "the default root" point at a fresh fixture
    too, so the "defaults included" case is exercised hermetically.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self):
        # Reset to the production defaults after every test in this class.
        yield
        mt.configure_roots()

    def _seed(self, root: Path, rel: str, body: str, manifest: str) -> None:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
        meta_rel = rel.removesuffix(".yaml") + ".meta.yaml"
        _write(root / meta_rel, manifest)

    def test_extra_root_adds_new_scaffold(self, tmp_path: Path, monkeypatch) -> None:
        # Default root has one scaffold; user root adds a second under
        # the same category. After merge, the category lists both.
        default_root = tmp_path / "default"
        user_root = tmp_path / "user"
        self._seed(
            default_root,
            "datasets/builtin.yaml",
            "id: $X\n",
            'title: "Builtin"\nfields:\n  - name: X\n    required: true\n',
        )
        self._seed(
            user_root,
            "datasets/mine.yaml",
            "id: $X\n",
            'title: "Mine"\nfields:\n  - name: X\n    required: true\n',
        )
        monkeypatch.setattr(mt, "META_ROOT", str(default_root))
        mt.configure_roots([str(user_root)])

        [datasets] = mt.discover()
        ids = sorted(t.id for t in datasets.templates)
        assert ids == ["datasets/builtin", "datasets/mine"]

    def test_extra_root_overrides_same_id(self, tmp_path: Path, monkeypatch) -> None:
        # Both roots have a scaffold at datasets/x. User wins.
        default_root = tmp_path / "default"
        user_root = tmp_path / "user"
        self._seed(
            default_root,
            "datasets/x.yaml",
            "from: default\n",
            'title: "Default version"\nfields: []\n',
        )
        self._seed(
            user_root,
            "datasets/x.yaml",
            "from: user\n",
            'title: "User version"\nfields: []\n',
        )
        monkeypatch.setattr(mt, "META_ROOT", str(default_root))
        mt.configure_roots([str(user_root)])

        # discover() merge surfaces the user title
        [datasets] = mt.discover()
        [tpl] = datasets.templates
        assert tpl.title == "User version"

        # get() also lands on the user version, so render() reads its body
        rendered = mt.render("datasets/x", {})
        assert rendered == "from: user\n"

    def test_disable_default_drops_bundled(self, tmp_path: Path, monkeypatch) -> None:
        # With --no-default-meta-templates, the bundled root is invisible
        # even if it has scaffolds in it.
        default_root = tmp_path / "default"
        user_root = tmp_path / "user"
        self._seed(
            default_root,
            "datasets/builtin.yaml",
            "id: $X\n",
            'title: "Builtin"\nfields:\n  - name: X\n    required: true\n',
        )
        self._seed(
            user_root,
            "datasets/mine.yaml",
            "id: $X\n",
            'title: "Mine"\nfields:\n  - name: X\n    required: true\n',
        )
        monkeypatch.setattr(mt, "META_ROOT", str(default_root))
        mt.configure_roots([str(user_root)], disable_default=True)

        cats = mt.discover()
        ids = [t.id for c in cats for t in c.templates]
        assert ids == ["datasets/mine"]
        with pytest.raises(KeyError):
            mt.get("datasets/builtin")

    def test_categories_merge_recursively(self, tmp_path: Path, monkeypatch) -> None:
        # Same category name in both roots → children + templates merge.
        # _category.yaml from the *first* root wins on display label.
        default_root = tmp_path / "default"
        user_root = tmp_path / "user"
        _write(
            default_root / "datasets" / "_category.yaml",
            'title: "Default Datasets"\n',
        )
        self._seed(
            default_root,
            "datasets/builtin.yaml",
            "id: $X\n",
            'title: "Builtin"\nfields:\n  - name: X\n    required: true\n',
        )
        _write(
            user_root / "datasets" / "_category.yaml",
            'title: "User Datasets"\n',
        )
        self._seed(
            user_root,
            "datasets/mine.yaml",
            "id: $X\n",
            'title: "Mine"\nfields:\n  - name: X\n    required: true\n',
        )
        monkeypatch.setattr(mt, "META_ROOT", str(default_root))
        mt.configure_roots([str(user_root)])

        [datasets] = mt.discover()
        # First-wins on label.
        assert datasets.title == "User Datasets"
        # Templates from both roots present.
        assert {t.id for t in datasets.templates} == {
            "datasets/builtin",
            "datasets/mine",
        }

    def test_nonexistent_extra_silently_dropped(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # A typo'd --meta-template-dir shouldn't kill discovery.
        default_root = tmp_path / "default"
        self._seed(
            default_root,
            "datasets/x.yaml",
            "id: $X\n",
            'title: "X"\nfields:\n  - name: X\n    required: true\n',
        )
        monkeypatch.setattr(mt, "META_ROOT", str(default_root))
        mt.configure_roots([str(tmp_path / "does/not/exist")])

        [datasets] = mt.discover()
        assert [t.id for t in datasets.templates] == ["datasets/x"]


def test_render_undeclared_marker_in_body_raises(tmp_path: Path) -> None:
    # If the meta-template body references $UNDECLARED but the manifest
    # doesn't declare it, that's a bug in the meta-template — render
    # should refuse, not silently leave the marker in the output.
    _write(tmp_path / "meta" / "bad" / "tpl.yaml", "x: $UNDECLARED\n")
    _write(
        tmp_path / "meta" / "bad" / "tpl.meta.yaml",
        """
        title: "Bad"
        fields: []
        """,
    )
    with pytest.raises(KeyError):
        mt.render("bad/tpl", {}, meta_root=str(tmp_path / "meta"))
