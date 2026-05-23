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
