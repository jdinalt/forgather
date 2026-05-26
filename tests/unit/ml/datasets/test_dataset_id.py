"""Tests for the dataset_id helper used by DiLoCo work-unit dispatch."""

from __future__ import annotations

import pytest

from forgather.ml.datasets.dataset_id import compute_dataset_id


def test_returns_16_hex_chars():
    out = compute_dataset_id("foo/bar")
    assert isinstance(out, str)
    assert len(out) == 16
    assert all(c in "0123456789abcdef" for c in out)


def test_deterministic():
    a = compute_dataset_id("foo/bar", name="en", split="train")
    b = compute_dataset_id("foo/bar", name="en", split="train")
    assert a == b


def test_arg_order_independent():
    # Same inputs in different keyword order → same id.
    a = compute_dataset_id(
        "foo/bar", name="en", split="train", data_files=["a.json", "b.json"]
    )
    b = compute_dataset_id(
        "foo/bar", data_files=["b.json", "a.json"], split="train", name="en"
    )
    assert a == b


def test_data_files_list_sorted():
    # ``data_files`` list entries are sorted internally — same id
    # regardless of input order.
    a = compute_dataset_id("p", data_files=["x", "y", "z"])
    b = compute_dataset_id("p", data_files=["z", "y", "x"])
    c = compute_dataset_id("p", data_files=["y", "z", "x"])
    assert a == b == c


def test_data_files_dict_sorted():
    a = compute_dataset_id("p", data_files={"train": "t.json", "val": "v.json"})
    b = compute_dataset_id("p", data_files={"val": "v.json", "train": "t.json"})
    assert a == b


def test_different_path_different_id():
    a = compute_dataset_id("foo/bar")
    b = compute_dataset_id("foo/baz")
    assert a != b


def test_different_name_different_id():
    a = compute_dataset_id("p", name="en")
    b = compute_dataset_id("p", name="de")
    assert a != b


def test_different_split_different_id():
    a = compute_dataset_id("p", split="train")
    b = compute_dataset_id("p", split="validation")
    assert a != b


def test_different_revision_different_id():
    a = compute_dataset_id("p", revision="v1")
    b = compute_dataset_id("p", revision="v2")
    assert a != b


def test_whitespace_normalized_in_str_fields():
    a = compute_dataset_id("foo/bar", name="  en  ", split="train")
    b = compute_dataset_id("foo/bar", name="en", split="  train  ")
    c = compute_dataset_id("foo/bar", name="en", split="train")
    assert a == b == c


def test_empty_string_treated_as_none():
    # After strip(), empty string → None → same as omitting the field.
    a = compute_dataset_id("p", name="")
    b = compute_dataset_id("p")
    assert a == b


def test_none_vs_missing_equivalent():
    a = compute_dataset_id("p", name=None, split=None)
    b = compute_dataset_id("p")
    assert a == b


def test_path_required():
    with pytest.raises(ValueError, match="path is required"):
        compute_dataset_id("")
    with pytest.raises(ValueError, match="path is required"):
        compute_dataset_id("   ")  # whitespace-only


def test_path_whitespace_stripped():
    a = compute_dataset_id("  foo/bar  ")
    b = compute_dataset_id("foo/bar")
    assert a == b
