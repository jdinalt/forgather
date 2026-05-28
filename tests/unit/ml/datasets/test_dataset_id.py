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


# --- Slice bounds (added when dispatch moved into the composable; the hash
# absorbs slice_start / slice_end so two slices of the same source key
# distinct work-unit queues).


def test_slice_bounds_change_id():
    base = compute_dataset_id("p", split="train")
    a = compute_dataset_id("p", split="train", slice_start=100, slice_end=200)
    b = compute_dataset_id("p", split="train", slice_start=200, slice_end=300)
    assert base != a
    assert a != b


def test_slice_bounds_none_matches_base():
    # Explicit None slice args produce the same hash as omitting them entirely.
    a = compute_dataset_id("p", split="train")
    b = compute_dataset_id("p", split="train", slice_start=None, slice_end=None)
    assert a == b


def test_open_slice_distinct_from_zero_length():
    # `None` (open at this end) hashes differently from explicit `0` — the
    # intent differs and we want it to.
    open_left = compute_dataset_id("p", split="train", slice_end=1000)
    explicit_zero = compute_dataset_id(
        "p", split="train", slice_start=0, slice_end=1000
    )
    assert open_left != explicit_zero


def test_slice_bounds_integer_coercion():
    # Bounds passed as numpy-like int or Python int produce identical ids.
    a = compute_dataset_id("p", split="train", slice_start=10, slice_end=20)
    b = compute_dataset_id(
        "p", split="train", slice_start=int(10.0), slice_end=int(20.0)
    )
    assert a == b
