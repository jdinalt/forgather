"""Tests for ``dataset_server.wire.to_jsonable``.

These exercise the non-trivial type-conversion branches introduced to
fix ``TypeError: Object of type datetime is not JSON serializable``
raised by ``/v1/datasets/{handle}/iter`` on HF datasets that carry
datetime-typed columns (e.g. C4's ``timestamp``).

Tests are dependency-light and do not require a running server.
"""

from __future__ import annotations

import base64
import datetime as dt
import decimal
import json

import pytest
from dataset_server.wire import from_jsonable, to_jsonable


def _roundtrip(value):
    """Encode through ``to_jsonable`` -> json -> ``from_jsonable``."""
    encoded = json.dumps(to_jsonable(value))
    return from_jsonable(json.loads(encoded))


def test_passthrough_native_types_unchanged():
    example = {
        "id": 7,
        "score": 0.5,
        "text": "hello",
        "flag": True,
        "missing": None,
        "tokens": [1, 2, 3],
        "nested": {"a": [1.5, "x"], "b": None},
    }
    out = to_jsonable(example)
    # Native JSON types must round-trip identically (this also pins the
    # wire format so we don't accidentally rewrap strings/ints).
    assert json.loads(json.dumps(out)) == example


def test_datetime_converted_to_iso_string():
    ts = dt.datetime(2024, 5, 3, 12, 34, 56)
    out = to_jsonable({"timestamp": ts})
    assert out == {"timestamp": "2024-05-03T12:34:56"}
    # And the encoded form is plain JSON.
    assert json.loads(json.dumps(out)) == {"timestamp": "2024-05-03T12:34:56"}


def test_datetime_with_timezone_preserves_offset():
    ts = dt.datetime(2024, 5, 3, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = to_jsonable(ts)
    assert out == "2024-05-03T12:00:00+00:00"


def test_date_converted_to_iso_string():
    d = dt.date(2024, 5, 3)
    assert to_jsonable(d) == "2024-05-03"


def test_time_converted_to_iso_string():
    t = dt.time(9, 30, 15)
    assert to_jsonable(t) == "09:30:15"


def test_timedelta_converted_to_total_seconds():
    td = dt.timedelta(hours=1, minutes=30)
    out = to_jsonable(td)
    assert isinstance(out, float)
    assert out == 5400.0


def test_decimal_converted_to_string():
    assert to_jsonable(decimal.Decimal("1.23")) == "1.23"


def test_bytes_round_trip_through_jsonable():
    payload = b"\x00\x01ascii\xff"
    out = to_jsonable(payload)
    # Tagged-dict shape is part of the existing wire contract.
    assert isinstance(out, dict)
    assert "__bytes_b64__" in out
    assert base64.b64decode(out["__bytes_b64__"]) == payload
    # Full round-trip recovers the original bytes.
    assert _roundtrip(payload) == payload


def test_bytearray_treated_like_bytes():
    payload = bytearray(b"abc")
    assert _roundtrip(payload) == b"abc"


def test_nested_datetime_inside_list_of_dicts():
    """The motivating bug: nested ``datetime`` inside list-of-dicts."""
    example = {
        "url": "https://example.com",
        "events": [
            {"name": "open", "at": dt.datetime(2024, 1, 1, 0, 0, 0)},
            {"name": "click", "at": dt.datetime(2024, 1, 1, 0, 0, 5)},
        ],
        "meta": {"first_seen": dt.date(2024, 1, 1)},
    }
    decoded = _roundtrip(example)
    assert decoded == {
        "url": "https://example.com",
        "events": [
            {"name": "open", "at": "2024-01-01T00:00:00"},
            {"name": "click", "at": "2024-01-01T00:00:05"},
        ],
        "meta": {"first_seen": "2024-01-01"},
    }


def test_tuple_normalized_to_list():
    out = to_jsonable((1, 2, dt.date(2024, 1, 1)))
    assert out == [1, 2, "2024-01-01"]


def test_mixed_example_serializable_via_json_dumps():
    """End-to-end smoke: an example with every handled non-native type
    must successfully ``json.dumps`` (this is the exact call path used
    by ``_stream_examples`` in routes.py)."""
    example = {
        "timestamp": dt.datetime(2024, 5, 3, 12, 34, 56),
        "date": dt.date(2024, 5, 3),
        "time": dt.time(9, 30, 15),
        "elapsed": dt.timedelta(seconds=42),
        "price": decimal.Decimal("9.99"),
        "blob": b"\x00binary\xff",
        "tags": ["a", "b"],
        "n": 3,
    }
    line = json.dumps(to_jsonable(example))
    parsed = json.loads(line)
    assert parsed["timestamp"] == "2024-05-03T12:34:56"
    assert parsed["date"] == "2024-05-03"
    assert parsed["time"] == "09:30:15"
    assert parsed["elapsed"] == 42.0
    assert parsed["price"] == "9.99"
    assert parsed["blob"] == {
        "__bytes_b64__": base64.b64encode(b"\x00binary\xff").decode("ascii")
    }


# numpy is optional in the wire layer; if it's installed we exercise the
# scalar / array branches. Skip cleanly otherwise.
_np = pytest.importorskip("numpy", reason="numpy not installed")


def test_numpy_scalars_converted_to_python_natives():
    assert to_jsonable(_np.int64(7)) == 7
    assert to_jsonable(_np.float32(0.5)) == pytest.approx(0.5)
    assert to_jsonable(_np.bool_(True)) is True


def test_numpy_array_converted_to_nested_list():
    arr = _np.arange(4, dtype=_np.int32).reshape(2, 2)
    out = to_jsonable(arr)
    assert out == [[0, 1], [2, 3]]
    # And the encoded form is plain JSON.
    assert json.loads(json.dumps(out)) == [[0, 1], [2, 3]]


def test_numpy_array_nested_inside_dict():
    example = {"ids": _np.array([1, 2, 3], dtype=_np.int64)}
    assert _roundtrip(example) == {"ids": [1, 2, 3]}
