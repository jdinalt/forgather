"""
Wire-format helpers shared between the server and `RemoteBackend`.

NDJSON-over-HTTP can carry str/int/float/list/dict/None directly via
``json.dumps``. ``bytes`` values get wrapped in a tagged dict so the
client can recover them. Other commonly-seen-but-not-JSON-native types
that HF datasets produce (``datetime``, ``decimal.Decimal``, numpy
scalars / arrays, ``PIL.Image.Image`` instances) are normalized to
JSON-friendly forms here so the streaming endpoint doesn't crash on
them. Anything still unhandled is wrapped in a generic
``__unserializable__`` envelope (with type name and a truncated
``repr``) rather than being silently lost or crashing the stream.
"""

from __future__ import annotations

import base64
import datetime as _dt
import decimal
from typing import Any

# numpy is optional. Many HF datasets surface numpy scalars / arrays
# (e.g. ``np.int64`` ids, ``np.ndarray`` audio buffers), but we don't
# want to add a hard dependency on numpy for the wire layer.
try:  # pragma: no cover - import guard
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - import guard
    _np = None  # type: ignore
    _HAS_NUMPY = False


# PIL is also optional. Many HF image / multimodal datasets carry
# ``PIL.Image.Image`` instances even on primarily-text rows (e.g. PNG
# thumbnails attached as metadata). Imported lazily so the wire layer
# stays usable on installs without PIL.
try:  # pragma: no cover - import guard
    from PIL import Image as _PIL_Image  # type: ignore

    _HAS_PIL = True
except Exception:  # pragma: no cover - import guard
    _PIL_Image = None  # type: ignore
    _HAS_PIL = False


# Defensive guard against pathological / cyclic structures. HF examples
# in practice never recurse this deep, so anything past this bound is
# almost certainly a bug — we fall back to ``str(value)`` rather than
# blowing the stack.
_MAX_DEPTH = 64

# Cap on the ``__repr__`` field in the generic unserializable envelope.
# Picked to be informative for diagnosing the offending column without
# bloating the NDJSON stream when the offender has a multi-KB repr
# (e.g. a giant array printed via numpy/torch reprs).
_MAX_REPR_LEN = 256


def _safe_repr(obj: Any) -> str:
    """Return a truncated ``repr(obj)`` that never raises.

    Used by the generic-fallback envelope so a single misbehaving
    column can't crash the iter stream. Mirrors the spirit of the
    existing ``_parse_error`` handling in the iter proxy: degrade
    gracefully, don't kill the explore session.
    """
    try:
        text = repr(obj)
    except Exception as exc:  # pragma: no cover - exotic __repr__
        return f"<repr unavailable: {type(exc).__name__}>"
    if len(text) > _MAX_REPR_LEN:
        return text[:_MAX_REPR_LEN] + "...<truncated>"
    return text


def to_jsonable(value: Any, _depth: int = 0) -> Any:
    if _depth >= _MAX_DEPTH:
        return str(value)

    # Fast paths for JSON-native scalars. Listed explicitly so that
    # ``bool`` doesn't fall through to the numpy branch (``bool`` is a
    # subclass of ``int`` but ``isinstance(True, _np.bool_)`` is False
    # anyway — listing here is just for clarity and speed).
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, bytearray):
        return {"__bytes_b64__": base64.b64encode(bytes(value)).decode("ascii")}

    # datetime.datetime is a subclass of datetime.date, so check it
    # first to get the full timestamp rather than just the date part.
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, _dt.time):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        return value.total_seconds()

    if isinstance(value, decimal.Decimal):
        return str(value)

    if _HAS_NUMPY:
        # numpy scalar types expose .item() which returns the closest
        # native Python type (int/float/bool/complex/str). Arrays expose
        # .tolist() which recursively converts to nested lists of
        # Python scalars. Re-run through ``to_jsonable`` so any nested
        # non-JSON-native values (rare for numpy, but possible for
        # object dtype arrays) get normalized too.
        if isinstance(value, _np.ndarray):
            return to_jsonable(value.tolist(), _depth + 1)
        if isinstance(value, _np.generic):
            return to_jsonable(value.item(), _depth + 1)

    if isinstance(value, dict):
        return {k: to_jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, _depth + 1) for v in value]

    # PIL image branch: emit a descriptive envelope rather than the
    # raw bytes. The webui's Explore tab is for inspection, not image
    # rendering — shipping multi-megapixel pixel buffers over NDJSON
    # would dwarf the actual text content and make the pretty-printed
    # JSON unreadable. A separate endpoint can serve image bytes if
    # we ever need to render thumbnails.
    if _HAS_PIL and isinstance(value, _PIL_Image.Image):
        return {
            "__pil_image__": True,
            "format": value.format,
            "mode": value.mode,
            "size": list(value.size),
        }

    # Generic catch-all envelope. Anything that reaches this point is
    # not JSON-native and not one of the types we've explicitly handled
    # above. Returning a descriptive envelope (rather than the raw
    # object) keeps ``json.dumps`` from blowing up the iter stream and
    # lets the user still see *what* showed up in the table.
    return {
        "__unserializable__": True,
        "__type__": f"{type(value).__module__}.{type(value).__name__}",
        "__repr__": _safe_repr(value),
    }


def from_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if "__bytes_b64__" in value and len(value) == 1:
            return base64.b64decode(value["__bytes_b64__"])
        return {k: from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_jsonable(v) for v in value]
    return value
