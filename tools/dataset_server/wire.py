"""
Wire-format helpers shared between the server and `RemoteBackend`.

NDJSON-over-HTTP can carry str/int/float/list/dict/None directly via
``json.dumps``. ``bytes`` values get wrapped in a tagged dict so the
client can recover them. Anything else surfaces a ``TypeError`` from
``json.dumps`` rather than being silently lost.
"""

from __future__ import annotations

import base64
from typing import Any


def to_jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def from_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if "__bytes_b64__" in value and len(value) == 1:
            return base64.b64decode(value["__bytes_b64__"])
        return {k: from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_jsonable(v) for v in value]
    return value
