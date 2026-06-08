"""Pluggable byte transport for the DiLoCo bulk legs (issue #154).

The three bulk operations (pseudo-gradients up, averaged weights down) share one
shape: send a framed byte payload to an endpoint and get a framed byte payload
back. ``BulkBytesTransport`` is that seam — it knows nothing about state dicts,
worker ids, or the barrier; the client owns framing/serialization and the
protocol, the transport only moves bytes.

``HttpBytesTransport`` is the default: the historical ``urllib`` round-trip,
reusing the client's URL/header/SSL resolution so behavior is byte-for-byte
identical to the prior inline path. A future ``GrpcBytesTransport`` implements the
same interface over a streaming RPC, selected by ``/info`` negotiation, without
the client knowing the transport.
"""

import enum
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable, Optional, Protocol

logger = logging.getLogger(__name__)


class BulkOp(enum.Enum):
    """A bulk operation, identified by its canonical control-plane path.

    The value doubles as the HTTP path; a non-HTTP transport maps the op to its
    own method (e.g. a gRPC stub) internally, so callers stay transport-agnostic.
    """

    SUBMIT_PSEUDOGRAD = "/submit_pseudograd"
    SUBMIT_FRAGMENT = "/submit_fragment_pseudograd"
    GLOBAL_PARAMS = "/global_params"


# ``GLOBAL_PARAMS`` is a bodyless GET (fetch current weights); the submit legs
# POST a payload. A transport maps the op to its own verb from this table.
_OP_IS_UPLOAD = {
    BulkOp.SUBMIT_PSEUDOGRAD: True,
    BulkOp.SUBMIT_FRAGMENT: True,
    BulkOp.GLOBAL_PARAMS: False,
}

#: Reverse map from canonical path to op, for callers still keyed on a path.
PATH_TO_OP = {op.value: op for op in BulkOp}


class BulkBytesTransport(Protocol):
    """Round-trips a framed byte payload for a bulk op and returns the response
    bytes. Stateless w.r.t. the protocol — framing and deserialization stay in
    the caller."""

    def round_trip(
        self, op: BulkOp, payload: Optional[bytes] = None, *, retries: int = 0
    ) -> bytes: ...

    def close(self) -> None: ...


class HttpBytesTransport:
    """Bulk round-trip over ``urllib`` (the default, always-available transport).

    Reuses the client's resolvers rather than re-deriving routing/auth/TLS: the
    bulk plane shares the control plane's URL building, bearer-omission, and
    cleartext bulk-listener offload (issue #90), and routing can change at
    runtime when the server advertises a bulk URL on ``/register``. Injecting the
    bound resolvers keeps that single-sourced and the behavior identical to the
    prior inline ``_request_tensor``.
    """

    #: Content type for an upload body; matches the historical request.
    _UPLOAD_CONTENT_TYPE = "application/octet-stream"

    def __init__(
        self,
        *,
        url_for: Callable[[str], str],
        headers_for: Callable[..., dict],
        ssl_for: Callable[[str], object],
        timeout: float,
        retry_delay: float,
        scheme_hint: Callable[[], str],
    ):
        self._url_for = url_for
        self._headers_for = headers_for
        self._ssl_for = ssl_for
        self._timeout = timeout
        self._retry_delay = retry_delay
        self._scheme_hint = scheme_hint

    def round_trip(
        self, op: BulkOp, payload: Optional[bytes] = None, *, retries: int = 0
    ) -> bytes:
        path = op.value
        method = "POST" if _OP_IS_UPLOAD[op] else "GET"
        url = self._url_for(path)
        delay = self._retry_delay

        for attempt in range(retries + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                method=method,
                headers=self._headers_for(
                    self._UPLOAD_CONTENT_TYPE if payload else None, path=path
                ),
            )
            try:
                with urllib.request.urlopen(
                    req,
                    timeout=self._timeout,
                    context=self._ssl_for(url),
                ) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                # HTTP error (4xx/5xx) - read the response body for diagnostics
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                    error_detail = json.loads(error_body).get("error", error_body)
                except Exception:
                    error_detail = str(e)
                raise ConnectionError(
                    f"Server returned HTTP {e.code} for {url}: {error_detail}"
                ) from e
            except urllib.error.URLError as e:
                if attempt < retries:
                    logger.warning(
                        f"Tensor request to {url} failed "
                        f"(attempt {attempt + 1}/{retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ConnectionError(
                        f"Failed to connect to DiLoCo server at {url}: {e}"
                        f"{self._scheme_hint()}"
                    ) from e

    def close(self) -> None:
        # urllib opens a connection per request; nothing persistent to close.
        pass
