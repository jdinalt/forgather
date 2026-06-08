"""Server-side gRPC bulk transport for DiLoCo (issue #154).

The opt-in high-throughput path for the bulk legs. Rather than re-implement the
barrier / validation / per-worker slicing logic (the synchronization-critical
heart of the server), the gRPC servicer **reuses the exact HTTP request
handlers** via an in-memory ``_CapturingHandler``: it reassembles the streamed
request chunks into the same ``[len][header][blob]`` frame the HTTP path uses,
feeds it to the handler, and captures the framed response bytes + status. The
handler's blocking barrier works unchanged — the servicer runs in gRPC's thread
pool, exactly as the HTTP path runs per-request threads.

``grpc`` is a hard dependency; this module is imported lazily by the server only
when the gRPC listener is enabled, so a server that never enables it pays no
import cost.
"""

import io
import json
import logging
from concurrent import futures
from typing import Optional

import grpc

from forgather.ml.diloco.proto import bulk_pb2, bulk_pb2_grpc

logger = logging.getLogger(__name__)

#: Response chunk size. gRPC's default max message is 4 MiB; stream well under it.
CHUNK_BYTES = 1 << 20  # 1 MiB

# HTTP status -> gRPC status, for translating a handler's error response.
_HTTP_TO_GRPC = {
    400: grpc.StatusCode.INVALID_ARGUMENT,
    404: grpc.StatusCode.NOT_FOUND,
    409: grpc.StatusCode.FAILED_PRECONDITION,
    500: grpc.StatusCode.INTERNAL,
    504: grpc.StatusCode.DEADLINE_EXCEEDED,
}


class _CapturingHandler:
    """In-memory stand-in for ``BaseHTTPRequestHandler`` covering only the
    surface the bulk handlers touch (via ``_read_request_body`` /
    ``_send_json_response`` / ``_send_tensor_response``): a readable request
    body, a writable response buffer, and the status/header sink. Lets the gRPC
    servicer drive the unmodified HTTP handlers and read back what they wrote."""

    def __init__(self, body: bytes):
        self.rfile = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body))}
        self.wfile = io.BytesIO()
        self.status: Optional[int] = None
        self._headers: dict = {}

    def send_response(self, code: int) -> None:
        self.status = code

    def send_header(self, key: str, value: str) -> None:
        self._headers[key] = value

    def end_headers(self) -> None:
        pass

    def response_bytes(self) -> bytes:
        return self.wfile.getvalue()

    def error_message(self) -> str:
        """The handler's JSON error body, reduced to its ``error`` text."""
        try:
            return json.loads(self.response_bytes().decode("utf-8")).get(
                "error", "request failed"
            )
        except Exception:
            return "request failed"


class DiLoCoBulkServicer(bulk_pb2_grpc.DiLoCoBulkServicer):
    """Drives the server's HTTP bulk handlers from gRPC streams.

    ``authenticate`` is injected by the listener (mirrors the HTTP auth gate:
    mTLS peer cert or bearer metadata); ``None`` means the listener is running
    open (trusted LAN), matching the HTTP bulk listener's posture.
    """

    def __init__(self, server, authenticate=None, chunk_bytes: int = CHUNK_BYTES):
        self._server = server
        self._authenticate = authenticate
        self._chunk = chunk_bytes

    # --- RPCs -----------------------------------------------------------------

    def SubmitPseudograd(self, request_iterator, context):
        self._auth(context)
        body = b"".join(chunk.data for chunk in request_iterator)
        cap = _CapturingHandler(body)
        self._server._handle_submit_pseudograd(cap)
        yield from self._respond(cap, context)

    def SubmitFragment(self, request_iterator, context):
        self._auth(context)
        body = b"".join(chunk.data for chunk in request_iterator)
        cap = _CapturingHandler(body)
        self._server._handle_submit_fragment_pseudograd(cap)
        yield from self._respond(cap, context)

    def GlobalParams(self, request, context):
        self._auth(context)
        cap = _CapturingHandler(b"")  # the HTTP GET has no request body
        self._server._handle_get_global_params(cap)
        yield from self._respond(cap, context)

    # --- helpers --------------------------------------------------------------

    def _auth(self, context) -> None:
        if self._authenticate is not None:
            self._authenticate(context)  # raises/aborts on failure

    def _respond(self, cap: _CapturingHandler, context):
        """Translate the captured HTTP response: stream the framed bytes on 200,
        else abort with the mapped gRPC status + the handler's error text."""
        if cap.status == 200:
            data = cap.response_bytes()
            for i in range(0, len(data), self._chunk):
                yield bulk_pb2.Chunk(data=data[i : i + self._chunk])
            return
        code = _HTTP_TO_GRPC.get(cap.status, grpc.StatusCode.UNKNOWN)
        context.abort(code, cap.error_message())


def make_server_credentials(
    cert_file: str, key_file: str, ca_file: Optional[str]
) -> "grpc.ServerCredentials":
    """Build gRPC server TLS credentials from the control plane's cert/key.

    ``require_client_auth=False``: TLS provides encryption + server
    authentication, and the client authenticates by **bearer over TLS** (gRPC
    has no ``CERT_OPTIONAL`` equivalent — a client cert is only exposed under
    ``require_client_auth=True``, which would reject every non-cert client at the
    handshake, so the HTTP control plane's mTLS-or-bearer model doesn't translate
    to the worker-only bulk plane). ``ca_file`` is accepted for signature
    symmetry / a future require-client-auth mode but unused here.
    """
    with open(cert_file, "rb") as f:
        cert_pem = f.read()
    with open(key_file, "rb") as f:
        key_pem = f.read()
    return grpc.ssl_server_credentials([(key_pem, cert_pem)])


def make_grpc_server(
    server,
    host: str,
    *,
    authenticate=None,
    server_credentials: Optional["grpc.ServerCredentials"] = None,
    max_workers: int = 16,
):
    """Build (but do not start) a gRPC server serving the bulk service.

    Binds an ephemeral port (``:0``) — secure when ``server_credentials`` is
    given, else cleartext. Returns ``(grpc_server, port)``.
    """
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    bulk_pb2_grpc.add_DiLoCoBulkServicer_to_server(
        DiLoCoBulkServicer(server, authenticate=authenticate), grpc_server
    )
    target = f"{host}:0"
    if server_credentials is not None:
        port = grpc_server.add_secure_port(target, server_credentials)
    else:
        port = grpc_server.add_insecure_port(target)
    return grpc_server, port
