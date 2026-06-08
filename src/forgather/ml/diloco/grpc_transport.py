"""Client-side gRPC bulk transport for DiLoCo (issue #154).

``GrpcBytesTransport`` implements the same ``BulkBytesTransport`` seam as
``HttpBytesTransport`` (``round_trip(op, payload) -> bytes`` + ``close``), so the
client swaps it in for the bulk legs without any other change. It is selected
only when the server advertises ``transport: "grpc"`` via ``/info``; otherwise
the client keeps the HTTP transport. Lives in its own module (lazily imported on
that negotiation) so the grpc import stays off the common path.

The op maps to a streaming RPC: the framed ``[len][header][blob]`` payload is
chunked into the request stream and the response stream is concatenated back —
the framing the client already builds rides through opaque, so the safetensors
wire-format negotiation is unchanged.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

#: Request chunk size; match the server's CHUNK_BYTES, stay under gRPC's 4 MiB.
CHUNK_BYTES = 1 << 20  # 1 MiB


def client_channel_credentials(verify_tls: bool = True):
    """Build gRPC client channel credentials from this host's TLS config, or
    ``None`` when TLS isn't provisioned (cleartext channel).

    The gRPC analogue of ``urllib_ssl_context``: loads the cluster CA bundle to
    verify the server's cert and, when this host has its own cert/key, presents
    them for mTLS. Returns ``None`` when no CA bundle exists, so the caller falls
    back to an insecure channel (the cleartext trusted-LAN posture).
    """
    import grpc

    from forgather.tls import load_config

    cfg = load_config()
    bundle = cfg.effective_bundle()
    if bundle is None:
        return None
    with open(str(bundle), "rb") as f:
        ca_pem = f.read()
    cert_pem = key_pem = None
    # Present this node's cert/key when provisioned (mirrors
    # urllib_ssl_context.load_cert_chain on the HTTP client). The gRPC bulk
    # server runs server-auth-only TLS today (require_client_auth=False), so this
    # client cert is currently unused on the wire — kept for symmetry and a
    # future require-client-auth mode.
    if cfg.is_provisioned():
        with open(str(cfg.server_cert), "rb") as f:
            cert_pem = f.read()
        with open(str(cfg.server_key), "rb") as f:
            key_pem = f.read()
    creds = grpc.ssl_channel_credentials(
        root_certificates=ca_pem,
        private_key=key_pem,
        certificate_chain=cert_pem,
    )
    if not verify_tls:
        # gRPC has no clean per-channel skip-verify (unlike the urllib client's
        # CERT_NONE escape hatch). Warn rather than silently honor it: a worker
        # relying on --no-verify-tls for a tunneled endpoint gets HTTP-works /
        # gRPC-CA-verified, so the request is not honored on the gRPC leg.
        logger.warning(
            "gRPC: verify_tls=False not supported; the bulk channel still "
            "verifies the server against the cluster CA."
        )
    return creds


class GrpcBytesTransport:
    """Bulk round-trip over a gRPC streaming channel.

    ``credentials`` is a ``grpc.ChannelCredentials`` for a TLS channel, or
    ``None`` for cleartext (the trusted-LAN posture the gRPC listener uses
    today). A failed RPC surfaces as ``ConnectionError`` — the same type the
    HTTP transport raises — so it flows into the worker's existing
    retry/reconnect handling.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        credentials=None,
        bearer: Optional[str] = None,
        timeout: float = 600,
        retry_delay: float = 1.0,
        chunk_bytes: int = CHUNK_BYTES,
    ):
        import grpc

        from forgather.ml.diloco.proto import bulk_pb2, bulk_pb2_grpc

        self._grpc = grpc
        self._pb2 = bulk_pb2
        self._endpoint = endpoint
        self._timeout = timeout
        self._retry_delay = retry_delay
        self._chunk = chunk_bytes
        # Bearer rides as call metadata, and ONLY over a secure channel — a
        # token over cleartext is theater (and the cleartext listener is open
        # anyway). Mirrors the HTTP client omitting the bearer on the bulk plane.
        if bearer and credentials is not None:
            self._metadata = [("authorization", f"bearer {bearer}")]
        else:
            self._metadata = None
        if credentials is not None:
            self._channel = grpc.secure_channel(endpoint, credentials)
        else:
            self._channel = grpc.insecure_channel(endpoint)
        self._stub = bulk_pb2_grpc.DiLoCoBulkStub(self._channel)

    def _chunks(self, payload: bytes):
        for i in range(0, len(payload), self._chunk):
            yield self._pb2.Chunk(data=payload[i : i + self._chunk])

    def _invoke(self, op, payload):
        # Import here to avoid a module-level dependency on the enum's module.
        from forgather.ml.diloco.bulk_transport import BulkOp

        md = self._metadata
        if op is BulkOp.GLOBAL_PARAMS:
            return self._stub.GlobalParams(
                self._pb2.GlobalParamsRequest(), timeout=self._timeout, metadata=md
            )
        if op is BulkOp.SUBMIT_PSEUDOGRAD:
            return self._stub.SubmitPseudograd(
                self._chunks(payload or b""), timeout=self._timeout, metadata=md
            )
        if op is BulkOp.SUBMIT_FRAGMENT:
            return self._stub.SubmitFragment(
                self._chunks(payload or b""), timeout=self._timeout, metadata=md
            )
        raise ValueError(f"unsupported bulk op: {op!r}")

    def round_trip(
        self, op, payload: Optional[bytes] = None, *, retries: int = 0
    ) -> bytes:
        import time

        grpc = self._grpc
        delay = self._retry_delay
        for attempt in range(retries + 1):
            try:
                resp = self._invoke(op, payload)
                return b"".join(chunk.data for chunk in resp)
            except grpc.RpcError as e:
                code = e.code() if hasattr(e, "code") else None
                # Retry only transient unavailability (mirrors the HTTP URLError
                # retry); other statuses (the server's mapped 4xx/5xx) are
                # terminal and surface immediately.
                transient = code == grpc.StatusCode.UNAVAILABLE
                if transient and attempt < retries:
                    logger.warning(
                        f"gRPC bulk {op.value} to {self._endpoint} unavailable "
                        f"(attempt {attempt + 1}/{retries + 1}); retrying in "
                        f"{delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                detail = e.details() if hasattr(e, "details") else str(e)
                raise ConnectionError(
                    f"gRPC bulk {op.value} to {self._endpoint} failed "
                    f"({code}): {detail}"
                ) from e

    def close(self) -> None:
        self._channel.close()
