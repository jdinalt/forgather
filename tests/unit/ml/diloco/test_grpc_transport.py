"""Tests for the gRPC bulk transport (issue #154).

The gRPC listener serves the same three bulk legs as HTTP, negotiated via /info.
The server-side servicer reuses the exact HTTP handlers through an in-memory
``_CapturingHandler``; the client swaps ``GrpcBytesTransport`` in for the bulk
legs while the control plane (register/heartbeat/info) stays on HTTP.
"""

import struct
import threading
import time

import pytest
import torch

from forgather.ml.diloco.bulk_transport import BulkOp
from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.grpc_bulk import _CapturingHandler
from forgather.ml.diloco.server import DiLoCoServer
from forgather.ml.diloco.wire_serialize import serialize_state_dict

from .conftest import make_initial_checkpoint


def _make_sd(dim=8, seed=42):
    torch.manual_seed(seed)
    return {f"layer{i}.weight": torch.randn(dim, dim) for i in range(2)}


def _grpc_server(tmp_path, sd, num_workers=1, wire_format="safetensors"):
    ckpt = make_initial_checkpoint(sd, tmp_path)
    server = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=num_workers,
        port=0,
        grpc_enabled=True,
        wire_format=wire_format,
        outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=1.0),
    )
    server.start()
    time.sleep(0.2)
    return server


def _negotiated_client(server, wire_format="safetensors"):
    """Build a client the way the callback would, from the server's /info."""
    info = DiLoCoClient(f"localhost:{server.port}", timeout=10).get_info()
    return DiLoCoClient(
        f"localhost:{server.port}",
        timeout=10,
        wire_format=wire_format,
        transport=info["transport"],
        grpc_endpoint=info["grpc_endpoint"],
    )


class TestNegotiation:
    def test_grpc_server_advertises_endpoint(self, tmp_path):
        server = _grpc_server(tmp_path, _make_sd())
        try:
            info = DiLoCoClient(f"localhost:{server.port}", timeout=10).get_info()
            assert info["transport"] == "grpc"
            assert info["grpc_endpoint"] == f"127.0.0.1:{server.grpc_port}"
            # gRPC supersedes the cleartext bulk listener.
            assert server._bulk_enabled is False
        finally:
            server.stop()

    def test_default_server_advertises_http(self, tmp_path):
        sd = _make_sd()
        ckpt = make_initial_checkpoint(sd, tmp_path)
        server = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=1.0),
        )
        server.start()
        time.sleep(0.2)
        try:
            info = DiLoCoClient(f"localhost:{server.port}", timeout=10).get_info()
            assert info["transport"] == "http"
            assert info["grpc_endpoint"] is None
        finally:
            server.stop()

    def test_client_selects_grpc_transport(self, tmp_path):
        server = _grpc_server(tmp_path, _make_sd())
        try:
            client = _negotiated_client(server)
            assert type(client._transport).__name__ == "GrpcBytesTransport"
            client._transport.close()
        finally:
            server.stop()


class TestEndToEnd:
    def test_register_http_then_bulk_over_grpc(self, tmp_path):
        sd = _make_sd()
        server = _grpc_server(tmp_path, sd)
        try:
            client = _negotiated_client(server)
            # register: HTTP control plane (download leg).
            params = client.register("w0")
            for k in sd:
                assert torch.allclose(params[k], sd[k].float(), atol=1e-6), k
            # global_params: gRPC download.
            gp = client.get_global_params()
            for k in sd:
                assert torch.allclose(gp[k], sd[k].float(), atol=1e-6), k
            # submit: gRPC upload + download + barrier + outer step.
            pg = {k: torch.full_like(v, 0.1) for k, v in sd.items()}
            out = client.submit_pseudogradients("w0", pg)
            for k in sd:
                assert torch.allclose(out[k], sd[k].float() - 0.1, atol=1e-5), k
            client._transport.close()
        finally:
            server.stop()

    def test_large_payload_spans_chunks(self, tmp_path):
        # A model bigger than one 1 MiB chunk exercises the chunk/reassemble path.
        sd = _make_sd(dim=400)  # 2 x 400x400 fp32 ~ 2.5 MB
        server = _grpc_server(tmp_path, sd)
        try:
            client = _negotiated_client(server)
            client.register("w0")
            pg = {k: torch.zeros_like(v) for k, v in sd.items()}
            out = client.submit_pseudogradients("w0", pg)  # zero grad -> unchanged
            for k in sd:
                assert torch.allclose(out[k], sd[k].float(), atol=1e-5), k
            client._transport.close()
        finally:
            server.stop()

    def test_two_workers_sync_over_grpc(self, tmp_path):
        sd = _make_sd()
        server = _grpc_server(tmp_path, sd, num_workers=2)
        try:
            c0 = _negotiated_client(server)
            c1 = _negotiated_client(server)
            c0.register("w0")
            c1.register("w1")
            pg0 = {k: torch.full_like(v, 0.2) for k, v in sd.items()}
            pg1 = {k: torch.full_like(v, 0.4) for k, v in sd.items()}
            results = [None, None]

            def submit(i, c, wid, pg):
                results[i] = c.submit_pseudogradients(wid, pg)

            t0 = threading.Thread(target=submit, args=(0, c0, "w0", pg0))
            t1 = threading.Thread(target=submit, args=(1, c1, "w1", pg1))
            t0.start()
            t1.start()
            t0.join(timeout=10)
            t1.join(timeout=10)
            # mean grad 0.3 -> new = old - 0.3; both workers agree.
            for k in sd:
                assert torch.allclose(results[0][k], sd[k].float() - 0.3, atol=1e-5)
                assert torch.allclose(results[0][k], results[1][k])
            c0._transport.close()
            c1._transport.close()
        finally:
            server.stop()

    def test_error_maps_to_connection_error(self, tmp_path):
        """An unregistered worker's submit -> server 404 -> gRPC NOT_FOUND ->
        ConnectionError on the client (the type the worker's retry loop expects)."""
        server = _grpc_server(tmp_path, _make_sd())
        try:
            client = _negotiated_client(server)
            pg = {k: torch.zeros_like(v) for k, v in _make_sd().items()}
            with pytest.raises(ConnectionError):
                client.submit_pseudogradients("ghost", pg)
            client._transport.close()
        finally:
            server.stop()


class TestCapturingHandler:
    def test_drives_an_http_handler(self, tmp_path):
        """The adapter feeds a framed request to a real server handler and
        captures the framed tensor response + 200 status."""
        sd = _make_sd()
        server = _grpc_server(tmp_path, sd)
        try:
            # register with a matching-codec client (server is safetensors).
            DiLoCoClient(
                f"localhost:{server.port}", timeout=10, wire_format="safetensors"
            ).register("w0")
            pg = {k: torch.full_like(v, 0.1) for k, v in sd.items()}
            header = b'{"worker_id": "w0", "fmt": "safetensors"}'
            body = (
                struct.pack("!I", len(header))
                + header
                + serialize_state_dict(pg, "safetensors")
            )
            cap = _CapturingHandler(body)
            server._handle_submit_pseudograd(cap)
            assert cap.status == 200
            assert len(cap.response_bytes()) > 0  # framed tensor payload
        finally:
            server.stop()

    def test_error_response_captured(self, tmp_path):
        sd = _make_sd()
        server = _grpc_server(tmp_path, sd)
        try:
            # A well-framed but ghost-worker submit: deserialize succeeds (the
            # ghost check is after it), then the handler writes a 404 JSON error.
            pg = {k: torch.zeros_like(v) for k, v in sd.items()}
            header = b'{"worker_id": "ghost", "fmt": "safetensors"}'
            body = (
                struct.pack("!I", len(header))
                + header
                + serialize_state_dict(pg, "safetensors")
            )
            cap = _CapturingHandler(body)
            server._handle_submit_pseudograd(cap)
            assert cap.status == 404
            assert "not registered" in cap.error_message()
        finally:
            server.stop()
