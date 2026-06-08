"""TLS + bearer auth for the gRPC bulk transport (issue #154).

The gRPC bulk listener follows the control-plane TLS posture: a TLS server runs
gRPC over TLS with the same cert/key (encryption + server auth), and the worker
authenticates by a bearer token over the secure channel. (gRPC TLS has no
``CERT_OPTIONAL`` equivalent, so the bulk plane uses bearer rather than the
control plane's mTLS-or-bearer.) Built on the same CA-provisioning fixtures as
``test_server_mtls.py``.
"""

from __future__ import annotations

import time

import pytest
import torch

from forgather.ml.diloco.bulk_transport import BulkOp
from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.grpc_transport import GrpcBytesTransport
from forgather.ml.diloco.server import DiLoCoServer
from forgather.tls import load_config
from forgather.tls.ca import (
    create_ca,
    install_server_cert,
    mint_server_cert,
    rebuild_bundle,
)
from forgather.tls.config import save_config
from forgather.tls.runtime import server_tls_files, stdlib_ssl_context

from .conftest import make_initial_checkpoint


@pytest.fixture
def tls_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGATHER_TLS_DIR", str(tmp_path / "tls"))
    return tmp_path / "tls"


def _provisioned_cfg(tls_root):
    cfg = load_config()
    cfg.enabled = True
    cfg.san_hostnames = ["localhost"]
    cfg.san_ips = ["127.0.0.1"]
    create_ca(cfg, common_name="Test CA")
    minted = mint_server_cert(cfg, hostnames=["localhost"], ips=["127.0.0.1"])
    install_server_cert(cfg, minted)
    rebuild_bundle(cfg)
    save_config(cfg)
    return load_config()


def _sd():
    torch.manual_seed(0)
    return {"layer.weight": torch.randn(4, 4)}


@pytest.fixture
def grpc_tls_server(tmp_path, tls_root):
    """A DiLoCo server with TLS + bearer + a gRPC bulk listener over TLS."""
    cfg = _provisioned_cfg(tls_root)
    ctx = stdlib_ssl_context()
    assert ctx is not None
    cert, key, ca = server_tls_files()
    assert cert and key and ca
    ckpt = make_initial_checkpoint(_sd(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
        auth_token="bearer-fallback",
        ssl_context=ctx,
        tls_cert_file=cert,
        tls_key_file=key,
        tls_ca_file=ca,
        grpc_enabled=True,
        wire_format="safetensors",
        outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=1.0),
    )
    s.start()
    time.sleep(0.2)
    yield s, cfg
    s.stop()


def _ca_only_credentials(cfg):
    """A verify-only client channel credential: trust the cluster CA but present
    NO client cert (so the only way in is a bearer token)."""
    import grpc

    with open(str(cfg.effective_bundle()), "rb") as f:
        ca_pem = f.read()
    return grpc.ssl_channel_credentials(root_certificates=ca_pem)


def test_grpc_listener_is_tls(grpc_tls_server):
    server, _cfg = grpc_tls_server
    # The listener bound a port and the control plane is https.
    assert server.grpc_port is not None
    info = DiLoCoClient(f"https://localhost:{server.port}", timeout=10).get_info()
    assert info["transport"] == "grpc"
    assert info["grpc_endpoint"] == f"127.0.0.1:{server.grpc_port}"


def test_bearer_sync_over_grpc(grpc_tls_server):
    """A worker authenticates by bearer over the secure gRPC channel and syncs
    end to end. The client infers TLS from the https control scheme and sends the
    bearer as call metadata (over TLS only)."""
    server, _cfg = grpc_tls_server
    addr = f"https://localhost:{server.port}"
    info = DiLoCoClient(addr, timeout=10, token="bearer-fallback").get_info()
    client = DiLoCoClient(
        addr,
        timeout=10,
        token="bearer-fallback",
        wire_format="safetensors",
        transport=info["transport"],
        grpc_endpoint=info["grpc_endpoint"],
    )
    assert type(client._transport).__name__ == "GrpcBytesTransport"
    sd = _sd()
    client.register("w0")  # HTTPS control plane (bearer)
    pg = {k: torch.full_like(v, 0.1) for k, v in sd.items()}
    out = client.submit_pseudogradients("w0", pg)  # gRPC over TLS + bearer
    for k in sd:
        assert torch.allclose(out[k], sd[k].float() - 0.1, atol=1e-5), k
    client.close()


def test_bearer_over_tls_accepted(grpc_tls_server):
    """A client with NO cert but the matching bearer is accepted over TLS."""
    server, cfg = grpc_tls_server
    t = GrpcBytesTransport(
        f"127.0.0.1:{server.grpc_port}",
        credentials=_ca_only_credentials(cfg),
        bearer="bearer-fallback",
        timeout=10,
    )
    try:
        # global_params needs no registration; a non-empty framed response
        # proves the call authenticated and returned weights.
        assert len(t.round_trip(BulkOp.GLOBAL_PARAMS)) > 0
    finally:
        t.close()


def test_no_cert_no_bearer_rejected(grpc_tls_server):
    """No client cert and no bearer over TLS -> UNAUTHENTICATED -> ConnectionError."""
    server, cfg = grpc_tls_server
    t = GrpcBytesTransport(
        f"127.0.0.1:{server.grpc_port}",
        credentials=_ca_only_credentials(cfg),
        bearer=None,
        timeout=10,
    )
    try:
        with pytest.raises(ConnectionError):
            t.round_trip(BulkOp.GLOBAL_PARAMS)
    finally:
        t.close()


def test_wrong_bearer_rejected(grpc_tls_server):
    server, cfg = grpc_tls_server
    t = GrpcBytesTransport(
        f"127.0.0.1:{server.grpc_port}",
        credentials=_ca_only_credentials(cfg),
        bearer="not-the-token",
        timeout=10,
    )
    try:
        with pytest.raises(ConnectionError):
            t.round_trip(BulkOp.GLOBAL_PARAMS)
    finally:
        t.close()


def test_authenticate_grpc_context_unit():
    """The bearer auth gate: matching bearer accepted; wrong/missing rejected;
    no token -> open."""
    from forgather.ml.diloco.auth import authenticate_grpc_context

    class _Ctx:
        def __init__(self, md=None):
            self._md = md or []

        def invocation_metadata(self):
            return self._md

    tok = "secret"
    assert authenticate_grpc_context(_Ctx([("authorization", "bearer secret")]), tok)
    # gRPC lowercases keys, but be tolerant of case in the value scheme.
    assert authenticate_grpc_context(_Ctx([("authorization", "Bearer secret")]), tok)
    assert not authenticate_grpc_context(_Ctx([("authorization", "bearer nope")]), tok)
    assert not authenticate_grpc_context(_Ctx(), tok)
    assert authenticate_grpc_context(_Ctx(), None)  # auth disabled -> open
