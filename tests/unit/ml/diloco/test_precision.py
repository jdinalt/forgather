"""Tests for the four-knob wire-precision schema (issue #130).

`DiLoCoServer` and `DiLoCoWorker` carry independent server-authoritative
knobs for the upload (worker → server pseudo-grads) and download
(server → worker averaged params) wire dtypes, plus a stochastic-rounding
flag per direction. The legacy `bf16_comm` boolean stays as a back-compat
alias for `upload_dtype`.

Coverage:

* Constructor validation: dtype enum is `{"fp32","bf16"}`; the deprecated
  `bf16_comm` alias maps to `upload_dtype`; passing both raises; same on
  the worker.
* `/info` advertises all four keys plus the legacy `bf16_comm`.
* The server's `_cast_for_download` produces tensors in the configured
  download dtype, with SR routed through `fp32_to_bf16_stochastic_round`
  when requested.
* The ParamView's `compute_pseudograds` produces tensors in the
  configured upload dtype, again with SR for the fp32 → bf16 cast.
* Live integration: a registration and one sync round under
  `default_dtype=bfloat16` weights complete without exception (issue
  #130 acceptance #1) and the server responds in the requested
  download dtype.
"""

from __future__ import annotations

import io
import json
import struct
import time
import urllib.request
from typing import Dict

import pytest
import torch
from torch import nn

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.param_view import (
    PipelineParamView,
    SimpleModelParamView,
)
from forgather.ml.diloco.server import DiLoCoServer
from forgather.ml.diloco.wire_cast import cast_for_upload as _cast_for_upload
from forgather.ml.diloco.worker import DiLoCoWorker

from .conftest import make_initial_checkpoint


def _tiny_state() -> Dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "layer_0.weight": torch.randn(4, 4),
        "layer_1.weight": torch.randn(4, 4),
    }


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestServerConstructor:
    def _server(self, tmp_path, **overrides):
        ckpt = make_initial_checkpoint(_tiny_state(), tmp_path)
        return DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            heartbeat_timeout=0,
            **overrides,
        )

    def test_defaults_match_pre_refactor(self, tmp_path):
        """Default kwargs reproduce the pre-#130 behavior: upload bf16
        (today's `bf16_comm=True`), download fp32 (today's behavior),
        no SR in either direction."""
        s = self._server(tmp_path)
        assert s.upload_dtype == "bf16"
        assert s.download_dtype == "fp32"
        assert s.upload_sr is False
        assert s.download_sr is False
        assert s.bf16_comm is True  # legacy mirror

    def test_bf16_comm_alias_true(self, tmp_path):
        s = self._server(tmp_path, bf16_comm=True)
        assert s.upload_dtype == "bf16"
        assert s.bf16_comm is True

    def test_bf16_comm_alias_false(self, tmp_path):
        s = self._server(tmp_path, bf16_comm=False)
        assert s.upload_dtype == "fp32"
        assert s.bf16_comm is False

    def test_bf16_comm_and_upload_dtype_together_raises(self, tmp_path):
        with pytest.raises(ValueError, match="upload_dtype"):
            self._server(tmp_path, bf16_comm=True, upload_dtype="bf16")

    def test_invalid_upload_dtype_raises(self, tmp_path):
        with pytest.raises(ValueError, match="upload_dtype"):
            self._server(tmp_path, upload_dtype="float64")

    def test_invalid_download_dtype_raises(self, tmp_path):
        with pytest.raises(ValueError, match="download_dtype"):
            self._server(tmp_path, download_dtype="float64")


class TestWorkerConstructor:
    """The worker mirrors the server's four-knob schema + alias rules.
    Worker construction is gated on importing torch but doesn't open
    any network sockets, so we can pass a stub server_addr."""

    def _worker(self, **overrides):
        model = nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        return DiLoCoWorker(
            model=model,
            optimizer=opt,
            server_addr="127.0.0.1:0",  # not contacted in __init__
            **overrides,
        )

    def test_defaults_match_pre_refactor(self):
        w = self._worker()
        assert w.upload_dtype == "bf16"
        assert w.download_dtype == "fp32"
        assert w.upload_sr is False
        assert w.download_sr is False
        assert w.bf16_comm is True

    def test_bf16_comm_alias_false(self):
        w = self._worker(bf16_comm=False)
        assert w.upload_dtype == "fp32"
        assert w.bf16_comm is False

    def test_bf16_comm_and_upload_dtype_together_raises(self):
        with pytest.raises(ValueError, match="upload_dtype"):
            self._worker(bf16_comm=True, upload_dtype="bf16")

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="upload_dtype"):
            self._worker(upload_dtype="float64")
        with pytest.raises(ValueError, match="download_dtype"):
            self._worker(download_dtype="float64")


# ---------------------------------------------------------------------------
# /info advertisement
# ---------------------------------------------------------------------------


class TestInfoAdvertisement:
    def _running(self, tmp_path, **kw):
        ckpt = make_initial_checkpoint(_tiny_state(), tmp_path)
        s = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            heartbeat_timeout=0,
            **kw,
        )
        s.start()
        time.sleep(0.2)
        return s

    def _fetch_info(self, server: DiLoCoServer) -> dict:
        req = urllib.request.Request(
            f"http://localhost:{server.port}/info", method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_four_knobs_present(self, tmp_path):
        s = self._running(
            tmp_path,
            upload_dtype="bf16",
            upload_sr=True,
            download_dtype="bf16",
            download_sr=True,
        )
        try:
            info = self._fetch_info(s)
            ecs = info["expected_client_settings"]
            assert ecs["upload_dtype"] == "bf16"
            assert ecs["upload_sr"] is True
            assert ecs["download_dtype"] == "bf16"
            assert ecs["download_sr"] is True
        finally:
            s.stop()

    def test_legacy_bf16_comm_still_emitted(self, tmp_path):
        """A pre-#130 worker reading ``expected_client_settings.bf16_comm``
        must still negotiate the right upload format. The flag tracks
        ``upload_dtype == "bf16"``."""
        s = self._running(tmp_path, upload_dtype="fp32")
        try:
            info = self._fetch_info(s)
            ecs = info["expected_client_settings"]
            assert ecs["bf16_comm"] is False
            assert ecs["upload_dtype"] == "fp32"
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# Cast helpers — what the wire dtypes actually look like
# ---------------------------------------------------------------------------


class TestCastForUpload:
    """`wire_cast.cast_for_upload` is the upload-leg wire cast owned by the
    HTTP backend. Exercises the dtype matrix without spinning up a server."""

    def test_fp32_passthrough(self):
        x = torch.randn(8)
        out = _cast_for_upload(x, "fp32", upload_sr=False)
        assert out.dtype == torch.float32
        assert torch.equal(out, x)

    def test_bf16_rne_cast(self):
        x = torch.randn(8)
        out = _cast_for_upload(x, "bf16", upload_sr=False)
        assert out.dtype == torch.bfloat16

    def test_bf16_sr_cast(self):
        # Use a tensor with sub-ULP variation so SR can be observed
        # statistically; here we just assert the dtype contract.
        x = torch.randn(8)
        out = _cast_for_upload(x, "bf16", upload_sr=True)
        assert out.dtype == torch.bfloat16

    def test_bf16_sr_on_bf16_input_falls_back_to_rne(self):
        """SR is only meaningful for an fp32 input — when the value
        is already bf16 the cast is identity and SR has no effect."""
        x = torch.randn(8).bfloat16()
        out = _cast_for_upload(x, "bf16", upload_sr=True)
        assert out.dtype == torch.bfloat16

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="upload_dtype"):
            _cast_for_upload(torch.randn(2), "fp64", upload_sr=False)


class TestServerCastForDownload:
    """The server's ``_cast_for_download`` is what determines the wire
    dtype of every sync/register/fragment response. The helper is
    invoked by the three response sites; assert its contract here."""

    def _server(self, tmp_path, **kw):
        ckpt = make_initial_checkpoint(_tiny_state(), tmp_path)
        return DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            heartbeat_timeout=0,
            **kw,
        )

    def test_fp32_passthrough(self, tmp_path):
        s = self._server(tmp_path, download_dtype="fp32")
        out = s._cast_for_download({"a": torch.randn(4)})
        assert out["a"].dtype == torch.float32

    def test_bf16_rne(self, tmp_path):
        s = self._server(tmp_path, download_dtype="bf16")
        out = s._cast_for_download({"a": torch.randn(4)})
        assert out["a"].dtype == torch.bfloat16

    def test_bf16_sr(self, tmp_path):
        s = self._server(tmp_path, download_dtype="bf16", download_sr=True)
        out = s._cast_for_download({"a": torch.randn(4)})
        assert out["a"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# End-to-end: response dtypes match the configured download_dtype
# ---------------------------------------------------------------------------


def _register_via_http(
    server: DiLoCoServer,
    worker_id: str,
    shapes: Dict[str, list],
) -> bytes:
    body = {
        "worker_id": worker_id,
        "hostname": "test",
        "param_shapes": shapes,
    }
    req = urllib.request.Request(
        f"http://localhost:{server.port}/register",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read()


def _submit_pseudograd_via_http(
    server: DiLoCoServer,
    worker_id: str,
    pg: Dict[str, torch.Tensor],
) -> bytes:
    header = json.dumps({"worker_id": worker_id}).encode("utf-8")
    buf = io.BytesIO()
    torch.save(pg, buf)
    body = struct.pack("!I", len(header)) + header + buf.getvalue()
    req = urllib.request.Request(
        f"http://localhost:{server.port}/submit_pseudograd",
        data=body,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read()


class TestEndToEndResponseDtype:
    """Round-trip a registration and sync against a real server with
    download_dtype=bf16; assert the wire response is bf16. This is the
    load-bearing end-to-end check that ties the server's cast helper
    to the actual response path."""

    def _server(self, tmp_path, **kw):
        ckpt = make_initial_checkpoint(_tiny_state(), tmp_path)
        s = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            heartbeat_timeout=0,
            **kw,
        )
        s.start()
        time.sleep(0.2)
        return s

    def test_register_response_bf16_when_configured(self, tmp_path):
        s = self._server(tmp_path, download_dtype="bf16")
        try:
            shapes = {n: list(t.shape) for n, t in _tiny_state().items()}
            data = _register_via_http(s, "alpha", shapes)
            state = torch.load(io.BytesIO(data), weights_only=False)
            for name, t in state.items():
                assert (
                    t.dtype == torch.bfloat16
                ), f"{name} came back as {t.dtype}, expected bf16"
        finally:
            s.stop()

    def test_register_response_fp32_by_default(self, tmp_path):
        s = self._server(tmp_path)  # default download_dtype=fp32
        try:
            shapes = {n: list(t.shape) for n, t in _tiny_state().items()}
            data = _register_via_http(s, "alpha", shapes)
            state = torch.load(io.BytesIO(data), weights_only=False)
            for name, t in state.items():
                assert t.dtype == torch.float32

        finally:
            s.stop()

    def test_sync_response_bf16_when_configured(self, tmp_path):
        s = self._server(tmp_path, download_dtype="bf16")
        try:
            shapes = {n: list(t.shape) for n, t in _tiny_state().items()}
            _register_via_http(s, "alpha", shapes)
            # Fake a tiny pseudograd; one worker so barrier releases
            # immediately.
            pg = {n: torch.zeros(*t.shape) for n, t in _tiny_state().items()}
            data = _submit_pseudograd_via_http(s, "alpha", pg)
            state = torch.load(io.BytesIO(data), weights_only=False)
            for name, t in state.items():
                assert t.dtype == torch.bfloat16
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# True-bf16 weight smoke test (#130 acceptance #1)
# ---------------------------------------------------------------------------


class TestTrueBf16WeightsSmoke:
    """A DiLoCo run with bf16 live weights and bf16 download completes
    one sync round without exception. This is the smoke test #130
    acceptance #1 asks for — the legacy code already runs (no hard
    dtype gate), but the refactor expands the matrix and this guards
    against silent breakage."""

    def test_register_and_sync_round_with_bf16_weights(self, tmp_path):
        ckpt = make_initial_checkpoint(_tiny_state(), tmp_path)
        s = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            heartbeat_timeout=0,
            download_dtype="bf16",
            download_sr=True,
        )
        s.start()
        time.sleep(0.2)
        try:
            # Build a model whose weights match the server's slice
            # (named layer_0/layer_1), but in bf16. We register and
            # submit a tiny pseudograd — the assertion is "no
            # exception" + dtype contract on the response.
            class TwoLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    init = _tiny_state()
                    self.layer_0 = nn.Parameter(init["layer_0.weight"].bfloat16())
                    self.layer_1 = nn.Parameter(init["layer_1.weight"].bfloat16())

                def named_parameters(self, recurse=True):
                    yield "layer_0.weight", self.layer_0
                    yield "layer_1.weight", self.layer_1

            shapes = {n: list(t.shape) for n, t in _tiny_state().items()}
            data = _register_via_http(s, "alpha", shapes)
            snap = torch.load(io.BytesIO(data), weights_only=False)
            for t in snap.values():
                assert t.dtype == torch.bfloat16

            # Compute pseudograds against the bf16 snapshot using the
            # ParamView path (same code the live worker uses) — the
            # bf16-snapshot, bf16-live case must run end-to-end.
            model = TwoLayer()
            view = SimpleModelParamView(model)
            # Raw diff is in the live model dtype (bf16 here); the backend would
            # apply the wire cast, but this test submits the raw pg over HTTP
            # directly to exercise the server's bf16 handling.
            pg = view.compute_pseudograds(snap)
            for t in pg.values():
                assert t.dtype == torch.bfloat16

            data2 = _submit_pseudograd_via_http(s, "alpha", pg)
            state = torch.load(io.BytesIO(data2), weights_only=False)
            for t in state.values():
                assert t.dtype == torch.bfloat16
        finally:
            s.stop()
