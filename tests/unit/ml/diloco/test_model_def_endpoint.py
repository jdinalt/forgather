"""Integration tests for the server ``GET /model_def`` endpoint and the
worker-side staging helper (issue #53).

A DiLoCo worker fetches the model definition (config + custom code +
tokenizer, never weights) from the server, stages it into its own output
dir, and builds the model from there. These tests run an in-process server
over a real checkpoint dir and exercise: the endpoint serves the right
files with the hash header, the persisted checkpoint dir + folded
model_hash, auth is required, and ``stage_model_def`` caches / invalidates.
"""

import os
import time

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient, DiLoCoModelMismatchError
from forgather.ml.diloco.model_def import MODEL_HASH_HEADER
from forgather.ml.diloco.model_stage import STAGE_SUBDIR, STAMP_NAME, stage_model_def
from forgather.ml.diloco.server import DiLoCoServer
from forgather.ml.sharded_checkpoint import save_checkpoint as _save_checkpoint


def _make_state_dict(dim=8, num_layers=2, seed=42):
    torch.manual_seed(seed)
    return {f"layer{i}.weight": torch.randn(dim, dim) for i in range(num_layers)}


def _make_model_dir(base, sd, *, code="class Model: pass"):
    """A self-contained model dir: weights + config + custom code + tokenizer."""
    ckpt = os.path.join(str(base), "model")
    _save_checkpoint(ckpt, sd, safetensors=True)
    with open(os.path.join(ckpt, "config.json"), "w") as fh:
        fh.write('{"hidden_size": 8, "model_type": "demo"}')
    with open(os.path.join(ckpt, "modeling_demo.py"), "w") as fh:
        fh.write(code)
    with open(os.path.join(ckpt, "tokenizer_config.json"), "w") as fh:
        fh.write("{}")
    return ckpt


def _start_server(tmp_path, ckpt, **kwargs):
    def simple_sgd(params):
        return torch.optim.SGD(params, lr=1.0, momentum=0.0)

    server = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        outer_optimizer_factory=simple_sgd,
        **kwargs,
    )
    server.start()
    time.sleep(0.2)
    return server


@pytest.fixture
def server(tmp_path):
    sd = _make_state_dict()
    ckpt = _make_model_dir(tmp_path, sd)
    srv = _start_server(tmp_path, ckpt)
    yield srv, ckpt
    srv.stop()


def test_load_state_persists_checkpoint_dir(server):
    srv, ckpt = server
    assert srv._loaded_checkpoint_dir == os.path.realpath(ckpt)


def test_model_hash_folds_in_definition(tmp_path):
    """Two servers with identical weights but different custom code must
    advertise different model_hash (the bundle content is folded in)."""
    sd = _make_state_dict()
    c1 = _make_model_dir(tmp_path / "a", sd, code="class Model: pass")
    c2 = _make_model_dir(tmp_path / "b", sd, code="class Model: x = 1")
    s1 = _start_server(tmp_path / "a", c1)
    s2 = _start_server(tmp_path / "b", c2)
    try:
        assert s1._model_hash != s2._model_hash
    finally:
        s1.stop()
        s2.stop()


def test_model_def_serves_definition_with_hash_header(server):
    srv, ckpt = server
    client = DiLoCoClient(f"localhost:{srv.port}", timeout=10)
    dest = os.path.join(srv.output_dir, "fetched")
    returned_hash = client.fetch_model_def(dest)
    got = sorted(os.listdir(dest))
    # Definition files present; weights absent.
    assert "config.json" in got
    assert "modeling_demo.py" in got
    assert "tokenizer_config.json" in got
    assert not any(g.endswith(".safetensors") for g in got)
    assert not any(g == "server_state.pt" for g in got)
    # The hash matches the server's advertised model_hash.
    assert returned_hash == srv._model_hash
    assert client.get_info()["model_hash"] == srv._model_hash


def test_model_def_requires_auth(tmp_path):
    sd = _make_state_dict()
    ckpt = _make_model_dir(tmp_path, sd)
    srv = _start_server(tmp_path, ckpt, auth_token="s3cret")
    try:
        # Wrong token -> rejected before any bytes are served.
        bad = DiLoCoClient(f"localhost:{srv.port}", timeout=10, token="wrong")
        with pytest.raises(Exception):
            bad.fetch_model_def(os.path.join(str(tmp_path), "nope"))
        # Correct token -> served.
        good = DiLoCoClient(f"localhost:{srv.port}", timeout=10, token="s3cret")
        good.fetch_model_def(os.path.join(str(tmp_path), "ok"))
        assert os.path.exists(os.path.join(str(tmp_path), "ok", "config.json"))
    finally:
        srv.stop()


def test_stage_model_def_miss_then_hit(server, tmp_path):
    srv, ckpt = server
    out = tmp_path / "worker_out"
    out.mkdir()
    addr = f"localhost:{srv.port}"
    # Miss: stages the bundle into <out>/diloco_model_def with a stamp.
    local = stage_model_def(addr, str(out))
    assert local == os.path.join(str(out), STAGE_SUBDIR)
    assert os.path.exists(os.path.join(local, "config.json"))
    stamp_path = os.path.join(local, STAMP_NAME)
    assert os.path.exists(stamp_path)
    with open(stamp_path) as fh:
        assert fh.read().strip() == srv._model_hash

    # Hit: a second call with a matching stamp reuses without re-extracting.
    # Prove reuse by mutating a staged file and confirming it is NOT replaced.
    sentinel = os.path.join(local, "config.json")
    with open(sentinel, "w") as fh:
        fh.write("REUSED")
    local2 = stage_model_def(addr, str(out))
    assert local2 == local
    with open(sentinel) as fh:
        assert fh.read() == "REUSED"  # not re-fetched


@pytest.mark.skipif(
    not os.path.isdir("models/tiny") or not os.path.exists("models/tiny/config.json"),
    reason="models/tiny generated model dir not present",
)
def test_end_to_end_stage_and_build_empty_on_meta(tmp_path):
    """The DiLoCo worker's real path: a server started from a multi-file
    custom model serves its definition; the worker stages it and builds an
    EMPTY model on meta whose parameter set matches the server's (so the
    /register fingerprint would pass). Exercises the two-file/multi-file
    trust_remote_code closure end to end."""
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    srv = _start_server(tmp_path, os.path.realpath("models/tiny"))
    try:
        out = tmp_path / "worker_out"
        out.mkdir()
        addr = f"localhost:{srv.port}"

        # Tokenizer-first ordering: in the real config the tokenizer
        # resolves at dataset preprocessing, BEFORE the model is built. The
        # shared staging singleton handles this — the first consumer
        # triggers the one fetch; the model construction below reuses it.
        local = stage_model_def(addr, str(out))
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        assert tok is not None

        # The whole .py closure (many files) was staged, plus config +
        # tokenizer; no weights.
        staged = set(os.listdir(local))
        assert "config.json" in staged
        assert "tokenizer.json" in staged
        assert sum(1 for f in staged if f.endswith(".py")) >= 5
        assert not any(f.endswith((".bin", ".safetensors")) for f in staged)

        # Second staging call (the model's config arg) must reuse the cache:
        # mutate a staged file and confirm it is NOT re-fetched.
        with open(os.path.join(local, "config.json")) as fh:
            cfg_text = fh.read()
        local2 = stage_model_def(addr, str(out))
        assert local2 == local
        with open(os.path.join(local2, "config.json")) as fh:
            assert fh.read() == cfg_text  # untouched -> single fetch

        # Build the empty model on meta from the staged definition.
        config = AutoConfig.from_pretrained(local, trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        # Its parameter set matches what the server holds (the coarse model
        # the worker would register with).
        model_params = {n for n, _ in model.named_parameters()}
        server_params = set(srv._param_names)
        assert model_params == server_params, model_params ^ server_params
    finally:
        srv.stop()


def test_stage_model_def_invalidates_on_hash_change(tmp_path):
    """When the server is restarted on a different model (new hash), a stale
    stamp forces a clean re-fetch — no silent reuse."""
    sd = _make_state_dict()
    ckpt = _make_model_dir(tmp_path, sd, code="class Model: pass")
    out = tmp_path / "worker_out"
    out.mkdir()

    srv1 = _start_server(tmp_path, ckpt)
    addr = f"localhost:{srv1.port}"
    local = stage_model_def(addr, str(out))
    first_hash = open(os.path.join(local, STAMP_NAME)).read().strip()
    srv1.stop()

    # Restart on a DIFFERENT definition at the SAME output_dir/port-ish addr.
    ckpt2 = _make_model_dir(tmp_path / "v2", sd, code="class Model: y = 2")
    srv2 = _start_server(tmp_path / "v2", ckpt2)
    try:
        addr2 = f"localhost:{srv2.port}"
        local2 = stage_model_def(addr2, str(out))
        second_hash = open(os.path.join(local2, STAMP_NAME)).read().strip()
        assert second_hash != first_hash
        assert second_hash == srv2._model_hash
        assert "y = 2" in open(os.path.join(local2, "modeling_demo.py")).read()
    finally:
        srv2.stop()
