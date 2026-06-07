"""Tests for the collective backend selection in DiLoCoCallback (#154).

Covers `_make_collective_backend`: the `DILOCO_BACKEND=collective` knob, the
requirement of a torch.distributed world, the DDP-wrap rejection, the
init-checkpoint precedence (`/info` vs `DILOCO_INIT_CHECKPOINT`), the
outer-optimizer reproduction, and the fail-loud guards. The success-path tests
run inside a single-rank gloo group (no GPU needed).
"""

import os
import socket

import pytest
import torch

from forgather.ml.diloco.collective_backend import CollectiveBackend
from forgather.ml.trainer.callbacks.diloco_callback import DiLoCoCallback

_COLLECTIVE_ENV = ("DILOCO_BACKEND", "DILOCO_INIT_CHECKPOINT")
_SGD_INFO = {"name": "SGD", "lr": 0.7, "momentum": 0.9, "nesterov": True}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_cb(monkeypatch, **env):
    for k in _COLLECTIVE_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return DiLoCoCallback(server_addr="dummy:8512")


def _settings(num_fragments=1, ckpt="/info/ckpt", outer=_SGD_INFO):
    return {
        "num_fragments": num_fragments,
        "model_checkpoint_dir": ckpt,
        "outer_optimizer": outer,
    }


@pytest.fixture
def gloo_world():
    """A single-rank gloo process group so the collective branch's dist checks
    pass without a real torchrun world or GPUs."""
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(_free_port()),
        RANK="0",
        WORLD_SIZE="1",
    )
    dist.init_process_group(backend="gloo")
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_collective_requires_dist(monkeypatch):
    # No torch.distributed world -> fail loud (collective needs a process group).
    cb = _make_cb(monkeypatch, DILOCO_BACKEND="collective")
    assert cb.backend_kind == "collective"
    with pytest.raises(ValueError, match="torch.distributed"):
        cb._make_sync_backend(_settings())


def test_collective_builds_backend_from_info_checkpoint(monkeypatch, gloo_world):
    cb = _make_cb(monkeypatch, DILOCO_BACKEND="collective")
    backend = cb._make_sync_backend(_settings(ckpt="/info/ckpt"))
    assert isinstance(backend, CollectiveBackend)
    assert backend.init_checkpoint == "/info/ckpt"  # advertised by /info
    assert backend.group_size == 1 and backend.rank == 0
    # The replicated outer optimizer reproduces the server's (from /info).
    opt = backend.outer_opt_factory([torch.zeros(1, requires_grad=True)])
    pg = opt.param_groups[0]
    assert (pg["lr"], pg["momentum"], pg["nesterov"]) == (0.7, 0.9, True)


def test_env_init_checkpoint_overrides_info(monkeypatch, gloo_world):
    cb = _make_cb(
        monkeypatch,
        DILOCO_BACKEND="collective",
        DILOCO_INIT_CHECKPOINT="/override/ckpt",
    )
    backend = cb._make_sync_backend(_settings(ckpt="/info/ckpt"))
    assert backend.init_checkpoint == "/override/ckpt"


def test_rejects_ddp_wrapped_model(monkeypatch, gloo_world):
    from torch.nn.parallel import DistributedDataParallel as DDP

    cb = _make_cb(monkeypatch, DILOCO_BACKEND="collective")
    model = DDP(torch.nn.Linear(3, 3))  # gloo/CPU single-rank wrap
    with pytest.raises(ValueError, match="DistributedDataParallel"):
        cb._make_sync_backend(_settings(), model)


def test_rejects_fragments(monkeypatch, gloo_world):
    cb = _make_cb(monkeypatch, DILOCO_BACKEND="collective")
    with pytest.raises(ValueError, match="streaming fragments"):
        cb._make_sync_backend(_settings(num_fragments=2))


def test_no_init_checkpoint_raises(monkeypatch, gloo_world):
    cb = _make_cb(monkeypatch, DILOCO_BACKEND="collective")
    with pytest.raises(ValueError, match="init checkpoint"):
        cb._make_sync_backend(_settings(ckpt=None))


def test_non_sgd_outer_optimizer_raises(monkeypatch, gloo_world):
    cb = _make_cb(monkeypatch, DILOCO_BACKEND="collective")
    with pytest.raises(ValueError):
        cb._make_sync_backend(_settings(outer={"name": "Adam", "lr": 1e-3}))
