"""Tests for the shared-memory backend selection in DiLoCoCallback (#154).

Covers `_make_sync_backend`: the `DILOCO_BACKEND` knob, the group-rendezvous env
validation, the init-checkpoint precedence (`/info` advertised vs env override),
the outer-optimizer reproduction from `/info`, and the fail-loud guards.
"""

import os

import pytest
import torch

from forgather.ml.diloco.shared_memory_backend import SharedMemoryBackend
from forgather.ml.trainer.callbacks.diloco_callback import DiLoCoCallback

_SHM_ENV = (
    "DILOCO_BACKEND",
    "DILOCO_SHM_GROUP_DIR",
    "DILOCO_SHM_GROUP_SIZE",
    "DILOCO_SHM_INIT_CHECKPOINT",
)


def _make_cb(monkeypatch, **env):
    for k in _SHM_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return DiLoCoCallback(server_addr="dummy:8512")


_SGD_INFO = {"name": "SGD", "lr": 0.7, "momentum": 0.9, "nesterov": True}


def _settings(num_fragments=1, ckpt="/info/ckpt", outer=_SGD_INFO):
    return {
        "num_fragments": num_fragments,
        "model_checkpoint_dir": ckpt,
        "outer_optimizer": outer,
    }


class TestMakeSyncBackend:
    def test_http_default_returns_none(self, monkeypatch):
        cb = _make_cb(monkeypatch)  # no DILOCO_BACKEND -> http
        assert cb.backend_kind == "http"
        assert cb._make_sync_backend(_settings()) is None

    def test_shared_memory_builds_backend_from_info_checkpoint(
        self, monkeypatch, tmp_path
    ):
        cb = _make_cb(
            monkeypatch,
            DILOCO_BACKEND="shared_memory",
            DILOCO_SHM_GROUP_DIR=str(tmp_path),
            DILOCO_SHM_GROUP_SIZE="2",
        )
        backend = cb._make_sync_backend(_settings(ckpt="/info/ckpt"))
        assert isinstance(backend, SharedMemoryBackend)
        assert backend.group_size == 2
        assert backend.group_dir == os.path.realpath(str(tmp_path))
        assert backend.init_checkpoint == "/info/ckpt"  # advertised by /info
        # The aggregator reproduces the server's outer optimizer (from /info).
        opt = backend.outer_opt_factory([torch.zeros(1, requires_grad=True)])
        pg = opt.param_groups[0]
        assert (pg["lr"], pg["momentum"], pg["nesterov"]) == (0.7, 0.9, True)

    def test_env_init_checkpoint_overrides_info(self, monkeypatch, tmp_path):
        cb = _make_cb(
            monkeypatch,
            DILOCO_BACKEND="shared_memory",
            DILOCO_SHM_GROUP_DIR=str(tmp_path),
            DILOCO_SHM_GROUP_SIZE="2",
            DILOCO_SHM_INIT_CHECKPOINT="/override/ckpt",
        )
        backend = cb._make_sync_backend(_settings(ckpt="/info/ckpt"))
        assert backend.init_checkpoint == "/override/ckpt"

    def test_missing_group_env_raises_at_construction(self, monkeypatch):
        with pytest.raises(ValueError):
            _make_cb(monkeypatch, DILOCO_BACKEND="shared_memory")  # no dir/size

    def test_invalid_backend_raises(self, monkeypatch):
        with pytest.raises(ValueError):
            _make_cb(monkeypatch, DILOCO_BACKEND="bogus")

    def test_num_fragments_gt1_raises(self, monkeypatch, tmp_path):
        cb = _make_cb(
            monkeypatch,
            DILOCO_BACKEND="shared_memory",
            DILOCO_SHM_GROUP_DIR=str(tmp_path),
            DILOCO_SHM_GROUP_SIZE="2",
        )
        with pytest.raises(ValueError):
            cb._make_sync_backend(_settings(num_fragments=2))

    def test_no_init_checkpoint_raises(self, monkeypatch, tmp_path):
        cb = _make_cb(
            monkeypatch,
            DILOCO_BACKEND="shared_memory",
            DILOCO_SHM_GROUP_DIR=str(tmp_path),
            DILOCO_SHM_GROUP_SIZE="2",
        )
        with pytest.raises(ValueError):
            cb._make_sync_backend(_settings(ckpt=None))

    def test_missing_outer_optimizer_raises(self, monkeypatch, tmp_path):
        # Older server that doesn't advertise its outer-opt config -> fail loud.
        cb = _make_cb(
            monkeypatch,
            DILOCO_BACKEND="shared_memory",
            DILOCO_SHM_GROUP_DIR=str(tmp_path),
            DILOCO_SHM_GROUP_SIZE="2",
        )
        with pytest.raises(ValueError):
            cb._make_sync_backend(_settings(outer=None))

    def test_non_sgd_outer_optimizer_raises(self, monkeypatch, tmp_path):
        cb = _make_cb(
            monkeypatch,
            DILOCO_BACKEND="shared_memory",
            DILOCO_SHM_GROUP_DIR=str(tmp_path),
            DILOCO_SHM_GROUP_SIZE="2",
        )
        with pytest.raises(ValueError):
            cb._make_sync_backend(_settings(outer={"name": "Adam", "lr": 1e-3}))


class TestReportSyncStateKnob:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("DILOCO_REPORT_SYNC_STATE", raising=False)
        assert _make_cb(monkeypatch).report_sync_state is True

    def test_disabled(self, monkeypatch):
        monkeypatch.setenv("DILOCO_REPORT_SYNC_STATE", "0")
        assert _make_cb(monkeypatch).report_sync_state is False
