"""Tests for the shared-memory backend selection in DiLoCoCallback (#154).

Covers `_make_sync_backend`: the `DILOCO_BACKEND` knob, the group-rendezvous env
validation, the init-checkpoint precedence (`/info` advertised vs env override),
the outer-optimizer reproduction from `/info`, and the fail-loud guards.
"""

import os

import pytest

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


def _settings(num_fragments=1, shm_group_dir=None, shm_group_size=None):
    """An /info-derived settings dict. Under Flavor 2 the follower reads the
    region dir + group size from /info (``shm_group_dir`` / ``shm_group_size``)
    and does not seed weights or build the outer optimizer (the server does)."""
    return {
        "num_fragments": num_fragments,
        "model_checkpoint_dir": "/info/ckpt",
        "outer_optimizer": _SGD_INFO,
        "shm_group_dir": shm_group_dir,
        "shm_group_size": shm_group_size,
    }


class TestMakeSyncBackend:
    def test_http_default_returns_none(self, monkeypatch):
        cb = _make_cb(monkeypatch)  # no DILOCO_BACKEND -> http
        assert cb.backend_kind == "http"
        assert cb._make_sync_backend(_settings()) is None

    def test_shared_memory_follower_from_info(self, monkeypatch, tmp_path):
        # Flavor 2: dir + size come from the server's /info; the worker is a
        # pure follower (no self-elected aggregation), seeds no weights, and
        # does not build the outer optimizer (the server owns it).
        cb = _make_cb(monkeypatch, DILOCO_BACKEND="shared_memory")  # no env dir/size
        backend = cb._make_sync_backend(
            _settings(shm_group_dir=str(tmp_path), shm_group_size=3)
        )
        assert isinstance(backend, SharedMemoryBackend)
        assert backend.group_size == 3
        assert backend.group_dir == os.path.realpath(str(tmp_path))
        assert backend.follower_only is True
        assert backend.init_checkpoint is None

    def test_env_overrides_info_dir_size(self, monkeypatch, tmp_path):
        # The env vars remain an explicit override over /info.
        cb = _make_cb(
            monkeypatch,
            DILOCO_BACKEND="shared_memory",
            DILOCO_SHM_GROUP_DIR=str(tmp_path),
            DILOCO_SHM_GROUP_SIZE="2",
        )
        backend = cb._make_sync_backend(
            _settings(shm_group_dir="/info/other", shm_group_size=9)
        )
        assert backend.group_dir == os.path.realpath(str(tmp_path))
        assert backend.group_size == 2

    def test_missing_dir_size_raises_at_build(self, monkeypatch):
        # Construction no longer requires the group env (it comes from /info);
        # build fails loud only when NEITHER /info nor env supplies dir/size.
        cb = _make_cb(monkeypatch, DILOCO_BACKEND="shared_memory")
        assert cb.backend_kind == "shared_memory"
        with pytest.raises(ValueError):
            cb._make_sync_backend(_settings())  # no shm_group_dir / shm_group_size

    def test_invalid_backend_raises(self, monkeypatch):
        with pytest.raises(ValueError):
            _make_cb(monkeypatch, DILOCO_BACKEND="bogus")

    def test_num_fragments_gt1_raises(self, monkeypatch, tmp_path):
        cb = _make_cb(monkeypatch, DILOCO_BACKEND="shared_memory")
        with pytest.raises(ValueError):
            cb._make_sync_backend(
                _settings(
                    num_fragments=2, shm_group_dir=str(tmp_path), shm_group_size=2
                )
            )


class TestReportSyncStateKnob:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("DILOCO_REPORT_SYNC_STATE", raising=False)
        assert _make_cb(monkeypatch).report_sync_state is True

    def test_disabled(self, monkeypatch):
        monkeypatch.setenv("DILOCO_REPORT_SYNC_STATE", "0")
        assert _make_cb(monkeypatch).report_sync_state is False
