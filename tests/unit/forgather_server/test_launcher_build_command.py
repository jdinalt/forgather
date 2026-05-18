"""Tests for launcher.build_command's argv shape.

The function reads the config's materialized meta block to pick up
``nproc_per_node`` and ``forgather_dir``; we mock that out here so the
test doesn't need a real project tree on disk. What we're verifying is
the standalone-vs-rdzv argv switch, which is the multi-node submit
path's main risk surface.
"""

from unittest.mock import MagicMock, patch

import forgather_server.launcher as launcher


def _patched_meta(forgather_dir="/forgather", nproc_per_node=2):
    """Stub MetaConfig + get_env + Latent.materialize chain."""
    meta = MagicMock()
    meta.config_path.return_value = "/tmp/cfg.yaml"
    meta.system_path = None
    env = MagicMock()
    loaded = MagicMock()
    loaded.config.meta = MagicMock()
    env.load.return_value = loaded
    materialized = {
        "nproc_per_node": nproc_per_node,
        "forgather_dir": forgather_dir,
    }
    return meta, env, materialized


class TestStandaloneMode:
    def test_emits_standalone_when_rdzv_args_none(self, monkeypatch):
        meta, env, materialized = _patched_meta()
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        cmd = launcher.build_command(
            project_dir="/proj",
            config_name="train.yaml",
            dynamic_args={},
            rdzv_args=None,
        )
        assert cmd[0] == "torchrun"
        assert "--standalone" in cmd
        # Standalone mode does not emit any rdzv flags.
        assert "--rdzv-backend" not in cmd
        assert "--node-rank" not in cmd
        assert "--nnodes" not in cmd

    def test_default_path_unchanged_when_arg_omitted(self, monkeypatch):
        # Backwards compat: build_command was called with positional
        # args before this PR; the new rdzv_args parameter must be
        # optional so existing callers keep working.
        meta, env, materialized = _patched_meta()
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        cmd = launcher.build_command("/proj", "train.yaml", {})
        assert "--standalone" in cmd


class TestRdzvMode:
    def test_emits_rdzv_block(self, monkeypatch):
        meta, env, materialized = _patched_meta(nproc_per_node=4)
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        rdzv = {
            "nnodes": 2,
            "node_rank": 1,
            "rdzv_backend": "c10d",
            "rdzv_endpoint": "wopr:29400",
            "rdzv_id": "abc123",
            "nproc_per_node": 1,  # peer has fewer GPUs than the config
        }
        cmd = launcher.build_command(
            project_dir="/proj",
            config_name="train.yaml",
            dynamic_args={},
            rdzv_args=rdzv,
        )
        assert "--standalone" not in cmd
        # All rdzv block flags present in argv-pair form.
        idx = cmd.index("--nnodes")
        assert cmd[idx + 1] == "2"
        idx = cmd.index("--node-rank")
        assert cmd[idx + 1] == "1"
        idx = cmd.index("--rdzv-backend")
        assert cmd[idx + 1] == "c10d"
        idx = cmd.index("--rdzv-endpoint")
        assert cmd[idx + 1] == "wopr:29400"
        idx = cmd.index("--rdzv-id")
        assert cmd[idx + 1] == "abc123"
        # Cluster-supplied nproc_per_node wins over the config's.
        idx = cmd.index("--nproc-per-node")
        assert cmd[idx + 1] == "1"

    def test_default_rdzv_backend_is_c10d(self, monkeypatch):
        meta, env, materialized = _patched_meta()
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        rdzv = {
            "nnodes": 2,
            "node_rank": 0,
            "rdzv_endpoint": "wopr:29400",
            "rdzv_id": "abc",
        }
        cmd = launcher.build_command("/proj", "train.yaml", {}, rdzv_args=rdzv)
        idx = cmd.index("--rdzv-backend")
        assert cmd[idx + 1] == "c10d"

    def test_falls_back_to_config_nproc_when_rdzv_omits_it(self, monkeypatch):
        meta, env, materialized = _patched_meta(nproc_per_node=8)
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        rdzv = {
            "nnodes": 2,
            "node_rank": 0,
            "rdzv_endpoint": "wopr:29400",
            "rdzv_id": "abc",
            # nproc_per_node intentionally absent
        }
        cmd = launcher.build_command("/proj", "train.yaml", {}, rdzv_args=rdzv)
        idx = cmd.index("--nproc-per-node")
        assert cmd[idx + 1] == "8"

    def test_emits_rdzv_conf_is_host_true(self, monkeypatch):
        # is_host=True must reach torchrun as
        # ``--rdzv-conf is_host=true`` so c10d skips the broken
        # gethostname-based autodetection (resolves to 127.0.1.1 on
        # Debian/Ubuntu, which never matches the rdzv endpoint).
        meta, env, materialized = _patched_meta()
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        rdzv = {
            "nnodes": 2,
            "node_rank": 0,
            "rdzv_endpoint": "wopr:29400",
            "rdzv_id": "abc",
            "is_host": True,
        }
        cmd = launcher.build_command("/proj", "train.yaml", {}, rdzv_args=rdzv)
        idx = cmd.index("--rdzv-conf")
        assert cmd[idx + 1] == "is_host=true"

    def test_emits_rdzv_conf_is_host_false(self, monkeypatch):
        meta, env, materialized = _patched_meta()
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        rdzv = {
            "nnodes": 2,
            "node_rank": 1,
            "rdzv_endpoint": "wopr:29400",
            "rdzv_id": "abc",
            "is_host": False,
        }
        cmd = launcher.build_command("/proj", "train.yaml", {}, rdzv_args=rdzv)
        idx = cmd.index("--rdzv-conf")
        assert cmd[idx + 1] == "is_host=false"

    def test_emits_local_addr_when_present(self, monkeypatch):
        # local_addr must reach torchrun as ``--local-addr <ip>`` so
        # rank 0's elastic agent writes an IP into the c10d store as
        # MASTER_ADDR instead of falling back to socket.getfqdn(),
        # which yields an unresolvable hostname on LANs without DNS.
        meta, env, materialized = _patched_meta()
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        rdzv = {
            "nnodes": 2,
            "node_rank": 0,
            "rdzv_endpoint": "192.168.9.95:29400",
            "rdzv_id": "abc",
            "local_addr": "192.168.9.95",
        }
        cmd = launcher.build_command("/proj", "train.yaml", {}, rdzv_args=rdzv)
        idx = cmd.index("--local-addr")
        assert cmd[idx + 1] == "192.168.9.95"

    def test_omits_local_addr_when_absent(self, monkeypatch):
        # Backwards-compat: callers that never set local_addr must keep
        # working — torch falls back to its own hostname lookup.
        meta, env, materialized = _patched_meta()
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        rdzv = {
            "nnodes": 2,
            "node_rank": 0,
            "rdzv_endpoint": "wopr:29400",
            "rdzv_id": "abc",
        }
        cmd = launcher.build_command("/proj", "train.yaml", {}, rdzv_args=rdzv)
        assert "--local-addr" not in cmd

    def test_omits_rdzv_conf_when_is_host_absent(self, monkeypatch):
        # Backwards-compat: callers that never set is_host must keep
        # working — torch's autodetection runs and we don't inject a
        # config flag we didn't ask for.
        meta, env, materialized = _patched_meta()
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        rdzv = {
            "nnodes": 2,
            "node_rank": 0,
            "rdzv_endpoint": "wopr:29400",
            "rdzv_id": "abc",
        }
        cmd = launcher.build_command("/proj", "train.yaml", {}, rdzv_args=rdzv)
        assert "--rdzv-conf" not in cmd

    def test_train_script_path_present_in_both_modes(self, monkeypatch):
        meta, env, materialized = _patched_meta(forgather_dir="/fg")
        monkeypatch.setattr(launcher, "MetaConfig", lambda *a, **kw: meta)
        monkeypatch.setattr(launcher, "get_env", lambda *a, **kw: env)
        monkeypatch.setattr(
            launcher.Latent, "materialize", lambda x: materialized
        )
        for rdzv in (
            None,
            {
                "nnodes": 2,
                "node_rank": 0,
                "rdzv_endpoint": "wopr:29400",
                "rdzv_id": "abc",
            },
        ):
            cmd = launcher.build_command(
                "/proj", "train.yaml", {}, rdzv_args=rdzv
            )
            assert "/fg/scripts/train_script.py" in cmd
            assert "-p" in cmd
            assert "train.yaml" == cmd[-1]
