"""Unit tests for ``scheduler._resolve_diloco_server_token``.

Mirrors ``test_scheduler_dataset_server_token.py``: per-port persisted
token reused across restarts; ``regen=True`` rotates. Pins the same
contract for the DiLoCo spawn path (issue #90).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def _patch_config_dir(tmp_path: Path):
    return patch(
        "forgather.ml.diloco.auth.forgather_config_dir",
        return_value=str(tmp_path),
    )


def test_first_spawn_mints_and_persists(tmp_path):
    from forgather_server import scheduler

    from forgather.ml.diloco.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        token = scheduler._resolve_diloco_server_token(port=18512, regen=False)
        path = standalone_token_file(18512)

    assert len(token) == 64
    assert path.is_file()
    assert path.read_text().strip() == token
    assert (path.stat().st_mode & 0o777) == 0o600


def test_second_spawn_reuses_persisted_token(tmp_path):
    from forgather_server import scheduler

    from forgather.ml.diloco.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        first = scheduler._resolve_diloco_server_token(port=18513, regen=False)
        second = scheduler._resolve_diloco_server_token(port=18513, regen=False)
        path = standalone_token_file(18513)

    assert first == second
    assert path.read_text().strip() == first


def test_regen_rotates_existing_token(tmp_path):
    from forgather_server import scheduler

    from forgather.ml.diloco.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        original = scheduler._resolve_diloco_server_token(port=18514, regen=False)
        rotated = scheduler._resolve_diloco_server_token(port=18514, regen=True)
        path = standalone_token_file(18514)

    assert original != rotated
    assert path.read_text().strip() == rotated


def test_empty_file_is_treated_as_missing(tmp_path):
    from forgather_server import scheduler

    from forgather.ml.diloco.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        path = standalone_token_file(18515)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        token = scheduler._resolve_diloco_server_token(port=18515, regen=False)

    assert len(token) == 64
    assert path.read_text().strip() == token


def test_build_diloco_command_includes_auth_token_file():
    """The CLI builder forwards the per-port token file path so the
    actual token never appears in argv (visible via ``ps``)."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(
        output_dir="/tmp/out",
        num_workers=1,
        port=8512,
        auth_token_file="/some/per/port/token.token",
    )
    assert "--auth-token-file" in cmd
    idx = cmd.index("--auth-token-file")
    assert cmd[idx + 1] == "/some/per/port/token.token"


def test_build_diloco_command_no_auth():
    """no_auth=True emits --no-auth and skips --auth-token-file."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(
        output_dir="/tmp/out",
        num_workers=1,
        port=8512,
        no_auth=True,
        auth_token_file="ignored-when-no-auth",
    )
    assert "--no-auth" in cmd
    assert "--auth-token-file" not in cmd


def test_build_diloco_command_is_local_only():
    """The scheduler-spawned server MUST run foreground (--local-only), or it
    would hit its own orchestrator auto-detect and re-enqueue itself — a
    self-enqueue loop of dead jobs. Regression guard."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(output_dir="/tmp/out", num_workers=1, port=8512)
    assert "--local-only" in cmd


def test_build_diloco_command_bulk_cleartext_flag():
    """The cleartext-bulk flag surfaces on the spawn argv when enabled."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(
        output_dir="/tmp/out",
        num_workers=1,
        port=8512,
        bulk_cleartext=True,
    )
    assert "--bulk-cleartext" in cmd
    # No port number is emitted — the server picks an ephemeral one.
    assert "--bulk-port" not in cmd


def test_build_diloco_command_run_name():
    """--run-name surfaces on the spawn argv when set, and is omitted when not."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(
        output_dir="/tmp/out",
        num_workers=1,
        port=8512,
        run_name="lr0.7-2w",
    )
    assert "--run-name" in cmd
    assert cmd[cmd.index("--run-name") + 1] == "lr0.7-2w"

    cmd2 = build_diloco_server_command(output_dir="/tmp/out", num_workers=1, port=8512)
    assert "--run-name" not in cmd2


def test_build_diloco_command_group_settings():
    """sync_every/num_fragments/bf16_comm (server-authoritative, adopted by
    workers from /info) surface on the spawn argv."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(
        output_dir="/tmp/out",
        num_workers=1,
        port=8512,
        sync_every=250,
        num_fragments=4,
        bf16_comm=False,
    )
    assert "--sync-every" in cmd
    assert cmd[cmd.index("--sync-every") + 1] == "250"
    assert "--num-fragments" in cmd
    assert cmd[cmd.index("--num-fragments") + 1] == "4"
    assert "--no-bf16" in cmd


def test_build_diloco_command_group_settings_defaults():
    """Defaults: --sync-every emitted (always meaningful), but
    --num-fragments (1) and --no-bf16 (bf16 on) are omitted for a readable
    argv."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(
        output_dir="/tmp/out",
        num_workers=1,
        port=8512,
    )
    assert "--sync-every" in cmd
    assert "--num-fragments" not in cmd
    assert "--no-bf16" not in cmd


def test_build_diloco_command_no_bulk_cleartext_omits_flag():
    """Default (bulk_cleartext off) → no bulk flag on the argv."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(
        output_dir="/tmp/out",
        num_workers=1,
        port=8512,
    )
    assert "--bulk-cleartext" not in cmd
    assert "--bulk-port" not in cmd


def test_build_diloco_command_bulk_transport_flags():
    """Bulk transport (issue #154): --wire-format only on divergence from the
    pickle default; --grpc when the gRPC listener is requested."""
    from forgather_server.diloco_server_ops import build_diloco_server_command

    cmd = build_diloco_server_command(
        output_dir="/tmp/out",
        num_workers=1,
        port=8512,
        wire_format="safetensors",
        grpc_enabled=True,
    )
    assert "--wire-format" in cmd
    assert cmd[cmd.index("--wire-format") + 1] == "safetensors"
    assert "--grpc" in cmd

    # Defaults (pickle / no gRPC) emit neither, keeping the argv readable.
    cmd2 = build_diloco_server_command(output_dir="/tmp/out", num_workers=1, port=8512)
    assert "--wire-format" not in cmd2
    assert "--grpc" not in cmd2


def test_server_job_params_to_command_threads_bulk_transport():
    """End-to-end orchestrator path: the CLI args -> _server_job_params dict ->
    build_diloco_server_command argv carries wire_format + grpc."""
    import argparse

    from forgather_server.diloco_server_ops import build_diloco_server_command

    from forgather.cli.diloco_orch import _server_job_params

    args = argparse.Namespace(
        output_dir="/tmp/out",
        port=8512,
        num_workers=2,
        host="127.0.0.1",
        save_every=10,
        save_total_limit=3,
        outer_lr=0.7,
        outer_momentum=0.9,
        no_nesterov=False,
        heartbeat_timeout=120,
        min_workers=1,
        sync_every=500,
        num_fragments=1,
        bf16_comm=None,
        wire_format="safetensors",
        grpc_enabled=True,
    )
    p = _server_job_params(args)
    assert p["wire_format"] == "safetensors"
    assert p["grpc_enabled"] is True
    # The scheduler hands these job_params keys to the command builder.
    cmd = build_diloco_server_command(
        output_dir=p["output_dir"],
        num_workers=p["num_workers"],
        port=p["port"],
        wire_format=p["wire_format"],
        grpc_enabled=p["grpc_enabled"],
    )
    assert cmd[cmd.index("--wire-format") + 1] == "safetensors"
    assert "--grpc" in cmd


def test_scheduler_forwards_bulk_transport_to_launcher(tmp_path):
    """The scheduler hop: _build_diloco_server reads wire_format/grpc_enabled
    from job_params and forwards them to spawn_diloco_server_process."""
    from unittest.mock import patch

    from forgather_server import scheduler

    item = type(
        "Item",
        (),
        {
            "job_params": {
                "output_dir": "/tmp/out",
                "num_workers": 2,
                "port": 8512,
                "no_auth": True,  # skip token-file resolution
                "wire_format": "safetensors",
                "grpc_enabled": True,
            }
        },
    )()
    with patch.object(scheduler.launcher, "spawn_diloco_server_process") as spawn:
        scheduler._build_diloco_server(item, [], tmp_path / "tty.log")
    assert spawn.call_args.kwargs["wire_format"] == "safetensors"
    assert spawn.call_args.kwargs["grpc_enabled"] is True


# ---------------------------------------------------------------------------
# Token injection into the training worker's env (issue #90 follow-up):
# a worker pointed at a routable (non-loopback) DiLoCo URL can't use the
# loopback per-port file, so the scheduler forwards the token via env.
# ---------------------------------------------------------------------------


def _fake_diloco_job(host, port, token, routable_host=None, status="running"):
    return type(
        "FakeRec",
        (),
        {
            "queue_id": f"q-{port}",
            "job_type": "diloco_server",
            "status": status,
            "auth_token": token,
            "job_params": {
                "host": host,
                "port": port,
                **({"routable_host": routable_host} if routable_host else {}),
            },
        },
    )()


def test_token_for_server_addr_matches_routable_url(monkeypatch):
    """Server binds 0.0.0.0, scheduler stamped routable_host; a training
    worker given the routable URL resolves the token by JobRecord match."""
    from forgather_server import scheduler

    monkeypatch.setattr(
        scheduler.job_records,
        "list_records",
        lambda: [_fake_diloco_job("0.0.0.0", 8512, "tok-routable", "192.168.9.43")],
    )
    assert (
        scheduler._diloco_token_for_server_addr("https://192.168.9.43:8512")
        == "tok-routable"
    )
    # Bare host:port (no scheme) resolves the same way.
    assert (
        scheduler._diloco_token_for_server_addr("192.168.9.43:8512") == "tok-routable"
    )


def test_token_for_server_addr_loopback_url(monkeypatch):
    from forgather_server import scheduler

    monkeypatch.setattr(
        scheduler.job_records,
        "list_records",
        lambda: [_fake_diloco_job("127.0.0.1", 8512, "tok-loop")],
    )
    assert (
        scheduler._diloco_token_for_server_addr("http://localhost:8512") == "tok-loop"
    )


def test_token_for_server_addr_no_match_returns_none(monkeypatch):
    """A genuinely remote server (no local record) → None; the worker
    relies on its own explicit token / env."""
    from forgather_server import scheduler

    monkeypatch.setattr(
        scheduler.job_records,
        "list_records",
        lambda: [_fake_diloco_job("0.0.0.0", 8512, "tok", "192.168.9.43")],
    )
    assert scheduler._diloco_token_for_server_addr("https://10.20.30.40:8512") is None


def test_token_for_server_addr_skips_terminal_records(monkeypatch):
    from forgather_server import scheduler

    monkeypatch.setattr(
        scheduler.job_records,
        "list_records",
        lambda: [_fake_diloco_job("127.0.0.1", 8512, "tok", status="done")],
    )
    assert scheduler._diloco_token_for_server_addr("http://localhost:8512") is None


def test_env_builder_injects_token(monkeypatch):
    """_diloco_env_from_job_params forwards FORGATHER_DILOCO_SERVER_TOKEN
    when a matching local server record exists."""
    from forgather_server import scheduler

    monkeypatch.setattr(
        scheduler.job_records,
        "list_records",
        lambda: [_fake_diloco_job("0.0.0.0", 8512, "tok-x", "192.168.9.43")],
    )
    env = scheduler._diloco_env_from_job_params(
        {"server_addr": "https://192.168.9.43:8512"}, "queue-1"
    )
    assert env["DILOCO_SERVER"] == "https://192.168.9.43:8512"
    assert env["FORGATHER_DILOCO_SERVER_TOKEN"] == "tok-x"


def test_env_builder_no_token_when_no_match(monkeypatch):
    from forgather_server import scheduler

    monkeypatch.setattr(scheduler.job_records, "list_records", lambda: [])
    env = scheduler._diloco_env_from_job_params(
        {"server_addr": "https://10.0.0.9:8512"}, "queue-1"
    )
    assert "FORGATHER_DILOCO_SERVER_TOKEN" not in env
