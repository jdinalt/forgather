"""Plumbing: --grace-period reaches the spawned diloco server argv."""


def _build(**kw):
    from forgather_server import diloco_server_ops

    return diloco_server_ops.build_diloco_server_command(
        output_dir="/tmp/m", num_workers=2, async_mode=True, **kw
    )


def test_grace_period_emitted_when_set():
    cmd = _build(grace_period=2.0)
    assert "--grace-period" in cmd
    assert cmd[cmd.index("--grace-period") + 1] == "2.0"


def test_grace_period_omitted_when_zero():
    assert "--grace-period" not in _build(grace_period=0.0)
    # default (no kwarg) is also off
    assert "--grace-period" not in _build()
