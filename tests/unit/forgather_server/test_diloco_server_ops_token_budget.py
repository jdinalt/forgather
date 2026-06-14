"""Plumbing: --token-budget reaches the spawned diloco server argv."""


def _build(**kw):
    from forgather_server import diloco_server_ops

    return diloco_server_ops.build_diloco_server_command(
        output_dir="/tmp/m", num_workers=2, **kw
    )


def test_token_budget_emitted_when_set():
    cmd = _build(token_budget=1_000_000)
    assert "--token-budget" in cmd
    assert cmd[cmd.index("--token-budget") + 1] == "1000000"


def test_token_budget_omitted_when_zero():
    assert "--token-budget" not in _build(token_budget=0)
    assert "--token-budget" not in _build()
