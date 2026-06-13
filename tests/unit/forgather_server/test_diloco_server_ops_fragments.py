"""Plumbing: --fragment-assignment reaches the spawned diloco server argv."""


def _build(**kw):
    from forgather_server import diloco_server_ops

    return diloco_server_ops.build_diloco_server_command(
        output_dir="/tmp/m", num_workers=2, async_mode=False, **kw
    )


def test_fragment_assignment_emitted_when_nondefault_and_streaming():
    cmd = _build(num_fragments=3, fragment_assignment="sequential")
    assert "--fragment-assignment" in cmd
    assert cmd[cmd.index("--fragment-assignment") + 1] == "sequential"


def test_fragment_assignment_omitted_when_default():
    # default 'strided' is the server default -> omitted (argv stays clean)
    assert "--fragment-assignment" not in _build(num_fragments=3)
    assert "--fragment-assignment" not in _build(
        num_fragments=3, fragment_assignment="strided"
    )


def test_fragment_assignment_omitted_without_streaming():
    # meaningless without fragments -> not emitted even if non-default
    assert "--fragment-assignment" not in _build(
        num_fragments=1, fragment_assignment="sequential"
    )
