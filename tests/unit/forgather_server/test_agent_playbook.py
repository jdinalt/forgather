"""Agent playbook: loader + list_playbook / read_playbook meta tools."""

from __future__ import annotations

import pytest

from forgather_server.agent import playbook, tools_meta
from forgather_server.agent.registry import META, READ, ToolRegistry

_EXPECTED = {
    "configs", "training", "datasets", "results", "evaluation",
    "services", "inference", "diloco", "filesystem",
}


def test_topics_present_with_summaries():
    topics = {t["topic"]: t["summary"] for t in playbook.topics()}
    assert _EXPECTED <= set(topics)
    for t in _EXPECTED:
        assert topics[t] and topics[t] != t  # a real one-line summary, not just the stem


def test_read_returns_content():
    txt = playbook.read("training")
    assert "run_train" in txt and "resolve_output_dir" in txt


def test_read_unknown_raises_listing_topics():
    with pytest.raises(ValueError) as e:
        playbook.read("nope")
    assert "available" in str(e.value) and "training" in str(e.value)


def test_read_rejects_path_traversal():
    with pytest.raises(ValueError):
        playbook.read("../runtime")


def test_meta_tools_registered():
    reg = ToolRegistry()
    tools_meta.register_all(reg)
    by = {s.name: s for s in reg.specs()}
    for name in ("list_playbook", "read_playbook"):
        assert by[name].risk == READ and by[name].tier == META


def test_read_playbook_handler():
    reg = ToolRegistry()
    tools_meta.register_all(reg)
    by = {s.name: s for s in reg.specs()}
    out = by["read_playbook"].handler({"topic": "inference"})
    assert out["topic"] == "inference" and "from_checkpoint" in out["content"]
    topics = by["list_playbook"].handler({})["topics"]
    assert _EXPECTED <= {t["topic"] for t in topics}
    with pytest.raises(ValueError):
        by["read_playbook"].handler({"topic": ""})


def test_prompt_points_at_playbook():
    from forgather_server.agent import runtime

    sp = runtime.SYSTEM_PROMPT
    assert "read_playbook" in sp and "list_playbook" in sp
    # Judgment-based nudge: read the playbook before the consequential action,
    # not mechanically first (rigid "first call" was a net negative in testing).
    assert "consequential action" in sp
    # The prompt is lean — the long per-task procedures are gone.
    assert len(sp) < 6000
