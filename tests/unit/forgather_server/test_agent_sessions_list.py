"""GET /api/agent/sessions list endpoint + session.list_conversations."""

from __future__ import annotations

from forgather_server.agent import session
from forgather_server.routes import agent as agent_routes


def test_sessions_list_endpoint_reports_seeded_sessions():
    c1 = session.get_or_create(None)
    c1.messages.append({"role": "user", "content": []})
    c1.touch()
    c2 = session.get_or_create(None)

    out = agent_routes.agent_sessions_list()
    by = {s["session_id"]: s for s in out["sessions"]}
    assert c1.session_id in by and c2.session_id in by
    assert by[c1.session_id]["message_count"] == 1
    assert by[c2.session_id]["awaiting_approval"] is False


def test_session_delete_endpoint():
    import pytest
    from fastapi import HTTPException

    c = session.get_or_create(None)
    out = agent_routes.agent_session_delete(c.session_id)
    assert out["deleted"] == c.session_id
    assert session.get_conversation(c.session_id) is None
    # Deleting again 404s.
    with pytest.raises(HTTPException):
        agent_routes.agent_session_delete(c.session_id)
