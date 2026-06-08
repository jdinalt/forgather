"""Agent playbook: task-specific procedures the agent retrieves on demand.

Keeping every task's how-to in the system prompt taxes context on every request
(badly so for limited-context local models). Instead the base prompt stays lean
and points here; the agent calls ``read_playbook(topic)`` for the procedure when
it actually starts that task (the knowledge analogue of the tool-disclosure
``tool_help``).

Entries are markdown files under ``playbook/<topic>.md`` — extensible and
reviewable without code changes. The first non-blank line of each file (its
``# topic — summary`` heading) is the one-line summary shown by list_playbook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_DIR = Path(__file__).parent / "playbook"


def _summary(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s
    return fallback


def topics() -> List[Dict[str, str]]:
    """[{topic, summary}] for every playbook entry, sorted by topic."""
    out: List[Dict[str, str]] = []
    if not _DIR.is_dir():
        return out
    for p in sorted(_DIR.glob("*.md")):
        out.append({"topic": p.stem, "summary": _summary(p.read_text(), p.stem)})
    return out


def read(topic: str) -> str:
    """Full markdown for one topic; ValueError (listing topics) if unknown."""
    topic = (topic or "").strip()
    p = _DIR / f"{topic}.md"
    # Guard against path tricks — only a bare topic name maps to a file.
    if "/" in topic or "\\" in topic or not p.is_file():
        avail = ", ".join(t["topic"] for t in topics())
        raise ValueError(f"unknown playbook topic {topic!r}; available: {avail}")
    return p.read_text()
