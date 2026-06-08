"""``forgather agent`` — drive the server's in-process AI agent from the CLI.

A thin client over the existing ``/api/agent/*`` endpoints, built for
interactive testing: send a message playing the user, watch the agent's
text / tool calls / results stream, and — at every approval gate — make the
Approve/Reject call yourself (no auto-approve). Conversation + pending-action
state lives in the running server, so each command here is one step:

    forgather agent profiles                      # list connection profiles
    forgather agent use <profile_id>              # activate one to test against
    forgather agent message "build the X dataset" # start a turn -> prints session id
    forgather agent approve <action_id>           # your call on a proposed action
    forgather agent reject  <action_id> --reason "use config Y instead"
    forgather agent message --session <id> "..."  # follow-up guidance
    forgather agent sessions                       # list active session ids
    forgather agent history <id>                  # dump the conversation

Each turn streams until the agent finishes (done — often an answer or a
clarifying question) or pauses for approval; a final STATE: line says which,
and what to run next. Needs a running ``forgather server``; auth + base URL are
resolved the same way as every other forgather CLI client.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from .server_client import ServerClient


def _clip(text: Any, n: int = 600) -> str:
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    s = s.replace("\n", "\n    ")
    return s if len(s) <= n else s[:n] + f" …[+{len(s) - n} chars]"


def _stream(client: ServerClient, path: str, body: Dict[str, Any], *, as_json: bool) -> Dict[str, Any]:
    """POST an SSE endpoint and render its event stream. Returns collected
    state: {session_id, pending:[action_card,...], done, error, usage}."""
    import requests

    url = client._url(path)
    try:
        r = client.session.post(url, json=body, stream=True, timeout=(10, None))
    except (requests.ConnectionError, requests.Timeout):
        print(f"could not reach forgather-server at {client.base}; is it running? "
              "(start with: forgather server)", file=sys.stderr)
        return {"error": "unreachable"}
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        print(f"server: {detail}", file=sys.stderr)
        return {"error": detail}

    state: Dict[str, Any] = {"session_id": body.get("session_id"), "pending": [],
                             "done": None, "error": None, "usage": None}
    text_buf: List[str] = []

    def flush_text():
        if text_buf:
            print("assistant> " + "".join(text_buf).strip())
            text_buf.clear()

    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        ev = json.loads(raw[5:].lstrip())
        if as_json:
            print(json.dumps(ev))
        t = ev.get("type")
        if t in ("session", "awaiting_approval", "recorded", "action_resolved", "done"):
            state["session_id"] = ev.get("session_id") or state["session_id"]
        if as_json:
            if t == "action_card":
                state["pending"].append(ev)
            elif t == "done":
                state["done"] = ev
            elif t == "error":
                state["error"] = ev.get("message")
            continue
        # Human-readable rendering.
        if t == "text":
            text_buf.append(ev.get("text", ""))
        elif t == "tool_use":
            flush_text()
            print(f"  -> {ev.get('name')}({json.dumps(ev.get('input') or {}, default=str)})")
        elif t == "tool_result":
            flush_text()
            tag = "ERROR " if ev.get("is_error") else ""
            print(f"     <- {tag}{_clip(ev.get('content'))}")
        elif t == "ui_directive":
            flush_text()
            print(f"  ~ ui:{ev.get('action')} {json.dumps(ev.get('payload') or {}, default=str)}")
        elif t == "action_card":
            flush_text()
            state["pending"].append(ev)
            extra = ev.get("extra") or {}
            line = f"  [?] APPROVAL action={ev.get('action_id')} risk={ev.get('risk')}: {ev.get('title')}"
            print(line)
            if ev.get("summary"):
                print(f"      {ev['summary']}")
            if extra.get("command"):
                print(f"      command: {extra['command']}")
            else:
                shown = {k: v for k, v in extra.items() if v is not None and k != "warning"}
                if shown:
                    print(f"      args: {_clip(shown)}")
            if ev.get("before") is not None or ev.get("after") is not None:
                print(f"      diff: {ev.get('path')} ({len(ev.get('before') or '')} -> {len(ev.get('after') or '')} chars)")
        elif t == "action_resolved":
            print(f"  = resolved {ev.get('action_id')} approved={ev.get('approved')}"
                  + (f" -> {_clip(ev.get('result'))}" if ev.get("result") else "")
                  + (f" ERROR {ev.get('error')}" if ev.get("error") else ""))
        elif t == "usage":
            state["usage"] = ev
        elif t == "awaiting_approval":
            pass  # the action_card(s) already printed; outstanding list is implied
        elif t == "done":
            flush_text()
            state["done"] = ev
        elif t == "error":
            flush_text()
            state["error"] = ev.get("message")
            print(f"ERROR: {ev.get('message')}", file=sys.stderr)
    flush_text()

    _print_state(state)
    return state


def _print_state(state: Dict[str, Any]) -> None:
    sid = state.get("session_id")
    u = state.get("usage")
    usage = ""
    if u:
        usage = (f"  [tokens in={u.get('input_tokens')} out={u.get('output_tokens')}"
                 f" ctx={u.get('context_window')}]")
    # Pending approvals win — that's the agent waiting on YOU.
    outstanding = [c for c in state.get("pending", [])]
    # Drop any that were resolved in this same stream is not tracked here; the
    # server enforces correctness, so just report what was proposed.
    if outstanding and state.get("done") is None:
        ids = " ".join(c.get("action_id") for c in outstanding)
        print(f"\nSTATE: session={sid}  AWAITING_APPROVAL  actions=[{ids}]{usage}")
        print("  -> decide with: forgather agent approve <action_id>   (or: "
              "reject <action_id> --reason \"...\")")
    elif state.get("error"):
        print(f"\nSTATE: session={sid}  ERROR: {state['error']}{usage}")
    else:
        d = state.get("done") or {}
        reason = d.get("reason")
        if d.get("incomplete"):
            print(f"\nSTATE: session={sid}  INCOMPLETE (reason={reason}){usage}")
            print(f"  -> resume with: forgather agent continue --session {sid}")
        else:
            print(f"\nSTATE: session={sid}  DONE (reason={reason}){usage}")
            print(f"  -> reply with:  forgather agent message --session {sid} \"...\"")


# ---- subcommands -----------------------------------------------------------


def _cmd_profiles(client: ServerClient, args) -> int:
    data = client._get("/agent/profiles").json()
    active = data.get("active_id")
    profiles = data.get("profiles") or []
    if not profiles:
        print("no agent profiles configured (add one in the webui: Agent -> settings)")
        return 0
    print(f"{'':2}{'id':<14} {'provider':<10} {'model':<28} base_url")
    for p in profiles:
        mark = "* " if p.get("id") == active else "  "
        print(f"{mark}{p.get('id', ''):<14} {p.get('provider', ''):<10} "
              f"{(p.get('model') or '(auto)'):<28} {p.get('base_url') or '(default)'}"
              f"   {p.get('label', '')}")
    print(f"\nactive: {active or '(none)'}")
    return 0


def _cmd_use(client: ServerClient, args) -> int:
    client._post(f"/agent/profiles/{args.profile_id}/activate")
    print(f"activated profile {args.profile_id}")
    st = client._get("/agent/status").json()
    print(f"agent: provider={st.get('provider')} model={st.get('model') or '(auto)'} "
          f"base_url={st.get('base_url') or '(default)'} "
          f"disclosure={st.get('disclosure_mode')}")
    return 0


def _cmd_status(client: ServerClient, args) -> int:
    st = client._get("/agent/status").json()
    print(json.dumps(st, indent=2))
    return 0


def _cmd_message(client: ServerClient, args) -> int:
    body = {"message": args.text}
    if args.session:
        body["session_id"] = args.session
    state = _stream(client, "/agent/message", body, as_json=args.json)
    return 1 if state.get("error") else 0


def _cmd_approve(client: ServerClient, args) -> int:
    state = _stream(client, "/agent/approve", {"action_id": args.action_id}, as_json=args.json)
    return 1 if state.get("error") else 0


def _cmd_reject(client: ServerClient, args) -> int:
    # The server's reject takes only the action_id; surface the reason locally
    # so the tester's intent is captured in the transcript/output.
    if args.reason:
        print(f"(reject reason: {args.reason})")
    state = _stream(client, "/agent/reject", {"action_id": args.action_id}, as_json=args.json)
    return 1 if state.get("error") else 0


def _cmd_continue(client: ServerClient, args) -> int:
    if not args.session:
        print("--session is required for continue", file=sys.stderr)
        return 2
    state = _stream(client, "/agent/continue", {"session_id": args.session}, as_json=args.json)
    return 1 if state.get("error") else 0


def _cmd_sessions(client: ServerClient, args) -> int:
    import time

    data = client._get("/agent/sessions").json()
    sessions = data.get("sessions") or []
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    if not sessions:
        print('no active sessions (start one: forgather agent message "...")')
        return 0
    print(f"{'session_id':<38} {'msgs':>4} {'awaiting':>8}  updated")
    for s in sessions:
        upd = s.get("updated_at")
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(upd)) if upd else "-"
        print(f"{s.get('session_id', ''):<38} {s.get('message_count', 0):>4} "
              f"{str(bool(s.get('awaiting_approval'))):>8}  {when}")
    return 0


def _cmd_history(client: ServerClient, args) -> int:
    data = client._get(f"/agent/sessions/{args.session_id}").json()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    for m in data.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            print(f"{role}> {_clip(content)}")
            continue
        for block in content or []:
            bt = block.get("type") if isinstance(block, dict) else None
            if bt == "text":
                print(f"{role}> {_clip(block.get('text'))}")
            elif bt == "tool_use":
                print(f"{role}>   -> {block.get('name')}({json.dumps(block.get('input') or {}, default=str)})")
            elif bt == "tool_result":
                print(f"{role}>   <- {_clip(block.get('content'))}")
    if data.get("awaiting_approval"):
        print(f"\n(awaiting approval: {data['awaiting_approval']})")
    return 0


def agent_cmd(args) -> int:
    sub = getattr(args, "agent_subcommand", None)
    if not sub:
        print("usage: forgather agent {profiles|use|status|message|approve|reject|"
              "continue|sessions|history} ... (forgather agent --help)", file=sys.stderr)
        return 2
    client = ServerClient.from_args(args)
    handlers = {
        "profiles": _cmd_profiles,
        "use": _cmd_use,
        "status": _cmd_status,
        "message": _cmd_message,
        "approve": _cmd_approve,
        "reject": _cmd_reject,
        "continue": _cmd_continue,
        "sessions": _cmd_sessions,
        "history": _cmd_history,
    }
    try:
        return handlers[sub](client, args)
    except Exception as e:  # AuthRequired / ServerUnreachable / RuntimeError
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
