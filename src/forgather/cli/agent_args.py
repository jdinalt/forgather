"""Argument parser for ``forgather agent`` (drive the server's AI agent)."""

import argparse
from argparse import RawTextHelpFormatter


def create_agent_parser(global_args):
    parser = argparse.ArgumentParser(
        prog="forgather agent",
        description="Drive the forgather-server AI agent from the CLI (interactive testing)",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: $FORGATHER_SERVER_URL or http://127.0.0.1:8765)",
    )
    sub = parser.add_subparsers(dest="agent_subcommand", help="Agent subcommands")

    sub.add_parser("profiles", help="List connection profiles (which one is active)",
                   formatter_class=RawTextHelpFormatter)

    p_use = sub.add_parser("use", help="Activate a profile to test against",
                           formatter_class=RawTextHelpFormatter)
    p_use.add_argument("profile_id", help="Profile id (see 'forgather agent profiles')")

    sub.add_parser("status", help="Show the active agent's connection status",
                   formatter_class=RawTextHelpFormatter)

    p_msg = sub.add_parser(
        "message", help="Send a user message; stream the agent's turn",
        formatter_class=RawTextHelpFormatter,
        description="Send a message playing the user. Streams the agent's text /\n"
                    "tool calls / results until it finishes (often an answer or a\n"
                    "clarifying question) or pauses for approval. The first call\n"
                    "(no --session) prints a new session id to reuse for follow-ups.",
    )
    p_msg.add_argument("text", help="The user message")
    p_msg.add_argument("--session", default=None, help="Continue an existing session id")
    p_msg.add_argument("--json", action="store_true", help="Emit raw event JSONL")

    p_ap = sub.add_parser("approve", help="Approve a pending action (your call)",
                          formatter_class=RawTextHelpFormatter)
    p_ap.add_argument("action_id", help="Action id from an AWAITING_APPROVAL turn")
    p_ap.add_argument("--json", action="store_true", help="Emit raw event JSONL")

    p_rj = sub.add_parser("reject", help="Reject a pending action (your call)",
                          formatter_class=RawTextHelpFormatter)
    p_rj.add_argument("action_id", help="Action id from an AWAITING_APPROVAL turn")
    p_rj.add_argument("--reason", default=None, help="Why (printed for the transcript)")
    p_rj.add_argument("--json", action="store_true", help="Emit raw event JSONL")

    p_co = sub.add_parser("continue", help="Resume an incomplete turn",
                          formatter_class=RawTextHelpFormatter)
    p_co.add_argument("--session", default=None, help="Session id to resume")
    p_co.add_argument("--json", action="store_true", help="Emit raw event JSONL")

    p_ss = sub.add_parser("sessions", help="List active session ids + metadata",
                          formatter_class=RawTextHelpFormatter)
    p_ss.add_argument("--json", action="store_true", help="Emit raw JSON")

    p_hi = sub.add_parser("history", help="Dump a session's conversation",
                          formatter_class=RawTextHelpFormatter)
    p_hi.add_argument("session_id", help="Session id")
    p_hi.add_argument("--json", action="store_true", help="Emit raw JSON")

    return parser
