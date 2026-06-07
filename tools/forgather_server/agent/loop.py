"""Provider-agnostic agent loop + server-side approval gate.

The loop drives the classic tool-use cycle: stream an assistant turn,
execute the tool calls it emits, feed the results back, repeat until a
turn emits no tool calls (done), an approval is required (pause), or the
iteration cap is hit.

The **approval gate is enforced here, server-side** — never in the
browser. ``read`` tools run automatically. ``propose``/``confirm`` tools
only ever have their *preview* computed inline (no side effect); the
``commit`` closure is stashed in a ``PendingApproval`` and run later, by
``apply_decision``, after the user approves. The model cannot make a
change permanent on its own: a paused turn owes a tool_result for every
tool_use it emitted, and the loop refuses to call the provider again until
every outstanding approval is resolved.

A tool call that names an unknown tool, fails schema validation, or
carries malformed JSON (``ToolCall.parse_error``) is returned to the model
as an *error tool_result* — it never crashes the turn. This tolerance is
what makes a flakier local vLLM model safe to drive.

The loop emits a stream of plain-dict *agent events* (``text``,
``tool_use``, ``tool_result``, ``action_card``, ``awaiting_approval``,
``usage``, ``done``, ``error``) that ``routes/agent.py`` serializes to SSE.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .providers.base import ChatProvider, Done, TextDelta, ToolCall, Usage
from .registry import READ, Proposal, ToolRegistry
from .session import (
    Conversation,
    PendingApproval,
    PendingTurn,
    get_conversation,
    get_turn_lock,
    new_action_id,
    peek_pending,
    pop_pending,
    register_pending,
)

log = logging.getLogger("forgather_server.agent.loop")

# Cap a single tool_result fed back to the model so a giant file dump
# doesn't blow the context. Generous; read tools that can be large
# (render_pp) should paginate at the tool layer if needed.
_MAX_RESULT_CHARS = 60_000


class AgentLoop:
    def __init__(
        self,
        provider: ChatProvider,
        registry: ToolRegistry,
        *,
        system: Optional[str] = None,
        max_iterations: int = 12,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.system = system
        self.max_iterations = max_iterations

    # ---- public entry points -------------------------------------------

    async def run_user_message(
        self, conv: Conversation, text: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Append a user message and run the agentic loop to a stopping point.

        Holds the per-conversation turn lock for the whole turn so a
        concurrent request on the same session (sidebar + full view share a
        session_id) can't interleave mutations of the message list.
        """
        async with get_turn_lock(conv.session_id):
            if conv.pending_turn is not None:
                yield {
                    "type": "error",
                    "message": "this conversation is awaiting approval of a "
                    "proposed change; approve or reject it first",
                }
                return
            conv.messages.append(
                {"role": "user", "content": [{"type": "text", "text": text}]}
            )
            conv.touch()
            async for ev in self._run_turns(conv):
                yield ev

    async def continue_turn(
        self, conv: Conversation
    ) -> AsyncIterator[Dict[str, Any]]:
        """Resume a turn that ended incomplete (max_tokens / iteration cap).

        If the conversation ended on an assistant message (output truncated),
        nudge it forward with a minimal user turn; if it ended on tool results
        (iteration cap), just re-run the loop. No-op error if there's nothing
        to continue or an approval is pending.
        """
        async with get_turn_lock(conv.session_id):
            if conv.pending_turn is not None:
                yield {
                    "type": "error",
                    "message": "this conversation is awaiting approval; "
                    "approve or reject it first",
                }
                return
            if not conv.messages:
                yield {"type": "error", "message": "nothing to continue"}
                return
            if conv.messages[-1].get("role") == "assistant":
                conv.messages.append(
                    {"role": "user", "content": [{"type": "text", "text": "Please continue."}]}
                )
            conv.touch()
            async for ev in self._run_turns(conv):
                yield ev

    async def apply_decision(
        self, action_id: str, *, approve: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Resolve one pending approval; resume the turn when the last clears.

        On approve, runs the stashed ``commit`` closure (the only place a
        propose/confirm side effect ever happens). On reject, feeds an error
        tool_result back so the model can adapt. Either way, once the turn
        has no outstanding approvals, the loop resumes and streams the
        continuation; otherwise it emits a ``recorded`` event and stops.

        Peeks the approval first and only *consumes* it once it holds the
        turn lock and has re-confirmed the conversation still awaits it — so
        a racing reset bails out without dropping a still-valid approval.
        """
        approval = peek_pending(action_id)
        if approval is None:
            yield {"type": "error", "message": f"no such pending action: {action_id}"}
            return
        conv = get_conversation(approval.session_id)
        if conv is None or conv.pending_turn is None:
            yield {"type": "error", "message": "the conversation is no longer awaiting this action"}
            return

        async with get_turn_lock(conv.session_id):
            # Re-validate under the lock, then consume the approval.
            approval = pop_pending(action_id)
            if approval is None:
                yield {"type": "error", "message": f"no such pending action: {action_id}"}
                return
            tool_use_id = approval.tool_use_id
            if conv.pending_turn is None or tool_use_id not in conv.pending_turn.outstanding:
                yield {
                    "type": "error",
                    "message": "the conversation is no longer awaiting this action",
                }
                return
            async for ev in self._resolve_one(conv, action_id, approval, approve):
                yield ev

    async def _resolve_one(
        self, conv, action_id: str, approval, approve: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        tool_use_id = approval.tool_use_id
        if approve:
            try:
                result_str = await self._run_commit(approval.proposal)
                conv.pending_turn.results[tool_use_id] = self.provider.format_tool_result(
                    tool_use_id, self._clip(result_str)
                )
                yield {
                    "type": "action_resolved",
                    "action_id": action_id,
                    "approved": True,
                    "result": result_str,
                }
            except Exception as e:  # commit failed — tell the model, don't crash
                log.exception("commit failed for action %s", action_id)
                conv.pending_turn.results[tool_use_id] = self.provider.format_tool_result(
                    tool_use_id, f"commit failed: {type(e).__name__}: {e}", is_error=True
                )
                yield {
                    "type": "action_resolved",
                    "action_id": action_id,
                    "approved": True,
                    "error": f"{type(e).__name__}: {e}",
                }
        else:
            conv.pending_turn.results[tool_use_id] = self.provider.format_tool_result(
                tool_use_id,
                "The user rejected this proposed change. Do not retry it "
                "without a different approach; ask what they would prefer.",
                is_error=True,
            )
            yield {"type": "action_resolved", "action_id": action_id, "approved": False}

        conv.pending_turn.outstanding.discard(tool_use_id)
        conv.touch()

        if conv.pending_turn.outstanding:
            yield {
                "type": "recorded",
                "session_id": conv.session_id,
                "outstanding": sorted(conv.pending_turn.outstanding),
            }
            return

        # All approvals for this turn are resolved — finalize and resume.
        self._flush_pending_turn(conv)
        async for ev in self._run_turns(conv):
            yield ev

    # ---- core loop -----------------------------------------------------

    async def _run_turns(self, conv: Conversation) -> AsyncIterator[Dict[str, Any]]:
        tools = self.registry.anthropic_tools()
        for _ in range(self.max_iterations):
            text_parts: List[str] = []
            tool_calls: List[ToolCall] = []
            stop_reason: Optional[str] = None

            try:
                async for ev in self.provider.stream_turn(
                    conv.messages, tools, system=self.system
                ):
                    if isinstance(ev, TextDelta):
                        text_parts.append(ev.text)
                        yield {"type": "text", "text": ev.text}
                    elif isinstance(ev, ToolCall):
                        tool_calls.append(ev)
                    elif isinstance(ev, Usage):
                        yield {
                            "type": "usage",
                            "input_tokens": ev.input_tokens,
                            "output_tokens": ev.output_tokens,
                            "context_window": ev.context_window,
                        }
                    elif isinstance(ev, Done):
                        stop_reason = ev.stop_reason
                        break
            except Exception as e:
                log.exception("provider stream failed")
                yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
                return

            assistant_msg = self._assistant_message(text_parts, tool_calls)

            if not tool_calls:
                conv.messages.append(assistant_msg)
                conv.touch()
                # ``incomplete`` lets the UI offer a "Continue" control when the
                # model's output was cut off by the token budget rather than
                # finishing on its own (max_tokens vs end_turn).
                yield {
                    "type": "done",
                    "session_id": conv.session_id,
                    "reason": stop_reason,
                    "incomplete": stop_reason == "max_tokens",
                }
                return

            pending_turn = PendingTurn(assistant_message=assistant_msg)
            for tc in tool_calls:
                yield {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
                async for ev in self._handle_tool_call(conv, tc, pending_turn):
                    yield ev

            if pending_turn.outstanding:
                conv.pending_turn = pending_turn
                conv.touch()
                yield {
                    "type": "awaiting_approval",
                    "session_id": conv.session_id,
                    "outstanding": sorted(pending_turn.outstanding),
                }
                return

            # Every tool was read-only (or errored inline) — append and loop.
            conv.messages.append(assistant_msg)
            conv.messages.append(self._results_message(assistant_msg, pending_turn.results))
            conv.touch()

        # Hit the tool-iteration cap mid-work. Not an error — surface it as an
        # incomplete turn so the UI can offer "Continue" (which resumes the
        # loop from the tool results already in the conversation).
        yield {
            "type": "done",
            "session_id": conv.session_id,
            "reason": "max_iterations",
            "incomplete": True,
        }

    async def _handle_tool_call(
        self, conv: Conversation, tc: ToolCall, pending_turn: PendingTurn
    ) -> AsyncIterator[Dict[str, Any]]:
        spec = self.registry.get(tc.name)
        if spec is None:
            async for ev in self._record_error(tc, pending_turn, f"unknown tool: {tc.name!r}"):
                yield ev
            return
        if tc.parse_error:
            async for ev in self._record_error(
                tc, pending_turn, f"could not parse tool arguments: {tc.parse_error}"
            ):
                yield ev
            return
        missing = self._missing_required(spec.json_schema, tc.arguments)
        if missing:
            async for ev in self._record_error(
                tc, pending_turn, f"missing required argument(s): {', '.join(missing)}"
            ):
                yield ev
            return

        if spec.risk == READ:
            try:
                out = await self._call(spec.handler, tc.arguments)
            except Exception as e:
                log.exception("read tool %s failed", tc.name)
                async for ev in self._record_error(
                    tc, pending_turn, f"{type(e).__name__}: {e}"
                ):
                    yield ev
                return
            content = self._clip(self._stringify(out))
            pending_turn.results[tc.id] = self.provider.format_tool_result(tc.id, content)
            yield {
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": content,
                "is_error": False,
            }
            return

        # propose / confirm — compute the preview only, gate the commit.
        try:
            proposal = await self._call(spec.handler, tc.arguments)
        except Exception as e:
            log.exception("propose tool %s failed", tc.name)
            async for ev in self._record_error(tc, pending_turn, f"{type(e).__name__}: {e}"):
                yield ev
            return
        if not isinstance(proposal, Proposal):
            async for ev in self._record_error(
                tc, pending_turn, "tool did not return a Proposal (internal error)"
            ):
                yield ev
            return

        action_id = new_action_id()
        register_pending(
            PendingApproval(
                action_id=action_id,
                session_id=conv.session_id,
                tool_use_id=tc.id,
                tool_name=tc.name,
                risk=spec.risk,
                proposal=proposal,
            )
        )
        pending_turn.outstanding.add(tc.id)
        yield {"type": "action_card", **proposal.to_card(action_id, spec.risk)}

    # ---- helpers -------------------------------------------------------

    async def _record_error(
        self, tc: ToolCall, pending_turn: PendingTurn, message: str
    ) -> AsyncIterator[Dict[str, Any]]:
        pending_turn.results[tc.id] = self.provider.format_tool_result(
            tc.id, message, is_error=True
        )
        yield {
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": message,
            "is_error": True,
        }

    def _flush_pending_turn(self, conv: Conversation) -> None:
        pt = conv.pending_turn
        assert pt is not None
        conv.messages.append(pt.assistant_message)
        conv.messages.append(self._results_message(pt.assistant_message, pt.results))
        conv.pending_turn = None

    @staticmethod
    def _assistant_message(
        text_parts: List[str], tool_calls: List[ToolCall]
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        text = "".join(text_parts)
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            )
        return {"role": "assistant", "content": content}

    @staticmethod
    def _results_message(
        assistant_msg: Dict[str, Any], results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Order results to match the assistant's tool_use order — the
        # Messages API expects a tool_result for every tool_use, and order
        # keeps things readable.
        order = [
            b["id"] for b in assistant_msg["content"] if b.get("type") == "tool_use"
        ]
        blocks = [results[tid] for tid in order if tid in results]
        return {"role": "user", "content": blocks}

    async def _call(self, fn, arguments: Dict[str, Any]) -> Any:
        res = fn(arguments)
        if inspect.isawaitable(res):
            res = await res
        return res

    async def _run_commit(self, proposal: Proposal) -> str:
        if proposal.commit is None:
            return "(no-op: nothing to apply)"
        res = proposal.commit()
        if inspect.isawaitable(res):
            res = await res
        return res if isinstance(res, str) else self._stringify(res)

    @staticmethod
    def _missing_required(schema: Dict[str, Any], arguments: Dict[str, Any]) -> List[str]:
        required = schema.get("required") or []
        return [k for k in required if k not in arguments]

    @staticmethod
    def _stringify(out: Any) -> str:
        if isinstance(out, str):
            return out
        try:
            return json.dumps(out, default=str, indent=2)
        except (TypeError, ValueError):
            return str(out)

    @staticmethod
    def _clip(text: str) -> str:
        if len(text) <= _MAX_RESULT_CHARS:
            return text
        return text[:_MAX_RESULT_CHARS] + f"\n... [truncated {len(text) - _MAX_RESULT_CHARS} chars]"
