import { useEffect, useRef, useState } from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  ChatMessage,
  GenerationParams,
  runChatCompletion,
  streamChatCompletion,
} from "../inference-client";
import { persistGet, persistSet } from "../persist";
import { InferenceState } from "./InferencePanel";

interface Props {
  state: InferenceState;
}

type Status =
  | { kind: "idle" }
  | { kind: "streaming"; startedAt: number; tokens: number }
  | { kind: "generating"; startedAt: number }
  | { kind: "done"; tokens: number; durationMs: number }
  | { kind: "stopped"; tokens: number; durationMs: number }
  | { kind: "error"; message: string };

interface PersistedChat {
  systemText: string;
  systemOpen: boolean;
  messages: ChatMessage[];
}

const STORAGE_KEY = "forgather-inference-chat-v1";

function loadPersisted(): PersistedChat {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return { systemText: "", systemOpen: false, messages: [] };
  try {
    const parsed = JSON.parse(raw) as Partial<PersistedChat>;
    return {
      systemText:
        typeof parsed.systemText === "string" ? parsed.systemText : "",
      systemOpen: !!parsed.systemOpen,
      messages: Array.isArray(parsed.messages)
        ? (parsed.messages.filter(
            (m) =>
              m &&
              typeof (m as ChatMessage).content === "string" &&
              ((m as ChatMessage).role === "user" ||
                (m as ChatMessage).role === "assistant"),
          ) as ChatMessage[])
        : [],
    };
  } catch {
    return { systemText: "", systemOpen: false, messages: [] };
  }
}

export function InferenceChatPanel({ state }: Props) {
  const initial = loadPersisted();
  const [systemText, setSystemText] = useState(initial.systemText);
  const [systemOpen, setSystemOpen] = useState(initial.systemOpen);
  const [messages, setMessages] = useState<ChatMessage[]>(initial.messages);
  const [draft, setDraft] = useState("");
  const [stream, setStream] = useState(true);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  // Index into ``messages`` of the user turn currently being edited (or
  // null for normal compose). When set, Send becomes "Save & re-run":
  // the message at that index is replaced with the draft and everything
  // after it is dropped before re-running the conversation.
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const draftTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  // Track the previous busy state so we can restore textarea focus
  // exactly on the busy→idle transition. Without this, Ctrl+Enter sends
  // and the textarea's ``disabled`` flip drops focus, forcing the user
  // to click back in to keep chatting.
  const wasBusyRef = useRef(false);

  const busy = status.kind === "streaming" || status.kind === "generating";

  useEffect(() => {
    if (wasBusyRef.current && !busy) {
      draftTextareaRef.current?.focus();
    }
    wasBusyRef.current = busy;
  }, [busy]);

  // Persistence — write the parts that should survive a reload, drop
  // transient state (draft, status, editingIndex). One writer covers
  // every state change.
  useEffect(() => {
    const payload: PersistedChat = { systemText, systemOpen, messages };
    persistSet(STORAGE_KEY, JSON.stringify(payload));
  }, [systemText, systemOpen, messages]);

  // Keep the transcript pinned to the bottom while new tokens arrive,
  // and after a send/regenerate. Scrolling to the absolute bottom each
  // tick is fine here — the message list is short and there's no
  // infinite-scroll behavior to fight.
  useEffect(() => {
    const el = transcriptRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, status.kind]);

  /** Run the model against the supplied conversation. The caller is
   *  responsible for getting ``msgs`` into its desired pre-call shape
   *  (append a user turn, drop the trailing assistant turn for
   *  regenerate, etc.). This function appends the new assistant turn,
   *  drives streaming if requested, and updates status. */
  const runTurn = async (msgs: ChatMessage[]) => {
    const payload: ChatMessage[] = systemText.trim()
      ? [{ role: "system", content: systemText.trim() }, ...msgs]
      : msgs;
    const params: GenerationParams = stripEmpty(state.params);
    const ac = new AbortController();
    abortRef.current = ac;
    const started = Date.now();
    let tokenCount = 0;

    if (stream) {
      // Show a placeholder assistant bubble immediately so the UI
      // doesn't look frozen during the first-token latency.
      setMessages([...msgs, { role: "assistant", content: "" }]);
      setStatus({ kind: "streaming", startedAt: started, tokens: 0 });
      try {
        for await (const delta of streamChatCompletion(
          state.baseUrl,
          state.model,
          payload,
          params,
          ac.signal,
        )) {
          tokenCount += 1;
          setMessages((prev) => {
            // Append the delta to the trailing assistant message.
            const next = prev.slice();
            const last = next[next.length - 1];
            if (last && last.role === "assistant") {
              next[next.length - 1] = {
                role: "assistant",
                content: last.content + delta,
              };
            }
            return next;
          });
          setStatus({
            kind: "streaming",
            startedAt: started,
            tokens: tokenCount,
          });
        }
        setStatus({
          kind: "done",
          tokens: tokenCount,
          durationMs: Date.now() - started,
        });
      } catch (err) {
        if (ac.signal.aborted) {
          setStatus({
            kind: "stopped",
            tokens: tokenCount,
            durationMs: Date.now() - started,
          });
        } else {
          setStatus({
            kind: "error",
            message: err instanceof Error ? err.message : String(err),
          });
        }
      } finally {
        abortRef.current = null;
      }
    } else {
      setStatus({ kind: "generating", startedAt: started });
      try {
        const full = await runChatCompletion(
          state.baseUrl,
          state.model,
          payload,
          params,
          ac.signal,
        );
        setMessages([...msgs, { role: "assistant", content: full }]);
        setStatus({
          kind: "done",
          tokens: -1,
          durationMs: Date.now() - started,
        });
      } catch (err) {
        if (ac.signal.aborted) {
          setStatus({
            kind: "stopped",
            tokens: -1,
            durationMs: Date.now() - started,
          });
        } else {
          setStatus({
            kind: "error",
            message: err instanceof Error ? err.message : String(err),
          });
        }
      } finally {
        abortRef.current = null;
      }
    }
  };

  const onSend = () => {
    const text = draft.trim();
    if (!text || busy) return;

    if (editingIndex !== null) {
      // Truncate at the edited turn and replace it with the new text.
      const truncated = messages.slice(0, editingIndex);
      truncated.push({ role: "user", content: text });
      setMessages(truncated);
      setDraft("");
      setEditingIndex(null);
      void runTurn(truncated);
      return;
    }

    const next: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setDraft("");
    void runTurn(next);
  };

  const onStop = () => abortRef.current?.abort();

  const canRegenerate =
    !busy &&
    messages.length > 0 &&
    messages[messages.length - 1].role === "assistant";

  const onRegenerate = () => {
    if (!canRegenerate) return;
    // Drop the trailing assistant message and re-run from the previous
    // user turn. Streaming will append a fresh assistant bubble.
    const truncated = messages.slice(0, -1);
    setMessages(truncated);
    void runTurn(truncated);
  };

  const onReset = () => {
    if (busy) return;
    if (messages.length === 0 && !systemText) return;
    if (!window.confirm("Clear chat history and system message?")) return;
    setMessages([]);
    setSystemText("");
    setEditingIndex(null);
    setDraft("");
    setStatus({ kind: "idle" });
  };

  const onDeleteMessage = (index: number) => {
    if (busy) return;
    if (!window.confirm("Delete this message?")) return;
    setMessages((prev) => prev.filter((_, i) => i !== index));
    // If we were editing a message that no longer exists, exit edit mode.
    if (editingIndex !== null) {
      if (editingIndex === index) {
        setEditingIndex(null);
        setDraft("");
      } else if (editingIndex > index) {
        setEditingIndex(editingIndex - 1);
      }
    }
  };

  const onEditUser = (index: number) => {
    if (busy) return;
    const msg = messages[index];
    if (!msg || msg.role !== "user") return;
    setEditingIndex(index);
    setDraft(msg.content);
  };

  const onCancelEdit = () => {
    setEditingIndex(null);
    setDraft("");
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter inserts a newline; Ctrl/Cmd+Enter sends. This matches the
    // convention used by most modern chat UIs and keeps multi-line
    // prompts (code blocks, paragraphs) easy to compose.
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      onSend();
    }
  };

  const sendLabel = editingIndex !== null ? "Save & re-run" : "Send";

  return (
    <div className="inference-chat">
      <div className="inference-chat-bar">
        <button
          type="button"
          className="secondary inference-chat-system-toggle"
          onClick={() => setSystemOpen((v) => !v)}
          title="Edit the system message prepended to every request"
        >
          <span className="tri">{systemOpen ? "▾" : "▸"}</span>
          System
          {systemText.trim() && <span className="inference-chat-system-dot" />}
        </button>
        <label
          className="dyn-checkbox"
          title="Uncheck for modes incompatible with streaming, e.g. beam search"
        >
          <input
            type="checkbox"
            checked={stream}
            onChange={(e) => setStream(e.target.checked)}
            disabled={busy}
          />
          stream
        </label>
        <button
          type="button"
          className="secondary"
          onClick={onRegenerate}
          disabled={!canRegenerate}
          title="Re-run the last assistant turn"
        >
          Regenerate
        </button>
        <button
          type="button"
          className="secondary"
          onClick={onStop}
          disabled={!busy}
          title="Abort the in-flight request"
        >
          Stop
        </button>
        <button
          type="button"
          className="secondary"
          onClick={onReset}
          disabled={busy || (messages.length === 0 && !systemText)}
          title="Clear all chat history and the system message"
        >
          Reset
        </button>
        <div className="muted inference-status">
          <StatusLine status={status} />
        </div>
      </div>

      {systemOpen && (
        <div className="inference-chat-system">
          <textarea
            value={systemText}
            onChange={(e) => setSystemText(e.target.value)}
            placeholder="Optional system message — sent before every user turn."
            spellCheck={false}
            rows={3}
          />
        </div>
      )}

      <div className="inference-chat-messages" ref={transcriptRef}>
        {messages.length === 0 && (
          <div className="muted inference-chat-empty">
            No messages yet — type below and Send (or Ctrl+Enter).
          </div>
        )}
        {messages.map((msg, i) => (
          <Message
            key={i}
            msg={msg}
            disabled={busy}
            onDelete={() => onDeleteMessage(i)}
            onEdit={msg.role === "user" ? () => onEditUser(i) : undefined}
            isEditing={editingIndex === i}
          />
        ))}
      </div>

      <div className="inference-chat-input">
        <textarea
          ref={draftTextareaRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            editingIndex !== null
              ? "Edit your message — Save & re-run truncates the conversation here."
              : "Message the model. Ctrl+Enter to send."
          }
          spellCheck={false}
          disabled={busy}
          rows={3}
        />
        <div className="inference-chat-input-buttons">
          {editingIndex !== null && (
            <button
              type="button"
              className="secondary"
              onClick={onCancelEdit}
              disabled={busy}
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            onClick={onSend}
            disabled={busy || !draft.trim() || !state.baseUrl}
          >
            {sendLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

function Message({
  msg,
  disabled,
  onDelete,
  onEdit,
  isEditing,
}: {
  msg: ChatMessage;
  disabled: boolean;
  onDelete: () => void;
  onEdit?: () => void;
  isEditing: boolean;
}) {
  const cls =
    "inference-chat-msg inference-chat-msg-" +
    msg.role +
    (isEditing ? " editing" : "");
  return (
    <div className={cls}>
      <div className="inference-chat-msg-head">
        <span className="inference-chat-msg-role">{msg.role}</span>
        <span className="inference-chat-msg-actions">
          {onEdit && (
            <button
              type="button"
              className="tiny"
              onClick={onEdit}
              disabled={disabled}
              title="Edit and re-run from this point"
            >
              edit
            </button>
          )}
          <button
            type="button"
            className="tiny"
            onClick={onDelete}
            disabled={disabled}
            title="Delete this message"
          >
            ×
          </button>
        </span>
      </div>
      <div className="inference-chat-msg-body">
        {msg.role === "assistant" ? (
          msg.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.content}
            </ReactMarkdown>
          ) : (
            <span className="muted">…</span>
          )
        ) : (
          <pre className="inference-chat-msg-user-text">{msg.content}</pre>
        )}
      </div>
    </div>
  );
}

function StatusLine({ status }: { status: Status }) {
  const tokLabel = (n: number) =>
    n < 0 ? "— tokens" : `${n} token${n === 1 ? "" : "s"}`;
  switch (status.kind) {
    case "idle":
      return <span>ready</span>;
    case "streaming":
      return <span>streaming · {tokLabel(status.tokens)}</span>;
    case "generating":
      return <span>generating…</span>;
    case "done":
      return (
        <span>
          done · {tokLabel(status.tokens)} ·{" "}
          {(status.durationMs / 1000).toFixed(1)}s
        </span>
      );
    case "stopped":
      return (
        <span>
          stopped · {tokLabel(status.tokens)} ·{" "}
          {(status.durationMs / 1000).toFixed(1)}s
        </span>
      );
    case "error":
      return <span className="err">error · {status.message}</span>;
  }
}

/** Drop keys whose value is undefined, null, or an empty string / array
 *  so the server sees its per-model defaults. Mirrors the completion
 *  panel's helper. */
function stripEmpty(params: GenerationParams): GenerationParams {
  const out: GenerationParams = {};
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    if (typeof v === "string" && v === "") continue;
    if (Array.isArray(v) && v.length === 0) continue;
    (out as Record<string, unknown>)[k] = v;
  }
  return out;
}
