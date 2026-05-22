import { useEffect, useRef, useState } from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  ChatMessage,
  GenerationParams,
  runChatCompletion,
  streamChatCompletion,
  tokenizeChat,
} from "../inference-client";
import { persistGet, persistSet } from "../persist";
import { InferenceState } from "./InferencePanel";

interface Props {
  state: InferenceState;
  // Hand the rendered prompt off to the completion panel and switch
  // to its tab — used by the "Send to completion" toolbar button.
  onSendToCompletion: (rendered: string) => void;
}

type Status =
  | { kind: "idle" }
  | { kind: "streaming"; startedAt: number; tokens: number }
  | { kind: "generating"; startedAt: number }
  | { kind: "done"; tokens: number; durationMs: number }
  | { kind: "stopped"; tokens: number; durationMs: number }
  | { kind: "error"; message: string };

/** How the Impersonate button drives the model into producing a
 *  user-voice turn.
 *  - ``prefix``: server-side prefix continuation — the inference
 *    server renders the chat template with ``continue_final_message=True``
 *    and an empty trailing user turn, so the model literally generates
 *    inside an opened user-role span. Most reliable; requires the
 *    Forgather inference server (uses the non-standard ``next_role``
 *    request field).
 *  - ``swap``: client-side role-swap — flip user↔assistant in the
 *    transcript so the model "completes" what looks like the next
 *    assistant turn (which is the user's voice on the swapped history).
 *  - ``prompt``: client-side system-prompt steering — keep roles
 *    intact and append a trailing system instruction asking the model
 *    to write the user's next message. The most conservative option;
 *    least reliable. */
type ImpersonateMode = "prefix" | "swap" | "prompt";

interface PersistedChat {
  systemText: string;
  systemOpen: boolean;
  settingsOpen: boolean;
  impersonateMode: ImpersonateMode;
  messages: ChatMessage[];
}

const STORAGE_KEY = "forgather-inference-chat-v1";

function loadPersisted(): PersistedChat {
  const fallback: PersistedChat = {
    systemText: "",
    systemOpen: false,
    settingsOpen: false,
    impersonateMode: "prefix",
    messages: [],
  };
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw) as Partial<PersistedChat>;
    const mode: ImpersonateMode =
      parsed.impersonateMode === "prompt"
        ? "prompt"
        : parsed.impersonateMode === "swap"
          ? "swap"
          : "prefix";
    return {
      systemText:
        typeof parsed.systemText === "string" ? parsed.systemText : "",
      systemOpen: !!parsed.systemOpen,
      settingsOpen: !!parsed.settingsOpen,
      impersonateMode: mode,
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
    return fallback;
  }
}

export function InferenceChatPanel({ state, onSendToCompletion }: Props) {
  const initial = loadPersisted();
  const [systemText, setSystemText] = useState(initial.systemText);
  const [systemOpen, setSystemOpen] = useState(initial.systemOpen);
  const [settingsOpen, setSettingsOpen] = useState(initial.settingsOpen);
  const [impersonateMode, setImpersonateMode] = useState<ImpersonateMode>(
    initial.impersonateMode,
  );
  const [messages, setMessages] = useState<ChatMessage[]>(initial.messages);
  const [draft, setDraft] = useState("");
  const [stream, setStream] = useState(true);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  // Index into ``messages`` of the user turn currently being edited (or
  // null when not editing). Editing happens inline in the message bubble
  // — the bubble renders its own textarea bound to ``editDraft`` and
  // Save & re-run / Cancel buttons. Saving truncates everything after
  // the edited turn and re-runs the conversation.
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");
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
    // Reasoning is per-turn scratch from the model's thinking trace; it
    // is intentionally not persisted (and the existing loadPersisted
    // type-filter would discard it anyway). Serialize only {role,content}.
    const payload: PersistedChat = {
      systemText,
      systemOpen,
      settingsOpen,
      impersonateMode,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
    };
    persistSet(STORAGE_KEY, JSON.stringify(payload));
  }, [systemText, systemOpen, settingsOpen, impersonateMode, messages]);

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
   *  regenerate, etc.). The new turn is appended with role ``asRole``
   *  (default ``assistant``); when ``extraSystem`` is set, that text is
   *  appended as a final ``system`` message — used by Impersonate to
   *  steer the model into producing a user-voice turn. */
  const runTurn = async (
    msgs: ChatMessage[],
    opts?: {
      asRole?: "assistant" | "user";
      extraSystem?: string;
      // What actually gets sent to the API; defaults to ``msgs``. Used
      // by Impersonate's role-swap mode to ship a transformed
      // transcript while keeping the displayed history untouched.
      apiMessages?: ChatMessage[];
      // Server-side prefix-continuation hint: the inference server
      // renders the chat template open at this role's turn so the
      // model generates as that role. Affects the wire request only,
      // not how we display the result.
      nextRole?: "assistant" | "user";
    },
  ) => {
    const asRole = opts?.asRole ?? "assistant";
    const apiBase = opts?.apiMessages ?? msgs;
    const reqOpts = opts?.nextRole ? { nextRole: opts.nextRole } : undefined;
    const payload: ChatMessage[] = [];
    if (systemText.trim()) {
      payload.push({ role: "system", content: systemText.trim() });
    }
    payload.push(...apiBase);
    if (opts?.extraSystem) {
      payload.push({ role: "system", content: opts.extraSystem });
    }
    const params: GenerationParams = stripEmpty(state.params);
    const ac = new AbortController();
    abortRef.current = ac;
    const started = Date.now();
    let tokenCount = 0;

    if (stream) {
      // Show a placeholder bubble immediately so the UI doesn't look
      // frozen during the first-token latency.
      setMessages([...msgs, { role: asRole, content: "" }]);
      setStatus({ kind: "streaming", startedAt: started, tokens: 0 });
      try {
        for await (const delta of streamChatCompletion(
          state.baseUrl,
          state.model,
          payload,
          params,
          ac.signal,
          reqOpts,
          state.authToken || undefined,
        )) {
          tokenCount += 1;
          setMessages((prev) => {
            // Append the delta to the trailing placeholder turn.
            // ``reasoning`` and ``content`` accumulate independently so
            // the bubble can render them in different styles. Note that
            // interleaved reasoning/content from the model is normalized
            // here: each kind is collapsed into a single contiguous blob
            // and the panel always renders reasoning above content. This
            // matches how vLLM's qwen3 reasoning parser actually emits
            // tokens in practice; parsers that genuinely interleave the
            // two would need an ordered segment list instead.
            const next = prev.slice();
            const idx = next.length - 1;
            const last = next[idx];
            if (last && last.role === asRole) {
              // Re-read via index so future intermediate mutations
              // above this line don't get silently overwritten by the
              // spread.
              if (delta.kind === "reasoning") {
                next[idx] = {
                  ...next[idx],
                  reasoning: (next[idx].reasoning ?? "") + delta.text,
                };
              } else {
                next[idx] = {
                  ...next[idx],
                  content: next[idx].content + delta.text,
                };
              }
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
        const result = await runChatCompletion(
          state.baseUrl,
          state.model,
          payload,
          params,
          ac.signal,
          reqOpts,
          state.authToken || undefined,
        );
        setMessages([
          ...msgs,
          {
            role: asRole,
            content: result.content,
            ...(result.reasoning ? { reasoning: result.reasoning } : {}),
          },
        ]);
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
    const next: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setDraft("");
    void runTurn(next);
  };

  const onSaveEdit = () => {
    if (editingIndex === null || busy) return;
    const text = editDraft.trim();
    if (!text) return;
    const target = messages[editingIndex];
    if (!target) return;
    if (target.role === "user") {
      // Editing a user turn changes the input — truncate everything
      // after this point and re-run the conversation.
      const truncated = messages.slice(0, editingIndex);
      truncated.push({ role: "user", content: text });
      setMessages(truncated);
      setEditingIndex(null);
      setEditDraft("");
      void runTurn(truncated);
    } else {
      // Editing an assistant turn rewrites the dialog history in place
      // — useful for steering future generations or crafting few-shot
      // context. No re-run; the user explicitly chose this content.
      setMessages((prev) =>
        prev.map((m, i) =>
          i === editingIndex ? { role: "assistant", content: text } : m,
        ),
      );
      setEditingIndex(null);
      setEditDraft("");
    }
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

  // Impersonate is enabled whenever there's something to predict from
  // and we're not busy. Most useful after an assistant turn (predict the
  // user's reply), but also fine on an empty / user-led conversation
  // where the system prompt sets up a roleplay scenario.
  const canImpersonate = !busy;

  const onImpersonate = () => {
    if (!canImpersonate) return;
    if (impersonateMode === "prefix") {
      // Server-side prefix continuation: pass ``next_role: "user"`` so
      // the inference server renders the chat template with
      // ``continue_final_message=True`` and a trailing empty user
      // turn. The model literally generates inside an opened user-role
      // span — the cleanest way to coerce a user-voice turn out of a
      // chat-tuned model.
      void runTurn(messages, { asRole: "user", nextRole: "user" });
      return;
    }
    if (impersonateMode === "swap") {
      // Role-swap: flip user↔assistant in the transcript so the model
      // generates what looks to it like the next assistant turn — but
      // on the swapped history, that's actually the user's voice.
      // System messages pass through untouched (they define the scene
      // for both sides). Display history stays unchanged via
      // ``apiMessages``.
      const swapped: ChatMessage[] = messages.map((m) =>
        m.role === "user"
          ? { role: "assistant", content: m.content }
          : m.role === "assistant"
            ? { role: "user", content: m.content }
            : m,
      );
      void runTurn(messages, { asRole: "user", apiMessages: swapped });
      return;
    }
    // ``prompt`` mode: keep roles intact, append a trailing system
    // instruction telling the model to write a user turn. More
    // conservative than role-swap; less reliable on weaker chat models.
    const instruction =
      "INSTRUCTION: The next message in this conversation is from the USER, not the assistant. " +
      "Generate only the user's next message, in the user's voice. " +
      "Output the message text directly — no role labels, no quotes, no narration, no commentary, no meta-text. " +
      "Stop after the user's message; do not continue with an assistant response.";
    void runTurn(messages, { asRole: "user", extraSystem: instruction });
  };

  // Continue picks up where the conversation ends without inserting a
  // new user turn — the natural follow-up to Impersonate, since the
  // model just wrote a user turn and we now want its assistant reply.
  const canContinue =
    !busy &&
    messages.length > 0 &&
    messages[messages.length - 1].role === "user";

  const onContinue = () => {
    if (!canContinue) return;
    void runTurn(messages);
  };

  // Render the current conversation through the inference server's
  // chat template (via /tokenize) and append the result to the
  // completion textarea, switching tabs on success. Sends the same
  // system text the chat completion path would send so what shows up
  // in completion is byte-identical to what the model would actually
  // see for a "Send" or "Continue" press in chat. Disabled while a
  // request is in flight.
  const canSendToCompletion = !busy;

  const onSendToCompletionClick = async () => {
    if (!canSendToCompletion || !state.baseUrl) return;
    const payload: ChatMessage[] = systemText.trim()
      ? [{ role: "system", content: systemText.trim() }, ...messages]
      : messages;
    try {
      const r = await tokenizeChat(
        state.baseUrl,
        state.model,
        payload,
        undefined,
        state.authToken || undefined,
      );
      const rendered = (r.prompt ?? "").toString();
      if (!rendered) {
        setStatus({
          kind: "error",
          message: "Server returned no prompt text — check inference server logs.",
        });
        return;
      }
      onSendToCompletion(rendered);
    } catch (err) {
      setStatus({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const onReset = () => {
    if (busy) return;
    if (messages.length === 0 && !systemText) return;
    if (!window.confirm("Clear chat history and system message?")) return;
    setMessages([]);
    setSystemText("");
    setEditingIndex(null);
    setEditDraft("");
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
        setEditDraft("");
      } else if (editingIndex > index) {
        setEditingIndex(editingIndex - 1);
      }
    }
  };

  const onEditMessage = (index: number) => {
    if (busy) return;
    const msg = messages[index];
    if (!msg) return;
    if (msg.role !== "user" && msg.role !== "assistant") return;
    setEditingIndex(index);
    setEditDraft(msg.content);
  };

  const onCancelEdit = () => {
    setEditingIndex(null);
    setEditDraft("");
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
        <button
          type="button"
          className="secondary inference-chat-system-toggle"
          onClick={() => setSettingsOpen((v) => !v)}
          title="Configure chat-panel behavior"
        >
          <span className="tri">{settingsOpen ? "▾" : "▸"}</span>
          Settings
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
          onClick={onImpersonate}
          disabled={!canImpersonate}
          title="Generate the next user turn (predict what the user would say)"
        >
          Impersonate
        </button>
        <button
          type="button"
          className="secondary"
          onClick={onContinue}
          disabled={!canContinue}
          title="Generate the next assistant turn from the current state"
        >
          Continue
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
          onClick={onSendToCompletionClick}
          disabled={!canSendToCompletion || !state.baseUrl}
          title="Render the conversation via the chat template and open it in the completion tab"
        >
          To completion
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

      {settingsOpen && (
        <div className="inference-chat-settings">
          <div className="inference-chat-settings-row">
            <label htmlFor="impersonate-mode">Impersonate mode</label>
            <select
              id="impersonate-mode"
              value={impersonateMode}
              onChange={(e) =>
                setImpersonateMode(e.target.value as ImpersonateMode)
              }
            >
              <option value="prefix">
                Prefix continuation (server-side, recommended)
              </option>
              <option value="swap">Role-swap (client-side)</option>
              <option value="prompt">
                System-prompt instruction (client-side)
              </option>
            </select>
            <span className="muted inference-chat-settings-hint">
              {impersonateMode === "prefix"
                ? "Sends next_role=user to the inference server, which renders the chat template open at a user turn (continue_final_message). Most reliable; requires Forgather's inference server."
                : impersonateMode === "swap"
                  ? "Flips user↔assistant in the transcript so the model writes the next 'assistant' turn — which is the user's voice on the swapped history. Pure client-side, works with any OpenAI-compatible server."
                  : "Keeps roles intact and appends a system instruction asking the model to write the next user turn. Pure client-side; least reliable."}
            </span>
          </div>
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
            onEdit={() => onEditMessage(i)}
            isEditing={editingIndex === i}
            editDraft={editDraft}
            onEditDraftChange={setEditDraft}
            onSaveEdit={onSaveEdit}
            onCancelEdit={onCancelEdit}
          />
        ))}
      </div>

      <div className="inference-chat-input">
        <textarea
          ref={draftTextareaRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Message the model. Ctrl+Enter to send."
          spellCheck={false}
          disabled={busy}
          rows={3}
        />
        <div className="inference-chat-input-buttons">
          <button
            type="button"
            onClick={onSend}
            disabled={busy || !draft.trim() || !state.baseUrl}
          >
            Send
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
  editDraft,
  onEditDraftChange,
  onSaveEdit,
  onCancelEdit,
}: {
  msg: ChatMessage;
  disabled: boolean;
  onDelete: () => void;
  onEdit?: () => void;
  isEditing: boolean;
  editDraft: string;
  onEditDraftChange: (v: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
}) {
  const editTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  // Focus the inline editor when it opens so the user can start typing
  // immediately. Caret goes to the end so they can append without
  // having to manually move it.
  useEffect(() => {
    if (isEditing && editTextareaRef.current) {
      const ta = editTextareaRef.current;
      ta.focus();
      const n = ta.value.length;
      ta.setSelectionRange(n, n);
    }
  }, [isEditing]);

  const onEditKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      onSaveEdit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onCancelEdit();
    }
  };

  const cls =
    "inference-chat-msg inference-chat-msg-" +
    msg.role +
    (isEditing ? " editing" : "");
  return (
    <div className={cls}>
      <div className="inference-chat-msg-head">
        <span className="inference-chat-msg-role">{msg.role}</span>
        <span className="inference-chat-msg-actions">
          {onEdit && !isEditing && (
            <button
              type="button"
              className="tiny"
              onClick={onEdit}
              disabled={disabled}
              title={
                msg.role === "user"
                  ? "Edit and re-run from this point"
                  : "Edit this assistant message in place"
              }
            >
              edit
            </button>
          )}
          {!isEditing && (
            <button
              type="button"
              className="tiny"
              onClick={onDelete}
              disabled={disabled}
              title="Delete this message"
            >
              ×
            </button>
          )}
        </span>
      </div>
      <div className="inference-chat-msg-body">
        {isEditing ? (
          <div className="inference-chat-msg-edit">
            <textarea
              ref={editTextareaRef}
              value={editDraft}
              onChange={(e) => onEditDraftChange(e.target.value)}
              onKeyDown={onEditKeyDown}
              spellCheck={false}
              rows={Math.max(3, editDraft.split("\n").length)}
            />
            <div className="inference-chat-msg-edit-buttons">
              <button
                type="button"
                className="secondary"
                onClick={onCancelEdit}
                disabled={disabled}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={onSaveEdit}
                disabled={disabled || !editDraft.trim()}
                title={
                  msg.role === "user"
                    ? "Truncate the conversation here and re-run (Ctrl+Enter)"
                    : "Replace this assistant message in place (Ctrl+Enter)"
                }
              >
                {msg.role === "user" ? "Save & re-run" : "Save"}
              </button>
            </div>
          </div>
        ) : msg.role === "assistant" ? (
          <>
            {msg.reasoning ? (
              <pre className="inference-chat-msg-reasoning">
                {msg.reasoning}
              </pre>
            ) : null}
            {msg.content ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            ) : msg.reasoning ? null : (
              <span className="muted">…</span>
            )}
          </>
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
