import { useEffect, useRef, useState } from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { useDemoMode } from "../demoMode";
import {
  ChatMessage,
  GenerationParams,
  detokenizeTokens,
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
  const demoMode = useDemoMode();
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
  // Per-request max-new-tokens override — mirrors the Completion panel
  // so the two views stay consistent. Empty means "leave it to the
  // server / model defaults" so a too-small baked-in generation config
  // doesn't truncate a reply unnoticed. Seeded from state.params if a
  // preset or the Model panel set it.
  const [maxTokens, setMaxTokens] = useState<number | "">(
    state.params.max_tokens ?? "",
  );
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
  // "Stick to bottom" auto-follow. Tracks whether the transcript is at
  // (or very near) the bottom; when true, every new chunk pins the view
  // to the latest text. The user scrolling up flips it to false so they
  // can read history without being yanked back, and scrolling back into
  // the bottom band flips it on again. A floating "Jump to latest"
  // button surfaces while we're not stuck so they always have a way
  // back. Kept in a ref *and* state — the ref feeds the scroll effect
  // synchronously (state updates lag a frame), the state drives the
  // button's visibility.
  const stickToBottomRef = useRef(true);
  const [stickToBottom, setStickToBottom] = useState(true);
  // Distance from the bottom (px) at which we still consider the user
  // "at bottom." Catches sub-pixel/rounding gaps after a programmatic
  // scrollTop assignment, and forgives small inertial overshoots.
  const STICK_THRESHOLD_PX = 24;
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

  // Keep the transcript pinned to the bottom while new tokens arrive —
  // but only when the user hasn't scrolled up to read history. The
  // scroll handler below flips ``stickToBottomRef`` based on how far
  // from the bottom the viewport sits; this effect honors that. When
  // the user *sends* a new turn (transition into busy), force-stick
  // even if they were scrolled up — sending implicitly means "show me
  // the response."
  const prevBusyRef = useRef(busy);
  useEffect(() => {
    const el = transcriptRef.current;
    if (!el) return;
    const justStartedBusy = busy && !prevBusyRef.current;
    prevBusyRef.current = busy;
    if (justStartedBusy) {
      stickToBottomRef.current = true;
      setStickToBottom(true);
    }
    if (stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, status.kind, busy]);

  // Recompute "is at bottom" on every scroll. Programmatic scrolls fire
  // this too, but they land at the absolute bottom (distance 0) so the
  // threshold check still reports stuck — no special-casing needed.
  const onTranscriptScroll = () => {
    const el = transcriptRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distanceFromBottom <= STICK_THRESHOLD_PX;
    if (atBottom !== stickToBottomRef.current) {
      stickToBottomRef.current = atBottom;
      setStickToBottom(atBottom);
    }
  };

  const jumpToBottom = () => {
    const el = transcriptRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    stickToBottomRef.current = true;
    setStickToBottom(true);
  };

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
    const params: GenerationParams = stripEmpty({
      ...state.params,
      max_tokens:
        typeof maxTokens === "number" && maxTokens > 0 ? maxTokens : undefined,
    });
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

  // Last-resort fallback when /tokenize either fails outright or
  // returns no rendered prompt (e.g. vLLM, whose /tokenize doesn't
  // include the ``prompt`` Forgather extension — so the user can't get
  // the chat template back through it). Dump the conversation as
  // pretty-printed JSON with a banner so it's obvious this is *not*
  // the byte-identical chat-template output, and the user can hand-edit
  // from there.
  const jsonFallback = (
    payload: ChatMessage[],
    reason: string,
  ): string => {
    const banner =
      `# WARNING: server didn't return a rendered chat-template prompt ` +
      `(${reason}). Falling back to a JSON dump of the conversation — ` +
      `this is NOT what the model sees during a chat call. Hand-edit ` +
      `before sending.\n`;
    return banner + JSON.stringify(payload, null, 2) + "\n";
  };

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
      if (rendered) {
        onSendToCompletion(rendered);
        return;
      }
      // /tokenize succeeded but didn't include the rendered prompt —
      // the typical vLLM case. Round-trip the token ids through
      // /detokenize to recover the byte-accurate prompt string.
      if (Array.isArray(r.tokens) && r.tokens.length > 0) {
        try {
          const d = await detokenizeTokens(
            state.baseUrl,
            state.model,
            r.tokens,
            state.authToken || undefined,
          );
          const recovered = (d.prompt ?? "").toString();
          if (recovered) {
            onSendToCompletion(recovered);
            return;
          }
        } catch (detokErr) {
          // Fall through to the JSON dump below; include the reason
          // so the status line tells the user which path failed.
          const reason =
            detokErr instanceof Error ? detokErr.message : String(detokErr);
          onSendToCompletion(
            jsonFallback(
              payload,
              `/tokenize returned no prompt and /detokenize failed: ${reason}`,
            ),
          );
          setStatus({
            kind: "error",
            message: `/detokenize failed (${reason}) — pasted conversation JSON instead.`,
          });
          return;
        }
      }
      // Either no tokens to detokenize or /detokenize returned empty —
      // JSON dump is the only thing left.
      onSendToCompletion(
        jsonFallback(
          payload,
          "neither /tokenize nor /detokenize returned a rendered prompt",
        ),
      );
      setStatus({
        kind: "error",
        message:
          "Server didn't return a rendered prompt — pasted conversation JSON instead. Edit before sending.",
      });
    } catch (err) {
      // Hard error from /tokenize itself (404, auth, network). Still
      // give the user the JSON dump so they have a starting point,
      // and surface the underlying error so they can fix the config.
      const reason = err instanceof Error ? err.message : String(err);
      onSendToCompletion(jsonFallback(payload, `/tokenize failed: ${reason}`));
      setStatus({
        kind: "error",
        message: `/tokenize failed (${reason}) — pasted conversation JSON instead.`,
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

  // Hidden file input — clicked programmatically by onImport so the
  // user gets a native file picker without us rendering a stray input
  // in the toolbar.
  const importFileRef = useRef<HTMLInputElement | null>(null);

  /** Build the JSON payload that Export downloads / Import accepts.
   *  Includes the parts of the chat state that round-trip meaningfully
   *  — system text and the message list. Mirror of PersistedChat minus
   *  transient UI flags (panel open/close, impersonate mode) so the
   *  file doesn't surprise-reset preferences on import. ``version`` is
   *  a small hedge in case the schema changes; current readers accept
   *  any value but log a hint on mismatch. */
  const buildExportPayload = (): {
    version: number;
    exported_at: string;
    systemText: string;
    messages: { role: string; content: string }[];
  } => ({
    version: 1,
    exported_at: new Date().toISOString(),
    systemText,
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
  });

  const onExport = async () => {
    // Demo mode is a read-only surface; exporting the conversation
    // gives a public visitor a one-click way to siphon whatever the
    // operator has been demonstrating. Disable rather than gate on
    // server state — Export is a purely client-side operation.
    if (demoMode) {
      setStatus({
        kind: "error",
        message: "Export is disabled in demo mode.",
      });
      return;
    }
    if (messages.length === 0 && !systemText) {
      setStatus({
        kind: "error",
        message: "Nothing to export — the conversation is empty.",
      });
      return;
    }
    const payload = buildExportPayload();
    const json = JSON.stringify(payload, null, 2);
    // Date-stamp the filename so multiple exports stay sortable. Local
    // time is more useful than UTC for "which one did I just save."
    const now = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    const stamp =
      `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}` +
      `_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
    const suggestedName = `forgather-chat-${stamp}.json`;

    // Prefer the File System Access API when available (Chromium-family,
    // and gated to secure contexts) — gives the user a native Save-As
    // dialog instead of dropping into the browser's default Downloads
    // folder. Fall back to the classic anchor-click download elsewhere
    // (Firefox, Safari, http:// contexts).
    type SaveFilePickerWindow = Window &
      typeof globalThis & {
        showSaveFilePicker?: (opts: {
          suggestedName?: string;
          types?: { description: string; accept: Record<string, string[]> }[];
        }) => Promise<{
          createWritable: () => Promise<{
            write: (data: Blob | string) => Promise<void>;
            close: () => Promise<void>;
          }>;
        }>;
      };
    const win = window as SaveFilePickerWindow;
    if (typeof win.showSaveFilePicker === "function") {
      try {
        const handle = await win.showSaveFilePicker({
          suggestedName,
          types: [
            {
              description: "Forgather chat (JSON)",
              accept: { "application/json": [".json"] },
            },
          ],
        });
        const writable = await handle.createWritable();
        await writable.write(
          new Blob([json], { type: "application/json" }),
        );
        await writable.close();
        return;
      } catch (err) {
        // User cancelled the picker — silently no-op; any other error
        // falls through to the anchor-click fallback below so the user
        // still ends up with their file.
        if (err instanceof DOMException && err.name === "AbortError") return;
      }
    }

    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = suggestedName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Defer the revoke so the browser's download pipeline has a tick
    // to read from the blob URL. Without this, fast clicks on Firefox
    // occasionally produce a 0-byte download.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const onImportClick = () => {
    if (busy) return;
    if (messages.length > 0 || systemText.trim()) {
      if (
        !window.confirm(
          "Importing replaces the current conversation and system message. Continue?",
        )
      ) {
        return;
      }
    }
    importFileRef.current?.click();
  };

  const onImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset the input so picking the same file twice in a row still
    // fires onChange.
    e.target.value = "";
    if (!file) return;
    let text: string;
    try {
      text = await file.text();
    } catch (err) {
      setStatus({
        kind: "error",
        message: `Couldn't read file: ${err instanceof Error ? err.message : String(err)}`,
      });
      return;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (err) {
      setStatus({
        kind: "error",
        message: `Not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
      });
      return;
    }
    if (!parsed || typeof parsed !== "object") {
      setStatus({
        kind: "error",
        message: "Import file must be a JSON object.",
      });
      return;
    }
    // Permissive shape check — accept both our own export format and a
    // bare ``[{role, content}, …]`` array so users can paste in a
    // hand-edited transcript without ceremony.
    let importedSystem = "";
    let importedMessages: { role: string; content: string }[] = [];
    if (Array.isArray(parsed)) {
      importedMessages = parsed as { role: string; content: string }[];
    } else {
      const obj = parsed as Record<string, unknown>;
      if (typeof obj.systemText === "string") importedSystem = obj.systemText;
      if (Array.isArray(obj.messages)) {
        importedMessages = obj.messages as { role: string; content: string }[];
      } else {
        setStatus({
          kind: "error",
          message:
            "Import file is missing a ``messages`` array. Expected the export format or a bare [{role, content}] list.",
        });
        return;
      }
    }
    const cleaned: ChatMessage[] = [];
    for (const m of importedMessages) {
      if (!m || typeof m.content !== "string") continue;
      if (m.role !== "user" && m.role !== "assistant") continue;
      cleaned.push({ role: m.role, content: m.content });
    }
    setMessages(cleaned);
    setSystemText(importedSystem);
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
        <label title="Per-request override for the model's max new tokens. Leave empty to let the server / baked-in generation config decide.">
          Max new tokens
          <input
            type="number"
            min={1}
            value={maxTokens === "" ? "" : maxTokens}
            onChange={(e) => {
              const raw = e.target.value;
              if (raw === "") {
                setMaxTokens("");
                return;
              }
              const n = Number(raw);
              setMaxTokens(Number.isFinite(n) && n > 0 ? Math.floor(n) : "");
            }}
            placeholder="server default"
            disabled={busy}
            style={{ width: 110 }}
          />
        </label>
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
          onClick={onImportClick}
          disabled={busy}
          title="Replace the current conversation with one loaded from a JSON file"
        >
          Import
        </button>
        <button
          type="button"
          className="secondary"
          onClick={onExport}
          disabled={demoMode || (messages.length === 0 && !systemText)}
          title={
            demoMode
              ? "Export is disabled in demo mode"
              : "Save the current conversation (system + messages) to a JSON file"
          }
        >
          Export
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
        <input
          ref={importFileRef}
          type="file"
          accept="application/json,.json"
          style={{ display: "none" }}
          onChange={onImportFile}
        />
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

      <div className="inference-chat-messages-wrap">
        <div
          className="inference-chat-messages"
          ref={transcriptRef}
          onScroll={onTranscriptScroll}
        >
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
        {!stickToBottom && (
          <button
            type="button"
            className="inference-chat-jump-bottom"
            onClick={jumpToBottom}
            title="Jump to latest message — re-engages auto-follow while streaming"
          >
            ↓ Jump to latest
          </button>
        )}
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
