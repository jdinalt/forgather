import { useCallback, useEffect, useRef, useState } from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  GenerationParams,
  runCompletion,
  streamCompletion,
} from "../inference-client";
import { persistGet, persistSet } from "../persist";
import { DragHandle } from "./DragHandle";
import { InferenceState } from "./InferencePanel";

/** Persisted user prefs for the completion panel layout. Kept narrow:
 *  the markdown toggle + split percentage are the only things worth
 *  surviving a reload. Other state (prevInput, status) is transient. */
interface CompletionPrefs {
  markdownView: boolean;
  /** Percent of the body width allocated to the raw textarea when the
   *  split view is on. Remainder goes to the rendered markdown pane. */
  leftWidthPct: number;
}

const PREFS_KEY = "forgather-completion-prefs";
const DEFAULT_PREFS: CompletionPrefs = {
  markdownView: false,
  leftWidthPct: 50,
};

function loadPrefs(): CompletionPrefs {
  const raw = persistGet(PREFS_KEY);
  if (!raw) return DEFAULT_PREFS;
  try {
    const parsed = JSON.parse(raw) as Partial<CompletionPrefs>;
    return {
      markdownView:
        typeof parsed.markdownView === "boolean"
          ? parsed.markdownView
          : DEFAULT_PREFS.markdownView,
      leftWidthPct:
        typeof parsed.leftWidthPct === "number" &&
        parsed.leftWidthPct >= 15 &&
        parsed.leftWidthPct <= 85
          ? parsed.leftWidthPct
          : DEFAULT_PREFS.leftWidthPct,
    };
  } catch {
    return DEFAULT_PREFS;
  }
}

interface Props {
  state: InferenceState;
  // Controlled by the parent so the chat panel's "Send to completion"
  // button can append a rendered prompt and switch tabs without losing
  // whatever the user already had in the textarea.
  text: string;
  setText: React.Dispatch<React.SetStateAction<string>>;
}

type Status =
  | { kind: "idle" }
  | { kind: "streaming"; startedAt: number; tokens: number }
  | { kind: "generating"; startedAt: number }
  | {
      kind: "done";
      tokens: number;
      durationMs: number;
    }
  | { kind: "stopped"; tokens: number; durationMs: number }
  | { kind: "error"; message: string };

export function InferenceCompletionPanel({ state, text, setText }: Props) {
  // Per-request max-new-tokens override — convenient for "give me just
  // a few more tokens" without editing the main params. Empty means
  // "leave it to the server / model defaults". Seeded from state.params
  // if a preset (or the Model panel) set it; otherwise starts empty.
  const [maxTokens, setMaxTokens] = useState<number | "">(
    state.params.max_tokens ?? "",
  );
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  // Streaming is incompatible with some HF generation modes (notably
  // beam search: ``streamer`` + ``num_beams > 1`` raises). Expose an
  // explicit toggle so those presets can be run as a single POST.
  const [stream, setStream] = useState<boolean>(true);
  // Snapshot of the textarea contents from immediately before the most
  // recent generation. Lets the user re-run with the same prompt and
  // different generation params without manually deleting the previously
  // appended completion. Null until the first generation has started.
  const [prevInput, setPrevInput] = useState<string | null>(null);
  const [prefs, setPrefs] = useState<CompletionPrefs>(loadPrefs);
  const abortRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);

  // Drag handler: pixel delta → percentage of the body width. Clamped
  // to keep either pane from collapsing. State updates fire on every
  // pointermove; persistence is deferred to pointerup via
  // ``persistCurrentPrefs`` so a fast drag doesn't emit hundreds of
  // localStorage writes per second. Mirrors the analyze panel pattern.
  const onSplitXDelta = useCallback((dx: number) => {
    const body = bodyRef.current;
    if (!body) return;
    const w = body.getBoundingClientRect().width;
    if (w <= 0) return;
    setPrefs((prev) => {
      const next = Math.max(
        15,
        Math.min(85, prev.leftWidthPct + (dx / w) * 100),
      );
      if (next === prev.leftWidthPct) return prev;
      return { ...prev, leftWidthPct: next };
    });
  }, []);

  const persistPrefsRef = useRef(prefs);
  persistPrefsRef.current = prefs;
  const persistCurrentPrefs = useCallback(() => {
    persistSet(PREFS_KEY, JSON.stringify(persistPrefsRef.current));
  }, []);

  const updatePrefs = (patch: Partial<CompletionPrefs>) => {
    setPrefs((prev) => {
      const next = { ...prev, ...patch };
      persistSet(PREFS_KEY, JSON.stringify(next));
      return next;
    });
  };
  // Track the previous busy state so we can restore textarea focus
  // exactly on the busy→idle transition. Without this, Ctrl+Enter
  // sends and the textarea's ``disabled`` flip drops focus, forcing
  // the user to click back in to keep working. Same workaround as
  // InferenceChatPanel.
  const wasBusyRef = useRef(false);

  const busy = status.kind === "streaming" || status.kind === "generating";

  useEffect(() => {
    if (wasBusyRef.current && !busy) {
      textareaRef.current?.focus();
    }
    wasBusyRef.current = busy;
  }, [busy]);

  // ``promptOverride`` lets Regenerate run with the restored prompt
  // without waiting for the setText state update to flush — passing the
  // string directly avoids a race where the request would otherwise be
  // built from the still-stale ``text`` value.
  const runGeneration = async (promptOverride?: string) => {
    const prompt = promptOverride ?? text;
    setPrevInput(prompt);
    if (promptOverride !== undefined) {
      setText(promptOverride);
    }
    // Build the params payload: take the user's generation params, layer
    // the per-request max_tokens on top, drop any explicitly-empty keys
    // so the server sees its own defaults rather than null.
    // Conditional spread: empty per-request override means "don't
    // override," not "actively clear." Mirrors the chat panel — see
    // there for full rationale.
    const overrideMax =
      typeof maxTokens === "number" && maxTokens > 0
        ? { max_tokens: maxTokens }
        : {};
    const params: GenerationParams = stripEmpty({
      ...state.params,
      ...overrideMax,
    });
    const ac = new AbortController();
    abortRef.current = ac;
    const started = Date.now();
    let tokenCount = 0;

    if (stream) {
      setStatus({ kind: "streaming", startedAt: started, tokens: 0 });
      try {
        for await (const delta of streamCompletion(
          state.baseUrl,
          state.model,
          prompt,
          params,
          ac.signal,
          state.authToken || undefined,
          state.serverId,
        )) {
          tokenCount += 1;
          setText((prev) => prev + delta);
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
        const full = await runCompletion(
          state.baseUrl,
          state.model,
          prompt,
          params,
          ac.signal,
          state.authToken || undefined,
          state.serverId,
        );
        setText((prev) => prev + full);
        setStatus({
          kind: "done",
          // No per-token count in non-streaming mode — the server's
          // usage struct would give us completion_tokens, but we drop
          // the usage payload today. Reporting 0 would be misleading;
          // show "—" in the status line instead.
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

  const onContinue = () => runGeneration();
  const onRegenerate = () => {
    if (prevInput === null) return;
    void runGeneration(prevInput);
  };

  const onStop = () => {
    abortRef.current?.abort();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl/Cmd+Enter triggers Continue, matching the chat panel.
    // Plain Enter still inserts a newline so multi-line prompts work.
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (!busy && state.baseUrl) {
        void onContinue();
      }
    }
  };

  return (
    <div className="inference-completion">
      <div className="inference-completion-bar">
        <label title="Per-request override for max new tokens. Leave empty to use the client default of 2048 — OpenAI's /v1/completions spec defaults to 16 when the request omits max_tokens, so the client injects 2048 on this code path to keep raw-prompt extension from clipping silently. Set explicitly to override.">
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
            placeholder="2048 (client default)"
            disabled={busy}
            style={{ width: 130 }}
          />
        </label>
        <label className="dyn-checkbox" title="Uncheck for modes incompatible with streaming, e.g. beam search">
          <input
            type="checkbox"
            checked={stream}
            onChange={(e) => setStream(e.target.checked)}
            disabled={busy}
          />
          stream
        </label>
        <label
          className="dyn-checkbox"
          title="Render a side-by-side view with the raw text on the left and Markdown-rendered output on the right"
        >
          <input
            type="checkbox"
            checked={prefs.markdownView}
            onChange={(e) => updatePrefs({ markdownView: e.target.checked })}
          />
          markdown
        </label>
        <button onClick={onContinue} disabled={busy || !state.baseUrl}>
          {busy ? "Continuing…" : "Continue"}
        </button>
        <button
          className="secondary"
          onClick={onRegenerate}
          disabled={busy || !state.baseUrl || prevInput === null}
          title="Restore the prompt as it was before the last generation, then continue again"
        >
          Regenerate
        </button>
        <button
          className="secondary"
          onClick={onStop}
          disabled={!busy}
          title="Abort the in-flight request"
        >
          Stop
        </button>
        <button
          className="secondary"
          onClick={() => setText("")}
          disabled={busy}
          title="Clear the textarea"
        >
          Clear
        </button>
        <div className="muted inference-status">
          <StatusLine status={status} />
        </div>
      </div>
      <div className="inference-completion-body" ref={bodyRef}>
        <textarea
          ref={textareaRef}
          className="inference-textarea"
          style={
            prefs.markdownView
              ? { flex: `0 0 ${prefs.leftWidthPct}%` }
              : undefined
          }
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a prompt here, then click Continue (or Ctrl+Enter) to let the model extend it."
          spellCheck={false}
        />
        {prefs.markdownView && (
          <>
            <DragHandle
              axis="x"
              ariaLabel="Resize raw text vs Markdown panes"
              onDragDelta={onSplitXDelta}
              onDragEnd={persistCurrentPrefs}
              onDoubleClick={() => updatePrefs({ leftWidthPct: 50 })}
            />
            <div className="inference-completion-markdown">
              {text ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {text}
                </ReactMarkdown>
              ) : (
                <div className="muted analyze-empty">
                  Nothing to render yet — type a prompt and Continue.
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function StatusLine({ status }: { status: Status }) {
  const tokLabel = (n: number) =>
    n < 0 ? "— tokens" : `${n} token${n === 1 ? "" : "s"}`;
  // Throughput formatter — only meaningful when we actually have a
  // positive token count and a non-zero duration. Non-streaming
  // responses report ``tokens: -1`` (we drop usage from the wire
  // today), so the rate becomes "— tok/s" there.
  const rateLabel = (n: number, ms: number) => {
    if (n <= 0 || ms <= 0) return "— tok/s";
    const rate = n / (ms / 1000);
    // <10 → one decimal; ≥10 → integer. Reads better in the status
    // strip where horizontal real estate is tight.
    return `${rate < 10 ? rate.toFixed(1) : Math.round(rate)} tok/s`;
  };
  switch (status.kind) {
    case "idle":
      return <span>ready</span>;
    case "streaming": {
      const elapsedMs = Date.now() - status.startedAt;
      return (
        <span>
          streaming · {tokLabel(status.tokens)} ·{" "}
          {rateLabel(status.tokens, elapsedMs)}
        </span>
      );
    }
    case "generating":
      return <span>generating…</span>;
    case "done":
      return (
        <span>
          done · {tokLabel(status.tokens)} ·{" "}
          {(status.durationMs / 1000).toFixed(1)}s ·{" "}
          {rateLabel(status.tokens, status.durationMs)}
        </span>
      );
    case "stopped":
      return (
        <span>
          stopped · {tokLabel(status.tokens)} ·{" "}
          {(status.durationMs / 1000).toFixed(1)}s ·{" "}
          {rateLabel(status.tokens, status.durationMs)}
        </span>
      );
    case "error":
      return <span className="err">error · {status.message}</span>;
  }
}

/** Drop keys whose value is undefined, null, or an empty string / array
 *  so the server sees its per-model defaults. Mirrors the CLI client's
 *  behavior where omitted flags mean "use the template default". */
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
