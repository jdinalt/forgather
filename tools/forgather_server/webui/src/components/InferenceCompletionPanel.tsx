import { useEffect, useRef, useState } from "react";

import {
  GenerationParams,
  runCompletion,
  streamCompletion,
} from "../inference-client";
import { InferenceState } from "./InferencePanel";

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
  // a few more tokens" without editing the main params.
  const [maxTokens, setMaxTokens] = useState<number>(
    state.params.max_tokens ?? 256,
  );
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  // Streaming is incompatible with some HF generation modes (notably
  // beam search: ``streamer`` + ``num_beams > 1`` raises). Expose an
  // explicit toggle so those presets can be run as a single POST.
  const [stream, setStream] = useState<boolean>(true);
  const abortRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
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

  const onContinue = async () => {
    // Build the params payload: take the user's generation params, layer
    // the per-request max_tokens on top, drop any explicitly-empty keys
    // so the server sees its own defaults rather than null.
    const params: GenerationParams = stripEmpty({
      ...state.params,
      max_tokens: maxTokens > 0 ? maxTokens : undefined,
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
          text,
          params,
          ac.signal,
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
          text,
          params,
          ac.signal,
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
        <label>
          Max new tokens
          <input
            type="number"
            min={1}
            value={maxTokens}
            onChange={(e) =>
              setMaxTokens(Math.max(1, Number(e.target.value) || 1))
            }
            disabled={busy}
            style={{ width: 80 }}
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
        <button onClick={onContinue} disabled={busy || !state.baseUrl}>
          {busy ? "Continuing…" : "Continue"}
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
      <textarea
        ref={textareaRef}
        className="inference-textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Type a prompt here, then click Continue (or Ctrl+Enter) to let the model extend it."
        spellCheck={false}
      />
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
