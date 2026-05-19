import { useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  COLORMAPS,
  cssColor,
  getColormap,
  readableForeground,
} from "../colormaps";
import { scorePrompt, TokenScores } from "../inference-client";
import { persistGet, persistSet } from "../persist";
import { InferenceState } from "./InferencePanel";

interface Props {
  state: InferenceState;
}

type Status =
  | { kind: "idle" }
  | { kind: "scoring" }
  | { kind: "done"; durationMs: number; nTokens: number }
  | { kind: "stopped" }
  | { kind: "error"; message: string };

const DEFAULT_TOP_K = 10;

type ScaleMode = "auto" | "manual";
type Metric = "loss" | "entropy";

const STORAGE_KEY = "forgather-analyze-prefs";

interface AnalyzePrefs {
  metric: Metric;
  cmap: string;
  scaleMode: ScaleMode;
  manualLo: number;
  manualHi: number;
}

const DEFAULT_PREFS: AnalyzePrefs = {
  metric: "loss",
  cmap: "viridis",
  scaleMode: "auto",
  // Reasonable defaults for causal-LM loss in nats: <0.1 trivial,
  // >5 quite surprising. User can dial these in once data is on screen.
  // Entropy ranges higher (up to log(vocab) ≈ 10 nats for 32k vocab),
  // but loss is the default metric so these are loss-tuned. Users
  // adjust manually after switching to entropy.
  manualLo: 0,
  manualHi: 5,
};

function loadPrefs(): AnalyzePrefs {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return DEFAULT_PREFS;
  try {
    const parsed = JSON.parse(raw) as Partial<AnalyzePrefs>;
    return {
      metric:
        parsed.metric === "entropy" ? "entropy" : "loss",
      cmap:
        typeof parsed.cmap === "string" ? parsed.cmap : DEFAULT_PREFS.cmap,
      scaleMode:
        parsed.scaleMode === "manual" ? "manual" : "auto",
      manualLo:
        typeof parsed.manualLo === "number"
          ? parsed.manualLo
          : DEFAULT_PREFS.manualLo,
      manualHi:
        typeof parsed.manualHi === "number"
          ? parsed.manualHi
          : DEFAULT_PREFS.manualHi,
    };
  } catch {
    return DEFAULT_PREFS;
  }
}

export function InferenceAnalyzePanel({ state }: Props) {
  const [text, setText] = useState<string>("");
  const [topK, setTopK] = useState<number>(DEFAULT_TOP_K);
  const [scores, setScores] = useState<TokenScores | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [prefs, setPrefs] = useState<AnalyzePrefs>(loadPrefs);
  const [selectionLen, setSelectionLen] = useState<number>(0);
  const abortRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const busy = status.kind === "scoring";

  const updatePrefs = (patch: Partial<AnalyzePrefs>) => {
    setPrefs((prev) => {
      const next = { ...prev, ...patch };
      persistSet(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  const runAnalyze = async () => {
    // If the user has a non-empty selection in the textarea, score
    // just the selected substring. Big inputs (a whole novel pasted
    // in) are unusable to score in one go — selecting a paragraph
    // and analyzing only that is far more useful than truncating
    // arbitrarily. Selection state is read from the DOM at click
    // time, not from the cached selectionLen, so it's always fresh.
    const ta = textareaRef.current;
    let prompt = text;
    if (ta && ta.selectionStart !== ta.selectionEnd) {
      prompt = text.substring(ta.selectionStart, ta.selectionEnd);
    }
    if (!prompt.trim()) return;
    const ac = new AbortController();
    abortRef.current = ac;
    setStatus({ kind: "scoring" });
    const started = Date.now();
    try {
      const result = await scorePrompt(
        state.baseUrl,
        state.model,
        prompt,
        topK,
        ac.signal,
        state.authToken || undefined,
      );
      setScores(result);
      setStatus({
        kind: "done",
        durationMs: Date.now() - started,
        nTokens: result.tokens.length,
      });
    } catch (err) {
      if (ac.signal.aborted) {
        setStatus({ kind: "stopped" });
      } else {
        setStatus({
          kind: "error",
          message: err instanceof Error ? err.message : String(err),
        });
      }
    } finally {
      abortRef.current = null;
    }
  };

  const onStop = () => abortRef.current?.abort();

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (!busy && state.baseUrl) void runAnalyze();
    }
  };

  return (
    <div className="inference-analyze">
      <div className="inference-analyze-bar">
        <label>
          Top-K
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            onChange={(e) =>
              setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 1)))
            }
            disabled={busy}
            style={{ width: 60 }}
          />
        </label>
        <button
          onClick={runAnalyze}
          disabled={busy || !state.baseUrl || !text.trim()}
          title={
            selectionLen > 0
              ? `Score only the selected ${selectionLen.toLocaleString()} characters`
              : "Score the entire input"
          }
        >
          {busy
            ? "Analyzing…"
            : selectionLen > 0
              ? `Analyze selection (${selectionLen.toLocaleString()} ch)`
              : "Analyze"}
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
          onClick={() => {
            setText("");
            setScores(null);
            setStatus({ kind: "idle" });
          }}
          disabled={busy}
          title="Clear input and result"
        >
          Clear
        </button>
        <label title="Quantity color-encoded across tokens. Entropy is a Forgather extension and falls back to loss when the server doesn't provide it.">
          metric
          <select
            value={prefs.metric}
            onChange={(e) =>
              updatePrefs({ metric: e.target.value as Metric })
            }
          >
            <option value="loss">loss</option>
            <option value="entropy">entropy</option>
          </select>
        </label>
        <label title="Color encoding">
          colormap
          <select
            value={prefs.cmap}
            onChange={(e) => updatePrefs({ cmap: e.target.value })}
          >
            {COLORMAPS.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label title="Auto = 5th/95th percentile of this response. Manual = fixed loss range across all responses.">
          scale
          <select
            value={prefs.scaleMode}
            onChange={(e) =>
              updatePrefs({ scaleMode: e.target.value as ScaleMode })
            }
          >
            <option value="auto">auto</option>
            <option value="manual">manual</option>
          </select>
        </label>
        {prefs.scaleMode === "manual" && (
          <>
            <label title="Loss value mapped to the cold end of the colormap">
              min
              <input
                type="number"
                step="0.1"
                value={prefs.manualLo}
                onChange={(e) =>
                  updatePrefs({ manualLo: Number(e.target.value) || 0 })
                }
                style={{ width: 60 }}
              />
            </label>
            <label title="Loss value mapped to the hot end of the colormap">
              max
              <input
                type="number"
                step="0.1"
                value={prefs.manualHi}
                onChange={(e) =>
                  updatePrefs({ manualHi: Number(e.target.value) || 1 })
                }
                style={{ width: 60 }}
              />
            </label>
          </>
        )}
        <div className="muted inference-status">
          <StatusLine status={status} />
        </div>
      </div>
      <div className="inference-analyze-body">
        <textarea
          ref={textareaRef}
          className="inference-analyze-input"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            // Selection collapses on edit; mirror that so the button
            // label switches back to plain "Analyze".
            setSelectionLen(0);
          }}
          onSelect={(e) => {
            const el = e.currentTarget;
            setSelectionLen(
              Math.max(0, el.selectionEnd - el.selectionStart),
            );
          }}
          onBlur={() => {
            // Keep the cached length in sync with the DOM. When the
            // user clicks Analyze, focus moves and the textarea
            // retains its selection — selectionLen should stay
            // accurate so the button label is correct.
            const el = textareaRef.current;
            if (el) {
              setSelectionLen(
                Math.max(0, el.selectionEnd - el.selectionStart),
              );
            }
          }}
          onKeyDown={onKeyDown}
          placeholder="Paste or type text to score. Select a passage to score just that — useful for sampling out of a large pasted document. Ctrl+Enter to Analyze."
          spellCheck={false}
        />
        <div className="inference-analyze-output">
          {scores ? (
            <ScoredText scores={scores} prefs={prefs} />
          ) : (
            <div className="muted analyze-empty">
              {status.kind === "scoring"
                ? "Running forward pass…"
                : "No scores yet. Click Analyze to score the input."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusLine({ status }: { status: Status }) {
  switch (status.kind) {
    case "idle":
      return <span>ready</span>;
    case "scoring":
      return <span>scoring…</span>;
    case "done":
      return (
        <span>
          done · {status.nTokens} token{status.nTokens === 1 ? "" : "s"} ·{" "}
          {(status.durationMs / 1000).toFixed(2)}s
        </span>
      );
    case "stopped":
      return <span>stopped</span>;
    case "error":
      return <span className="err">error · {status.message}</span>;
  }
}

/** Render token sequence with loss-driven background color and hover
 *  tooltips. In "auto" scale mode, uses the 5th/95th-percentile bounds
 *  of the response's own losses for the color domain so a handful of
 *  outlier hard-to-predict tokens don't squash the gradient (matches
 *  `forgather logs plot`'s outlier-aware default). In "manual" mode,
 *  uses the user-supplied min/max so the same loss reads the same color
 *  across separate runs — important for comparing inputs side-by-side.
 *
 *  Hover state is lifted to this component (rather than each token
 *  owning its own nested tooltip span) so the tooltip can render
 *  ``position: fixed`` outside the scrolling output pane — otherwise
 *  ``overflow: auto`` on the pane clips it. Placement flips
 *  above/below the anchor and clamps to the viewport. */
function ScoredText({
  scores,
  prefs,
}: {
  scores: TokenScores;
  prefs: AnalyzePrefs;
}) {
  const { tokens, token_logprobs, top_logprobs, token_entropies } = scores;
  const [hover, setHover] = useState<HoverState | null>(null);

  // Entropy is a Forgather extension; an OpenAI/vLLM server won't
  // return it. If the user picked "entropy" and the response has none,
  // silently fall back to loss for coloring + display a one-line note
  // above the rendered text so they know why.
  const entropyAvailable = Array.isArray(token_entropies);
  const effectiveMetric: Metric =
    prefs.metric === "entropy" && !entropyAvailable ? "loss" : prefs.metric;
  const fellBack = prefs.metric !== effectiveMetric;

  // Pull the active per-position values for the chosen metric, with
  // null at positions that have no prediction (index 0).
  const values = useMemo<(number | null)[]>(() => {
    if (effectiveMetric === "entropy") {
      return (token_entropies ?? []).slice();
    }
    return token_logprobs.map((lp) =>
      typeof lp === "number" ? -lp : null,
    );
  }, [effectiveMetric, token_logprobs, token_entropies]);

  const { lo, hi } = useMemo(() => {
    if (prefs.scaleMode === "manual") {
      const a = Math.min(prefs.manualLo, prefs.manualHi);
      const b = Math.max(prefs.manualLo, prefs.manualHi);
      return { lo: a, hi: b > a ? b : a + 1 };
    }
    const samples: number[] = [];
    for (const v of values) {
      if (typeof v === "number") samples.push(v);
    }
    if (samples.length === 0) return { lo: 0, hi: 1 };
    const sorted = samples.slice().sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.05)] ?? sorted[0];
    const hi =
      sorted[Math.floor(sorted.length * 0.95)] ?? sorted[sorted.length - 1];
    return { lo, hi: hi > lo ? hi : lo + 1 };
  }, [values, prefs.scaleMode, prefs.manualLo, prefs.manualHi]);

  const cmap = useMemo(() => getColormap(prefs.cmap), [prefs.cmap]);

  return (
    <>
      {fellBack && (
        <div className="analyze-fallback-note muted">
          This server didn't return entropy — coloring by loss. (Entropy is a
          Forgather extension; OpenAI/vLLM don't expose it.)
        </div>
      )}
      <div className="scored-text">
        {tokens.map((tok, i) => {
          const lp = token_logprobs[i];
          const loss = typeof lp === "number" ? -lp : null;
          const entropy =
            entropyAvailable && typeof token_entropies![i] === "number"
              ? (token_entropies![i] as number)
              : null;
          const v = values[i];
          let bg = "transparent";
          let fg = "inherit";
          if (typeof v === "number") {
            const t = (v - lo) / (hi - lo);
            const rgb = cmap.fn(t);
            bg = cssColor(rgb);
            fg = readableForeground(rgb);
          }
          return (
            <TokenSpan
              key={i}
              token={tok}
              loss={loss}
              entropy={entropy}
              background={bg}
              foreground={fg}
              topLogprobs={top_logprobs[i]}
              onHoverChange={setHover}
            />
          );
        })}
      </div>
      {hover && <FloatingTooltip data={hover} />}
    </>
  );
}

interface HoverState {
  token: string;
  loss: number | null;
  entropy: number | null;
  topLogprobs: Record<string, number> | null;
  anchor: DOMRect;
}

interface TokenSpanProps {
  token: string;
  loss: number | null;
  entropy: number | null;
  background: string;
  foreground: string;
  topLogprobs: Record<string, number> | null;
  onHoverChange: (state: HoverState | null) => void;
}

function TokenSpan({
  token,
  loss,
  entropy,
  background,
  foreground,
  topLogprobs,
  onHoverChange,
}: TokenSpanProps) {
  // Container uses ``white-space: pre-wrap`` so spaces and newlines
  // inside the token render verbatim — no manual NBSP / <br> swap
  // needed. The whole token (including any embedded newline) keeps a
  // single background swatch, which makes "\n\n" tokens visible as
  // colored blank lines rather than vanishing.
  return (
    <span
      className="scored-token"
      style={{ background, color: foreground }}
      onMouseEnter={(e) =>
        onHoverChange({
          token,
          loss,
          entropy,
          topLogprobs,
          anchor: e.currentTarget.getBoundingClientRect(),
        })
      }
      onMouseLeave={() => onHoverChange(null)}
    >
      {token}
    </span>
  );
}

/** Viewport-fixed tooltip positioned relative to a token's bounding
 *  rect. Tries to sit just above the anchor; flips below if there isn't
 *  room (and clamps into the viewport if both fail — rare); clamps the
 *  horizontal position so it never overflows left/right edges.
 *
 *  Measurement happens in ``useLayoutEffect`` so the corrected position
 *  is committed before paint — the user never sees the tooltip flash at
 *  (0,0). The ``visibility: hidden`` first-frame guard covers the gap
 *  between insert and measure. */
function FloatingTooltip({ data }: { data: HoverState }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState<{
    top: number;
    left: number;
    ready: boolean;
  }>({ top: 0, left: 0, ready: false });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const margin = 6;

    // Prefer above the anchor; flip below if it'd clip the top.
    let top = data.anchor.top - rect.height - margin;
    if (top < margin) {
      const belowTop = data.anchor.bottom + margin;
      if (belowTop + rect.height <= vh - margin) {
        top = belowTop;
      } else {
        // Neither above nor below fits — clamp to viewport bounds.
        // Better to overlap the anchor a bit than to be clipped.
        top = Math.max(margin, vh - rect.height - margin);
      }
    }

    // Anchor the left edge to the token, then clamp into the viewport.
    let left = data.anchor.left;
    if (left + rect.width > vw - margin) {
      left = vw - rect.width - margin;
    }
    if (left < margin) left = margin;

    setPos({ top, left, ready: true });
  }, [data]);

  return (
    <div
      ref={ref}
      className="scored-token-tooltip"
      style={{
        top: pos.top,
        left: pos.left,
        visibility: pos.ready ? "visible" : "hidden",
      }}
    >
      <TooltipBody
        token={data.token}
        loss={data.loss}
        entropy={data.entropy}
        topLogprobs={data.topLogprobs}
      />
    </div>
  );
}

function TooltipBody({
  token,
  loss,
  entropy,
  topLogprobs,
}: {
  token: string;
  loss: number | null;
  entropy: number | null;
  topLogprobs: Record<string, number> | null;
}) {
  const ppx = loss !== null ? Math.exp(loss) : null;
  // Sort top-K by probability descending.
  const ranked = topLogprobs
    ? Object.entries(topLogprobs).sort((a, b) => b[1] - a[1])
    : [];
  return (
    <>
      <div className="tt-token">token: <code>{JSON.stringify(token)}</code></div>
      {loss !== null && (
        <div className="tt-metrics">
          loss: {loss.toFixed(3)} · perplexity: {ppx!.toFixed(2)}
          {entropy !== null && (
            <>
              {" · "}entropy: {entropy.toFixed(3)} nats
            </>
          )}
        </div>
      )}
      {ranked.length > 0 && (
        <>
          <div className="tt-divider">top {ranked.length}</div>
          <table className="tt-topk">
            <tbody>
              {ranked.map(([t, lp]) => (
                <tr key={t}>
                  <td className="tt-prob">{(Math.exp(lp) * 100).toFixed(1)}%</td>
                  <td className="tt-tok"><code>{JSON.stringify(t)}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {loss === null && (
        <div className="tt-metrics muted">first token — no prediction</div>
      )}
    </>
  );
}

