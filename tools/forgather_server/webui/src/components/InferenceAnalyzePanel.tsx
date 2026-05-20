import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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
  /** Inbound request from elsewhere in the app (e.g. the Datasets
   *  cell context menu) to replace the input text and run scoring
   *  immediately. ``key`` is a Date.now() nonce — keyed-effect dedup
   *  fires on each fresh request without re-firing on parent
   *  re-renders. */
  pendingAnalyze?: { text: string; key: number } | null;
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
  /** Exponential moving average factor applied to the color-encoded
   *  signal, in [0, 1). 0 = disabled (raw values); higher = smoother.
   *  Formula: ``s[i] = α·s[i-1] + (1-α)·raw[i]``. Useful for spotting
   *  region-scale trends in long inputs where the per-token signal is
   *  too noisy to read. Tooltip values stay raw. */
  emaAlpha: number;
  /** Toggle: show histogram of raw metric values below the scored text. */
  showHistogram: boolean;
  /** Percent of the body width allocated to the input textarea
   *  (remainder is the output pane). Persisted; drag the vertical
   *  handle between the panes to change. */
  leftWidthPct: number;
  /** Percent of the output pane's height allocated to the histogram
   *  when shown. Persisted; drag the horizontal handle. */
  histHeightPct: number;
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
  emaAlpha: 0,
  showHistogram: false,
  leftWidthPct: 50,
  histHeightPct: 35,
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
      emaAlpha:
        typeof parsed.emaAlpha === "number" &&
        parsed.emaAlpha >= 0 &&
        parsed.emaAlpha < 1
          ? parsed.emaAlpha
          : DEFAULT_PREFS.emaAlpha,
      showHistogram:
        typeof parsed.showHistogram === "boolean"
          ? parsed.showHistogram
          : DEFAULT_PREFS.showHistogram,
      leftWidthPct:
        typeof parsed.leftWidthPct === "number" &&
        parsed.leftWidthPct >= 15 &&
        parsed.leftWidthPct <= 85
          ? parsed.leftWidthPct
          : DEFAULT_PREFS.leftWidthPct,
      histHeightPct:
        typeof parsed.histHeightPct === "number" &&
        parsed.histHeightPct >= 10 &&
        parsed.histHeightPct <= 80
          ? parsed.histHeightPct
          : DEFAULT_PREFS.histHeightPct,
    };
  } catch {
    return DEFAULT_PREFS;
  }
}

export function InferenceAnalyzePanel({ state, pendingAnalyze }: Props) {
  const [text, setText] = useState<string>("");
  const [topK, setTopK] = useState<number>(DEFAULT_TOP_K);
  const [scores, setScores] = useState<TokenScores | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [prefs, setPrefs] = useState<AnalyzePrefs>(loadPrefs);
  const [selectionLen, setSelectionLen] = useState<number>(0);
  const abortRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const outputPaneRef = useRef<HTMLDivElement | null>(null);

  // Drag handlers convert per-move pixel deltas to percentage updates
  // against the relevant parent dimension. Clamped to keep either pane
  // from collapsing to zero. The pref write-back is the same throttling
  // we already use for other prefs — state update + persist.
  const onSplitXDelta = useCallback(
    (dx: number) => {
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
        const updated = { ...prev, leftWidthPct: next };
        persistSet(STORAGE_KEY, JSON.stringify(updated));
        return updated;
      });
    },
    [],
  );
  const onSplitYDelta = useCallback(
    (dy: number) => {
      const pane = outputPaneRef.current;
      if (!pane) return;
      const h = pane.getBoundingClientRect().height;
      if (h <= 0) return;
      setPrefs((prev) => {
        // dy positive = handle moved down = histogram shrinks
        const next = Math.max(
          10,
          Math.min(80, prev.histHeightPct - (dy / h) * 100),
        );
        if (next === prev.histHeightPct) return prev;
        const updated = { ...prev, histHeightPct: next };
        persistSet(STORAGE_KEY, JSON.stringify(updated));
        return updated;
      });
    },
    [],
  );

  const busy = status.kind === "scoring";

  const updatePrefs = (patch: Partial<AnalyzePrefs>) => {
    setPrefs((prev) => {
      const next = { ...prev, ...patch };
      persistSet(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  const runAnalyze = async (textOverride?: string) => {
    // An explicit override (e.g. from the cross-section pendingAnalyze
    // path) wins over both ``text`` state and any textarea selection
    // — the caller has already decided what to score. Otherwise, if
    // the user has a non-empty selection in the textarea, score just
    // the selected substring. Big inputs (a whole novel pasted in) are
    // unusable to score in one go — selecting a paragraph and
    // analyzing only that is far more useful than truncating
    // arbitrarily. Selection state is read from the DOM at click
    // time, not from the cached selectionLen, so it's always fresh.
    let prompt: string;
    if (textOverride !== undefined) {
      prompt = textOverride;
    } else {
      const ta = textareaRef.current;
      prompt = text;
      if (ta && ta.selectionStart !== ta.selectionEnd) {
        prompt = text.substring(ta.selectionStart, ta.selectionEnd);
      }
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

  // Consume pendingAnalyze when its key is fresh. Replaces the
  // textarea contents with the inbound text and immediately runs
  // scoring on it. Ref-gated by key so flipping tabs back to Analyze
  // later doesn't re-trigger. ``runAnalyze`` is read via a ref so the
  // effect doesn't refire every render the function identity changes.
  const lastPendingKeyRef = useRef<number | null>(null);
  const runAnalyzeRef = useRef(runAnalyze);
  runAnalyzeRef.current = runAnalyze;
  useEffect(() => {
    if (!pendingAnalyze) return;
    if (pendingAnalyze.key === lastPendingKeyRef.current) return;
    lastPendingKeyRef.current = pendingAnalyze.key;
    setText(pendingAnalyze.text);
    setSelectionLen(0);
    void runAnalyzeRef.current(pendingAnalyze.text);
  }, [pendingAnalyze]);

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
          onClick={() => void runAnalyze()}
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
        <label
          className="dyn-checkbox"
          title="Show a histogram of raw metric values below the scored text. Bars are colored by the same colormap and scale so they double as a legend."
        >
          <input
            type="checkbox"
            checked={prefs.showHistogram}
            onChange={(e) =>
              updatePrefs({ showHistogram: e.target.checked })
            }
          />
          histogram
        </label>
        <label title="Exponential moving average over the color-encoded signal. 0 = off (raw values); higher = smoother. Formula: s[i] = α·s[i-1] + (1-α)·raw[i]. Tooltip values stay raw.">
          smooth
          <input
            type="number"
            min={0}
            max={0.99}
            step={0.05}
            value={prefs.emaAlpha}
            onChange={(e) => {
              const raw = Number(e.target.value);
              const v = Number.isFinite(raw)
                ? Math.min(0.99, Math.max(0, raw))
                : 0;
              updatePrefs({ emaAlpha: v });
            }}
            style={{ width: 60 }}
          />
        </label>
        <div className="muted inference-status">
          <StatusLine status={status} />
        </div>
      </div>
      <div className="inference-analyze-body" ref={bodyRef}>
        <textarea
          ref={textareaRef}
          className="inference-analyze-input"
          style={{ flex: `0 0 ${prefs.leftWidthPct}%` }}
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
        <DragHandle
          axis="x"
          ariaLabel="Resize input vs output panes"
          onDragDelta={onSplitXDelta}
          onDoubleClick={() => updatePrefs({ leftWidthPct: 50 })}
        />
        <div
          className="inference-analyze-output"
          ref={outputPaneRef}
        >
          <div
            className="inference-analyze-scored-pane"
            style={
              prefs.showHistogram
                ? { flex: `1 1 ${100 - prefs.histHeightPct}%`, minHeight: 0 }
                : { flex: 1, minHeight: 0 }
            }
          >
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
          {prefs.showHistogram && (
            <>
              <DragHandle
                axis="y"
                ariaLabel="Resize scored text vs histogram panes"
                onDragDelta={onSplitYDelta}
                onDoubleClick={() => updatePrefs({ histHeightPct: 35 })}
              />
              <div
                className="inference-analyze-hist-pane"
                style={{
                  flex: `0 0 ${prefs.histHeightPct}%`,
                  minHeight: 0,
                }}
              >
                {scores ? (
                  <HistogramView scores={scores} prefs={prefs} />
                ) : (
                  <div className="muted analyze-empty">
                    No data — run Analyze first.
                  </div>
                )}
              </div>
            </>
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
  // null at positions that have no prediction (index 0). When the EMA
  // factor is > 0, smooth the signal with s[i] = α·s[i-1] + (1-α)·v[i],
  // seeding from the first valid value. Null positions pass through
  // untouched (they have no prediction) and reset the EMA chain — a
  // contiguous block of nulls in the middle of the input shouldn't
  // bridge unrelated regions of text.
  const values = useMemo<(number | null)[]>(() => {
    const raw: (number | null)[] =
      effectiveMetric === "entropy"
        ? (token_entropies ?? []).slice()
        : token_logprobs.map((lp) =>
            typeof lp === "number" ? -lp : null,
          );
    if (prefs.emaAlpha <= 0) return raw;
    const alpha = Math.min(0.99, prefs.emaAlpha);
    const out: (number | null)[] = [];
    let prev: number | null = null;
    for (const v of raw) {
      if (typeof v !== "number") {
        out.push(null);
        prev = null;
        continue;
      }
      const s: number =
        prev === null ? v : alpha * prev + (1 - alpha) * v;
      out.push(s);
      prev = s;
    }
    return out;
  }, [
    effectiveMetric,
    token_logprobs,
    token_entropies,
    prefs.emaAlpha,
  ]);

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

/** Pointer-capture drag handle. ``axis="x"`` is a thin vertical strip
 *  that drags horizontally (col-resize); ``axis="y"`` is a horizontal
 *  strip that drags vertically (row-resize). Emits per-move pixel
 *  deltas — parent does the geometry math. Double-click invokes
 *  ``onDoubleClick`` (used to reset to default split). Pattern lifted
 *  from JobsPanel's split handle. */
function DragHandle({
  axis,
  ariaLabel,
  onDragDelta,
  onDoubleClick,
}: {
  axis: "x" | "y";
  ariaLabel: string;
  onDragDelta: (delta: number) => void;
  onDoubleClick?: () => void;
}) {
  const lastRef = useRef<{ x: number; y: number; pointerId: number } | null>(
    null,
  );
  return (
    <div
      className={axis === "x" ? "analyze-split-x" : "analyze-split-y"}
      role="separator"
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
      aria-label={ariaLabel}
      title="Drag to resize · double-click to reset"
      onPointerDown={(e) => {
        e.preventDefault();
        (e.currentTarget as Element).setPointerCapture(e.pointerId);
        lastRef.current = {
          x: e.clientX,
          y: e.clientY,
          pointerId: e.pointerId,
        };
        document.body.style.cursor =
          axis === "x" ? "col-resize" : "row-resize";
        document.body.style.userSelect = "none";
      }}
      onPointerMove={(e) => {
        const last = lastRef.current;
        if (!last) return;
        const delta =
          axis === "x" ? e.clientX - last.x : e.clientY - last.y;
        if (delta !== 0) onDragDelta(delta);
        lastRef.current = { ...last, x: e.clientX, y: e.clientY };
      }}
      onPointerUp={(e) => {
        if (!lastRef.current) return;
        lastRef.current = null;
        try {
          (e.currentTarget as Element).releasePointerCapture(e.pointerId);
        } catch {
          /* already released */
        }
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }}
      onPointerCancel={(e) => {
        lastRef.current = null;
        try {
          (e.currentTarget as Element).releasePointerCapture(e.pointerId);
        } catch {
          /* already released */
        }
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }}
      onDoubleClick={onDoubleClick}
    />
  );
}

const HIST_BINS = 30;

/** SVG histogram of the *raw* metric values (no smoothing — the
 *  histogram exists to show the underlying distribution, so applying
 *  the EMA filter would defeat the point). Bins span the data's
 *  observed range; bars are colored by their bin-center value mapped
 *  through the current colormap + scale, which makes the histogram
 *  double as a colormap legend.
 *
 *  Pure SVG with a viewBox so it scales cleanly with the pane's
 *  current size — no canvas, no dep on a charting library. */
function HistogramView({
  scores,
  prefs,
}: {
  scores: TokenScores;
  prefs: AnalyzePrefs;
}) {
  const { token_logprobs, token_entropies } = scores;

  const entropyAvailable = Array.isArray(token_entropies);
  const effectiveMetric: Metric =
    prefs.metric === "entropy" && !entropyAvailable ? "loss" : prefs.metric;

  // Raw (unsmoothed) values for the chosen metric.
  const rawValues = useMemo<number[]>(() => {
    const out: number[] = [];
    if (effectiveMetric === "entropy") {
      for (const e of token_entropies ?? []) {
        if (typeof e === "number") out.push(e);
      }
    } else {
      for (const lp of token_logprobs) {
        if (typeof lp === "number") out.push(-lp);
      }
    }
    return out;
  }, [effectiveMetric, token_logprobs, token_entropies]);

  // Color domain = same lo/hi the tokens use. Auto mode here is also
  // 5th/95th percentile but computed off raw values (the histogram
  // shows raw, so the scale should match that). Manual mode uses the
  // user-supplied bounds verbatim.
  const { colorLo, colorHi } = useMemo(() => {
    if (prefs.scaleMode === "manual") {
      const a = Math.min(prefs.manualLo, prefs.manualHi);
      const b = Math.max(prefs.manualLo, prefs.manualHi);
      return { colorLo: a, colorHi: b > a ? b : a + 1 };
    }
    if (rawValues.length === 0) return { colorLo: 0, colorHi: 1 };
    const sorted = rawValues.slice().sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.05)] ?? sorted[0];
    const hi =
      sorted[Math.floor(sorted.length * 0.95)] ?? sorted[sorted.length - 1];
    return { colorLo: lo, colorHi: hi > lo ? hi : lo + 1 };
  }, [rawValues, prefs.scaleMode, prefs.manualLo, prefs.manualHi]);

  const cmap = useMemo(() => getColormap(prefs.cmap), [prefs.cmap]);

  const bins = useMemo(() => {
    if (rawValues.length === 0) {
      return { counts: [] as number[], xMin: 0, xMax: 1, max: 0 };
    }
    let xMin = Infinity;
    let xMax = -Infinity;
    for (const v of rawValues) {
      if (v < xMin) xMin = v;
      if (v > xMax) xMax = v;
    }
    if (xMin === xMax) xMax = xMin + 1;
    const counts = new Array<number>(HIST_BINS).fill(0);
    const span = xMax - xMin;
    for (const v of rawValues) {
      const t = (v - xMin) / span;
      const idx = Math.min(HIST_BINS - 1, Math.max(0, Math.floor(t * HIST_BINS)));
      counts[idx] += 1;
    }
    let max = 0;
    for (const c of counts) if (c > max) max = c;
    return { counts, xMin, xMax, max };
  }, [rawValues]);

  if (rawValues.length === 0) {
    return <div className="muted analyze-empty">No scored values yet.</div>;
  }

  // SVG layout — viewBox is fixed; SVG fills the pane via CSS. Padding
  // values picked so axis labels don't crop at small heights.
  const VW = 600;
  const VH = 200;
  const PAD_L = 36;
  const PAD_R = 8;
  const PAD_T = 6;
  const PAD_B = 22;
  const plotW = VW - PAD_L - PAD_R;
  const plotH = VH - PAD_T - PAD_B;
  const barW = plotW / bins.counts.length;
  const colorSpan = colorHi - colorLo;
  const metricLabel =
    effectiveMetric === "entropy" ? "entropy (nats)" : "loss (nats)";
  const mid = (bins.xMin + bins.xMax) / 2;

  return (
    <div className="analyze-histogram-wrap">
      <div className="analyze-histogram-meta muted">
        n = {rawValues.length} · {metricLabel}
      </div>
      <svg
        className="analyze-histogram-svg"
        viewBox={`0 0 ${VW} ${VH}`}
        preserveAspectRatio="none"
      >
        {/* axes */}
        <line
          x1={PAD_L}
          y1={PAD_T}
          x2={PAD_L}
          y2={PAD_T + plotH}
          stroke="currentColor"
          strokeOpacity={0.4}
          vectorEffect="non-scaling-stroke"
        />
        <line
          x1={PAD_L}
          y1={PAD_T + plotH}
          x2={PAD_L + plotW}
          y2={PAD_T + plotH}
          stroke="currentColor"
          strokeOpacity={0.4}
          vectorEffect="non-scaling-stroke"
        />
        {/* bars — colored by bin-center mapped through the colormap so
            the histogram doubles as a legend for the color encoding */}
        {bins.counts.map((c, i) => {
          const binCenter =
            bins.xMin + ((i + 0.5) / bins.counts.length) * (bins.xMax - bins.xMin);
          const t = (binCenter - colorLo) / colorSpan;
          const fill = cssColor(cmap.fn(t));
          const h = (c / bins.max) * plotH;
          return (
            <rect
              key={i}
              x={PAD_L + i * barW}
              y={PAD_T + plotH - h}
              width={Math.max(0.5, barW - 0.5)}
              height={h}
              fill={fill}
            >
              <title>{`[${(bins.xMin + (i / bins.counts.length) * (bins.xMax - bins.xMin)).toFixed(3)}, ${(bins.xMin + ((i + 1) / bins.counts.length) * (bins.xMax - bins.xMin)).toFixed(3)}): ${c}`}</title>
            </rect>
          );
        })}
        {/* y-axis ticks: 0 and max */}
        <text
          x={PAD_L - 4}
          y={PAD_T + 8}
          fontSize={10}
          fill="currentColor"
          fillOpacity={0.7}
          textAnchor="end"
        >
          {bins.max}
        </text>
        <text
          x={PAD_L - 4}
          y={PAD_T + plotH}
          fontSize={10}
          fill="currentColor"
          fillOpacity={0.7}
          textAnchor="end"
        >
          0
        </text>
        {/* x-axis ticks: min, mid, max — formatted by data magnitude */}
        <text
          x={PAD_L}
          y={PAD_T + plotH + 14}
          fontSize={10}
          fill="currentColor"
          fillOpacity={0.7}
        >
          {fmt(bins.xMin)}
        </text>
        <text
          x={PAD_L + plotW / 2}
          y={PAD_T + plotH + 14}
          fontSize={10}
          fill="currentColor"
          fillOpacity={0.7}
          textAnchor="middle"
        >
          {fmt(mid)}
        </text>
        <text
          x={PAD_L + plotW}
          y={PAD_T + plotH + 14}
          fontSize={10}
          fill="currentColor"
          fillOpacity={0.7}
          textAnchor="end"
        >
          {fmt(bins.xMax)}
        </text>
      </svg>
    </div>
  );

  function fmt(v: number): string {
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }
}

