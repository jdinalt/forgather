import { useEffect, useRef, useState } from "react";

import { GenerationParams } from "../inference-client";
import { persistGet, persistSet } from "../persist";
import { InferenceAnalyzePanel } from "./InferenceAnalyzePanel";
import { InferenceChatPanel } from "./InferenceChatPanel";
import { InferenceCompletionPanel } from "./InferenceCompletionPanel";
import { InferenceModelPanel } from "./InferenceModelPanel";

type SubTab = "model" | "completion" | "chat" | "analyze";

export interface InferenceState {
  baseUrl: string;
  /** Bearer token forwarded to the upstream as ``Authorization: Bearer …``
   *  (via the proxy's X-Inference-Auth-Token side-channel so the user's
   *  Authorization to the forgather-server doesn't leak). Auto-populated
   *  from a JobRecord when picking a local server; user-editable for
   *  external OpenAI-compatible servers. Empty = no upstream auth. */
  authToken: string;
  model: string;
  params: GenerationParams;
}

const STORAGE_KEY = "forgather-inference-state";
// max_tokens is intentionally absent — the design intent is "server
// dictates the default." Forgather's own inference server falls back
// to the model's baked-in generation_config with a generous floor;
// vLLM and most other OpenAI-spec servers default to "rest of the
// context window." Either way the client shouldn't lock a number in.
export const DEFAULT_GENERATION_PARAMS: GenerationParams = {};
const DEFAULT_STATE: InferenceState = {
  baseUrl: "http://localhost:8137/v1",
  authToken: "",
  model: "",
  params: DEFAULT_GENERATION_PARAMS,
};

function loadState(): InferenceState {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return DEFAULT_STATE;
  try {
    const parsed = JSON.parse(raw) as Partial<InferenceState>;
    const params: GenerationParams =
      parsed.params && typeof parsed.params === "object"
        ? { ...(parsed.params as GenerationParams) }
        : { ...DEFAULT_STATE.params };
    // Migration: previously-shipped DEFAULT_GENERATION_PARAMS seeded
    // ``max_tokens: 256`` (long-standing) and briefly ``max_tokens: 1024``
    // (a one-build mistake). Both contradicted the "server dictates"
    // design intent. Strip either stale value from returning sessions
    // so the UI's "server picks" claim is actually true; users who
    // explicitly picked 256/1024 will have to re-set, which is a fair
    // trade against the much larger population who never touched it.
    if (params.max_tokens === 256 || params.max_tokens === 1024) {
      delete params.max_tokens;
    }
    return {
      baseUrl:
        typeof parsed.baseUrl === "string"
          ? parsed.baseUrl
          : DEFAULT_STATE.baseUrl,
      // ``authToken`` is intentionally NOT restored. The token comes
      // from authoritative sources on demand each session: the job
      // listing for locally-spawned servers (auto-filled by pickRow),
      // or the explicit reveal endpoint for user-added entries
      // (fetched on Show / Copy click). Restoring a persisted token
      // would (a) survive reloads even when the source server is
      // unreachable / has rotated its token, and (b) silently extend
      // user-added bearers' lifetime beyond the explicit-operator-
      // action window the reveal design intends. Returning sessions
      // pay one re-pick or one Show click — cheap in exchange.
      authToken: "",
      model: typeof parsed.model === "string" ? parsed.model : "",
      params,
    };
  } catch {
    return DEFAULT_STATE;
  }
}

interface InferencePanelProps {
  /** Cross-section trigger from the Datasets cell context menu:
   *  ``{text, key}`` where ``key`` is a Date.now() nonce. When this
   *  changes (key flips), the panel switches to the Analyze sub-tab
   *  and the AnalyzePanel picks up the text and runs scoring. Cleared
   *  by AnalyzePanel via ``onAnalyzeConsumed`` once it has handled
   *  the request so unmount/remount can't re-fire the same payload. */
  pendingAnalyze?: { text: string; key: number } | null;
  /** Called by AnalyzePanel after consuming pendingAnalyze so the
   *  parent can reset App state. Mirrors the existing
   *  pendingExplore / onPreselectConsumed pattern. */
  onAnalyzeConsumed?: () => void;
  /** Cross-section trigger from the Jobs view's "Open in Inference"
   *  button: pre-select the running server identified by this job
   *  id + base URL. Switches to the Model sub-tab so the picker is
   *  visible. ``baseUrl`` is needed because cluster-mode picker
   *  rows key on a base_url hash, not the job id — see
   *  InferenceModelPanel for the dual-matching logic. Same key-
   *  nonce pattern as pendingAnalyze. */
  pendingServerPick?: {
    jobId: string;
    baseUrl: string;
    key: number;
  } | null;
  /** Called by InferenceModelPanel after the picker row was found and
   *  selected, so App can reset the pending state. */
  onServerPickConsumed?: () => void;
}

/** Top-level Inference view. Holds the shared state (base URL, chosen
 *  model, generation params) so the Model sub-panel can configure it and
 *  the Completion sub-panel can consume it. Persists via localStorage so
 *  settings survive page reloads. */
export function InferencePanel({
  pendingAnalyze,
  onAnalyzeConsumed,
  pendingServerPick,
  onServerPickConsumed,
}: InferencePanelProps = {}) {
  const [tab, setTab] = useState<SubTab>("model");
  // Flip to the Analyze tab whenever a fresh pendingAnalyze arrives.
  // Per-key dedup so flipping back to Inference later doesn't bounce
  // the user out of whatever tab they navigated to.
  const lastAnalyzeKeyRef = useRef<number | null>(null);
  useEffect(() => {
    if (pendingAnalyze && pendingAnalyze.key !== lastAnalyzeKeyRef.current) {
      lastAnalyzeKeyRef.current = pendingAnalyze.key;
      setTab("analyze");
    }
  }, [pendingAnalyze]);
  // Flip to the Model tab whenever a fresh server pick arrives — the
  // picker rows live on that subtab. The actual row selection happens
  // inside InferenceModelPanel via the pendingServerPick prop.
  const lastServerPickKeyRef = useRef<number | null>(null);
  useEffect(() => {
    if (
      pendingServerPick &&
      pendingServerPick.key !== lastServerPickKeyRef.current
    ) {
      lastServerPickKeyRef.current = pendingServerPick.key;
      setTab("model");
    }
  }, [pendingServerPick]);
  const [state, setState] = useState<InferenceState>(loadState);
  // Lifted up so the chat panel can hand a rendered prompt to the
  // completion textarea ("Send to completion") and switch tabs.
  // Deliberately not persisted — completion text is a scratchpad,
  // never useful across reloads in practice.
  const [completionText, setCompletionText] = useState("");

  useEffect(() => {
    // Strip ``authToken`` before persisting — see loadState's
    // matching comment. Bearer tokens come from authoritative
    // sources on demand each session and must not leak into
    // localStorage (which would survive reloads, browser
    // restarts, and "user picked a different server but didn't
    // refresh" gaps).
    const { authToken: _drop, ...persistable } = state;
    void _drop;
    persistSet(STORAGE_KEY, JSON.stringify(persistable));
  }, [state]);

  const onSendToCompletion = (rendered: string) => {
    setCompletionText((prev) => (prev ? prev + rendered : rendered));
    setTab("completion");
  };

  return (
    <div className="inference-panel">
      <header className="viewer-header inference-header">
        <div className="inference-header-title">
          <strong>Inference</strong>
          <span className="muted"> — {state.baseUrl}</span>
          {state.model && (
            <span className="muted"> · {state.model}</span>
          )}
          <nav className="tabs">
            <button
              className={tab === "model" ? "active" : ""}
              onClick={() => setTab("model")}
            >
              model
            </button>
            <button
              className={tab === "completion" ? "active" : ""}
              onClick={() => setTab("completion")}
            >
              completion
            </button>
            <button
              className={tab === "chat" ? "active" : ""}
              onClick={() => setTab("chat")}
            >
              chat
            </button>
            <button
              className={tab === "analyze" ? "active" : ""}
              onClick={() => setTab("analyze")}
              title="Score input text per-token (loss + top-K predictions)"
            >
              analyze
            </button>
          </nav>
        </div>
      </header>

      {/* Both sub-panels stay mounted — completion in particular must
          keep its textarea state (and any in-flight stream) across a
          tab flip to "model". */}
      <div style={{ display: tab === "model" ? "block" : "none", flex: 1, minHeight: 0, overflow: "auto" }}>
        <InferenceModelPanel
          state={state}
          setState={setState}
          pendingServerPick={pendingServerPick}
          onServerPickConsumed={onServerPickConsumed}
        />
      </div>
      <div
        style={{
          display: tab === "completion" ? "flex" : "none",
          flex: 1,
          minHeight: 0,
          flexDirection: "column",
        }}
      >
        <InferenceCompletionPanel
          state={state}
          text={completionText}
          setText={setCompletionText}
        />
      </div>
      <div
        style={{
          display: tab === "chat" ? "flex" : "none",
          flex: 1,
          minHeight: 0,
          flexDirection: "column",
        }}
      >
        <InferenceChatPanel
          state={state}
          onSendToCompletion={onSendToCompletion}
        />
      </div>
      <div
        style={{
          display: tab === "analyze" ? "flex" : "none",
          flex: 1,
          minHeight: 0,
          flexDirection: "column",
        }}
      >
        <InferenceAnalyzePanel
          state={state}
          pendingAnalyze={pendingAnalyze}
          onPendingAnalyzeConsumed={onAnalyzeConsumed}
        />
      </div>
    </div>
  );
}
