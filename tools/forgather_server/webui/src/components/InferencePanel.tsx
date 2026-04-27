import { useEffect, useState } from "react";

import { GenerationParams } from "../inference-client";
import { persistGet, persistSet } from "../persist";
import { InferenceChatPanel } from "./InferenceChatPanel";
import { InferenceCompletionPanel } from "./InferenceCompletionPanel";
import { InferenceModelPanel } from "./InferenceModelPanel";

type SubTab = "model" | "completion" | "chat";

export interface InferenceState {
  baseUrl: string;
  model: string;
  params: GenerationParams;
}

const STORAGE_KEY = "forgather-inference-state";
// max_tokens default of 256 is deliberate — the server defaults to 16
// for completion, which produces a surprising "typed one sentence and
// it stopped" experience in a free-form textbox.
export const DEFAULT_GENERATION_PARAMS: GenerationParams = { max_tokens: 256 };
const DEFAULT_STATE: InferenceState = {
  baseUrl: "http://localhost:8137/v1",
  model: "",
  params: DEFAULT_GENERATION_PARAMS,
};

function loadState(): InferenceState {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return DEFAULT_STATE;
  try {
    const parsed = JSON.parse(raw) as Partial<InferenceState>;
    return {
      baseUrl:
        typeof parsed.baseUrl === "string"
          ? parsed.baseUrl
          : DEFAULT_STATE.baseUrl,
      model: typeof parsed.model === "string" ? parsed.model : "",
      params:
        parsed.params && typeof parsed.params === "object"
          ? (parsed.params as GenerationParams)
          : DEFAULT_STATE.params,
    };
  } catch {
    return DEFAULT_STATE;
  }
}

/** Top-level Inference view. Holds the shared state (base URL, chosen
 *  model, generation params) so the Model sub-panel can configure it and
 *  the Completion sub-panel can consume it. Persists via localStorage so
 *  settings survive page reloads. */
export function InferencePanel() {
  const [tab, setTab] = useState<SubTab>("model");
  const [state, setState] = useState<InferenceState>(loadState);

  useEffect(() => {
    persistSet(STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  return (
    <div className="inference-panel">
      <header className="viewer-header">
        <div>
          <strong>Inference</strong>
          <span className="muted"> — {state.baseUrl}</span>
          {state.model && (
            <span className="muted"> · {state.model}</span>
          )}
        </div>
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
        </nav>
      </header>

      {/* Both sub-panels stay mounted — completion in particular must
          keep its textarea state (and any in-flight stream) across a
          tab flip to "model". */}
      <div style={{ display: tab === "model" ? "block" : "none", flex: 1, minHeight: 0, overflow: "auto" }}>
        <InferenceModelPanel state={state} setState={setState} />
      </div>
      <div
        style={{
          display: tab === "completion" ? "flex" : "none",
          flex: 1,
          minHeight: 0,
          flexDirection: "column",
        }}
      >
        <InferenceCompletionPanel state={state} />
      </div>
      <div
        style={{
          display: tab === "chat" ? "flex" : "none",
          flex: 1,
          minHeight: 0,
          flexDirection: "column",
        }}
      >
        <InferenceChatPanel state={state} />
      </div>
    </div>
  );
}
