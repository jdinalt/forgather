/** Editor for the token-meter price table (estimates only).
 *
 *  The cost shown in the meter is cumulative tokens x these per-Mtok rates. This
 *  modal edits the *override* layer (persisted to
 *  ~/.config/forgather/server/agent_pricing.json, hot-reloaded — no restart);
 *  anything not overridden falls back to the built-in defaults shown for
 *  reference. Format is a JSON object of `"model-id-prefix": [input, output]`
 *  in USD per million tokens. */

import { useEffect, useState } from "react";

import {
  PricingTables,
  getAgentPricing,
  putAgentPricing,
} from "../agent-client";
import { ModalBackdrop } from "./ModalBackdrop";

type RateTable = Record<string, [number, number]>;

export function AgentPricingModal({
  onClose,
  onSaved,
}: {
  onClose: () => void;
  onSaved?: () => void;
}) {
  const [defaults, setDefaults] = useState<RateTable>({});
  const [text, setText] = useState("{}");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getAgentPricing()
      .then((t: PricingTables) => {
        setDefaults(t.defaults || {});
        setText(JSON.stringify(t.overrides || {}, null, 2));
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const save = async () => {
    setErr(null);
    let parsed: unknown;
    try {
      parsed = JSON.parse(text || "{}");
    } catch (e) {
      setErr(`Not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      setErr('Expected an object like { "claude-opus-4-8": [5, 25] }');
      return;
    }
    setBusy(true);
    try {
      await putAgentPricing(parsed as RateTable);
      onSaved?.();
      onClose();
    } catch (e) {
      // Surfaces the server's 400 detail (malformed/negative entry).
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const loadDefaults = () => setText(JSON.stringify(defaults, null, 2));
  const clear = () => setText("{}");

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal agent-pricing-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Edit price table"
      >
        <header className="modal-header">
          <h3>Token price table</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <p className="muted" style={{ marginTop: 0 }}>
            Estimates only — the billing dashboard is authoritative. Override the
            built-in rates here; entries are{" "}
            <code>"model-id-prefix": [input, output]</code> in USD per million
            tokens (cache rates are derived: read 0.1x, write 1.25x). Saved
            immediately — no restart. Models not listed fall back to defaults; a
            model in neither shows no cost.
          </p>

          <textarea
            className="agent-pricing-text"
            spellCheck={false}
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={12}
          />

          <div className="agent-pricing-actions">
            <button className="btn-link" onClick={loadDefaults}>
              Load defaults into editor
            </button>
            <button className="btn-link" onClick={clear}>
              Clear overrides
            </button>
          </div>

          {err && (
            <div className="err pad">
              <pre>{err}</pre>
            </div>
          )}

          <details className="agent-pricing-ref">
            <summary>Built-in defaults (reference)</summary>
            <pre>{JSON.stringify(defaults, null, 2)}</pre>
          </details>
        </div>

        <footer className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn-send" disabled={busy} onClick={save}>
            {busy ? "Saving…" : "Save"}
          </button>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
