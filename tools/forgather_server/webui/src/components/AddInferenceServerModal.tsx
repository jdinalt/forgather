import { useState } from "react";

import { AddInferenceServerRequest, api } from "../api";
import { useDemoMode } from "../demoMode";
import { ModalBackdrop } from "./ModalBackdrop";

/** Modal for registering an external inference-server URL.
 *
 *  Mirrors AddServerModal in DatasetsPanel.tsx: label / base URL /
 *  bearer token / per-entry TLS-verify toggle. The Forgather server
 *  persists these in a small JSON registry so the URL + token survive
 *  restarts, freeing the operator from re-typing them every session
 *  when working with external OpenAI-compatible servers (vLLM, a
 *  teammate's box, an external provider). */
export function AddInferenceServerModal({
  onClose,
  onAdded,
  initialBaseUrl,
  initialAuthToken,
}: {
  onClose: () => void;
  onAdded: () => void;
  /** Pre-fill the URL — useful when the operator just typed a URL into
   *  the Server-URL panel and now wants to save it. */
  initialBaseUrl?: string;
  initialAuthToken?: string;
}) {
  const demoMode = useDemoMode();
  const [label, setLabel] = useState("");
  const [baseUrl, setBaseUrl] = useState(initialBaseUrl ?? "");
  const [authToken, setAuthToken] = useState(initialAuthToken ?? "");
  const [showAuthToken, setShowAuthToken] = useState(false);
  // Per-entry TLS policy. Default secure (verify chain + hostname);
  // operator can opt out for SSH-tunneled / out-of-band-secured
  // remotes whose cert doesn't validate against the local CA.
  const [verifyTls, setVerifyTls] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setPending(true);
    setError(null);
    try {
      const req: AddInferenceServerRequest = {
        label: label.trim(),
        base_url: baseUrl.trim(),
        auth_token: authToken.trim(),
        verify_tls: verifyTls,
      };
      await api.addUserInferenceServer(req);
      onAdded();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Add inference server"
      >
        <header className="modal-header">
          <h3>Add inference server</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        {/* Wrap inputs in a <form> with autoComplete="off" plus the
            new-password trick on the token field so Chrome doesn't try
            to autofill the URL as a username for a saved password. */}
        <form
          className="modal-body"
          autoComplete="off"
          onSubmit={(e) => {
            e.preventDefault();
            if (!pending && baseUrl.trim()) void submit();
          }}
        >
          <div className="submit-row">
            <label className="wide">
              Label
              <input
                type="text"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. vllm host, teammate's box"
                autoComplete="off"
                name="inf-label"
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Base URL
              <input
                type="text"
                className="wide"
                inputMode="url"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://infhost:8137/v1"
                autoComplete="off"
                spellCheck={false}
                name="inf-base-url"
              />
            </label>
          </div>
          <div className="muted" style={{ marginTop: -4, marginBottom: 10 }}>
            Saved here for one-click selection in the picker above. The
            URL + token live in <code>~/.config/forgather/server/</code>
            with file mode 0600 — never sent to the browser after the
            initial save. User-added URLs aren't actively probed for
            reachability; use the Server-URL "Test" button to confirm
            after adding.
          </div>
          <div className="submit-row">
            <label className="wide">
              Auth token
              <div className="path-field">
                <input
                  // In demo mode force masked + read-only and hide the
                  // Show / Copy controls — the modal's submit is also
                  // disabled, but the panel could pre-fill ``initialAuthToken``
                  // and a careless Show click would dump the token to
                  // the visitor's screen.
                  type={demoMode || !showAuthToken ? "password" : "text"}
                  className="wide"
                  value={demoMode ? "" : authToken}
                  onChange={(e) => setAuthToken(e.target.value)}
                  readOnly={demoMode}
                  placeholder={
                    demoMode
                      ? "Token entry disabled in demo mode"
                      : "optional — leave blank if the server runs --no-auth"
                  }
                  autoComplete="new-password"
                  spellCheck={false}
                  name="inf-auth-token"
                />
                {!demoMode && (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => setShowAuthToken((v) => !v)}
                    title={showAuthToken ? "Hide token" : "Show token"}
                  >
                    {showAuthToken ? "Hide" : "Show"}
                  </button>
                )}
                {!demoMode && (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => {
                      if (!authToken) return;
                      navigator.clipboard?.writeText(authToken).catch(() => {});
                    }}
                    disabled={!authToken}
                    title="Copy token to clipboard"
                  >
                    Copy
                  </button>
                )}
              </div>
            </label>
          </div>
          <div className="submit-row">
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={verifyTls}
                onChange={(e) => setVerifyTls(e.target.checked)}
              />
              <span>Verify TLS chain + hostname</span>
            </label>
            {!verifyTls && (
              <div
                className="muted"
                style={{
                  marginTop: 4,
                  paddingLeft: 24,
                  color: "var(--warning, #b87000)",
                }}
              >
                ⚠ Chain validation off. The upstream cert is no longer
                authenticated by TLS — only enable this when the
                channel is secured by other means (SSH tunnel, VPN,
                air-gapped LAN). Bearer auth still applies.
              </div>
            )}
          </div>
        </form>
        <footer className="modal-footer">
          <div className="muted current-path">{error ?? ""}</div>
          <div className="btn-row">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void submit()}
              disabled={demoMode || pending || !baseUrl.trim()}
              title={
                demoMode
                  ? "Read-only demo mode — try the live tool to register a server"
                  : undefined
              }
            >
              {pending ? "Adding…" : "Add"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
