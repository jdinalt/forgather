/** Manage agent connection profiles: create/edit/delete, switch the active
 *  one (no server restart), import a self-signed cert, and pick the model
 *  from the server's live list. */

import { useEffect, useState } from "react";

import {
  AgentProfile,
  AgentProfileWrite,
  activateProfile,
  createProfile,
  deleteProfile,
  fetchServerCert,
  getProfiles,
  listAgentModels,
  updateProfile,
} from "../agent-client";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  onClose: () => void;
  onChanged: () => void; // refresh status/switcher after a change
}

interface FormState {
  label: string;
  provider: string;
  model: string;
  base_url: string;
  api_key: string; // blank = unchanged when editing
  api_key_env: string;
  verify_tls: boolean;
  max_tokens: number;
  max_iterations: number;
}

const BLANK: FormState = {
  label: "",
  provider: "anthropic",
  model: "",
  base_url: "",
  api_key: "",
  api_key_env: "ANTHROPIC_API_KEY",
  verify_tls: true,
  max_tokens: 4096,
  max_iterations: 12,
};

function formFromProfile(p: AgentProfile): FormState {
  return {
    label: p.label,
    provider: p.provider,
    model: p.model,
    base_url: p.base_url,
    api_key: "",
    api_key_env: p.api_key_env,
    verify_tls: p.verify_tls,
    max_tokens: p.max_tokens,
    max_iterations: p.max_iterations,
  };
}

export function AgentSettingsModal({ onClose, onChanged }: Props) {
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null); // null = new
  const [form, setForm] = useState<FormState>(BLANK);
  const [models, setModels] = useState<string[]>([]);
  const [pendingCertPem, setPendingCertPem] = useState<string | null>(null);
  const [hasImportedCert, setHasImportedCert] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const refresh = async (selectAfter?: string | null) => {
    const { active_id, profiles } = await getProfiles();
    setProfiles(profiles);
    setActiveId(active_id);
    if (selectAfter !== undefined) selectProfile(selectAfter, profiles);
    else if (selectedId === null && profiles.length) selectProfile(profiles[0].id, profiles);
  };

  const selectProfile = (id: string | null, list = profiles) => {
    setErr(null);
    setNote(null);
    setModels([]);
    setPendingCertPem(null);
    setSelectedId(id);
    if (id === null) {
      setForm(BLANK);
      setHasImportedCert(false);
      return;
    }
    const p = list.find((x) => x.id === id);
    if (p) {
      setForm(formFromProfile(p));
      setHasImportedCert(p.has_imported_cert);
    }
  };

  useEffect(() => {
    refresh().catch((e) => setErr(String(e?.message ?? e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((f) => ({ ...f, [k]: v }));

  const writeBody = (): AgentProfileWrite => {
    const body: AgentProfileWrite = {
      label: form.label,
      provider: form.provider,
      model: form.model,
      base_url: form.base_url,
      api_key_env: form.api_key_env,
      verify_tls: form.verify_tls,
      max_tokens: Number(form.max_tokens),
      max_iterations: Number(form.max_iterations),
    };
    if (form.api_key) body.api_key = form.api_key; // only overwrite when typed
    if (pendingCertPem !== null) body.ca_cert_pem = pendingCertPem; // import or clear
    return body;
  };

  const save = async () => {
    setBusy(true);
    setErr(null);
    setNote(null);
    try {
      if (selectedId === null) {
        const created = await createProfile(writeBody());
        await refresh(created.id);
      } else {
        await updateProfile(selectedId, writeBody());
        await refresh(selectedId);
      }
      onChanged();
      setNote("Saved.");
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const onDelete = async () => {
    if (selectedId === null) return;
    setBusy(true);
    try {
      await deleteProfile(selectedId);
      await refresh(null);
      onChanged();
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const onActivate = async (id: string) => {
    setBusy(true);
    try {
      await activateProfile(id);
      await refresh(id);
      onChanged();
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const loadModels = async () => {
    setBusy(true);
    setErr(null);
    try {
      const list = await listAgentModels({
        profile_id: selectedId ?? undefined,
        provider: form.provider,
        base_url: form.base_url,
        api_key: form.api_key || undefined,
        api_key_env: form.api_key_env,
        verify_tls: form.verify_tls,
        ca_cert_pem: pendingCertPem ?? undefined,
      });
      setModels(list);
      // Remember last selection; auto-select first if it's gone (weak binding).
      if (form.model && !list.includes(form.model)) {
        set("model", list[0] ?? "");
      } else if (!form.model && list.length) {
        set("model", list[0]);
      }
      setNote(`${list.length} model(s) available.`);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const importCert = async () => {
    if (!form.base_url) {
      setErr("Set base_url first.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const info = await fetchServerCert(form.base_url);
      const ok = window.confirm(
        `Import certificate from ${info.host}:${info.port}?\n\n` +
          `SHA-256 fingerprint:\n${info.sha256}\n\n` +
          `Only accept this if the fingerprint matches your vLLM server's ` +
          `certificate. The agent will then verify against it (with hostname ` +
          `checking off, suitable for a LAN self-signed cert).`,
      );
      if (ok) {
        setPendingCertPem(info.pem);
        setHasImportedCert(true);
        set("verify_tls", true);
        setNote(`Certificate ready to save (fingerprint ${info.sha256.slice(0, 17)}…).`);
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const clearCert = () => {
    setPendingCertPem(""); // empty string clears on save
    setHasImportedCert(false);
    setNote("Certificate will be cleared on save.");
  };

  const isHttps = form.base_url.toLowerCase().startsWith("https");
  const modelOptions = form.model && !models.includes(form.model) ? [form.model, ...models] : models;

  return (
    <ModalBackdrop onClose={onClose}>
      <div className="modal agent-settings-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Agent settings">
        <header className="modal-header">
          <h3>Agent profiles</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">×</button>
        </header>

        <div className="modal-body agent-settings-body">
          <div className="agent-profile-list">
            <button className={"agent-profile-row" + (selectedId === null ? " selected" : "")} onClick={() => selectProfile(null)}>
              + New profile
            </button>
            {profiles.map((p) => (
              <div key={p.id} className={"agent-profile-row" + (selectedId === p.id ? " selected" : "")}>
                <button className="agent-profile-pick" onClick={() => selectProfile(p.id)}>
                  {p.id === activeId && <span className="agent-active-dot" title="Active">●</span>}
                  <span className="agent-profile-label">{p.label}</span>
                  <span className="agent-profile-sub">{p.base_url || "Claude"}</span>
                </button>
                {p.id !== activeId && (
                  <button className="btn-link" disabled={busy} onClick={() => onActivate(p.id)}>
                    Use
                  </button>
                )}
              </div>
            ))}
          </div>

          <div className="agent-profile-editor">
            <label>Label
              <input value={form.label} onChange={(e) => set("label", e.target.value)} />
            </label>
            <label>Provider
              <select value={form.provider} onChange={(e) => set("provider", e.target.value)}>
                <option value="anthropic">anthropic (Claude or vLLM)</option>
              </select>
            </label>
            <label>Base URL <span className="muted">(blank = Claude; else vLLM, e.g. https://kitt:8000)</span>
              <input value={form.base_url} placeholder="https://kitt:8000" onChange={(e) => set("base_url", e.target.value)} />
            </label>

            <div className="agent-model-row">
              <label className="grow">Model <span className="muted">(blank = auto: first available)</span>
                <select value={form.model} onChange={(e) => set("model", e.target.value)}>
                  <option value="">(auto — first available)</option>
                  {modelOptions.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </label>
              <button className="btn-secondary" disabled={busy} onClick={loadModels}>Load models</button>
            </div>

            <label>API key {selectedId !== null && <span className="muted">(blank = unchanged)</span>}
              <input type="password" value={form.api_key} placeholder={selectedId !== null ? "••••••••" : ""} onChange={(e) => set("api_key", e.target.value)} />
            </label>
            <label>…or API key env var
              <input value={form.api_key_env} onChange={(e) => set("api_key_env", e.target.value)} />
            </label>

            {isHttps && (
              <div className="agent-tls-block">
                <label className="agent-checkbox">
                  <input type="checkbox" checked={form.verify_tls} onChange={(e) => set("verify_tls", e.target.checked)} />
                  Verify TLS certificate
                </label>
                {!form.verify_tls && (
                  <div className="agent-tls-warn">⚠ Accepts any certificate — vulnerable to MITM. Prefer importing the cert.</div>
                )}
                <div className="agent-cert-row">
                  <span className="muted">{hasImportedCert ? "Certificate imported" : "No certificate imported"}</span>
                  <button className="btn-link" disabled={busy} onClick={importCert}>Import certificate…</button>
                  {hasImportedCert && <button className="btn-link" disabled={busy} onClick={clearCert}>Clear</button>}
                </div>
              </div>
            )}

            <div className="agent-num-row">
              <label>Max tokens
                <input type="number" value={form.max_tokens} onChange={(e) => set("max_tokens", Number(e.target.value))} />
              </label>
              <label>Max tool iterations
                <input type="number" value={form.max_iterations} onChange={(e) => set("max_iterations", Number(e.target.value))} />
              </label>
            </div>

            {err && <div className="err pad"><pre>{err}</pre></div>}
            {note && <div className="agent-note">{note}</div>}

            <div className="agent-editor-buttons">
              <button className="btn-send" disabled={busy || !form.label} onClick={save}>
                {selectedId === null ? "Create" : "Save"}
              </button>
              {selectedId !== null && selectedId !== activeId && (
                <button className="btn-secondary" disabled={busy} onClick={() => onActivate(selectedId)}>Activate</button>
              )}
              {selectedId !== null && (
                <button className="btn-reject" disabled={busy} onClick={onDelete}>Delete</button>
              )}
            </div>
          </div>
        </div>
      </div>
    </ModalBackdrop>
  );
}
