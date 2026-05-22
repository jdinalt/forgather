import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import {
  api,
  ClusterInferenceServer,
  InferenceServerUser,
  Job,
} from "../api";
import {
  GenerationParams,
  ModelEntry,
  ServerCheckResult,
  checkServer,
  listModels,
} from "../inference-client";
import { AddInferenceServerModal } from "./AddInferenceServerModal";
import { DEFAULT_GENERATION_PARAMS, InferenceState } from "./InferencePanel";

interface Props {
  state: InferenceState;
  setState: (fn: (prev: InferenceState) => InferenceState) => void;
}

interface HealthState {
  kind: "unknown" | "ok" | "auth-failed" | "down";
  /** Error message when not "ok" / "unknown". Distinguishes auth rejection
   *  (401/403) from network/server errors so the hint can guide the user
   *  to the right field — bad token vs. wrong URL vs. server down. */
  message?: string;
}

export function InferenceModelPanel({ state, setState }: Props) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  // Token field defaults to masked. Operators frequently want to copy
  // the token out and paste it into a curl / external client, so we
  // expose a Show toggle (and a Copy button) rather than make them
  // fight a hidden field.
  const [showAuthToken, setShowAuthToken] = useState(false);
  const [health, setHealth] = useState<HealthState>({ kind: "unknown" });
  // User-added inference-server registry. The Add modal opens on
  // "+ Add server"; rows render in their own section below the
  // running-servers list and clicking one fills baseUrl + token
  // exactly like a running-server pick.
  const [addServerOpen, setAddServerOpen] = useState(false);
  // Which row was last picked. Disambiguates selection when a
  // user-registered URL coincides with a spawned/cluster server's URL
  // (e.g. register your local vLLM, later spawn a forgather inference
  // job on the same port) — without this, both rows would render
  // selected because URL string equality matches both. Cleared when
  // the operator edits the URL field directly.
  const [pickedRow, setPickedRow] = useState<
    | { kind: "running"; id: string }
    | { kind: "user"; id: string }
    | null
  >(null);
  const qc = useQueryClient();
  // Name of the last-loaded/saved preset, so Save defaults to overwriting
  // it and Delete has an obvious target. "" = no preset in play (fresh /
  // reset).
  const [activePreset, setActivePreset] = useState<string>("");

  const presetsQ = useQuery({
    queryKey: ["generation-configs"],
    queryFn: api.listGenerationConfigs,
  });

  const activeIsBuiltin = useMemo(() => {
    if (!activePreset) return false;
    return (
      presetsQ.data?.presets.find((p) => p.name === activePreset)?.builtin ??
      false
    );
  }, [activePreset, presetsQ.data]);

  const loadPreset = async (name: string) => {
    try {
      const entry = await api.getGenerationConfig(name);
      setState((prev) => ({
        ...prev,
        params: entry.params as GenerationParams,
      }));
      setActivePreset(name);
    } catch (e) {
      alert(`Load failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const savePreset = async () => {
    const suggested = activePreset || "my-preset";
    const name = window.prompt(
      "Save generation parameters as (letters, digits, space, - _ . ( )):",
      suggested,
    );
    if (!name) return;
    try {
      await api.putGenerationConfig(
        name,
        state.params as Record<string, unknown>,
      );
      setActivePreset(name);
      qc.invalidateQueries({ queryKey: ["generation-configs"] });
    } catch (e) {
      alert(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const deletePreset = async () => {
    if (!activePreset) return;
    if (!window.confirm(`Delete preset "${activePreset}"?`)) return;
    try {
      await api.deleteGenerationConfig(activePreset);
      setActivePreset("");
      qc.invalidateQueries({ queryKey: ["generation-configs"] });
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const resetParams = () => {
    setState((prev) => ({ ...prev, params: { ...DEFAULT_GENERATION_PARAMS } }));
    setActivePreset("");
  };

  // Cluster-mode detection — same pattern DatasetsPanel uses. TanStack
  // dedups by queryKey so this shares cache with App.tsx's gate query.
  const clusterSelfQ = useQuery({
    queryKey: ["cluster-self"],
    queryFn: api.getClusterSelf,
    refetchInterval: 30000,
  });
  // ``isFetched`` rather than ``!!data`` so the picker can wait for
  // the cluster gate to resolve. Without this, ``clusterActive`` is
  // briefly ``false`` (data is undefined during the first fetch),
  // the local-jobs query fires, the picker shows local-only rows,
  // and then swaps to the cluster set when the gate resolves — a
  // visible flicker on every load.
  const clusterGateResolved = clusterSelfQ.isFetched;
  const clusterActive = !!clusterSelfQ.data;

  // Local-jobs query (single-node setups). Only enabled once we've
  // confirmed the cluster gate is NOT active, so cluster masters
  // never see the local-only intermediate state.
  const jobsQ = useQuery({
    queryKey: ["jobs", false],
    queryFn: () => api.listJobs(false),
    refetchInterval: 5000,
    enabled: clusterGateResolved && !clusterActive,
  });

  // Cluster-aggregated picker contents. Includes servers running on
  // every reachable peer plus this one's own. On non-master nodes the
  // endpoint proxies to the master so every webui sees the same list.
  const clusterServersQ = useQuery({
    queryKey: ["cluster", "inference_servers"],
    queryFn: api.getClusterInferenceServers,
    refetchInterval: 5000,
    enabled: clusterGateResolved && clusterActive,
  });

  // User-added external servers (vLLM, remote OpenAI-compatible boxes,
  // etc.) — surfaced beside the live cluster/local list so the operator
  // doesn't have to retype URL + token every session.
  const userServersQ = useQuery({
    queryKey: ["inference-servers-user"],
    queryFn: api.listUserInferenceServers,
  });

  const removeUserServer = useMutation({
    mutationFn: (id: string) => api.deleteUserInferenceServer(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["inference-servers-user"] });
    },
  });

  // Unified picker rows. Both shapes (local Job and ClusterInferenceServer)
  // boil down to: a URL to dial, an auth token to attach, a display
  // label, and an optional health badge. The render path doesn't need
  // to care which source the row came from. Returns ``null`` while the
  // cluster gate is still resolving so the caller can render a
  // placeholder instead of an empty list (which would render as "no
  // live inference jobs" and is misleading during the gate fetch).
  const pickerRows = useMemo<PickerRow[] | null>(() => {
    if (!clusterGateResolved) return null;
    if (clusterActive) {
      return (clusterServersQ.data ?? []).map(_clusterEntryToRow);
    }
    return (jobsQ.data ?? [])
      .filter((j) => j.job_type === "inference" && j.alive)
      .map(_localJobToRow)
      .filter((r): r is PickerRow => r !== null);
  }, [clusterGateResolved, clusterActive, clusterServersQ.data, jobsQ.data]);

  const modelsQ = useQuery({
    // Token in the key so a paste/clear of the token retriggers the fetch
    // — reflects what the upstream actually sees rather than caching a
    // pre-token result.
    queryKey: ["inference-models", state.baseUrl, state.authToken],
    queryFn: () => listModels(state.baseUrl, state.authToken || undefined),
    enabled: false, // manual — user hits "Fetch models"
  });

  const healthM = useMutation({
    mutationFn: (): Promise<ServerCheckResult> =>
      checkServer(state.baseUrl, state.authToken || undefined),
    onSuccess: (result) => {
      if (result.kind === "ok") setHealth({ kind: "ok" });
      else if (result.kind === "auth-failed")
        setHealth({ kind: "auth-failed", message: result.message });
      else setHealth({ kind: "down", message: result.message });
    },
    onError: (err: unknown) =>
      setHealth({
        kind: "down",
        message: err instanceof Error ? err.message : String(err),
      }),
  });

  const setBaseUrl = (baseUrl: string) => {
    setState((prev) => ({ ...prev, baseUrl }));
    setHealth({ kind: "unknown" });
    // Manual edit detaches from any picked row.
    setPickedRow(null);
  };
  const setAuthToken = (authToken: string) => {
    setState((prev) => ({ ...prev, authToken }));
    setHealth({ kind: "unknown" });
  };
  const setModel = (model: string) =>
    setState((prev) => ({ ...prev, model }));
  const setParams = (patch: Partial<GenerationParams>) =>
    setState((prev) => ({ ...prev, params: { ...prev.params, ...patch } }));

  const pickRow = (row: PickerRow) => {
    setState((prev) => ({
      ...prev,
      baseUrl: row.url,
      authToken: row.authToken,
    }));
    setHealth({ kind: "unknown" });
    setPickedRow({ kind: "running", id: row.id });
    // Re-fetch models against the new URL.
    setTimeout(() => modelsQ.refetch(), 0);
  };

  // User-added entries don't carry the bearer token in the listing
  // (tokens stay server-side; the listing only exposes ``has_auth_token``).
  // The proxy resolves the token from the registry by base_url match —
  // see ``_auth_headers_for`` in routes/inference_proxy.py. We still
  // set authToken to "" so the local state doesn't carry a stale token
  // from a previously-picked entry.
  const pickUserServer = (s: InferenceServerUser) => {
    setState((prev) => ({
      ...prev,
      baseUrl: s.base_url,
      authToken: "",
    }));
    setHealth({ kind: "unknown" });
    setPickedRow({ kind: "user", id: s.id });
    setTimeout(() => modelsQ.refetch(), 0);
  };

  const userServers = userServersQ.data ?? [];

  // When the selected user-server is removed (via "×" or out-of-band),
  // clear baseUrl + the picked-row pin so action buttons stop firing
  // against an orphaned URL. Mirrors the parallel effect in
  // DatasetsPanel's DatasetServersTab.
  useEffect(() => {
    if (pickedRow?.kind !== "user") return;
    if (userServers.some((s) => s.id === pickedRow.id)) return;
    setPickedRow(null);
    setState((prev) => ({ ...prev, baseUrl: "", authToken: "" }));
    setHealth({ kind: "unknown" });
  }, [pickedRow, userServers, setState]);

  return (
    <div className="inference-model-panel">
      {/* Running inference servers — top of the list because the typical
          flow is: pick the server you just started, then pick a model on
          it, then tweak the URL only if you need a non-discovered host.
          In cluster mode this list also includes servers running on
          other peers; the master polls every member every ~10s. */}
      <section>
        <h4 className="dyn-heading">
          Running inference servers
          <span className="muted"> ({pickerRows?.length ?? 0})</span>
          {clusterActive && (
            <span className="muted" title="Aggregated across all cluster peers">
              {" "}
              · cluster
            </span>
          )}
        </h4>
        {pickerRows === null && (
          <div className="muted pane-state-small">Loading…</div>
        )}
        {pickerRows !== null && pickerRows.length === 0 && (
          <div className="muted pane-state-small">
            {clusterActive
              ? "No live inference jobs anywhere in the cluster — start one from the Models panel."
              : "No live inference jobs — start one from the Models panel."}
          </div>
        )}
        <ul className="inference-server-list">
          {(pickerRows ?? []).map((row) => {
            // When a row was explicitly picked, use its id to disambiguate
            // — a user-registered URL that coincides with a spawned/cluster
            // server's URL would otherwise highlight both rows. Fall back
            // to URL string match only when nothing has been picked (e.g.
            // baseUrl was restored from localStorage on page load).
            const selected =
              pickedRow !== null
                ? pickedRow.kind === "running" && pickedRow.id === row.id
                : row.url === state.baseUrl;
            return (
              <li
                key={row.id}
                className={
                  "inference-server-row" + (selected ? " selected" : "")
                }
                onClick={() => pickRow(row)}
                title={row.titleHint}
              >
                <span className="inf-server-model">{row.label}</span>
                <span className="muted">{row.hostPortLabel}</span>
                <span className="muted inf-server-id">{row.id}</span>
                {row.healthy === false && (
                  <span
                    className="muted"
                    title={row.healthError || "unhealthy"}
                  >
                    ⚠
                  </span>
                )}
                {row.peerHint && (
                  <span className="muted" title={`Peer node ${row.peerHint}`}>
                    {row.peerHint.slice(0, 8)}
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      {/* User-added servers — persistent across restarts. Lets the
          operator one-click-pick external OpenAI-compatible servers
          (vLLM, remote inference, a teammate's box) without
          retyping URL + token every session. The proxy resolves
          the token from the registry by URL match, so the browser
          never holds tokens for user-added entries. */}
      <section>
        <h4 className="dyn-heading">
          User-added servers
          <span className="muted"> ({userServers.length})</span>
          <button
            style={{ marginLeft: 12 }}
            onClick={() => setAddServerOpen(true)}
            title="Register an external inference-server URL + token"
          >
            + Add server
          </button>
        </h4>
        {userServers.length === 0 && (
          <div className="muted pane-state-small">
            No user-added servers. Use “+ Add server” to register a remote URL.
          </div>
        )}
        <ul className="inference-server-list">
          {userServers.map((s) => {
            // Same disambiguation as the running-servers list above.
            const selected =
              pickedRow !== null
                ? pickedRow.kind === "user" && pickedRow.id === s.id
                : s.base_url === state.baseUrl;
            return (
              <li
                key={s.id}
                className={
                  "inference-server-row" + (selected ? " selected" : "")
                }
                onClick={() => pickUserServer(s)}
                title={s.base_url}
              >
                <span className="inf-server-model">{s.label}</span>
                <span className="muted">{s.base_url}</span>
                {s.has_auth_token && (
                  <span className="muted">auth ✓</span>
                )}
                <button
                  className="tiny"
                  style={{ marginLeft: "auto" }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (
                      window.confirm(
                        `Remove ${s.label} from the registry?`,
                      )
                    ) {
                      removeUserServer.mutate(s.id);
                    }
                  }}
                  title="Remove this entry"
                >
                  ×
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      {/* Models on the selected server */}
      <section>
        <h4 className="dyn-heading">
          Models
          <button
            className="secondary"
            style={{ marginLeft: 10, fontSize: 11 }}
            onClick={() => modelsQ.refetch()}
            disabled={modelsQ.isFetching || !state.baseUrl}
          >
            {modelsQ.isFetching ? "Fetching…" : "Fetch models"}
          </button>
        </h4>
        {modelsQ.error && (
          <>
            <div className="err pane-state-small">
              {errorMessage(modelsQ.error)}
            </div>
            <FetchDebug url={modelsEndpoint(state.baseUrl)} />
          </>
        )}
        {modelsQ.data && modelsQ.data.length === 0 && (
          <div className="muted pane-state-small">
            Server returned an empty model list.
          </div>
        )}
        <ul className="inference-model-list">
          {(modelsQ.data ?? []).map((m: ModelEntry) => {
            const selected = m.id === state.model;
            return (
              <li
                key={m.id}
                className={"inference-model-row" + (selected ? " selected" : "")}
                onClick={() => setModel(m.id)}
              >
                <span className="inf-model-id">{m.id}</span>
                {m.owned_by && <span className="muted">{m.owned_by}</span>}
              </li>
            );
          })}
        </ul>
      </section>

      {/* URL — escape hatch for pointing at a server we didn't launch
          (remote, vLLM, OpenAI, etc.). Picking a server above
          auto-fills both fields; for an external OpenAI-compatible
          server, paste the URL and API key here. */}
      <section>
        <h4 className="dyn-heading">Server URL</h4>
        <div className="submit-row">
          <label className="wide">
            Base URL
            <div className="path-field">
              <input
                type="text"
                className="wide"
                value={state.baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://localhost:8137/v1"
              />
              <button
                type="button"
                className="secondary"
                onClick={() => healthM.mutate()}
                disabled={healthM.isPending}
              >
                {healthM.isPending ? "Testing…" : "Test"}
              </button>
            </div>
          </label>
        </div>
        <div className="submit-row">
          <label className="wide">
            Auth token
            {/* path-field wraps input + buttons in a flex row so the
                input stretches to fit a full bearer token (64 hex
                chars). Show toggles masking; Copy lifts the token
                straight to the clipboard for use in an external
                client (curl / OpenAI SDK / etc.). */}
            <div className="path-field">
              <input
                type={showAuthToken ? "text" : "password"}
                className="wide"
                value={state.authToken}
                onChange={(e) => setAuthToken(e.target.value)}
                placeholder="Bearer token (auto-filled from local server, or paste API key for external)"
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                className="secondary"
                onClick={() => setShowAuthToken((v) => !v)}
                title={showAuthToken ? "Hide token" : "Show token"}
              >
                {showAuthToken ? "Hide" : "Show"}
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  if (!state.authToken) return;
                  navigator.clipboard
                    ?.writeText(state.authToken)
                    .catch(() => {});
                }}
                disabled={!state.authToken}
                title="Copy token to clipboard"
              >
                Copy
              </button>
            </div>
          </label>
        </div>
        <div className="inference-health-status">
          <HealthStatus health={health} pending={healthM.isPending} />
        </div>
        {health.kind === "down" && (
          <FetchDebug url={modelsEndpoint(state.baseUrl)} />
        )}
      </section>

      {/* Generation parameters */}
      <section>
        <h4 className="dyn-heading">Generation parameters</h4>
        <div className="submit-row preset-row">
          <label>
            Preset
            <select
              value={activePreset}
              onChange={(e) => {
                const name = e.target.value;
                if (name) loadPreset(name);
                else setActivePreset("");
              }}
              title="Load a saved generation-parameter preset"
            >
              <option value="">(none)</option>
              {(presetsQ.data?.presets ?? []).map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                  {p.builtin ? " (built-in)" : ""}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="secondary"
            onClick={savePreset}
            title="Save current parameters as a preset"
          >
            Save as…
          </button>
          <button
            type="button"
            className="secondary"
            onClick={deletePreset}
            disabled={!activePreset || activeIsBuiltin}
            title={
              activeIsBuiltin
                ? "Built-in presets can't be deleted. Save under a new name to customize."
                : "Delete the selected preset"
            }
          >
            Delete
          </button>
          <button
            type="button"
            className="secondary"
            onClick={resetParams}
            title="Reset all generation parameters to defaults"
          >
            Reset
          </button>
        </div>
        <div className="submit-row">
          <NumField
            label="max_tokens"
            value={state.params.max_tokens}
            onChange={(v) => setParams({ max_tokens: v })}
            placeholder="256"
          />
          <NumField
            label="temperature"
            value={state.params.temperature}
            onChange={(v) => setParams({ temperature: v })}
            step="any"
            placeholder="model default"
          />
          <NumField
            label="top_p"
            value={state.params.top_p}
            onChange={(v) => setParams({ top_p: v })}
            step="any"
            placeholder="model default"
          />
        </div>
        <div className="submit-row">
          <NumField
            label="top_k"
            value={state.params.top_k}
            onChange={(v) => setParams({ top_k: v })}
            placeholder="model default"
          />
          <NumField
            label="repetition_penalty"
            value={state.params.repetition_penalty}
            onChange={(v) => setParams({ repetition_penalty: v })}
            step="any"
            placeholder="1.0"
          />
          <NumField
            label="typical_p"
            value={state.params.typical_p}
            onChange={(v) => setParams({ typical_p: v })}
            step="any"
            placeholder="off"
          />
        </div>
        <div className="submit-row">
          <label className="wide">
            stop (comma-separated)
            <input
              type="text"
              className="wide"
              value={(state.params.stop ?? []).join(", ")}
              onChange={(e) => {
                const parts = e.target.value
                  .split(",")
                  .map((s) => s.trim())
                  .filter(Boolean);
                setParams({ stop: parts.length > 0 ? parts : undefined });
              }}
              placeholder='e.g. "\\n\\n", "</s>"'
            />
          </label>
          <NumField
            label="seed"
            value={state.params.seed}
            onChange={(v) => setParams({ seed: v })}
            placeholder="random"
          />
        </div>

        <h4
          className="dyn-heading collapsible-heading"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          <span className="tri">{showAdvanced ? "▾" : "▸"}</span>
          Advanced
        </h4>
        {showAdvanced && (
          <>
            {/* Sampling cutoffs & penalties */}
            <div className="submit-row">
              <NumField
                label="min_p"
                value={state.params.min_p}
                onChange={(v) => setParams({ min_p: v })}
                step="any"
                placeholder="off"
              />
              <NumField
                label="epsilon_cutoff"
                value={state.params.epsilon_cutoff}
                onChange={(v) => setParams({ epsilon_cutoff: v })}
                step="any"
                placeholder="off"
              />
              <NumField
                label="eta_cutoff"
                value={state.params.eta_cutoff}
                onChange={(v) => setParams({ eta_cutoff: v })}
                step="any"
                placeholder="off"
              />
            </div>
            <div className="submit-row">
              <NumField
                label="presence_penalty"
                value={state.params.presence_penalty}
                onChange={(v) => setParams({ presence_penalty: v })}
                step="any"
                placeholder="0"
              />
              <NumField
                label="frequency_penalty"
                value={state.params.frequency_penalty}
                onChange={(v) => setParams({ frequency_penalty: v })}
                step="any"
                placeholder="0"
              />
              <NumField
                label="penalty_alpha"
                value={state.params.penalty_alpha}
                onChange={(v) => setParams({ penalty_alpha: v })}
                step="any"
                placeholder="off (contrastive)"
              />
            </div>
            {/* Length control */}
            <div className="submit-row">
              <NumField
                label="min_length"
                value={state.params.min_length}
                onChange={(v) => setParams({ min_length: v })}
                placeholder="0"
              />
              <NumField
                label="min_new_tokens"
                value={state.params.min_new_tokens}
                onChange={(v) => setParams({ min_new_tokens: v })}
                placeholder="0"
              />
              <NumField
                label="no_repeat_ngram_size"
                value={state.params.no_repeat_ngram_size}
                onChange={(v) => setParams({ no_repeat_ngram_size: v })}
                placeholder="0"
              />
            </div>
            <div className="submit-row">
              <NumField
                label="encoder_no_repeat_ngram_size"
                value={state.params.encoder_no_repeat_ngram_size}
                onChange={(v) =>
                  setParams({ encoder_no_repeat_ngram_size: v })
                }
                placeholder="0"
              />
              <NumField
                label="guidance_scale"
                value={state.params.guidance_scale}
                onChange={(v) => setParams({ guidance_scale: v })}
                step="any"
                placeholder="off (CFG)"
              />
            </div>
            {/* Beam search */}
            <div className="submit-row">
              <NumField
                label="num_beams"
                value={state.params.num_beams}
                onChange={(v) => setParams({ num_beams: v })}
                placeholder="1"
              />
              <NumField
                label="num_beam_groups"
                value={state.params.num_beam_groups}
                onChange={(v) => setParams({ num_beam_groups: v })}
                placeholder="1"
              />
              <NumField
                label="diversity_penalty"
                value={state.params.diversity_penalty}
                onChange={(v) => setParams({ diversity_penalty: v })}
                step="any"
                placeholder="0"
              />
            </div>
            <div className="submit-row">
              <NumField
                label="length_penalty"
                value={state.params.length_penalty}
                onChange={(v) => setParams({ length_penalty: v })}
                step="any"
                placeholder="1.0"
              />
              <TriBoolField
                label="early_stopping"
                value={state.params.early_stopping}
                onChange={(v) => setParams({ early_stopping: v })}
              />
            </div>
            {/* Decoding override */}
            <div className="submit-row">
              <TriBoolField
                label="do_sample"
                value={state.params.do_sample}
                onChange={(v) => setParams({ do_sample: v })}
              />
              <label className="dyn-checkbox">
                <input
                  type="checkbox"
                  checked={!!state.params.ignore_eos}
                  onChange={(e) =>
                    setParams({ ignore_eos: e.target.checked || undefined })
                  }
                />
                ignore_eos
              </label>
            </div>
          </>
        )}
      </section>

      {addServerOpen && (
        <AddInferenceServerModal
          onClose={() => setAddServerOpen(false)}
          // Pre-fill from the current Server-URL fields so an operator
          // who typed a URL into the panel and now wants to persist
          // it doesn't have to retype both halves.
          initialBaseUrl={state.baseUrl}
          initialAuthToken={state.authToken}
          onAdded={() => {
            qc.invalidateQueries({ queryKey: ["inference-servers-user"] });
            setAddServerOpen(false);
          }}
        />
      )}
    </div>
  );
}

function HealthStatus({
  health,
  pending,
}: {
  health: HealthState;
  pending: boolean;
}) {
  if (pending) return <span className="muted">Testing…</span>;
  if (health.kind === "unknown")
    return <span className="muted">Not tested</span>;
  if (health.kind === "ok")
    return (
      <span className="health-label health-ok">
        ✓ Reachable + token accepted (/models ok)
      </span>
    );
  if (health.kind === "auth-failed")
    return (
      <span className="health-label health-down">
        ✗ Token rejected — {health.message ?? "401"}
      </span>
    );
  return (
    <span className="health-label health-down">
      ✗ Unreachable — {health.message ?? "unknown error"}
    </span>
  );
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}


function modelsEndpoint(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "") + "/models";
}

/** Neutral diagnostic panel shown after a failed fetch.
 *
 *  Requests go through forgather-server's same-origin proxy, so browser
 *  CORS / PNA concerns don't apply — what remains is upstream
 *  reachability. Show the URL the proxy tried to reach and give the user
 *  a shell command that exercises the same path. */
function FetchDebug({ url }: { url: string }) {
  const curl = `curl -i '${url}'`;
  return (
    <div className="notice">
      <div>
        <strong>Upstream URL:</strong> <code>{url}</code>
      </div>
      <div style={{ marginTop: 6 }}>
        To check reachability from a shell:{" "}
        <code>{curl}</code>
      </div>
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
  step,
  placeholder,
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  step?: string;
  placeholder?: string;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        step={step}
        value={value ?? ""}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") {
            onChange(undefined);
            return;
          }
          const n = Number.parseFloat(raw);
          onChange(Number.isFinite(n) ? n : undefined);
        }}
        placeholder={placeholder}
      />
    </label>
  );
}

function TriBoolField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean | undefined;
  onChange: (v: boolean | undefined) => void;
}) {
  // Tri-state: unset → server/model default, true/false → explicit override.
  // Uses a plain select so the "unset" state is visible and recoverable;
  // a checkbox would collapse unset and false.
  const selectValue =
    value === undefined ? "" : value ? "true" : "false";
  return (
    <label>
      {label}
      <select
        value={selectValue}
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === "" ? undefined : v === "true");
        }}
      >
        <option value="">default</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    </label>
  );
}

function basename(p: string): string {
  const i = p.replace(/\/+$/, "").lastIndexOf("/");
  return i < 0 ? p : p.slice(i + 1);
}

/** Unified shape consumed by the picker rendering loop. Both local
 *  ``Job`` entries and ``ClusterInferenceServer`` entries adapt to
 *  this, so the JSX doesn't branch on source. */
interface PickerRow {
  /** Stable key + display id. Cluster: ``server_id``. Local: job id. */
  id: string;
  /** Full v1 base URL ready to set on state.baseUrl. */
  url: string;
  /** Upstream bearer token, or empty string for ``--no-auth`` upstreams. */
  authToken: string;
  /** Primary label shown in the row — the model name(s) the server
   *  is configured to host. */
  label: string;
  /** Muted secondary label: ``host:port``. */
  hostPortLabel: string;
  /** Health flag (cluster only — local jobs are assumed healthy when
   *  ``alive``). ``undefined`` ⇒ unknown; render nothing. */
  healthy?: boolean;
  /** Health-check error text used as the ⚠ icon's tooltip. */
  healthError?: string;
  /** Cluster peer that owns this server, when known. Renders as a
   *  short prefix badge so the operator can tell remote-peer servers
   *  apart from local ones at a glance. */
  peerHint?: string | null;
  /** Tooltip on the row itself — fuller URL / peer info when the
   *  short badges aren't enough. */
  titleHint?: string;
}

function _clusterEntryToRow(s: ClusterInferenceServer): PickerRow {
  // The cluster base_url already carries scheme://host:port — append
  // /v1 so it matches the picker's expected baseUrl shape.
  const url = `${s.base_url.replace(/\/+$/, "")}/v1`;
  // Label: prefer the configured model routing names (one entry per
  // ``-m`` flag at spawn), fall back to the human label, then the URL.
  const label =
    s.models && s.models.length > 0
      ? s.models.join(", ")
      : s.label || s.base_url;
  // host:port from the base URL. URL() handles bracketed IPv6.
  let hostPort = "";
  try {
    const parsed = new URL(s.base_url);
    hostPort = parsed.host;
  } catch {
    hostPort = s.base_url;
  }
  return {
    id: s.server_id,
    url,
    authToken: s.auth_token,
    label,
    hostPortLabel: hostPort,
    healthy: s.healthy,
    healthError: s.last_health_error,
    peerHint: s.peer_node_id,
    titleHint: s.peer_node_id ? `${s.base_url} — peer ${s.peer_node_id}` : s.base_url,
  };
}

function _localJobToRow(j: Job): PickerRow | null {
  const host =
    typeof j.job_params?.host === "string"
      ? (j.job_params.host as string)
      : "localhost";
  const port =
    typeof j.job_params?.port === "number"
      ? (j.job_params.port as number)
      : null;
  if (!port) return null;
  // 0.0.0.0 isn't dialable. Prefer the server-stamped ``routable_host``
  // (the inference server's LAN address); fall back to localhost only
  // when the server didn't supply one. For explicit binds (not
  // 0.0.0.0), trust what the operator typed.
  const routable =
    typeof j.job_params?.routable_host === "string"
      ? (j.job_params.routable_host as string)
      : null;
  const browserHost =
    host === "0.0.0.0" || host === "::" || !host
      ? routable || "localhost"
      : host;
  const scheme =
    typeof j.job_params?.scheme === "string"
      ? (j.job_params.scheme as string)
      : "http";
  const url = `${scheme}://${browserHost}:${port}/v1`;
  // Label: multi-model job_params carry ``models: [{name,...}]``;
  // single-model carry ``model_path``. Read either shape.
  let label = "?";
  const modelsField = j.job_params?.models;
  if (Array.isArray(modelsField) && modelsField.length > 0) {
    const names = modelsField
      .map((e) => (e && typeof (e as { name?: unknown }).name === "string"
        ? ((e as { name: string }).name)
        : ""))
      .filter(Boolean);
    if (names.length > 0) label = names.join(", ");
  } else if (typeof j.job_params?.model_path === "string") {
    label = basename(j.job_params.model_path as string);
  }
  return {
    id: j.id,
    url,
    authToken: j.auth_token ?? "",
    label,
    hostPortLabel: `${host}:${port}`,
  };
}
