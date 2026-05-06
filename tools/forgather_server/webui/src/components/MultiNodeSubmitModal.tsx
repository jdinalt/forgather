import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import {
  api,
  ClusterMembersResponse,
  ClusterMember,
  ClusterJobSubmitRequest,
} from "../api";
import { persistGet, persistSet } from "../persist";
import { ModalBackdrop } from "./ModalBackdrop";

/** Multi-node training submit modal.
 *
 *  v1 keeps the inputs minimal but covers the things the Samantha
 *  multi-node tutorial says you must get right:
 *   - Pick which nodes participate
 *   - Per-node nproc (defaults to 1)
 *   - Per-node NCCL_SOCKET_IFNAME (free-text or pick from probe)
 *   - rdzv host (defaults to master)
 *   - Acknowledge version mismatch when present
 *
 *  Project + config come in as text inputs because the cluster can
 *  span hosts with different mount layouts; pre-populating from a
 *  selected config in the local tree would be misleading on a
 *  non-shared-FS deployment. The user types the path that exists on
 *  every participating node (typical setup uses an NFS export at the
 *  same path on each).
 */

interface PersistedSettings {
  projectDir: string;
  config: string;
  rdzvPort: number;
  perNodeNproc: Record<string, number>;
  perNodeIface: Record<string, string>;
  selectedNodes: string[];
  rdzvNodeId: string;
}

const STORAGE_KEY = "forgather-multinode-submit-v1";

function loadPersisted(): Partial<PersistedSettings> {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function savePersisted(s: PersistedSettings) {
  persistSet(STORAGE_KEY, JSON.stringify(s));
}

interface Props {
  onClose: () => void;
  onSubmitted?: (clusterJobId: string) => void;
}

export function MultiNodeSubmitModal({ onClose, onSubmitted }: Props) {
  const qc = useQueryClient();
  const persisted = useMemo(() => loadPersisted(), []);
  const membersQ = useQuery<ClusterMembersResponse>({
    queryKey: ["cluster", "members"],
    queryFn: api.getClusterMembers,
    refetchInterval: 5000,
  });
  const members = membersQ.data?.members ?? [];
  const masterId = membersQ.data?.master_node_id ?? null;

  const [projectDir, setProjectDir] = useState(persisted.projectDir ?? "");
  const [config, setConfig] = useState(persisted.config ?? "");
  const [rdzvPort, setRdzvPort] = useState(persisted.rdzvPort ?? 29400);
  const [rdzvNodeId, setRdzvNodeId] = useState(
    persisted.rdzvNodeId ?? "",
  );
  const [selected, setSelected] = useState<Set<string>>(
    new Set(persisted.selectedNodes ?? []),
  );
  const [perNodeNproc, setPerNodeNproc] = useState<Record<string, number>>(
    persisted.perNodeNproc ?? {},
  );
  const [perNodeIface, setPerNodeIface] = useState<Record<string, string>>(
    persisted.perNodeIface ?? {},
  );
  const [allowMismatch, setAllowMismatch] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Default selected = all reachable when nothing was previously
  // persisted. Default rdzv host = master.
  useEffect(() => {
    if (!membersQ.data) return;
    if (selected.size === 0) {
      const defaultIds = membersQ.data.members
        .filter((m) => m.reachable)
        .map((m) => m.node_id);
      setSelected(new Set(defaultIds));
    }
    if (!rdzvNodeId && membersQ.data.master_node_id) {
      setRdzvNodeId(membersQ.data.master_node_id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [membersQ.data]);

  const submitMutation = useMutation({
    mutationFn: (req: ClusterJobSubmitRequest) => api.submitClusterJob(req),
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["cluster", "jobs"] });
      onSubmitted?.(resp.cluster_job.cluster_job_id);
      // If there were warnings the server didn't block, surface them
      // briefly so the operator notices.
      if (resp.warnings.length > 0) {
        alert(
          "Submitted with warnings:\n\n" + resp.warnings.join("\n"),
        );
      }
      onClose();
    },
    onError: (e) => setError(String(e)),
  });

  const handleSubmit = () => {
    setError(null);
    if (!projectDir.trim()) {
      setError("Project dir is required");
      return;
    }
    if (!config.trim()) {
      setError("Config is required");
      return;
    }
    if (selected.size === 0) {
      setError("Select at least one node");
      return;
    }
    const orderedSelected = members
      .filter((m) => selected.has(m.node_id))
      .map((m) => m.node_id);
    const req: ClusterJobSubmitRequest = {
      project_dir: projectDir.trim(),
      config: config.trim(),
      members: orderedSelected.map((id) => ({
        node_id: id,
        nproc_per_node: Math.max(1, perNodeNproc[id] ?? 1),
        nccl_socket_ifname: (perNodeIface[id] || "").trim() || null,
      })),
      rdzv_node_id: rdzvNodeId || undefined,
      rdzv_port: rdzvPort,
      allow_version_mismatch: allowMismatch,
    };
    savePersisted({
      projectDir: req.project_dir,
      config: req.config,
      rdzvPort: rdzvPort,
      perNodeNproc,
      perNodeIface,
      selectedNodes: orderedSelected,
      rdzvNodeId: rdzvNodeId,
    });
    submitMutation.mutate(req);
  };

  const versionWarnings = computeVersionWarnings(members, selected);

  return (
    <ModalBackdrop onClose={onClose}>
      <div className="modal-card multinode-modal">
        <div className="modal-header">
          <h2>Multi-node Training</h2>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="modal-body">
          <label className="modal-row">
            <span>Project directory</span>
            <input
              type="text"
              value={projectDir}
              onChange={(e) => setProjectDir(e.target.value)}
              placeholder="/path/visible/on/every/node"
            />
          </label>
          <label className="modal-row">
            <span>Config</span>
            <input
              type="text"
              value={config}
              onChange={(e) => setConfig(e.target.value)}
              placeholder="train.yaml"
            />
          </label>

          <div className="modal-section">
            <h3>Participants ({selected.size})</h3>
            <table className="multinode-members">
              <thead>
                <tr>
                  <th>Use</th>
                  <th>Node</th>
                  <th>nproc</th>
                  <th>NCCL iface</th>
                  <th>rdzv host</th>
                </tr>
              </thead>
              <tbody>
                {members.map((m) => {
                  const ifaces = (m.probe?.interfaces ?? []).filter(
                    (i) => i.is_up && !i.address.startsWith("127."),
                  );
                  const sel = selected.has(m.node_id);
                  return (
                    <tr
                      key={m.node_id}
                      className={
                        m.reachable ? "" : "unreachable"
                      }
                    >
                      <td>
                        <input
                          type="checkbox"
                          checked={sel}
                          disabled={!m.reachable}
                          onChange={(e) => {
                            const next = new Set(selected);
                            if (e.target.checked) next.add(m.node_id);
                            else next.delete(m.node_id);
                            setSelected(next);
                          }}
                        />
                      </td>
                      <td>
                        <strong>{m.hostname}</strong>{" "}
                        <code>
                          {m.address}:{m.port}
                        </code>
                        {m.node_id === masterId && (
                          <span className="node-tag node-tag-ok">master</span>
                        )}
                        {!m.reachable && (
                          <span className="node-tag node-tag-err">
                            unreachable
                          </span>
                        )}
                      </td>
                      <td>
                        <input
                          type="number"
                          min={1}
                          value={perNodeNproc[m.node_id] ?? 1}
                          onChange={(e) =>
                            setPerNodeNproc({
                              ...perNodeNproc,
                              [m.node_id]: Number(e.target.value) || 1,
                            })
                          }
                          style={{ width: 60 }}
                        />
                      </td>
                      <td>
                        {ifaces.length > 0 ? (
                          <select
                            value={perNodeIface[m.node_id] ?? ""}
                            onChange={(e) =>
                              setPerNodeIface({
                                ...perNodeIface,
                                [m.node_id]: e.target.value,
                              })
                            }
                          >
                            <option value="">(auto)</option>
                            {ifaces.map((i) => (
                              <option key={i.name} value={i.name}>
                                {i.name} — {i.address}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="text"
                            value={perNodeIface[m.node_id] ?? ""}
                            placeholder="eth0"
                            onChange={(e) =>
                              setPerNodeIface({
                                ...perNodeIface,
                                [m.node_id]: e.target.value,
                              })
                            }
                          />
                        )}
                      </td>
                      <td>
                        <input
                          type="radio"
                          name="rdzv-host"
                          checked={rdzvNodeId === m.node_id}
                          disabled={!m.reachable || !sel}
                          onChange={() => setRdzvNodeId(m.node_id)}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <label className="modal-row">
              <span>Rendezvous port</span>
              <input
                type="number"
                value={rdzvPort}
                onChange={(e) => setRdzvPort(Number(e.target.value) || 29400)}
                style={{ width: 100 }}
              />
            </label>
          </div>

          {versionWarnings.length > 0 && (
            <div className="modal-warning">
              <strong>Version mismatch across selected participants:</strong>
              <ul>
                {versionWarnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
              <label>
                <input
                  type="checkbox"
                  checked={allowMismatch}
                  onChange={(e) => setAllowMismatch(e.target.checked)}
                />{" "}
                I understand — submit anyway
              </label>
            </div>
          )}

          {error && (
            <div className="modal-error">
              <pre>{error}</pre>
            </div>
          )}
        </div>
        <div className="modal-footer">
          <button className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="primary"
            disabled={submitMutation.isPending}
            onClick={handleSubmit}
          >
            {submitMutation.isPending ? "Submitting…" : "Submit"}
          </button>
        </div>
      </div>
    </ModalBackdrop>
  );
}

function computeVersionWarnings(
  members: ClusterMember[],
  selected: Set<string>,
): string[] {
  const participants = members.filter((m) => selected.has(m.node_id));
  if (participants.length < 2) return [];
  const counts: Record<string, Record<string, number>> = {};
  for (const m of participants) {
    const versions = m.probe?.versions;
    if (!versions) continue;
    for (const key of ["forgather", "torch", "nccl", "transformers"]) {
      const v = versions[key];
      if (!v) continue;
      if (!counts[key]) counts[key] = {};
      counts[key][v] = (counts[key][v] ?? 0) + 1;
    }
  }
  const warnings: string[] = [];
  for (const [key, vals] of Object.entries(counts)) {
    if (Object.keys(vals).length > 1) {
      warnings.push(
        `${key}: ` +
          Object.entries(vals)
            .map(([v, n]) => `${v} (${n})`)
            .join(", "),
      );
    }
  }
  return warnings;
}
