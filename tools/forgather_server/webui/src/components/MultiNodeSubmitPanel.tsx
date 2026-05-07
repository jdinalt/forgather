import {
  ClusterMember,
  ClusterMembersResponse,
  MultinodeOverrides,
} from "../api";

/** A panel embedded inside the regular Submit dialog when the server
 *  is running in cluster mode. Lets the operator pick which nodes
 *  participate in the run, with per-node nproc + iface overrides and
 *  an rdzv host selection. The dialog's project/config inputs are
 *  the source of truth for the run's identity — this panel only adds
 *  the cluster-shaped knobs that the standalone "+ Multi-node
 *  training…" modal used to own.
 *
 *  Single-node behaviour: when the operator leaves only the local
 *  node checked, the parent submit treats this as a plain enqueue
 *  and skips the cluster fanout. The panel is purely additive — it
 *  doesn't change the default submit path.
 */

export interface MultiNodePanelState {
  rdzvPort: number;
  selected: Set<string>;
  perNodeNproc: Record<string, number>;
  perNodeIface: Record<string, string>;
  rdzvNodeId: string | null;
  allowMismatch: boolean;
}

export function emptyMultiNodeState(): MultiNodePanelState {
  return {
    rdzvPort: 29400,
    selected: new Set<string>(),
    perNodeNproc: {},
    perNodeIface: {},
    rdzvNodeId: null,
    allowMismatch: false,
  };
}

export function multiNodeStateFromOverrides(
  m: MultinodeOverrides,
): MultiNodePanelState {
  return {
    rdzvPort: m.rdzv_port,
    selected: new Set(m.selected_node_ids),
    perNodeNproc: { ...m.per_node_nproc },
    perNodeIface: { ...m.per_node_iface },
    rdzvNodeId: m.rdzv_node_id,
    allowMismatch: m.allow_version_mismatch,
  };
}

export function multiNodeStateToOverrides(
  s: MultiNodePanelState,
  members: ClusterMember[],
): MultinodeOverrides {
  // Keep selected_node_ids in members order so the persisted view is
  // deterministic across renders. node_rank assignment in the server
  // fanout follows request order — preserving that here means the
  // rendezvous host typically lands at rank 0 by default.
  const ordered = members
    .filter((m) => s.selected.has(m.node_id))
    .map((m) => m.node_id);
  return {
    rdzv_port: s.rdzvPort,
    selected_node_ids: ordered,
    per_node_nproc: s.perNodeNproc,
    per_node_iface: s.perNodeIface,
    rdzv_node_id: s.rdzvNodeId,
    allow_version_mismatch: s.allowMismatch,
  };
}

interface Props {
  members: ClusterMembersResponse;
  state: MultiNodePanelState;
  onChange: (next: MultiNodePanelState) => void;
  /** Default nproc to seed for newly-selected peers. Picked up from
   *  the parent's GPUs spinner so a config that asks for "2 GPUs"
   *  starts every selected node at 2. */
  defaultNproc: number;
}

export function MultiNodeSubmitPanel({
  members: membersData,
  state,
  onChange,
  defaultNproc,
}: Props) {
  const members = membersData.members;
  const selfId = membersData.self_node_id;
  const masterId = membersData.master_node_id;

  const update = (patch: Partial<MultiNodePanelState>) =>
    onChange({ ...state, ...patch });

  const toggleSelected = (m: ClusterMember) => {
    const next = new Set(state.selected);
    const nproc = { ...state.perNodeNproc };
    if (next.has(m.node_id)) {
      next.delete(m.node_id);
    } else {
      next.add(m.node_id);
      if (nproc[m.node_id] == null) {
        nproc[m.node_id] = defaultNproc;
      }
    }
    // If the rdzv host gets unselected, fall back to master so the
    // submit doesn't end up with a phantom rdzv pointer.
    let rdzvNodeId = state.rdzvNodeId;
    if (rdzvNodeId && !next.has(rdzvNodeId)) {
      rdzvNodeId = masterId && next.has(masterId) ? masterId : null;
    }
    update({ selected: next, perNodeNproc: nproc, rdzvNodeId });
  };

  const setNproc = (id: string, n: number) =>
    update({
      perNodeNproc: {
        ...state.perNodeNproc,
        [id]: Math.max(1, Math.floor(n) || 1),
      },
    });

  const setIface = (id: string, name: string) =>
    update({ perNodeIface: { ...state.perNodeIface, [id]: name } });

  const versionWarnings = computeVersionWarnings(members, state.selected);

  return (
    <div className="multinode-panel">
      <div className="modal-row">
        <label>
          Rendezvous port
          <input
            type="number"
            value={state.rdzvPort}
            onChange={(e) =>
              update({ rdzvPort: Number(e.target.value) || 29400 })
            }
            style={{ width: 90 }}
          />
        </label>
        <span className="muted">
          {state.selected.size} participant
          {state.selected.size === 1 ? "" : "s"}
        </span>
      </div>

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
            const sel = state.selected.has(m.node_id);
            const nproc = state.perNodeNproc[m.node_id] ?? defaultNproc;
            const iface = state.perNodeIface[m.node_id] ?? "";
            return (
              <tr
                key={m.node_id}
                className={m.reachable ? "" : "unreachable"}
              >
                <td>
                  <input
                    type="checkbox"
                    checked={sel}
                    disabled={!m.reachable}
                    onChange={() => toggleSelected(m)}
                  />
                </td>
                <td>
                  <strong>{m.hostname}</strong>{" "}
                  <code>
                    {m.address}:{m.port}
                  </code>
                  {m.node_id === selfId && (
                    <span className="node-tag node-tag-ok">this node</span>
                  )}
                  {m.node_id === masterId && (
                    <span className="node-tag node-tag-ok">master</span>
                  )}
                  {!m.reachable && (
                    <span className="node-tag node-tag-err">unreachable</span>
                  )}
                </td>
                <td>
                  <input
                    type="number"
                    min={1}
                    value={nproc}
                    disabled={!sel}
                    onChange={(e) => setNproc(m.node_id, Number(e.target.value))}
                    style={{ width: 60 }}
                  />
                </td>
                <td>
                  {ifaces.length > 0 ? (
                    <select
                      value={iface}
                      disabled={!sel}
                      onChange={(e) => setIface(m.node_id, e.target.value)}
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
                      value={iface}
                      placeholder="eth0"
                      disabled={!sel}
                      onChange={(e) => setIface(m.node_id, e.target.value)}
                    />
                  )}
                </td>
                <td>
                  <input
                    type="radio"
                    name="rdzv-host"
                    checked={state.rdzvNodeId === m.node_id}
                    disabled={!m.reachable || !sel}
                    onChange={() => update({ rdzvNodeId: m.node_id })}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

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
              checked={state.allowMismatch}
              onChange={(e) => update({ allowMismatch: e.target.checked })}
            />{" "}
            I understand — submit anyway
          </label>
        </div>
      )}

      <div className="submit-help muted">
        Multi-node submit assumes every selected node sees{" "}
        <strong>identical absolute paths</strong> for the project
        directory, dataset, output dir, and any cached artefacts —
        either via a shared filesystem or a synchronised local layout.
        The cluster master fans the submit out with the same{" "}
        <code>project_dir</code> / <code>config</code> to every peer.
      </div>
    </div>
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
