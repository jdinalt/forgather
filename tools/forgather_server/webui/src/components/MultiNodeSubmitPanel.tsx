import {
  ClusterGpusResponse,
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
  /** Cluster-wide GPU snapshot — used to cap each peer's GPUs spinner
   *  by the actual hardware on that node, and to show the idle count
   *  next to the input. May be undefined while the query is loading
   *  (the panel still renders, just without caps/idle counts). */
  clusterGpus?: ClusterGpusResponse;
  state: MultiNodePanelState;
  onChange: (next: MultiNodePanelState) => void;
  /** Default GPU count to seed for newly-selected peers when we
   *  don't yet know the node's hardware. Picked up from the parent
   *  modal so a config that asks for "2 GPUs" starts every node at
   *  2 (clamped down per-peer once the GPU snapshot arrives). */
  defaultGpus: number;
}

export function MultiNodeSubmitPanel({
  members: membersData,
  clusterGpus,
  state,
  onChange,
  defaultGpus,
}: Props) {
  const members = membersData.members;
  const selfId = membersData.self_node_id;
  const masterId = membersData.master_node_id;
  // Map: node_id → { max, idle } so per-node bounds and the
  // "(N idle of M)" hint are O(1) lookups while we render the
  // participants table. Idle uses the same gate as the single-node
  // Submit dialog: not excluded *and* not runtime-disabled — i.e.
  // GPUs the scheduler is willing to assign right now.
  const gpuStats = new Map<string, { max: number; idle: number }>();
  if (clusterGpus) {
    for (const node of clusterGpus.nodes) {
      const max = node.gpus.length;
      const idle = node.gpus.filter(
        (g) => !g.excluded && !g.disabled,
      ).length;
      gpuStats.set(node.node_id, { max, idle });
    }
  }

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
        // Seed at min(defaultGpus, this node's max) so we never start
        // selected with a value the node can't honour. Falls back to
        // defaultGpus when GPU stats haven't loaded yet.
        const stats = gpuStats.get(m.node_id);
        nproc[m.node_id] = stats
          ? Math.max(1, Math.min(defaultGpus, stats.max))
          : defaultGpus;
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

  const setGpus = (id: string, n: number) => {
    const stats = gpuStats.get(id);
    const cap = stats?.max ?? Number.POSITIVE_INFINITY;
    update({
      perNodeNproc: {
        ...state.perNodeNproc,
        [id]: Math.max(1, Math.min(cap, Math.floor(n) || 1)),
      },
    });
  };

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

      <div className="multinode-members-scroll">
        <table className="multinode-members">
          <thead>
            <tr>
              <th>Use</th>
              <th>Node</th>
              <th>GPUs</th>
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
              const stats = gpuStats.get(m.node_id);
              const gpus = state.perNodeNproc[m.node_id] ?? defaultGpus;
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
                      <span className="node-tag node-tag-err">
                        unreachable
                      </span>
                    )}
                  </td>
                  <td>
                    <input
                      type="number"
                      min={1}
                      max={stats?.max ?? undefined}
                      value={gpus}
                      disabled={!sel}
                      onChange={(e) =>
                        setGpus(m.node_id, Number(e.target.value))
                      }
                      style={{ width: 56 }}
                    />
                    {stats && (
                      <span className="muted">
                        {" "}
                        ({stats.idle} idle of {stats.max})
                      </span>
                    )}
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
