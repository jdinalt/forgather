import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  ClusterBandwidthEntry,
  ClusterBandwidthResponse,
  ClusterJob,
  ClusterMember,
  ClusterMembersResponse,
  ClusterGpusResponse,
  ClusterProbe,
  ClusterProbeInterface,
  GpuInfo,
  Job,
} from "../api";
import { GpuPanel, GpuCard } from "./GpuPanel";

/** Cluster-aware Nodes view (Phase 2).
 *
 *  Adds pre-flight surfaces on top of Phase 1's per-node GPU layout:
 *  package versions inline in each header (with diff-highlighting
 *  when a version drifts from the cluster majority), a collapsible
 *  network-interface list per node, and an on-demand pairwise
 *  bandwidth panel.
 */

// Versions we surface inline. Order = display order. ``python`` and
// ``platform`` are intentionally not in the inline list — they go in
// the tooltip — because they rarely drive a hang and would just
// clutter the row.
const HEADLINE_VERSION_KEYS: readonly string[] = [
  "forgather",
  "torch",
  "nccl",
  "transformers",
];


function computeVersionConsensus(
  members: ClusterMember[],
): Record<string, string> {
  // Most common reported value per version key. "Most common" rather
  // than "all the same" because in a 3-node cluster with one straggler
  // we want the divergence to be obvious — flagging two-against-one
  // is the right call.
  const counts: Record<string, Record<string, number>> = {};
  for (const m of members) {
    const versions = m.probe?.versions;
    if (!versions) continue;
    for (const [key, val] of Object.entries(versions)) {
      if (!val) continue;
      if (!counts[key]) counts[key] = {};
      counts[key][val] = (counts[key][val] ?? 0) + 1;
    }
  }
  const consensus: Record<string, string> = {};
  for (const [key, vals] of Object.entries(counts)) {
    let bestVal = "";
    let bestCount = 0;
    for (const [v, c] of Object.entries(vals)) {
      if (c > bestCount) {
        bestCount = c;
        bestVal = v;
      }
    }
    consensus[key] = bestVal;
  }
  return consensus;
}

function VersionChip({
  label,
  value,
  consensus,
}: {
  label: string;
  value: string | undefined;
  consensus: string;
}) {
  const missing = !value || value === "unavailable";
  const diverged = !missing && consensus && value !== consensus;
  let className = "version-chip";
  if (missing) className += " version-chip-muted";
  else if (diverged) className += " version-chip-warn";
  const tooltip = diverged
    ? `Cluster majority: ${consensus}\nThis node: ${value}`
    : missing
      ? `${label} not reported by this node`
      : `${label} ${value}`;
  return (
    <span className={className} title={tooltip}>
      <span className="version-chip-label">{label}</span>
      <span className="version-chip-value">
        {missing ? "—" : value}
      </span>
    </span>
  );
}

function VersionRow({
  probe,
  consensus,
}: {
  probe: ClusterProbe | null;
  consensus: Record<string, string>;
}) {
  if (!probe) {
    return (
      <span
        className="version-row version-row-pending"
        title="Pre-flight probe not yet received from this node"
      >
        probe pending…
      </span>
    );
  }
  return (
    <span className="version-row">
      {HEADLINE_VERSION_KEYS.map((key) => (
        <VersionChip
          key={key}
          label={key}
          value={probe.versions[key]}
          consensus={consensus[key] ?? ""}
        />
      ))}
    </span>
  );
}

/** Render an interface address as ``host/prefix`` (e.g.
 *  ``192.168.1.27/24``). The probe ships ``cidr`` as the network
 *  address with prefix, but what the operator wants to see is the
 *  host's own address combined with the prefix length — that's the
 *  conventional way ``ip addr`` and route tables present interface
 *  bindings, and it lets you read the netmask off without comparing
 *  two cells. */
function formatHostPrefix(address: string, cidr: string): string {
  if (!address) return "—";
  const slash = cidr.lastIndexOf("/");
  if (slash < 0) return address;
  return address + cidr.substring(slash);
}

function InterfaceList({ interfaces }: { interfaces: ClusterProbeInterface[] }) {
  if (interfaces.length === 0) {
    return (
      <div className="iface-empty muted">No IPv4 interfaces reported.</div>
    );
  }
  return (
    <table className="iface-table">
      <thead>
        <tr>
          <th>Interface</th>
          <th>Address</th>
          <th>Subnet</th>
          <th>Link</th>
        </tr>
      </thead>
      <tbody>
        {interfaces.map((i) => (
          <tr key={i.name + i.address} className={i.is_up ? "" : "down"}>
            <td>{i.name}</td>
            <td>
              <code title={`netmask ${i.netmask || "?"}`}>
                {formatHostPrefix(i.address, i.cidr)}
              </code>
            </td>
            <td>
              <code>{i.cidr || "—"}</code>
            </td>
            <td>
              {i.is_up ? "up" : "down"}
              {i.speed_mbps > 0 && (
                <span className="muted"> · {i.speed_mbps} Mbps</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function BandwidthPanel({
  members,
  selfNodeId,
}: {
  members: ClusterMember[];
  selfNodeId: string | null;
}) {
  const bwQ = useQuery<ClusterBandwidthResponse>({
    queryKey: ["cluster", "bandwidth"],
    queryFn: api.getClusterBandwidth,
    refetchInterval: false,
  });
  const refresh = useMutation({
    mutationFn: api.refreshClusterBandwidth,
    onSuccess: () => {
      // The mutation response holds the freshly-measured payload but
      // the GET endpoint also serves the same cache, so a single
      // refetch keeps the list state consistent with what other
      // tabs would see.
      bwQ.refetch();
    },
    onError: (e) => alert(`Bandwidth refresh failed: ${String(e)}`),
  });
  const byPeer = new Map<string, ClusterBandwidthEntry>();
  for (const m of bwQ.data?.measurements ?? []) {
    byPeer.set(m.peer_node_id, m);
  }
  // Show one row per non-self peer, regardless of whether we have a
  // measurement yet — empty rows make it obvious which peers haven't
  // been probed.
  const peers = members.filter((m) => m.node_id !== selfNodeId);
  return (
    <details className="bw-panel" open={false}>
      <summary>
        <span>Bandwidth (this node → peers)</span>
        <button
          className="bw-refresh"
          disabled={refresh.isPending}
          onClick={(e) => {
            e.preventDefault();
            refresh.mutate();
          }}
          title="Run a fresh single-stream throughput measurement to each peer. Sequential — takes ~few seconds per peer."
        >
          {refresh.isPending ? "Measuring…" : "Refresh"}
        </button>
      </summary>
      {peers.length === 0 ? (
        <div className="muted bw-empty">Only this node is in the cluster.</div>
      ) : (
        <table className="bw-table">
          <thead>
            <tr>
              <th>Peer</th>
              <th>Address</th>
              <th>Throughput</th>
              <th>Sample</th>
              <th>Measured</th>
            </tr>
          </thead>
          <tbody>
            {peers.map((m) => {
              const entry = byPeer.get(m.node_id);
              return (
                <tr
                  key={m.node_id}
                  className={entry?.error && entry.error !== "self" ? "err" : ""}
                >
                  <td>{m.hostname}</td>
                  <td>
                    <code>
                      {m.address}:{m.port}
                    </code>
                  </td>
                  <td>
                    {entry && !entry.error
                      ? `${entry.mbps.toFixed(1)} Mbps`
                      : entry?.error
                        ? entry.error
                        : "—"}
                  </td>
                  <td>
                    {entry && entry.bytes_transferred > 0
                      ? `${(entry.bytes_transferred / 1024 / 1024).toFixed(0)} MiB / ${entry.elapsed_seconds.toFixed(2)} s`
                      : "—"}
                  </td>
                  <td>
                    {entry?.timestamp
                      ? new Date(entry.timestamp * 1000).toLocaleTimeString()
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </details>
  );
}

function ClusterJobsPanel() {
  const qc = useQueryClient();
  const jobsQ = useQuery<ClusterJob[]>({
    queryKey: ["cluster", "jobs"],
    queryFn: api.listClusterJobs,
    refetchInterval: 5000,
  });
  const cancelMutation = useMutation({
    mutationFn: (clusterJobId: string) => api.cancelClusterJob(clusterJobId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cluster", "jobs"] }),
    onError: (e) => alert(`Cancel failed: ${String(e)}`),
  });
  const jobs = jobsQ.data ?? [];
  // "Active" now means the rolled-up status hasn't reached a
  // terminal state. Per-rank queue items can finish without the
  // bundle's own status field flipping (it only flips on cancel
  // or once all members are sticky-terminal), so we look at the
  // rollup the server computes from each member's live status.
  const isTerminal = (j: ClusterJob) => {
    const s = j.rolled_up_status ?? j.status;
    return s === "done" || s === "failed" || s === "cancelled";
  };
  const activeJobs = jobs.filter((j) => !isTerminal(j));
  return (
    <details className="cluster-jobs-panel" open={activeJobs.length > 0}>
      <summary>
        <span>
          Cluster Jobs ({activeJobs.length} active
          {jobs.length > activeJobs.length
            ? `, ${jobs.length - activeJobs.length} terminal`
            : ""}
          )
        </span>
        <span className="muted cj-hint">
          Submit via the config's <strong>Run…</strong> action — the
          cluster panel appears at the top of the dialog.
        </span>
      </summary>
      {jobs.length === 0 ? (
        <div className="muted cj-empty">
          {jobsQ.isError
            ? // The list-jobs endpoint is proxied to master from
              // non-master nodes. A transport error here usually
              // means master is unreachable — distinguish that from
              // "the cluster has had no submits yet" so the operator
              // doesn't think their submit silently disappeared.
              `Cluster jobs unavailable: ${String(jobsQ.error)}`
            : "No multi-node jobs submitted yet."}
        </div>
      ) : (
        <table className="cj-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Project / Config</th>
              <th>rdzv</th>
              <th>Members</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => {
              const overall = j.rolled_up_status ?? j.status;
              const terminal = isTerminal(j);
              return (
                <tr
                  key={j.cluster_job_id}
                  className={terminal ? "cancelled" : ""}
                >
                  <td>
                    <code title={j.cluster_job_id}>
                      {j.cluster_job_id.slice(0, 12)}…
                    </code>
                  </td>
                  <td>
                    <code>{j.project_dir}</code> / <code>{j.config}</code>
                  </td>
                  <td>
                    <code>{j.rdzv_endpoint}</code>{" "}
                    <span className="muted">id={j.rdzv_id.slice(0, 8)}</span>
                  </td>
                  <td>
                    {j.members.map((m) => (
                      <div key={m.node_id} className="cj-member">
                        <span>
                          rank {m.node_rank}: {m.hostname} ×
                          {m.nproc_per_node}
                        </span>{" "}
                        {m.current_status && (
                          <span className="muted">
                            [{m.current_status}]
                          </span>
                        )}
                      </div>
                    ))}
                  </td>
                  <td>{overall}</td>
                  <td>
                    {!terminal && (
                      <button
                        className="cj-cancel-btn"
                        disabled={cancelMutation.isPending}
                        onClick={() =>
                          cancelMutation.mutate(j.cluster_job_id)
                        }
                      >
                        Cancel
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </details>
  );
}

export function NodesPanel() {
  const membersQ = useQuery<ClusterMembersResponse>({
    queryKey: ["cluster", "members"],
    queryFn: api.getClusterMembers,
    refetchInterval: 5000,
  });
  const gpusQ = useQuery<ClusterGpusResponse>({
    queryKey: ["cluster", "gpus"],
    queryFn: api.getClusterGpus,
    refetchInterval: 5000,
  });
  const jobsQ = useQuery({
    queryKey: ["jobs", false],
    queryFn: () => api.listJobs(false),
    refetchInterval: 5000,
  });

  if (membersQ.isLoading) {
    return <div className="pane-state muted">Loading cluster members…</div>;
  }
  if (membersQ.isError) {
    return (
      <div className="pane-state err">
        <pre>{String(membersQ.error)}</pre>
      </div>
    );
  }
  const data = membersQ.data;
  if (!data || !data.cluster_name) {
    return (
      <div className="pane-state muted">Cluster mode is not active.</div>
    );
  }

  const gpusByNode = new Map<
    string,
    { gpus: GpuInfo[]; reachable: boolean; error: string | null }
  >();
  for (const entry of gpusQ.data?.nodes ?? []) {
    gpusByNode.set(entry.node_id, {
      gpus: entry.gpus,
      reachable: entry.reachable,
      error: entry.error ?? null,
    });
  }

  const jobByPid = new Map<number, Job>();
  // Per-node reserved GPU sets: a GPU is reserved by Forgather when any
  // running job lists it under gpu_indices. Keyed by node hostname so the
  // cluster nodes panel can mark a peer's GPU "busy" when it actually is.
  const reservedByNode = new Map<string, Set<number>>();
  for (const j of jobsQ.data ?? []) {
    if (j.pid != null) jobByPid.set(j.pid, j);
    if (
      (j.status === "starting" || j.status === "running") &&
      j.gpu_indices &&
      j.node
    ) {
      let set = reservedByNode.get(j.node);
      if (!set) {
        set = new Set<number>();
        reservedByNode.set(j.node, set);
      }
      for (const idx of j.gpu_indices) set.add(idx);
    }
  }

  const sortedMembers = [...data.members].sort((a, b) => {
    const score = (m: ClusterMember) => {
      if (m.node_id === data.master_node_id) return 0;
      if (m.reachable) return 1;
      return 2;
    };
    const sa = score(a);
    const sb = score(b);
    if (sa !== sb) return sa - sb;
    return a.hostname.localeCompare(b.hostname);
  });

  const masterHostname =
    data.master_node_id !== null
      ? data.members.find((m) => m.node_id === data.master_node_id)
          ?.hostname ?? data.master_node_id.slice(0, 8)
      : "(none)";

  const versionConsensus = computeVersionConsensus(data.members);
  const anyDivergence = data.members.some((m) =>
    m.probe?.versions
      ? HEADLINE_VERSION_KEYS.some(
          (k) =>
            versionConsensus[k] &&
            m.probe!.versions[k] &&
            m.probe!.versions[k] !== versionConsensus[k],
        )
      : false,
  );

  return (
    <div className="nodes-panel">
      <header className="nodes-panel-header">
        <h2>Cluster: {data.cluster_name}</h2>
        <span className="nodes-panel-meta">
          {data.members.length} node
          {data.members.length === 1 ? "" : "s"} · master: {masterHostname}
        </span>
        {anyDivergence && (
          <span
            className="node-tag node-tag-warn"
            title="At least one node reports a package version that does not match the cluster majority. Multi-node training is sensitive to this — see the per-node version chips below."
          >
            version mismatch
          </span>
        )}
      </header>
      <ClusterJobsPanel />
      <BandwidthPanel
        members={data.members}
        selfNodeId={data.self_node_id}
      />
      <div className="nodes-panel-rows">
        {sortedMembers.map((m) => {
          const isSelf = m.node_id === data.self_node_id;
          const isMaster = m.node_id === data.master_node_id;
          const gpuEntry = gpusByNode.get(m.node_id);
          return (
            <NodeGroup
              key={m.node_id}
              member={m}
              isSelf={isSelf}
              isMaster={isMaster}
              gpus={gpuEntry?.gpus ?? null}
              gpusError={gpuEntry?.error ?? null}
              jobByPid={jobByPid}
              reservedGpus={reservedByNode.get(m.hostname) ?? null}
              versionConsensus={versionConsensus}
            />
          );
        })}
      </div>
    </div>
  );
}

function NodeGroup({
  member,
  isSelf,
  isMaster,
  gpus,
  gpusError,
  jobByPid,
  reservedGpus,
  versionConsensus,
}: {
  member: ClusterMember;
  isSelf: boolean;
  isMaster: boolean;
  gpus: GpuInfo[] | null;
  gpusError: string | null;
  jobByPid: Map<number, Job>;
  /** GPU indices on this node currently held by a running Forgather job.
   *  ``null`` when no reservations apply (e.g. peer node with no jobs). */
  reservedGpus: Set<number> | null;
  versionConsensus: Record<string, string>;
}) {
  const qc = useQueryClient();
  const togglePolicy = useMutation({
    mutationFn: ({
      gpu_index,
      disabled,
    }: {
      gpu_index: number;
      disabled: boolean;
    }) =>
      api.setNodeGpuPolicy(member.node_id, gpu_index, { disabled }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cluster", "gpus"] }),
    onError: (e) => alert(`Policy update failed: ${String(e)}`),
  });
  const cpu = member.probe?.cpu;
  return (
    <section
      className={
        "node-group" + (member.reachable ? "" : " unreachable")
      }
    >
      <header className="node-group-header">
        <span className="node-group-title">{member.hostname}</span>
        {isMaster && <span className="node-tag node-tag-ok">master</span>}
        {!isMaster && (
          <span className="node-tag node-tag-muted">peer</span>
        )}
        {isSelf && (
          <span
            className="node-tag node-tag-muted"
            title="The webui you're looking at right now"
          >
            this server
          </span>
        )}
        {!member.reachable && (
          <span className="node-tag node-tag-err">unreachable</span>
        )}
        <span className="node-group-meta">
          <span className="node-address">
            {member.address}:{member.port}
          </span>
          {cpu && cpu.logical > 0 && (
            <span
              className="node-cpu muted"
              title={`${cpu.physical} physical cores · ${cpu.ram_gib} GiB RAM`}
            >
              {cpu.logical} cpu · {cpu.ram_gib} GiB
            </span>
          )}
        </span>
      </header>
      <div className="node-group-versions">
        <VersionRow probe={member.probe} consensus={versionConsensus} />
      </div>
      {member.probe?.interfaces && member.probe.interfaces.length > 0 && (
        <details className="node-group-interfaces">
          <summary>
            Interfaces ({member.probe.interfaces.length})
          </summary>
          <InterfaceList interfaces={member.probe.interfaces} />
        </details>
      )}
      <div className="node-group-body">
        {isSelf ? (
          <GpuPanel />
        ) : gpusError ? (
          <div className="node-group-error">
            GPUs unavailable: {gpusError}
          </div>
        ) : gpus === null ? (
          <div className="muted">Loading GPUs…</div>
        ) : gpus.length === 0 ? (
          <div className="muted">No GPUs reported.</div>
        ) : (
          <div className="gpu-grid">
            {gpus.map((g) => (
              <GpuCard
                key={g.index}
                g={g}
                jobByPid={jobByPid}
                reserved={reservedGpus?.has(g.index) ?? false}
                onToggleDisabled={() =>
                  togglePolicy.mutate({
                    gpu_index: g.index,
                    disabled: !g.disabled,
                  })
                }
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
