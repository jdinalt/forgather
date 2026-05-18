import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  ClusterBandwidthEntry,
  ClusterBandwidthResponse,
  ClusterJob,
  ClusterLatencyEntry,
  ClusterLatencyResponse,
  ClusterMember,
  ClusterMembersResponse,
  ClusterGpusResponse,
  ClusterProbe,
  ClusterProbeInterface,
  GpuInfo,
  Job,
  RUNNING_JOB_STATUSES,
} from "../api";
import { DatasetsClusterTab } from "./DatasetsPanel";
import type { SelectedLeaf } from "./DatasetsExploreTab";
/** Format a byte count as a compact MiB/GiB string. Local to the
 *  Nodes panel so the cluster view doesn't depend on GpuPanel's
 *  internal helper. */
function fmtMiB(bytes: number): string {
  const mib = bytes / (1024 * 1024);
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${Math.round(mib)} MiB`;
}

/** Cluster view — tabbed panel covering everything cluster-scoped:
 *
 *   - jobs:     bundle records (multi-node training jobs)
 *   - network:  pairwise bandwidth probe + (future) other network tools
 *   - nodes:    per-peer rollup with versions, interfaces, and GPUs
 *   - datasets: master-aggregated dataset_server / dataset inventory
 *
 *  Shown in the sidebar as "Cluster" (cluster-only). The sidebar
 *  group labelled "Nodes" is a separate surface that lists peers
 *  with SSO links.
 */

// Versions we surface inline. Order = display order. ``python`` and
// ``platform`` are intentionally not in the inline list — they go in
// the tooltip — because they rarely drive a hang and would just
// clutter the row.
//
// Exported so the sidebar Nodes group (ClusterSidebarPanel) can
// classify each peer's health using the same divergence rules as
// this view.
export const HEADLINE_VERSION_KEYS: readonly string[] = [
  "forgather",
  "torch",
  "nccl",
  "transformers",
];

/** Per-node health classification used by the sidebar dot.
 *
 *  - ``down``  — peer is not reachable over HTTP. Red.
 *  - ``warn``  — peer answers, but a headline version is missing or
 *                differs from the cluster majority. Yellow. This is
 *                the state that flagged kitt's nvml/nccl going
 *                AWOL after a driver glitch: the peer was still up
 *                but its version row no longer matched.
 *  - ``ok``    — reachable and no version mismatch (or the probe
 *                hasn't returned yet, in which case we don't
 *                preemptively warn). Green.
 *
 *  ``consensus`` is the dict from ``computeVersionConsensus``.
 */
export function nodeHealth(
  m: ClusterMember,
  consensus: Record<string, string>,
): "ok" | "warn" | "down" {
  if (!m.reachable) return "down";
  const versions = m.probe?.versions;
  if (!versions) return "ok";
  for (const key of HEADLINE_VERSION_KEYS) {
    const expected = consensus[key];
    if (!expected) continue; // no node reports this key — nothing to compare
    const value = versions[key];
    const missing = !value || value === "unavailable";
    if (missing || value !== expected) return "warn";
  }
  return "ok";
}

export function computeVersionConsensus(
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

function NetworkTab({
  members,
  selfNodeId,
}: {
  members: ClusterMember[];
  selfNodeId: string | null;
}) {
  const qc = useQueryClient();
  const bwQ = useQuery<ClusterBandwidthResponse>({
    queryKey: ["cluster", "bandwidth"],
    queryFn: api.getClusterBandwidth,
    refetchInterval: false,
  });
  const latQ = useQuery<ClusterLatencyResponse>({
    queryKey: ["cluster", "latency"],
    queryFn: api.getClusterLatency,
    refetchInterval: false,
  });
  // Tracks which peer is currently being probed. Set as the
  // per-peer requests run sequentially; ``null`` when the
  // orchestrator is idle. The row for the matching node_id renders
  // "Measuring…" in place of its result columns.
  const [activeProbe, setActiveProbe] = useState<string | null>(null);
  const [probeError, setProbeError] = useState<string | null>(null);

  const byBw = new Map<string, ClusterBandwidthEntry>();
  for (const m of bwQ.data?.measurements ?? []) byBw.set(m.peer_node_id, m);
  const byLat = new Map<string, ClusterLatencyEntry>();
  for (const m of latQ.data?.measurements ?? []) byLat.set(m.peer_node_id, m);

  // Show one row per non-self peer regardless of whether we have a
  // measurement yet — empty rows make it obvious which peers haven't
  // been probed.
  const peers = members.filter((m) => m.node_id !== selfNodeId);

  // Click handler: walks the peer list sequentially. For each peer
  // we set ``activeProbe`` to the peer's node_id, kick off the
  // latency probe (fast, ~50 ms steady-state on a LAN) followed by
  // the adaptive bandwidth probe (~3 s of steady-state), then move
  // to the next peer. Bandwidth probes are sequential because two
  // simultaneous bulk transfers would saturate the local NIC and
  // under-report each link's actual throughput; latency could go in
  // parallel, but keeping the ordering simple makes the per-row
  // progress feedback unambiguous.
  const probing = activeProbe !== null;
  const runAll = async () => {
    if (probing) return;
    setProbeError(null);
    try {
      for (const m of peers) {
        setActiveProbe(m.node_id);
        // Latency first (cheap; warms the connection pool for the
        // bandwidth pass), then bandwidth.
        try {
          await api.refreshClusterLatencyOne(m.node_id);
          // Push the latest result into the cached list query so the
          // row re-renders without waiting for the next refetch.
          qc.invalidateQueries({ queryKey: ["cluster", "latency"] });
        } catch (e) {
          // Don't abort the whole loop if one peer fails — show the
          // error in the row and keep going.
          console.warn(`latency probe for ${m.hostname} failed:`, e);
        }
        try {
          await api.refreshClusterBandwidthOne(m.node_id);
          qc.invalidateQueries({ queryKey: ["cluster", "bandwidth"] });
        } catch (e) {
          console.warn(`bandwidth probe for ${m.hostname} failed:`, e);
        }
      }
    } catch (e) {
      setProbeError(String(e));
    } finally {
      setActiveProbe(null);
    }
  };

  return (
    <div className="bw-panel cluster-tab-body">
      <div className="cluster-tab-heading">
        <span>Network (this node → peers)</span>
        <button
          className="bw-refresh"
          disabled={probing || peers.length === 0}
          onClick={runAll}
          title={
            "Measure latency (30 round-trips) and bandwidth " +
            "(adaptive ~2 s sample) against each peer in turn. " +
            "Sequential — two simultaneous bulk transfers would " +
            "saturate the local NIC."
          }
        >
          {probing ? "Measuring…" : "Refresh"}
        </button>
      </div>
      {probeError && <div className="err">{probeError}</div>}
      {peers.length === 0 ? (
        <div className="muted bw-empty">Only this node is in the cluster.</div>
      ) : (
        <table className="bw-table">
          <thead>
            <tr>
              <th>Peer</th>
              <th>Address</th>
              <th>Latency (min / med / max)</th>
              <th>Throughput</th>
              <th>Sample</th>
              <th>Measured</th>
            </tr>
          </thead>
          <tbody>
            {peers.map((m) => {
              const bw = byBw.get(m.node_id);
              const lat = byLat.get(m.node_id);
              const isActive = activeProbe === m.node_id;
              const rowErr =
                !isActive &&
                ((bw?.error && bw.error !== "self") ||
                  (lat?.error && lat.error !== "self"));
              const latestTs = Math.max(
                bw?.timestamp ?? 0,
                lat?.timestamp ?? 0,
              );
              return (
                <tr key={m.node_id} className={rowErr ? "err" : ""}>
                  <td>{m.hostname}</td>
                  <td>
                    <code>
                      {m.address}:{m.port}
                    </code>
                  </td>
                  <td>
                    {isActive ? (
                      <span className="muted">Measuring…</span>
                    ) : lat && !lat.error ? (
                      `${lat.min_ms.toFixed(2)} / ${lat.median_ms.toFixed(2)} / ${lat.max_ms.toFixed(2)} ms`
                    ) : lat?.error ? (
                      lat.error
                    ) : (
                      "—"
                    )}
                  </td>
                  <td>
                    {isActive ? (
                      <span className="muted">Measuring…</span>
                    ) : bw && !bw.error ? (
                      `${bw.mbps.toFixed(1)} Mbps`
                    ) : bw?.error ? (
                      bw.error
                    ) : (
                      "—"
                    )}
                  </td>
                  <td>
                    {isActive ? (
                      <span className="muted">…</span>
                    ) : bw && bw.bytes_transferred > 0 ? (
                      `${(bw.bytes_transferred / 1024 / 1024).toFixed(0)} MiB / ${bw.elapsed_seconds.toFixed(2)} s`
                    ) : (
                      "—"
                    )}
                  </td>
                  <td>
                    {latestTs
                      ? new Date(latestTs * 1000).toLocaleTimeString()
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function ClusterJobsTab() {
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
    <div className="cluster-jobs-panel cluster-tab-body">
      <div className="cluster-tab-heading">
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
      </div>
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
    </div>
  );
}

export function ClusterPanel({
  onOpenInExplore,
}: {
  /** Wired by App.tsx so a click on a dataset row in the Datasets
   *  tab here can navigate the outer view to Datasets and pre-select
   *  the row in Explore. Optional — when omitted, the rows render
   *  inert. */
  onOpenInExplore?: (leaf: SelectedLeaf) => void;
} = {}) {
  // Tab state must live above the early-returns below — once the
  // cluster comes online and the panel renders for the first time,
  // we want the user's tab selection to persist across loading
  // states (e.g. a transient membersQ.isLoading after a refetch).
  const [tab, setTab] = useState<ClusterTab>("nodes");
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
      RUNNING_JOB_STATUSES.has(j.status) &&
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
    <div className="cluster-panel">
      <header className="viewer-header cluster-panel-header">
        <div className="cluster-panel-header-title">
          <strong>Cluster</strong>
          <span className="muted"> — {data.cluster_name}</span>
          <span className="muted">
            {" · "}
            {data.members.length} node{data.members.length === 1 ? "" : "s"}
            {" · master: "}
            {masterHostname}
          </span>
          {anyDivergence && (
            <span
              className="node-tag node-tag-warn"
              title="At least one node reports a package version that does not match the cluster majority. Multi-node training is sensitive to this — see the per-node version chips on the Nodes tab."
            >
              version mismatch
            </span>
          )}
          <nav className="tabs">
            <button
              className={tab === "jobs" ? "active" : ""}
              onClick={() => setTab("jobs")}
            >
              jobs
            </button>
            <button
              className={tab === "network" ? "active" : ""}
              onClick={() => setTab("network")}
            >
              network
            </button>
            <button
              className={tab === "nodes" ? "active" : ""}
              onClick={() => setTab("nodes")}
            >
              nodes
            </button>
            <button
              className={tab === "datasets" ? "active" : ""}
              onClick={() => setTab("datasets")}
            >
              datasets
            </button>
          </nav>
        </div>
      </header>

      {/* All tab bodies stay mounted so each tab keeps its scroll
          position / in-flight queries across a tab flip — same idiom
          as InferencePanel / DatasetsPanel. */}
      <div
        style={{
          display: tab === "jobs" ? "block" : "none",
          flex: 1,
          minHeight: 0,
          overflow: "auto",
        }}
      >
        <ClusterJobsTab />
      </div>
      <div
        style={{
          display: tab === "network" ? "block" : "none",
          flex: 1,
          minHeight: 0,
          overflow: "auto",
        }}
      >
        <NetworkTab
          members={data.members}
          selfNodeId={data.self_node_id}
        />
      </div>
      <div
        style={{
          display: tab === "nodes" ? "block" : "none",
          flex: 1,
          minHeight: 0,
          overflow: "auto",
        }}
      >
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
      <div
        style={{
          display: tab === "datasets" ? "block" : "none",
          flex: 1,
          minHeight: 0,
          overflow: "auto",
        }}
      >
        <DatasetsClusterTab onOpenInExplore={onOpenInExplore} />
      </div>
    </div>
  );
}

type ClusterTab = "jobs" | "network" | "nodes" | "datasets";

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
      <NodeGpuList
        gpus={gpus}
        gpusError={gpusError}
        jobByPid={jobByPid}
        reservedGpus={reservedGpus}
        onToggleDisabled={(gpu_index, disabled) =>
          togglePolicy.mutate({ gpu_index, disabled })
        }
      />
    </section>
  );
}

/** Collapsible per-node GPU summary modelled on the Interfaces
 *  control. Expanded by default; one row per GPU; the summary surface
 *  carries the count + a one-line aggregate ("4 GPUs · 2 idle"). */
function NodeGpuList({
  gpus,
  gpusError,
  jobByPid,
  reservedGpus,
  onToggleDisabled,
}: {
  gpus: GpuInfo[] | null;
  gpusError: string | null;
  jobByPid: Map<number, Job>;
  reservedGpus: Set<number> | null;
  onToggleDisabled: (gpu_index: number, disabled: boolean) => void;
}) {
  if (gpusError) {
    return (
      <details className="node-group-gpus" open>
        <summary>GPUs (—)</summary>
        <div className="node-group-error">
          GPUs unavailable: {gpusError}
        </div>
      </details>
    );
  }
  if (gpus === null) {
    return (
      <details className="node-group-gpus" open>
        <summary>GPUs (loading…)</summary>
      </details>
    );
  }
  const total = gpus.length;
  const idleCount = gpus.filter(
    (g) =>
      !g.excluded &&
      !g.disabled &&
      !g.reserved &&
      !(reservedGpus?.has(g.index) ?? false),
  ).length;
  return (
    <details className="node-group-gpus" open>
      <summary>
        GPUs ({total}){total > 0 ? ` · ${idleCount} idle` : ""}
      </summary>
      {total === 0 ? (
        <div className="muted gpu-row-empty">No GPUs reported.</div>
      ) : (
        <table className="gpu-row-table">
          <thead>
            <tr>
              <th>GPU</th>
              <th>Name</th>
              <th>Memory</th>
              <th>Util</th>
              <th>Temp</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {gpus.map((g) => {
              // ``g.reserved`` is stamped by the owning peer (authoritative
              // for that node's running jobs); ``reservedGpus`` is the
              // local-jobs fallback used for older peers that don't ship
              // the field yet. OR them so a single source missing the
              // signal still flips the row to BUSY.
              const reserved =
                g.reserved || (reservedGpus?.has(g.index) ?? false);
              const idle = !g.excluded && !g.disabled && !reserved;
              const statusLabel = g.excluded
                ? "excluded"
                : g.disabled
                  ? "disabled"
                  : reserved
                    ? "busy"
                    : idle
                      ? "idle"
                      : "active";
              const procCount = g.processes.length;
              const procTitle = procCount
                ? g.processes
                    .map((p) => {
                      const job = jobByPid.get(p.pid);
                      return (
                        `pid ${p.pid}` +
                        (job ? ` (${job.config ?? job.id})` : "") +
                        ` · ${fmtMiB(p.used_mem_bytes)}`
                      );
                    })
                    .join("\n")
                : undefined;
              const clickTitle = g.excluded
                ? "Excluded via CUDA_VISIBLE_DEVICES — cannot toggle"
                : g.disabled
                  ? "Click to enable GPU (allow scheduling)"
                  : "Click to disable GPU (block scheduling)";
              return (
                // ``gpu-list-row`` is the right class for this table
                // row — avoid ``gpu-row`` which is used by GpuPanel
                // for a flexbox layout and would clobber table-row
                // display, breaking column alignment with the header.
                <tr
                  key={g.index}
                  className={"gpu-list-row gpu-list-row-" + statusLabel}
                  onClick={
                    g.excluded
                      ? undefined
                      : () => onToggleDisabled(g.index, !g.disabled)
                  }
                  style={g.excluded ? undefined : { cursor: "pointer" }}
                  title={clickTitle}
                >
                  <td className="gpu-row-idx">{g.index}</td>
                  <td className="gpu-row-name">{g.name}</td>
                  <td className="gpu-row-mem">
                    {fmtMiB(g.used_mem_bytes)} / {fmtMiB(g.total_mem_bytes)}
                  </td>
                  <td className="gpu-row-util">
                    {g.util_pct !== null ? `${g.util_pct}%` : "—"}
                  </td>
                  <td className="gpu-row-temp">
                    {g.temp_c !== null ? `${g.temp_c}°C` : "—"}
                  </td>
                  <td className="gpu-row-status">
                    <span className={"gpu-row-status-tag tag-" + statusLabel}>
                      {statusLabel}
                    </span>
                    {procCount > 0 && (
                      <span className="gpu-row-procs muted" title={procTitle}>
                        · {procCount} proc{procCount === 1 ? "" : "s"}
                      </span>
                    )}
                    {g.min_priority !== 0 && (
                      <span
                        className="gpu-row-priority muted"
                        title={`Only jobs with priority ≥ ${g.min_priority} may use this GPU`}
                      >
                        · ≥{g.min_priority}
                      </span>
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
