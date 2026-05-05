import { useQuery } from "@tanstack/react-query";
import {
  api,
  ClusterMember,
  ClusterMembersResponse,
  ClusterGpusResponse,
  GpuInfo,
} from "../api";

/** Per-node row in the cluster Nodes view.
 *
 *  Phase 1 keeps this minimal: identity, role, reachability, version
 *  placeholder (filled in Phase 2 when the probe layer lands), and a
 *  compact GPU summary line. The detailed GPU panel — including the
 *  policy/kill controls — only fires for the local node, since those
 *  actions are not yet routed through a by-node proxy.
 */
function NodeRow({
  member,
  isMaster,
  isSelf,
  gpus,
  gpusError,
}: {
  member: ClusterMember;
  isMaster: boolean;
  isSelf: boolean;
  gpus: GpuInfo[] | null;
  gpusError: string | null;
}) {
  const reachable = member.reachable;
  const role = isMaster ? "master" : "peer";
  const tag = (label: string, tone: "ok" | "warn" | "err" | "muted") => (
    <span className={`node-tag node-tag-${tone}`}>{label}</span>
  );
  return (
    <div className={"node-row" + (reachable ? "" : " unreachable")}>
      <div className="node-row-header">
        <strong className="node-hostname">{member.hostname}</strong>
        {tag(role, isMaster ? "ok" : "muted")}
        {isSelf && tag("this server", "muted")}
        {tag(
          reachable ? "reachable" : "unreachable",
          reachable ? "ok" : "err",
        )}
        <span className="node-address">
          {member.address}:{member.port}
        </span>
        <span className="node-version" title="forgather version">
          v{member.forgather_version || "unknown"}
        </span>
      </div>
      <div className="node-row-body">
        <NodeGpuSummary gpus={gpus} error={gpusError} />
      </div>
    </div>
  );
}

function NodeGpuSummary({
  gpus,
  error,
}: {
  gpus: GpuInfo[] | null;
  error: string | null;
}) {
  if (error) {
    return <div className="node-gpu-error">GPUs: {error}</div>;
  }
  if (gpus === null) {
    return <div className="node-gpu-loading">GPUs: …</div>;
  }
  if (gpus.length === 0) {
    return <div className="node-gpu-empty">No GPUs reported</div>;
  }
  return (
    <table className="node-gpu-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Name</th>
          <th>Mem</th>
          <th>Util</th>
          <th>Temp</th>
        </tr>
      </thead>
      <tbody>
        {gpus.map((g) => (
          <tr key={g.index} className={g.disabled ? "disabled" : undefined}>
            <td>{g.index}</td>
            <td>{g.name}</td>
            <td>
              {Math.round(g.used_mem_bytes / 1e9)} /{" "}
              {Math.round(g.total_mem_bytes / 1e9)} GB
            </td>
            <td>{g.util_pct ?? "—"}%</td>
            <td>{g.temp_c ?? "—"}°C</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function NodesPanel() {
  const membersQ = useQuery<ClusterMembersResponse>({
    queryKey: ["cluster", "members"],
    queryFn: api.getClusterMembers,
    // Match the backend's tick cadence so this view is never older
    // than ~5 s when the user is looking at it. Background refetch is
    // fine — the payload is small.
    refetchInterval: 5000,
  });
  const gpusQ = useQuery<ClusterGpusResponse>({
    queryKey: ["cluster", "gpus"],
    queryFn: api.getClusterGpus,
    refetchInterval: 5000,
  });

  if (membersQ.isLoading) {
    return <div className="nodes-panel">Loading cluster members…</div>;
  }
  if (membersQ.isError) {
    return (
      <div className="nodes-panel">
        <div className="error">Failed to load cluster: {String(membersQ.error)}</div>
      </div>
    );
  }
  const data = membersQ.data;
  if (!data || !data.cluster_name) {
    // Defensive: NodesPanel should not be mounted when standalone, but
    // if it is we render a friendly empty state rather than crashing.
    return <div className="nodes-panel">Cluster mode is not active.</div>;
  }
  const gpusByNode = new Map<string, { gpus: GpuInfo[]; error: string | null }>();
  for (const entry of gpusQ.data?.nodes ?? []) {
    gpusByNode.set(entry.node_id, {
      gpus: entry.gpus,
      error: entry.error ?? null,
    });
  }

  const members = [...data.members].sort((a, b) => {
    // Master first, then reachable peers, then unreachable. Within
    // each bucket sort by hostname for stable rendering.
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

  return (
    <div className="nodes-panel">
      <div className="nodes-panel-header">
        <h2>Cluster: {data.cluster_name}</h2>
        <div className="nodes-panel-meta">
          {data.members.length} member{data.members.length === 1 ? "" : "s"}
          {" · "}
          master:{" "}
          {data.master_node_id
            ? data.members.find((m) => m.node_id === data.master_node_id)
                ?.hostname ?? data.master_node_id.slice(0, 8)
            : "(none)"}
        </div>
      </div>
      <div className="nodes-panel-rows">
        {members.map((m) => {
          const entry = gpusByNode.get(m.node_id);
          return (
            <NodeRow
              key={m.node_id}
              member={m}
              isMaster={m.node_id === data.master_node_id}
              isSelf={m.node_id === data.self_node_id}
              gpus={entry ? entry.gpus : null}
              gpusError={entry?.error ?? null}
            />
          );
        })}
      </div>
    </div>
  );
}
