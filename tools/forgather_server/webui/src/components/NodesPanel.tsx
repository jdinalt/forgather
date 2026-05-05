import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  ClusterMember,
  ClusterMembersResponse,
  ClusterGpusResponse,
  GpuInfo,
  Job,
} from "../api";
import { GpuPanel, GpuCard } from "./GpuPanel";

/** Cluster-aware Nodes view.
 *
 *  Layout: a cluster header with name/master, then one bounded box
 *  per node containing the existing GPU-card layout. The local node
 *  embeds the full live ``<GpuPanel/>`` so kill/policy/context-menu
 *  controls and the WebSocket stream all keep working unchanged.
 *  Peer nodes render the same rich cards but read-only, driven by
 *  the master-side aggregator polled at 5 s. Peer mutations are
 *  intentionally disabled in Phase 1: cross-node policy/kill
 *  routing is part of the by-node proxy seam scheduled for later.
 *
 *  The cluster header doubles as the live/stale indicator: if the
 *  /api/cluster/members poll is in error or hasn't returned yet,
 *  the nodes list still shows whatever the previous refresh
 *  returned. Empty payload (cluster not active) shouldn't happen
 *  here — App.tsx only mounts NodesPanel when /api/cluster/self
 *  returned non-null — but we guard for it anyway.
 */
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
  // jobByPid is shared across all peer cards so the process chips
  // can show "config foo" instead of bare PIDs. The list is local
  // to the master — cross-node job attribution will be wired up
  // when the by-node proxy lands.
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
  for (const j of jobsQ.data ?? []) {
    if (j.pid != null) jobByPid.set(j.pid, j);
  }

  // Render order: master first, then reachable peers, then unreachable.
  // Within each bucket, sort by hostname so the layout is stable when
  // peers come and go.
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

  return (
    <div className="nodes-panel">
      <header className="nodes-panel-header">
        <h2>Cluster: {data.cluster_name}</h2>
        <span className="nodes-panel-meta">
          {data.members.length} node
          {data.members.length === 1 ? "" : "s"} · master: {masterHostname}
        </span>
      </header>
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
}: {
  member: ClusterMember;
  isSelf: boolean;
  isMaster: boolean;
  gpus: GpuInfo[] | null;
  gpusError: string | null;
  jobByPid: Map<number, Job>;
}) {
  const qc = useQueryClient();
  // Click-to-toggle the disabled flag, routed to the owning node via
  // the master-side proxy. The master short-circuits self-targets, so
  // this also works when the local node initiates the toggle on its
  // own card if we ever choose to use the polled view for self too.
  const togglePolicy = useMutation({
    mutationFn: ({
      gpu_index,
      disabled,
    }: {
      gpu_index: number;
      disabled: boolean;
    }) =>
      api.setNodeGpuPolicy(member.node_id, gpu_index, { disabled }),
    // The cluster GPU poll picks up the new state within 5 s; an
    // explicit invalidate makes the visual confirmation immediate.
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cluster", "gpus"] }),
    onError: (e) => alert(`Policy update failed: ${String(e)}`),
  });
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
          <span className="node-version" title="forgather version">
            v{member.forgather_version || "unknown"}
          </span>
        </span>
      </header>
      <div className="node-group-body">
        {isSelf ? (
          // Local node: keep the existing live GpuPanel — WS stream,
          // kill/policy controls, context menu all unchanged. The
          // outer node-group box just wraps it with the cluster
          // header.
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
