import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ClusterMember } from "../api";

/** Sidebar "Cluster" group — one row per known peer with a reachability
 *  dot and a click handler that mints a single-sign-on URL and opens it
 *  in a new tab. Hidden by App.tsx unless cluster mode is active. */
export function ClusterSidebarPanel({
  selfNodeId,
  masterNodeId,
}: {
  selfNodeId: string | null;
  masterNodeId: string | null;
}) {
  const membersQ = useQuery({
    queryKey: ["cluster", "members"],
    queryFn: api.getClusterMembers,
    refetchInterval: 5000,
  });
  // Per-row "opening…" state so a slow peer call doesn't freeze the
  // whole panel. We don't show a global error toast — the per-row
  // ``error`` string lives in this same map.
  const [pending, setPending] = useState<Record<string, string>>({});

  if (membersQ.isLoading) {
    return <div className="cluster-sidebar-empty muted">Loading…</div>;
  }
  if (membersQ.isError) {
    return (
      <div className="cluster-sidebar-empty err">
        {String(membersQ.error)}
      </div>
    );
  }
  const data = membersQ.data;
  if (!data || data.members.length === 0) {
    return <div className="cluster-sidebar-empty muted">No nodes.</div>;
  }
  // Master first, then reachable peers by hostname, unreachable last.
  // Same ordering as NodesPanel so the two views feel coherent.
  const sorted = [...data.members].sort((a, b) => {
    const score = (m: ClusterMember) => {
      if (m.node_id === masterNodeId) return 0;
      if (m.reachable) return 1;
      return 2;
    };
    const sa = score(a);
    const sb = score(b);
    if (sa !== sb) return sa - sb;
    return a.hostname.localeCompare(b.hostname);
  });

  async function open(m: ClusterMember) {
    if (m.node_id === selfNodeId) {
      // Self-SSO is meaningless — the backend would refuse anyway, but
      // refusing in the click handler avoids the round-trip.
      return;
    }
    setPending((p) => ({ ...p, [m.node_id]: "opening" }));
    try {
      const res = await api.peerSessionUrl(m.node_id);
      // Open the SSO URL in a new tab. window.open's strings are
      // already same-window-safe (no opener leakage for cross-origin
      // navigations in modern browsers), but ``noopener`` makes it
      // explicit and also drops window.opener so the peer tab can't
      // navigate this one.
      window.open(res.url, "_blank", "noopener,noreferrer");
      setPending((p) => {
        const { [m.node_id]: _, ...rest } = p;
        return rest;
      });
    } catch (e) {
      setPending((p) => ({ ...p, [m.node_id]: `error: ${String(e)}` }));
    }
  }

  return (
    <div className="cluster-sidebar">
      {sorted.map((m) => {
        const isSelf = m.node_id === selfNodeId;
        const isMaster = m.node_id === masterNodeId;
        const status = pending[m.node_id];
        const errored = status && status.startsWith("error");
        const titleParts = [
          `${m.hostname} (${m.address}:${m.port})`,
          isMaster ? "master" : "peer",
          isSelf ? "this server" : null,
          m.reachable ? "reachable" : "unreachable",
        ].filter(Boolean);
        return (
          <button
            key={m.node_id}
            className={
              "cluster-sidebar-row" +
              (isSelf ? " self" : "") +
              (!m.reachable ? " unreachable" : "") +
              (errored ? " errored" : "")
            }
            // Self-SSO would just hand the user a token for the same
            // origin they're already on; disable the click so the row
            // can't even be triggered.
            disabled={isSelf || !m.reachable || status === "opening"}
            onClick={() => open(m)}
            title={
              isSelf
                ? `${titleParts.join(" · ")}\n(this is the server you're already on)`
                : !m.reachable
                  ? `${titleParts.join(" · ")}\n(unreachable — cannot open)`
                  : status
                    ? status
                    : `${titleParts.join(" · ")}\nClick to open in a new tab`
            }
          >
            <span
              className={
                "cluster-sidebar-dot " +
                (m.reachable ? "ok" : "down")
              }
              aria-hidden="true"
            />
            <span className="cluster-sidebar-host">{m.hostname}</span>
            {isMaster && (
              <span className="cluster-sidebar-tag" title="cluster master">
                M
              </span>
            )}
            {isSelf && (
              <span
                className="cluster-sidebar-tag"
                title="the server you're already on"
              >
                ●
              </span>
            )}
            {status === "opening" && (
              <span className="cluster-sidebar-pending muted">…</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
