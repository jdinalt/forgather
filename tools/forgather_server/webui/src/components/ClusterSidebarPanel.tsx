import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ClusterMember } from "../api";
import {
  HEADLINE_VERSION_KEYS,
  computeVersionConsensus,
  nodeHealth,
} from "./ClusterPanel";

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
  // Compute version consensus once so each row can render its dot
  // colour against the cluster-wide majority. Surfaces the same
  // mismatch signal the Cluster view's Nodes tab already shows —
  // catches cases like a peer's driver glitch dropping its nvml
  // reading: still HTTP-reachable, but its version row no longer
  // matches and the operator should be warned.
  const consensus = computeVersionConsensus(data.members);
  // Master first, then reachable peers by hostname, unreachable last.
  // Same ordering as the Cluster view's Nodes tab so the two
  // surfaces feel coherent.
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
        const health = nodeHealth(m, consensus);
        // For a "warn" node, build a per-key summary of what
        // disagrees with the cluster majority so the tooltip
        // explains *what* the dot is flagging — kitt's "lost its
        // nvml" case was hard to diagnose without that detail.
        const warnDetails: string[] = [];
        if (health === "warn" && m.probe?.versions) {
          for (const key of HEADLINE_VERSION_KEYS) {
            const expected = consensus[key];
            if (!expected) continue;
            const value = m.probe.versions[key];
            const missing = !value || value === "unavailable";
            if (missing) {
              warnDetails.push(
                `${key}: missing on this node (cluster: ${expected})`,
              );
            } else if (value !== expected) {
              warnDetails.push(
                `${key}: ${value} (cluster: ${expected})`,
              );
            }
          }
        }
        const titleParts = [
          `${m.hostname} (${m.address}:${m.port})`,
          isMaster ? "master" : "peer",
          isSelf ? "this server" : null,
          health === "down"
            ? "unreachable"
            : health === "warn"
              ? "version mismatch"
              : "reachable",
        ].filter(Boolean);
        return (
          <button
            key={m.node_id}
            className={
              "cluster-sidebar-row" +
              (isSelf ? " self" : "") +
              (health === "down" ? " unreachable" : "") +
              (health === "warn" ? " warn" : "") +
              (errored ? " errored" : "")
            }
            // Self-SSO would just hand the user a token for the same
            // origin they're already on; disable the click so the row
            // can't even be triggered. ``warn`` peers stay clickable
            // — the operator may want to log into the affected node
            // to investigate.
            disabled={isSelf || !m.reachable || status === "opening"}
            onClick={() => open(m)}
            title={
              isSelf
                ? `${titleParts.join(" · ")}\n(this is the server you're already on)`
                : !m.reachable
                  ? `${titleParts.join(" · ")}\n(unreachable — cannot open)`
                  : status
                    ? status
                    : health === "warn"
                      ? `${titleParts.join(" · ")}\n${warnDetails.join("\n")}\nClick to open in a new tab`
                      : `${titleParts.join(" · ")}\nClick to open in a new tab`
            }
          >
            <span
              className={"cluster-sidebar-dot " + health}
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
