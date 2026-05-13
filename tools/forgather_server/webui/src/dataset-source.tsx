/**
 * Shared dataset-source selector logic for submit modals.
 *
 * Four modals all enqueue jobs that fetch examples through
 * ``fast_load_iterable_dataset``: SubmitModal (training), EvalModal,
 * ModelSubmitModal, and DatasetSubmitModal. Each wants the same UX:
 *
 *  - dropdown with "Local" + every known dataset_server,
 *  - in cluster mode, "Auto (cluster routing)" as the top option +
 *    default; the dropdown shows the master's cluster-wide server set
 *    (not just locally-known ones),
 *  - unreachable options visible but disabled,
 *  - seeded from cached overrides, falling back to the cluster-mode
 *    default ("auto" if available, otherwise "Local") when the saved
 *    choice is gone or offline,
 *  - the chosen ``DatasetSource`` returned for the submit payload.
 *
 * Putting it here keeps that surface in one place so a future tweak
 * (e.g. active reachability probing for user entries) lands in every
 * modal at once.
 */

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import {
  ClusterDatasetServer,
  DatasetServerLocal,
  DatasetServerUser,
  DatasetSource,
  api,
} from "./api";

/** True iff ``server_id`` resolves to a *reachable* dataset_server.
 *  Local servers carry an ``alive`` flag from the JobRecord scan; the
 *  registry doesn't currently probe user entries, so we treat them as
 *  "reachable when registered" until a clicked Status call says
 *  otherwise. The selector in DatasetSourceSelector only marks
 *  ``local:`` / cluster server_ids as ``(unreachable)`` for this reason. */
function isReachableId(
  server_id: string,
  locals: DatasetServerLocal[],
  users: DatasetServerUser[],
  cluster: ClusterDatasetServer[],
): boolean {
  const [kind, value] = server_id.split(":", 2);
  if (kind === "local")
    return locals.some((s) => s.queue_id === value && s.alive);
  if (kind === "user") return users.some((s) => s.id === value);
  if (kind === "cluster")
    return cluster.some((s) => s.server_id === value && s.healthy);
  // Bare server_id (no prefix) — treat as cluster-format and consult
  // the cluster inventory for backward compat with older saved values.
  return cluster.some((s) => s.server_id === server_id && s.healthy);
}

interface UseDatasetSourceOptions {
  /** Caller's cached value, typically pulled from server overrides
   *  or localStorage. The hook reads this once on the seed pass; the
   *  user's subsequent interactive picks win. */
  initial: DatasetSource | null;
  /** Hook waits for this to be ``true`` before seeding. Lets callers
   *  delay until their own async cache (overrides API) has loaded.
   *  Defaults to ``true``. */
  ready?: boolean;
}

interface UseDatasetSourceResult {
  /** Current selection. ``null`` = local. */
  source: DatasetSource | null;
  /** Setter the selector component calls on change. Useful for
   *  Reset-to-defaults handlers that need to clear the dropdown
   *  alongside everything else. */
  setSource: (s: DatasetSource | null) => void;
  /** Selector ready to drop into a ``.submit-row`` parent. Returned
   *  as a JSX element rather than a function component so re-renders
   *  on the parent don't remount the underlying ``<select>`` (which
   *  would close it mid-interaction). */
  selector: React.ReactNode;
}

/** Wires up the selector state + queries + offline-fallback seeding.
 *
 *  Callers own persistence (each modal already has its own store —
 *  server overrides for project-backed flows, localStorage for ad-hoc)
 *  and pass the cached value via ``initial``. The hook applies the
 *  "snap to local when the cached server is gone or not alive" rule
 *  exactly once, after both the dataset-server lists and the caller's
 *  cache (``ready``) have loaded.
 *
 *  In cluster mode the selector additionally surfaces an "Auto
 *  (cluster routing)" option and defaults to it when there's no
 *  cached preference.
 */
export function useDatasetSource(
  opts: UseDatasetSourceOptions,
): UseDatasetSourceResult {
  // Standalone-mode sources. Cheap to keep around even in cluster
  // mode — TanStack dedups by queryKey across components.
  const localsQ = useQuery({
    queryKey: ["dataset-servers-local"],
    queryFn: api.listLocalDatasetServers,
  });
  const usersQ = useQuery({
    queryKey: ["dataset-servers-user"],
    queryFn: api.listUserDatasetServers,
  });
  // Cluster mode detection + cluster-wide server list. ``getClusterSelf``
  // returns ``null`` outside cluster mode; we use that as the gate.
  const clusterSelfQ = useQuery({
    queryKey: ["cluster-self"],
    queryFn: api.getClusterSelf,
    refetchInterval: 30000,
  });
  const clusterActive = !!clusterSelfQ.data;
  const clusterServersQ = useQuery({
    queryKey: ["cluster", "dataset_servers"],
    queryFn: api.getClusterDatasetServers,
    refetchInterval: 10000,
    enabled: clusterActive,
  });
  const locals = localsQ.data ?? [];
  const users = usersQ.data ?? [];
  const clusterServers = clusterServersQ.data ?? [];

  const [source, setSource] = useState<DatasetSource | null>(null);
  const [seeded, setSeeded] = useState<boolean>(false);

  const ready = opts.ready ?? true;
  useEffect(() => {
    if (seeded) return;
    if (!ready) return;
    if (!localsQ.data || !usersQ.data) return;
    // Cluster mode: also wait for the cluster server list before
    // making the seed decision — otherwise a saved cluster server_id
    // would falsely look unreachable on the first render.
    if (clusterActive && !clusterServersQ.data) return;
    const cached = opts.initial;
    const clusterDefault: DatasetSource | null = clusterActive
      ? { kind: "auto" }
      : null;
    if (!cached) {
      setSource(clusterDefault);
    } else if (cached.kind === "auto") {
      // Honor the cached "auto" only when cluster mode is active —
      // otherwise it would be impossible to resolve and we'd just
      // surface a confusing error on submit. Fall back to local.
      setSource(clusterActive ? cached : null);
    } else if (cached.kind === "server") {
      if (isReachableId(cached.server_id, locals, users, clusterServers)) {
        setSource(cached);
      } else {
        setSource(clusterDefault);
      }
    } else {
      setSource(null);
    }
    setSeeded(true);
    // Intentionally not depending on ``opts.initial`` after the first
    // seed — once seeded, the user's interactive selection wins.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    localsQ.data,
    usersQ.data,
    clusterServersQ.data,
    clusterActive,
    seeded,
    ready,
  ]);

  const selector = (
    <DatasetSourceSelector
      value={source}
      onChange={setSource}
      locals={locals}
      users={users}
      clusterServers={clusterServers}
      clusterActive={clusterActive}
    />
  );

  return { source, setSource, selector };
}

interface DatasetSourceSelectorProps {
  value: DatasetSource | null;
  onChange: (v: DatasetSource | null) => void;
  locals: DatasetServerLocal[];
  users: DatasetServerUser[];
  /** Cluster-wide servers from the master's inventory. Empty in
   *  standalone mode. */
  clusterServers: ClusterDatasetServer[];
  /** Show the "Auto" option + cluster servers when true. */
  clusterActive: boolean;
}

/** Compact dropdown for the dataset-source choice. The select shows
 *  every known dataset_server even when unreachable, with the
 *  unreachable entries disabled so the operator can see what *would*
 *  be available — and gets a diagnostic in the title attribute on
 *  hover. The current selection always renders enabled, even if it
 *  has gone offline since the modal opened, so the user can see the
 *  problem before they switch. */
export function DatasetSourceSelector({
  value,
  onChange,
  locals,
  users,
  clusterServers,
  clusterActive,
}: DatasetSourceSelectorProps) {
  const currentId =
    value?.kind === "auto"
      ? "auto"
      : value?.kind === "server"
        ? value.server_id
        : "local";

  interface OptionRow {
    server_id: string;
    label: string;
    disabled: boolean;
    diagnostic: string | null;
  }
  const options: OptionRow[] = [];

  // Cluster mode: the master's view of every server. Includes both
  // peer-local-spawned and registry entries, deduped by base_url —
  // safer than rolling our own merge of the per-node lists.
  if (clusterActive) {
    for (const s of clusterServers) {
      options.push({
        server_id: `cluster:${s.server_id}`,
        label: `${s.label} — ${s.base_url}`,
        disabled: !s.healthy,
        diagnostic: s.healthy
          ? null
          : s.last_health_error || "cluster server is not reachable",
      });
    }
  } else {
    // Standalone: per-node lists, same UX as before.
    for (const s of locals) {
      options.push({
        server_id: `local:${s.queue_id}`,
        label: `${s.label} — ${s.base_url}`,
        disabled: !s.alive,
        diagnostic: s.alive ? null : "local dataset_server is not running",
      });
    }
    for (const s of users) {
      options.push({
        server_id: `user:${s.id}`,
        label: `${s.label} — ${s.base_url}`,
        disabled: false,
        diagnostic: null,
      });
    }
  }

  const onPick = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    if (id === "local") onChange(null);
    else if (id === "auto") onChange({ kind: "auto" });
    else onChange({ kind: "server", server_id: id });
  };

  const currentDiagnostic = (() => {
    if (!value) return null;
    if (value.kind === "auto") {
      if (!clusterActive)
        return "auto routing requires cluster mode on the forgather_server";
      return null;
    }
    if (value.kind !== "server") return null;
    if (
      value.server_id.startsWith("local:") &&
      !locals.some(
        (s) =>
          s.queue_id === value.server_id.slice("local:".length) && s.alive,
      )
    ) {
      return "the selected local dataset_server is not running";
    }
    if (
      value.server_id.startsWith("user:") &&
      !users.some((s) => s.id === value.server_id.slice("user:".length))
    ) {
      return "the selected registered dataset_server is gone";
    }
    if (value.server_id.startsWith("cluster:")) {
      const sid = value.server_id.slice("cluster:".length);
      const entry = clusterServers.find((s) => s.server_id === sid);
      if (!entry) return "the selected cluster server is no longer known";
      if (!entry.healthy)
        return entry.last_health_error || "cluster server is not reachable";
    }
    return null;
  })();

  const description = (() => {
    if (currentDiagnostic) {
      return `⚠ ${currentDiagnostic} — submit will fail with a clear error; pick a different source or switch to Local`;
    }
    if (value?.kind === "auto") {
      return (
        "Each rank's client asks the cluster router for a healthy " +
        "dataset_server at iter time; the master picks at random " +
        "across replicas and re-routes on failure."
      );
    }
    if (value?.kind === "server") {
      return "Training reads FORGATHER_DATASET_SERVER from the chosen server.";
    }
    return "Training loads datasets in-process (the default).";
  })();

  return (
    <div className="submit-row dataset-source-row">
      <label className="wide">
        Dataset source
        <select value={currentId} onChange={onPick}>
          {clusterActive && (
            <option value="auto">Auto (cluster routing)</option>
          )}
          <option value="local">Local (in-process loader)</option>
          {options.map((o) => (
            <option
              key={o.server_id}
              value={o.server_id}
              disabled={o.disabled && o.server_id !== currentId}
              title={o.diagnostic ?? ""}
            >
              {o.label}
              {o.disabled ? " (unreachable)" : ""}
            </option>
          ))}
        </select>
        <span className="muted">{description}</span>
      </label>
    </div>
  );
}
