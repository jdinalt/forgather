/**
 * Shared dataset-source selector logic for submit modals.
 *
 * Four modals all enqueue jobs that fetch examples through
 * ``fast_load_iterable_dataset``: SubmitModal (training), EvalModal,
 * ModelSubmitModal, and DatasetSubmitModal. Each wants the same UX:
 *
 *  - dropdown with "Local" + every known dataset_server,
 *  - unreachable options visible but disabled,
 *  - seeded from cached overrides, falling back to "Local" when the
 *    saved choice is gone or offline,
 *  - the chosen ``DatasetSource`` returned for the submit payload.
 *
 * Putting it here keeps that surface in one place so a future tweak
 * (e.g. active reachability probing for user entries) lands in every
 * modal at once.
 */

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import {
  DatasetServerLocal,
  DatasetServerUser,
  DatasetSource,
  api,
} from "./api";

/** True iff ``server_id`` resolves to a *reachable* dataset_server.
 *  Local servers carry an ``alive`` flag from the JobRecord scan; the
 *  registry doesn't currently probe user entries, so we treat them as
 *  reachable when they exist. */
export function isReachableId(
  server_id: string,
  locals: DatasetServerLocal[],
  users: DatasetServerUser[],
): boolean {
  const [kind, value] = server_id.split(":", 2);
  if (kind === "local") return locals.some((s) => s.queue_id === value && s.alive);
  if (kind === "user") return users.some((s) => s.id === value);
  return false;
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
  /** Setter the selector component calls on change. */
  setSource: (s: DatasetSource | null) => void;
  /** Selector component, ready to drop in a ``.submit-row`` parent. */
  Selector: () => JSX.Element;
}

/** Wires up the selector state + queries + offline-fallback seeding.
 *
 *  Callers own persistence (each modal already has its own store —
 *  server overrides for project-backed flows, localStorage for ad-hoc)
 *  and pass the cached value via ``initial``. The hook applies the
 *  "snap to local when the cached server is gone or not alive" rule
 *  exactly once, after both the dataset-server lists and the caller's
 *  cache (``ready``) have loaded.
 */
export function useDatasetSource(
  opts: UseDatasetSourceOptions,
): UseDatasetSourceResult {
  const localsQ = useQuery({
    queryKey: ["dataset-servers-local"],
    queryFn: api.listLocalDatasetServers,
  });
  const usersQ = useQuery({
    queryKey: ["dataset-servers-user"],
    queryFn: api.listUserDatasetServers,
  });
  const locals = localsQ.data ?? [];
  const users = usersQ.data ?? [];

  const [source, setSource] = useState<DatasetSource | null>(null);
  const [seeded, setSeeded] = useState<boolean>(false);

  const ready = opts.ready ?? true;
  useEffect(() => {
    if (seeded) return;
    if (!ready) return;
    if (!localsQ.data || !usersQ.data) return;
    const cached = opts.initial;
    if (!cached || cached.kind !== "server") {
      setSource(null);
    } else if (isReachableId(cached.server_id, localsQ.data, usersQ.data)) {
      setSource(cached);
    } else {
      setSource(null);
    }
    setSeeded(true);
    // Intentionally not depending on ``opts.initial`` after the first
    // seed — once seeded, the user's interactive selection wins.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [localsQ.data, usersQ.data, seeded, ready]);

  const Selector = () => (
    <DatasetSourceSelector
      value={source}
      onChange={setSource}
      locals={locals}
      users={users}
    />
  );

  return { source, setSource, Selector };
}

interface DatasetSourceSelectorProps {
  value: DatasetSource | null;
  onChange: (v: DatasetSource | null) => void;
  locals: DatasetServerLocal[];
  users: DatasetServerUser[];
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
}: DatasetSourceSelectorProps) {
  const currentId =
    value?.kind === "server" ? value.server_id : "local";

  interface OptionRow {
    server_id: string;
    label: string;
    disabled: boolean;
    diagnostic: string | null;
  }
  const options: OptionRow[] = [];
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

  const onPick = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    if (id === "local") onChange(null);
    else onChange({ kind: "server", server_id: id });
  };

  const currentDiagnostic = (() => {
    if (!value || value.kind !== "server") return null;
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
    return null;
  })();

  return (
    <div className="submit-row dataset-source-row">
      <label className="wide">
        Dataset source
        <select value={currentId} onChange={onPick}>
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
        <span className="muted">
          {currentDiagnostic
            ? `⚠ ${currentDiagnostic} — submit will fall back to local or fail with a clear error`
            : value?.kind === "server"
              ? "Training reads FORGATHER_DATASET_SERVER from the chosen server."
              : "Training loads datasets in-process (the default)."}
        </span>
      </label>
    </div>
  );
}
