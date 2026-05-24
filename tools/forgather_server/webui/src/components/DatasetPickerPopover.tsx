import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { api, ClusterDatasetEntry } from "../api";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  /** Called when the user picks an entry. The string passed is exactly
   *  what should land in the meta-template field — ``local/<name>`` for
   *  local registrations, the HF Hub id (``allenai/c4``) for cache
   *  entries — i.e. whatever ``load_dataset(path=...)`` accepts. */
  onPick: (datasetId: string) => void;
  onClose: () => void;
}

type SourceFilter = "all" | "local" | "hf";

/** Modal popover for picking a dataset from the cluster's aggregated
 *  inventory. Covers both the HuggingFace cache (``source: "hf"``) and
 *  dataset_server-registered local mappings (``source: "local"``).
 *
 *  Single-node mode: the inventory endpoint surfaces whatever the
 *  local node knows about — HF cache + local registrations from any
 *  dataset_server reachable from this host.
 *
 *  Cluster mode: the inventory is already cluster-aggregated and
 *  deduped across every reachable peer, so the picker shows the union
 *  for free with no per-mode branching.
 *
 *  ``source: "path"`` entries (ad-hoc absolute paths registered by
 *  clients) are filtered out — they're a power-user edge case and
 *  showing them in a "browse" UI would surface paths that aren't
 *  portable across machines. */
export function DatasetPickerPopover({ onPick, onClose }: Props) {
  const [filter, setFilter] = useState("");
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [activeIdx, setActiveIdx] = useState(0);
  const filterRef = useRef<HTMLInputElement>(null);

  const inv = useQuery({
    queryKey: ["cluster-dataset-inventory"],
    queryFn: () => api.getClusterDatasetInventory(),
    staleTime: 30 * 1000,
  });

  // All pickable entries (local + hf), sorted with local first since a
  // user filling in a config they just registered locally usually wants
  // their fresh entry, not the long HF tail.
  const allEntries = useMemo<ClusterDatasetEntry[]>(() => {
    const all = inv.data?.datasets ?? [];
    return all
      .filter((d) => d.source === "local" || d.source === "hf")
      .sort((a, b) => {
        if (a.source !== b.source) return a.source === "local" ? -1 : 1;
        return a.dataset_id.localeCompare(b.dataset_id);
      });
  }, [inv.data]);

  const counts = useMemo(
    () => ({
      all: allEntries.length,
      local: allEntries.filter((d) => d.source === "local").length,
      hf: allEntries.filter((d) => d.source === "hf").length,
    }),
    [allEntries],
  );

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    let out = allEntries;
    if (sourceFilter !== "all") {
      out = out.filter((d) => d.source === sourceFilter);
    }
    if (q) {
      out = out.filter((d) => d.dataset_id.toLowerCase().includes(q));
    }
    return out;
  }, [allEntries, filter, sourceFilter]);

  // Reset the active row when filter or source-filter changes so we
  // don't land on a row that the new filter just hid.
  useEffect(() => {
    setActiveIdx(0);
  }, [filter, sourceFilter]);

  useEffect(() => {
    filterRef.current?.focus();
  }, []);

  const selectAndClose = (entry: ClusterDatasetEntry) => {
    onPick(entry.dataset_id);
    onClose();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, Math.max(filtered.length - 1, 0)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const entry = filtered[activeIdx];
      if (entry) selectAndClose(entry);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  const isClusterActive = inv.data?.is_master === true;

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal dataset-picker"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Pick a dataset"
        onKeyDown={onKeyDown}
      >
        <header className="modal-header">
          <h3>Pick a dataset</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <input
            ref={filterRef}
            type="text"
            className="dataset-picker-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by id…"
            spellCheck={false}
          />

          <div className="dataset-picker-source-tabs" role="tablist">
            {(["all", "local", "hf"] as SourceFilter[]).map((s) => (
              <button
                key={s}
                role="tab"
                aria-selected={sourceFilter === s}
                className={`dataset-picker-source-tab${
                  sourceFilter === s ? " active" : ""
                }`}
                onClick={() => setSourceFilter(s)}
              >
                {s === "all" ? "All" : s === "local" ? "Local" : "HF cache"}
                <span className="muted"> ({counts[s]})</span>
              </button>
            ))}
            {isClusterActive && (
              <span className="muted dataset-picker-cluster-note">
                cluster-aggregated
              </span>
            )}
          </div>

          {inv.isLoading && (
            <div className="muted pad">Loading dataset inventory…</div>
          )}
          {inv.error != null && (
            <div className="err pad">
              <pre>{String(inv.error)}</pre>
            </div>
          )}
          {!inv.isLoading && !inv.error && allEntries.length === 0 && (
            <div className="muted pad">
              No datasets found in the inventory. Register a local one
              via <code>dataset_server --local NAME PATH</code> or load
              a HuggingFace dataset to populate the cache.
            </div>
          )}
          {!inv.isLoading &&
            !inv.error &&
            allEntries.length > 0 &&
            filtered.length === 0 && (
              <div className="muted pad">
                No matches for "{filter}"
                {sourceFilter !== "all" && (
                  <>
                    {" "}
                    in <code>{sourceFilter}</code>
                  </>
                )}
                .
              </div>
            )}

          <div className="dataset-picker-list" role="listbox">
            {filtered.map((d, i) => (
              <div
                key={`${d.source}:${d.dataset_id}`}
                role="option"
                aria-selected={i === activeIdx}
                className={`dataset-picker-row${
                  i === activeIdx ? " active" : ""
                }`}
                onClick={() => setActiveIdx(i)}
                onDoubleClick={() => selectAndClose(d)}
              >
                <div className="dataset-picker-row-head">
                  <span
                    className={`dataset-picker-source-badge src-${d.source}`}
                    title={
                      d.source === "local"
                        ? "Local dataset_server registration"
                        : "HuggingFace Hub cache"
                    }
                  >
                    {d.source === "local" ? "local" : "hf"}
                  </span>
                  <code className="dataset-picker-id">{d.dataset_id}</code>
                </div>
                <span className="muted dataset-picker-meta">
                  {d.length != null && <>{d.length.toLocaleString()} rows</>}
                  {d.column_names && d.column_names.length > 0 && (
                    <>
                      {d.length != null ? " · " : ""}
                      {d.column_names.join(", ")}
                    </>
                  )}
                  {d.server_ids.length > 1 && (
                    <> · on {d.server_ids.length} servers</>
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>

        <footer className="modal-footer">
          <div className="muted dataset-picker-hint">
            Double-click or press Enter to select. Esc to cancel.
          </div>
          <div className="btn-row">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              onClick={() => {
                const entry = filtered[activeIdx];
                if (entry) selectAndClose(entry);
              }}
              disabled={filtered.length === 0}
            >
              Select
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
