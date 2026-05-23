import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { api, ClusterDatasetEntry } from "../api";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  /** Called when the user picks an entry. The string passed is exactly
   *  what should land in the meta-template field (e.g. ``"local/stories"``). */
  onPick: (datasetId: string) => void;
  onClose: () => void;
}

/** Modal popover for picking a ``local/<name>`` dataset from the
 *  cluster's dataset inventory. Used by ``MetaTemplateFields`` when a
 *  field's manifest declares ``picker: "local_dataset"``.
 *
 *  Why a modal and not an anchored popover: meta-template fields can
 *  live inside another modal (NewConfig / NewProject), and anchoring a
 *  floating popover to a button that's already in a scrolled modal
 *  body is fiddly. A small centered dialog matches the rest of the
 *  app's "pick something from a list" UX (DirectoryBrowser et al.) and
 *  keeps the focus chain simple. */
export function LocalDatasetPickerPopover({ onPick, onClose }: Props) {
  const [filter, setFilter] = useState("");
  const [activeIdx, setActiveIdx] = useState(0);
  const filterRef = useRef<HTMLInputElement>(null);

  const inv = useQuery({
    queryKey: ["cluster-dataset-inventory"],
    queryFn: () => api.getClusterDatasetInventory(),
    staleTime: 30 * 1000,
  });

  // Just the local/<name> entries. The cluster aggregator dedupes by
  // dataset_id, so a name advertised by multiple servers shows once.
  const localEntries = useMemo<ClusterDatasetEntry[]>(() => {
    const all = inv.data?.datasets ?? [];
    return all
      .filter((d) => d.source === "local")
      .sort((a, b) => a.dataset_id.localeCompare(b.dataset_id));
  }, [inv.data]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return localEntries;
    return localEntries.filter((d) =>
      d.dataset_id.toLowerCase().includes(q),
    );
  }, [localEntries, filter]);

  // Reset the active row when the filter changes so we don't land on a
  // row that the new filter just hid.
  useEffect(() => {
    setActiveIdx(0);
  }, [filter]);

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

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal local-dataset-picker"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Pick a local dataset"
        onKeyDown={onKeyDown}
      >
        <header className="modal-header">
          <h3>Pick a local dataset</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <input
            ref={filterRef}
            type="text"
            className="local-dataset-picker-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by name…"
            spellCheck={false}
          />

          {inv.isLoading && (
            <div className="muted pad">Loading dataset inventory…</div>
          )}
          {inv.error != null && (
            <div className="err pad">
              <pre>{String(inv.error)}</pre>
            </div>
          )}
          {!inv.isLoading && !inv.error && localEntries.length === 0 && (
            <div className="muted pad">
              No <code>local/*</code> datasets found. Register one via{" "}
              <code>dataset_server --local NAME PATH</code> or the
              Datasets → Servers panel.
            </div>
          )}
          {!inv.isLoading &&
            !inv.error &&
            localEntries.length > 0 &&
            filtered.length === 0 && (
              <div className="muted pad">No matches for "{filter}".</div>
            )}

          <div className="local-dataset-picker-list" role="listbox">
            {filtered.map((d, i) => (
              <div
                key={d.dataset_id}
                role="option"
                aria-selected={i === activeIdx}
                className={`local-dataset-picker-row${
                  i === activeIdx ? " active" : ""
                }`}
                onClick={() => setActiveIdx(i)}
                onDoubleClick={() => selectAndClose(d)}
              >
                <code className="local-dataset-picker-id">{d.dataset_id}</code>
                <span className="muted local-dataset-picker-meta">
                  {d.length != null && (
                    <>
                      {d.length.toLocaleString()} rows
                    </>
                  )}
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
          <div className="muted local-dataset-picker-hint">
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
