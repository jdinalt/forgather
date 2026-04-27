import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api";
import { DirectoryBrowser } from "./DirectoryBrowser";
import { NewWorkspaceModal } from "./NewWorkspaceModal";

/** Top-level sidebar block listing the project-discovery roots, with
 *  Browse… (add a root) and 📁 New Workspace… (create a workspace
 *  inside one of them). Lifted out of ProjectTree so each sidebar
 *  group — Search Roots, Projects, Files — is an independent
 *  top-level collapsible. */
export function SearchRootsPanel() {
  const qc = useQueryClient();
  const rootsQ = useQuery({
    queryKey: ["search-roots"],
    queryFn: api.listSearchRoots,
  });

  const addRoot = useMutation({
    mutationFn: (path: string) => api.addSearchRoot(path),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["search-roots"] });
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });
  const removeRoot = useMutation({
    mutationFn: api.removeSearchRoot,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["search-roots"] });
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  const [browsing, setBrowsing] = useState(false);
  const [newWorkspaceOpen, setNewWorkspaceOpen] = useState(false);

  return (
    <div className="search-roots-panel">
      {rootsQ.isLoading && <div>Loading...</div>}
      {rootsQ.error && <div className="err">{String(rootsQ.error)}</div>}
      <ul className="roots-list">
        {rootsQ.data?.map((r) => (
          <li key={r.path} className={r.exists ? "" : "missing"}>
            <code>{r.path}</code>
            <button
              className="tiny"
              onClick={() => removeRoot.mutate(r.path)}
              title="Remove root"
            >
              ×
            </button>
          </li>
        ))}
      </ul>
      <div className="search-roots-actions">
        <button className="add-root-btn" onClick={() => setBrowsing(true)}>
          Browse…
        </button>
        <button
          className="add-root-btn"
          onClick={() => setNewWorkspaceOpen(true)}
          title="Create a new workspace under one of the search roots"
        >
          📁 New Workspace…
        </button>
      </div>
      {browsing && (
        <DirectoryBrowser
          onCancel={() => setBrowsing(false)}
          onPick={(path) => {
            addRoot.mutate(path);
            setBrowsing(false);
          }}
        />
      )}
      {newWorkspaceOpen && (
        <NewWorkspaceModal
          onCreated={() => {
            // Tree refresh comes from the modal's mutation onSuccess hook.
          }}
          onClose={() => setNewWorkspaceOpen(false)}
        />
      )}
    </div>
  );
}
