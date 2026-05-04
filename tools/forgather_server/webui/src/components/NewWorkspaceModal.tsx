import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api";
import { DirectoryBrowser } from "./DirectoryBrowser";
import { PathField } from "./PathField";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  /** Pre-select a parent search root in the dropdown. Used when the
   *  modal is opened from the Files-tree right-click on a directory
   *  whose enclosing search root we've already resolved. The user can
   *  still pick a different one. */
  initialParentDir?: string;
  /** Pre-fill the nested workspace-dir field. Used to drop the user
   *  into a half-built path (``relative/path/`` with trailing slash)
   *  so they only have to type the leaf name. */
  initialWorkspaceDirName?: string;
  onCreated: (workspace_dir: string) => void;
  onClose: () => void;
}

/** Mirrors ``forgather ws create``. The parent directory is constrained
 *  to one of the configured search roots, since a workspace planted
 *  outside of any search root would be invisible to discovery and
 *  promptly orphaned. The ``forgather_dir`` defaults to the server's
 *  "Forgather repo" quick-path so users on the bundled checkout don't
 *  have to type it. ``libs`` and ``search_paths`` are newline-separated
 *  text areas — multi-value forms are overkill for a CLI-equivalent
 *  prompt the user fills in once per workspace. */
const CREATE_ROOT_SENTINEL = "__create_new_root__";

export function NewWorkspaceModal({
  initialParentDir,
  initialWorkspaceDirName,
  onCreated,
  onClose,
}: Props) {
  const qc = useQueryClient();
  const rootsQ = useQuery({
    queryKey: ["search-roots"],
    queryFn: api.listSearchRoots,
  });
  const quickQ = useQuery({
    queryKey: ["fs-quick-paths"],
    queryFn: api.fsQuickPaths,
    staleTime: 5 * 60 * 1000,
  });

  const [parentDir, setParentDir] = useState<string>(initialParentDir ?? "");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [wsDirName, setWsDirName] = useState(initialWorkspaceDirName ?? "");
  const [forgatherDir, setForgatherDir] = useState("");
  // Pre-filled with the two libraries every workspace in the repo uses
  // (base + examples). Users can edit or clear; making them real defaults
  // is more useful than a placeholder that looks like one but isn't.
  const [libs, setLibs] = useState("base\nexamples");
  const [searchPaths, setSearchPaths] = useState("");
  const nameRef = useRef<HTMLInputElement>(null);

  // Inline "create new search root" sub-form. Activated when the user
  // picks the CREATE_ROOT_SENTINEL option in the dropdown. We collect
  // an existing parent dir + a name, then call POST /api/search-roots
  // with create=true so the server mkdirs the target and registers it
  // in one shot.
  const [creatingRoot, setCreatingRoot] = useState(false);
  const [newRootParent, setNewRootParent] = useState("");
  const [newRootName, setNewRootName] = useState("");
  const [browsingWsDir, setBrowsingWsDir] = useState(false);

  const createRoot = useMutation({
    mutationFn: () => {
      const trimmedName = newRootName.trim();
      if (!trimmedName) throw new Error("name is required");
      if (
        trimmedName.includes("/") ||
        trimmedName.includes("\\") ||
        trimmedName === "." ||
        trimmedName === ".."
      ) {
        throw new Error("name must be a bare directory name");
      }
      const fullPath = `${newRootParent.replace(/\/+$/, "")}/${trimmedName}`;
      return api.addSearchRoot(fullPath, true);
    },
    onSuccess: (root) => {
      qc.invalidateQueries({ queryKey: ["search-roots"] });
      setParentDir(root.path);
      setCreatingRoot(false);
      setNewRootParent("");
      setNewRootName("");
    },
  });

  // Default forgather_dir to the bundled "Forgather repo" quick-path so
  // most users can submit without touching the field. Don't stomp on a
  // value the user has typed.
  useEffect(() => {
    if (forgatherDir) return;
    const fg = quickQ.data?.find((q) => q.label === "Forgather repo");
    if (fg) setForgatherDir(fg.path);
  }, [quickQ.data, forgatherDir]);

  // Default parent_dir to the first existing search root. Same
  // don't-stomp rule.
  useEffect(() => {
    if (parentDir) return;
    const first = rootsQ.data?.find((r) => r.exists);
    if (first) setParentDir(first.path);
  }, [rootsQ.data, parentDir]);

  useEffect(() => {
    nameRef.current?.focus();
  }, []);

  // CLI-matching slugify: spaces -> underscores, lowercased, dots stripped.
  const derivedDirName = useMemo(
    () => name.trim().replace(/\s+/g, "_").toLowerCase().replace(/\./g, ""),
    [name],
  );
  const effectiveDirName = wsDirName.trim() || derivedDirName;
  const targetPreview =
    parentDir && effectiveDirName
      ? `${parentDir.replace(/\/+$/, "")}/${effectiveDirName}`
      : null;

  const splitLines = (s: string) =>
    s
      .split(/\r?\n/)
      .map((x) => x.trim())
      .filter((x) => x.length > 0);

  const create = useMutation({
    mutationFn: () =>
      api.newWorkspace({
        parent_dir: parentDir,
        name: name.trim(),
        description: description.trim(),
        workspace_dir_name: wsDirName.trim() || null,
        forgather_dir: forgatherDir.trim(),
        libs: splitLines(libs),
        search_paths: splitLines(searchPaths),
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      // Refresh only the *immediate parent* of the newly-created
      // workspace dir, so the Files tree's cached listing for that
      // parent picks up the new entry. With ``exact: false``, queries
      // whose key starts with ``["fs-browse", parent]`` match
      // (covering showHidden / files_too variations) without
      // cascading into sibling or unrelated directories' caches.
      const parent =
        r.workspace_dir.replace(/\/+$/, "").split("/").slice(0, -1).join("/") ||
        "/";
      qc.invalidateQueries({
        queryKey: ["fs-browse", parent],
        exact: false,
      });
      onCreated(r.workspace_dir);
      onClose();
    },
  });

  const canSubmit =
    !!parentDir &&
    !!name.trim() &&
    !!description.trim() &&
    !!forgatherDir.trim() &&
    !create.isPending;

  const submit = () => {
    if (!canSubmit) return;
    create.mutate();
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal new-workspace-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Create workspace"
      >
        <header className="modal-header">
          <h3>Create workspace</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="new-project-grid">
            <label className="full-row">
              <span className="muted">
                Parent (search root) *
                <span className="muted-hint">
                  {" "}
                  · workspace must live under a configured search root to be
                  discoverable
                </span>
              </span>
              <select
                value={creatingRoot ? CREATE_ROOT_SENTINEL : parentDir}
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === CREATE_ROOT_SENTINEL) {
                    setCreatingRoot(true);
                  } else {
                    setCreatingRoot(false);
                    setParentDir(v);
                  }
                }}
              >
                <option value="" disabled>
                  Select a search root…
                </option>
                {rootsQ.data?.map((r) => (
                  <option key={r.path} value={r.path} disabled={!r.exists}>
                    {r.path}
                    {!r.exists && " (missing)"}
                  </option>
                ))}
                <option value={CREATE_ROOT_SENTINEL}>
                  + Create new search root…
                </option>
              </select>
            </label>
            {creatingRoot && (
              <div className="new-root-inline full-row">
                <div className="muted-hint">
                  Pick an existing parent directory and give the new search
                  root a name; the directory will be created and registered.
                </div>
                <label className="full-row">
                  <span className="muted">Parent directory *</span>
                  <PathField
                    value={newRootParent}
                    onChange={setNewRootParent}
                    placeholder="/path/to/parent"
                    mode="dirs-only"
                    title="Pick parent directory"
                  />
                </label>
                <label className="full-row">
                  <span className="muted">New search root name *</span>
                  <input
                    type="text"
                    value={newRootName}
                    onChange={(e) => setNewRootName(e.target.value)}
                    placeholder="my_workspaces"
                    spellCheck={false}
                  />
                </label>
                {createRoot.error && (
                  <div className="err pad">
                    <pre>{String(createRoot.error)}</pre>
                  </div>
                )}
                <div className="btn-row">
                  <button
                    className="secondary"
                    onClick={() => {
                      setCreatingRoot(false);
                      setNewRootParent("");
                      setNewRootName("");
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => createRoot.mutate()}
                    disabled={
                      createRoot.isPending ||
                      !newRootParent.trim() ||
                      !newRootName.trim()
                    }
                  >
                    {createRoot.isPending ? "Creating…" : "Create root"}
                  </button>
                </div>
              </div>
            )}
            <label>
              <span className="muted">Name *</span>
              <input
                ref={nameRef}
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Workspace"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    submit();
                  }
                }}
              />
            </label>
            <label>
              <span className="muted">Description *</span>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Short description"
              />
            </label>
            <label>
              <span className="muted">
                Workspace dir
                <span className="muted-hint">
                  {" "}
                  · relative to parent; may be nested (mkdir -p). Use
                  Browse… to pick an existing subdirectory.
                </span>
              </span>
              <div className="path-field">
                <input
                  type="text"
                  value={wsDirName}
                  onChange={(e) => setWsDirName(e.target.value)}
                  placeholder={derivedDirName || "auto-derived from name"}
                  spellCheck={false}
                />
                <button
                  type="button"
                  className="secondary"
                  disabled={!parentDir}
                  onClick={() => setBrowsingWsDir(true)}
                >
                  Browse…
                </button>
              </div>
            </label>
            <label className="full-row">
              <span className="muted">Forgather dir *</span>
              <PathField
                value={forgatherDir}
                onChange={setForgatherDir}
                placeholder="/path/to/forgather"
                mode="dirs-only"
                title="Pick Forgather installation directory"
              />
            </label>
            <label className="full-row">
              <span className="muted">
                Libraries
                <span className="muted-hint">
                  {" "}
                  · one per line; resolved under{" "}
                  <code>forgather/templatelib/</code>. Defaults to{" "}
                  <code>base</code> + <code>examples</code> — edit or
                  clear as needed.
                </span>
              </span>
              <textarea
                value={libs}
                onChange={(e) => setLibs(e.target.value)}
                rows={3}
                spellCheck={false}
              />
            </label>
            <label className="full-row">
              <span className="muted">
                Additional search paths
                <span className="muted-hint">
                  {" "}
                  · one absolute path per line
                </span>
              </span>
              <textarea
                value={searchPaths}
                onChange={(e) => setSearchPaths(e.target.value)}
                rows={3}
                spellCheck={false}
              />
            </label>
          </div>

          {targetPreview && (
            <div className="new-template-preview">
              <span className="muted">will create</span>
              <code title={targetPreview}>{targetPreview}</code>
            </div>
          )}

          {create.error && (
            <div className="err pad">
              <pre>{String(create.error)}</pre>
            </div>
          )}
        </div>

        <footer className="modal-footer">
          <div />
          <div className="btn-row">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button onClick={submit} disabled={!canSubmit}>
              {create.isPending ? "Creating…" : "Create"}
            </button>
          </div>
        </footer>

        {browsingWsDir && parentDir && (
          <DirectoryBrowser
            initialPath={parentDir}
            mode="dirs-only"
            title={`Pick existing subdirectory under ${parentDir}`}
            onCancel={() => setBrowsingWsDir(false)}
            onPick={(picked) => {
              setBrowsingWsDir(false);
              const parentNorm = parentDir.replace(/\/+$/, "");
              const pickedNorm = picked.replace(/\/+$/, "");
              if (
                pickedNorm !== parentNorm &&
                !pickedNorm.startsWith(parentNorm + "/")
              ) {
                alert(
                  `Picked directory is not under the parent search root:\n\n${picked}\n\nNot under: ${parentDir}`,
                );
                return;
              }
              const rel =
                pickedNorm === parentNorm
                  ? ""
                  : pickedNorm.slice(parentNorm.length + 1);
              // Trailing "/" so the user knows to append the leaf name.
              // Empty rel (picked the parent itself) means leaf only — no
              // trailing slash needed; leave the field empty for them.
              setWsDirName(rel ? `${rel}/` : "");
            }}
          />
        )}
      </div>
    </ModalBackdrop>
  );
}
