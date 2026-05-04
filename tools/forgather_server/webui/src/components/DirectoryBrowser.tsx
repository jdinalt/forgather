import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../api";
import { ModalBackdrop } from "./ModalBackdrop";

export type BrowseMode = "dirs-only" | "files-and-dirs";

interface Props {
  initialPath?: string;
  /** Controls listing contents and the footer's "Use this directory" label.
   *  - ``dirs-only``: only directories listed; footer picks the current directory.
   *  - ``files-and-dirs``: files listed too; clicking a file selects it,
   *    directories still navigate. Footer still lets you pick the current
   *    directory (for path args where a directory is valid). */
  mode?: BrowseMode;
  title?: string;
  onCancel: () => void;
  onPick: (path: string) => void;
}

export function DirectoryBrowser({
  initialPath,
  mode = "dirs-only",
  title,
  onCancel,
  onPick,
}: Props) {
  const [path, setPath] = useState<string>(initialPath ?? "");
  const [showHidden, setShowHidden] = useState(false);
  const [pathInput, setPathInput] = useState<string>(initialPath ?? "");
  const [mkdirError, setMkdirError] = useState<string | null>(null);

  const filesToo = mode === "files-and-dirs";

  const qc = useQueryClient();
  const quickQ = useQuery({
    queryKey: ["fs-quick"],
    queryFn: api.fsQuickPaths,
  });

  const mkdir = useMutation({
    mutationFn: ({ parent, name }: { parent: string; name: string }) =>
      api.fsMkdir(parent, name),
    onSuccess: (r) => {
      // Refresh the listing so the new dir shows up immediately, then
      // navigate into it — that's almost always what the user wants
      // next.
      qc.invalidateQueries({ queryKey: ["fs-browse"] });
      setPath(r.path);
      setMkdirError(null);
    },
    onError: (e: unknown) => {
      setMkdirError(e instanceof Error ? e.message : String(e));
    },
  });

  const onNewFolder = () => {
    setMkdirError(null);
    if (!path) return;
    const name = window.prompt(
      `New folder under:\n${path}\n\nName:`,
      "",
    );
    if (name == null) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    mkdir.mutate({ parent: path, name: trimmed });
  };

  // Seed the initial path once quick-paths land (or from prop).
  useEffect(() => {
    if (!path && quickQ.data && quickQ.data.length) {
      const seed = quickQ.data[0].path;
      setPath(seed);
      setPathInput(seed);
    }
  }, [quickQ.data, path]);

  const listingQ = useQuery({
    queryKey: ["fs-browse", path, showHidden, filesToo],
    queryFn: () => api.fsBrowse(path, showHidden, filesToo),
    enabled: !!path,
  });

  // Keep input in sync with path as we navigate.
  useEffect(() => {
    setPathInput(path);
  }, [path]);

  const goTo = (p: string) => setPath(p);

  const heading = title ?? (mode === "dirs-only" ? "Add Search Root" : "Pick Path");
  const emptyMsg =
    mode === "dirs-only" ? "(no subdirectories)" : "(empty directory)";

  return (
    <ModalBackdrop onClose={onCancel}>
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Pick a path"
      >
        <header className="modal-header">
          <h3>{heading}</h3>
          <button className="tiny" onClick={onCancel} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="quick-row">
            {quickQ.data?.map((q) => (
              <button key={q.path} className="chip" onClick={() => goTo(q.path)}>
                {q.label}
              </button>
            ))}
            {listingQ.data?.parent && (
              <button
                className="chip"
                onClick={() => listingQ.data?.parent && goTo(listingQ.data.parent)}
              >
                Up
              </button>
            )}
            <label className="hidden-toggle">
              <input
                type="checkbox"
                checked={showHidden}
                onChange={(e) => setShowHidden(e.target.checked)}
              />
              Show hidden
            </label>
            <button
              type="button"
              className="chip"
              onClick={onNewFolder}
              disabled={!path || mkdir.isPending}
              title="Create a new directory under the current path"
            >
              + New Folder
            </button>
          </div>
          {mkdirError && (
            <div className="err pad" role="alert">
              <pre>{mkdirError}</pre>
            </div>
          )}

          <form
            className="path-row"
            onSubmit={(e) => {
              e.preventDefault();
              if (pathInput.trim()) {
                // In file-picker mode, "Go" on a path that points at a file
                // selects it directly; otherwise it's a navigate.
                const trimmed = pathInput.trim();
                if (filesToo) {
                  onPick(trimmed);
                } else {
                  goTo(trimmed);
                }
              }
            }}
          >
            <input
              type="text"
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              placeholder="/path/to/directory"
              spellCheck={false}
            />
            <button type="submit">{filesToo ? "Use" : "Go"}</button>
          </form>

          {listingQ.isLoading && <div className="muted pad">Loading…</div>}
          {listingQ.error && (
            <div className="err pad">
              <pre>{String(listingQ.error)}</pre>
            </div>
          )}
          {listingQ.data && (
            <ul className="dir-listing">
              {listingQ.data.entries.length === 0 && (
                <li className="muted pad">{emptyMsg}</li>
              )}
              {listingQ.data.entries.map((e) => (
                <li key={e.path}>
                  <button
                    className="dir-item"
                    onClick={() => (e.is_dir ? goTo(e.path) : onPick(e.path))}
                  >
                    <span className="glyph">{e.is_dir ? "▸" : "·"}</span>
                    {e.name}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <footer className="modal-footer">
          <div className="muted current-path" title={path}>
            {path}
          </div>
          <div className="btn-row">
            <button className="secondary" onClick={onCancel}>
              Cancel
            </button>
            <button
              onClick={() => path && onPick(path)}
              disabled={!path || !!listingQ.error}
            >
              Use This Directory
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
