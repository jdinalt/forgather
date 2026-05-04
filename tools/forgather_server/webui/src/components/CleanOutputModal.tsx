import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, ConfigInfo, ProjectInfo } from "../api";
import { DirectoryBrowser } from "./DirectoryBrowser";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  onClose: () => void;
}

/** Delete an output directory for a config — usually the default one
 *  resolved from the config's materialized meta block, but editable so
 *  the user can clean up locations that were overridden via --output-dir. */
export function CleanOutputModal({ project, config, onClose }: Props) {
  const qc = useQueryClient();
  const infoQ = useQuery({
    queryKey: ["output-dir", project.project_dir, config.name],
    queryFn: () => api.configOutputDir(project.project_dir, config.name),
  });

  const [path, setPath] = useState<string>("");
  const [browsing, setBrowsing] = useState(false);
  const [seeded, setSeeded] = useState(false);

  // Default to the resolved output_dir once it arrives; don't stomp on a
  // path the user has started editing.
  useEffect(() => {
    if (!seeded && infoQ.data) {
      setPath(infoQ.data.output_dir);
      setSeeded(true);
    }
  }, [infoQ.data, seeded]);

  const del = useMutation({
    mutationFn: (p: string) => api.deleteDir(p),
    onSuccess: (_data, deletedPath) => {
      // The output dir going away can change what pp/trefs render, and
      // the size info we just showed is now stale. Also drop the
      // project-tree caches that surface artifacts under this path so
      // the tree reflects the cleanup without a manual refresh.
      qc.invalidateQueries({ queryKey: ["output-dir"] });
      qc.invalidateQueries({
        queryKey: ["project-models", project.project_dir],
      });
      qc.invalidateQueries({ queryKey: ["model-runs", deletedPath] });
      qc.invalidateQueries({ queryKey: ["model-checkpoints", deletedPath] });
      qc.invalidateQueries({ queryKey: ["model-evaluations", deletedPath] });
      onClose();
    },
  });

  // Show the stats for the currently-typed path if it matches one of the
  // known dirs we already queried; otherwise just hide the stats block.
  const matchingInfo =
    infoQ.data == null
      ? null
      : path === infoQ.data.output_dir
        ? {
            exists: infoQ.data.output_dir_exists,
            size: infoQ.data.output_dir_size_bytes,
            entries: infoQ.data.output_dir_entry_count,
            label: "output_dir",
          }
        : path === infoQ.data.models_dir
          ? {
              exists: infoQ.data.models_dir_exists,
              size: infoQ.data.models_dir_size_bytes,
              entries: infoQ.data.models_dir_entry_count,
              label: "models_dir",
            }
          : null;

  const attemptDelete = () => {
    const target = path.trim();
    if (!target) return;
    const sizeHint = matchingInfo
      ? ` (${fmtBytes(matchingInfo.size)}, ${matchingInfo.entries} entries)`
      : "";
    if (!confirm(`Delete this directory?${sizeHint}\n\n${target}`)) return;
    del.mutate(target);
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal clean-output-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Clean output directory"
      >
        <header className="modal-header">
          <h3>Clean output directory</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-summary">
            <div>
              <span className="muted">config</span>
              <code>{config.name}</code>
            </div>
            <div>
              <span className="muted">project</span>
              <code>{project.project_dir}</code>
            </div>
          </div>

          {infoQ.isLoading && <div className="muted pad">Resolving…</div>}
          {infoQ.error && (
            <div className="err pad">
              <pre>{String(infoQ.error)}</pre>
            </div>
          )}

          {infoQ.data && (
            <div className="clean-presets">
              <button
                className="chip"
                onClick={() => setPath(infoQ.data!.output_dir)}
                title={infoQ.data.output_dir}
              >
                output_dir
              </button>
              <button
                className="chip"
                onClick={() => setPath(infoQ.data!.models_dir)}
                title={infoQ.data.models_dir}
              >
                models_dir (ALL models)
              </button>
            </div>
          )}

          <div className="clean-path-row">
            <input
              type="text"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="/path/to/directory"
              spellCheck={false}
            />
            <button
              type="button"
              className="secondary"
              onClick={() => setBrowsing(true)}
            >
              Browse…
            </button>
          </div>

          {matchingInfo && (
            <div className="clean-stats">
              {matchingInfo.exists ? (
                <>
                  <span className="muted">{matchingInfo.label}:</span>{" "}
                  <strong>{fmtBytes(matchingInfo.size)}</strong> across{" "}
                  <strong>{matchingInfo.entries}</strong> entries
                </>
              ) : (
                <span className="muted">
                  {matchingInfo.label}: does not exist (nothing to delete)
                </span>
              )}
            </div>
          )}

          {del.error && (
            <div className="err pad">
              <pre>{String(del.error)}</pre>
            </div>
          )}

          <div className="warning-box">
            This removes the directory and all checkpoints, runs, and logs
            inside it. It cannot be undone.
          </div>
        </div>

        <footer className="modal-footer">
          <div className="muted current-path" title={path}>
            {path}
          </div>
          <div className="btn-row">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              className="destructive"
              onClick={attemptDelete}
              disabled={del.isPending || !path.trim()}
            >
              {del.isPending ? "Deleting…" : "Delete"}
            </button>
          </div>
        </footer>

        {browsing && (
          <DirectoryBrowser
            initialPath={path || undefined}
            mode="files-and-dirs"
            title="Pick output directory to delete"
            onCancel={() => setBrowsing(false)}
            onPick={(p) => {
              setPath(p);
              setBrowsing(false);
            }}
          />
        )}
      </div>
    </ModalBackdrop>
  );
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}
