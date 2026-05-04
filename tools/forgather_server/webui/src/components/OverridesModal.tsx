import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, ConfigInfo, ProjectInfo } from "../api";
import { coerceArgs, DynamicArgsForm } from "./DynamicArgsForm";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  onClose: () => void;
}

export function OverridesModal({ project, config, onClose }: Props) {
  const qc = useQueryClient();

  const argsQ = useQuery({
    queryKey: ["dynamic-args", project.project_dir, config.name],
    queryFn: () => api.dynamicArgs(project.project_dir, config.name),
  });

  const overridesQ = useQuery({
    queryKey: ["overrides", project.project_dir, config.name],
    queryFn: () => api.getOverrides(project.project_dir, config.name),
  });

  const [values, setValues] = useState<Record<string, string>>({});
  const [seeded, setSeeded] = useState(false);

  // Seed form from cache once both schema and overrides arrive.
  useEffect(() => {
    if (seeded) return;
    if (!argsQ.data || !overridesQ.data) return;
    const cached = overridesQ.data.values;
    const schemaDests = new Set(argsQ.data.map((a) => a.dest));
    const seed: Record<string, string> = {};
    for (const [k, v] of Object.entries(cached)) {
      if (schemaDests.has(k) && v != null) {
        seed[k] = String(v);
      }
    }
    setValues(seed);
    setSeeded(true);
  }, [argsQ.data, overridesQ.data, seeded]);

  const invalidateAll = () => {
    qc.invalidateQueries({
      queryKey: ["overrides", project.project_dir, config.name],
    });
    qc.invalidateQueries({
      queryKey: ["pp", project.project_dir, config.name],
    });
    qc.invalidateQueries({
      queryKey: ["output-dir", project.project_dir, config.name],
    });
    // Overrides can move ``output_dir`` (e.g. a finetune that points
    // ``model_id_or_path`` at an external dir), which changes the
    // catalog's per-config grouping + checkpoint counts. Without this,
    // right-click conditional items (Serve / Eval / Convert / Finalize)
    // stay hidden until the user hits Refresh.
    qc.invalidateQueries({
      queryKey: ["project-models", project.project_dir],
    });
  };

  const saveMut = useMutation({
    mutationFn: (vals: Record<string, unknown>) =>
      api.setOverrides(project.project_dir, config.name, vals),
    onSuccess: () => {
      invalidateAll();
      onClose();
    },
  });

  const clearMut = useMutation({
    mutationFn: () => api.clearOverrides(project.project_dir, config.name),
    onSuccess: () => {
      invalidateAll();
      onClose();
    },
  });

  const handleSave = () => {
    const schema = argsQ.data ?? [];
    const coerced = coerceArgs(values, schema);
    saveMut.mutate(coerced);
  };

  const handleReset = () => {
    if (!confirm("Clear all overrides for this config?")) return;
    clearMut.mutate();
  };

  const busy = saveMut.isPending || clearMut.isPending;

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal overrides-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Set overrides"
      >
        <header className="modal-header">
          <h3>Set overrides for {config.name}</h3>
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

          <div className="notice overrides-notice">
            These values are applied automatically to pp / output-dir / config
            meta and pre-fill the Submit form. Blank fields fall back to
            template defaults.
          </div>

          {argsQ.isLoading && <div className="muted pad">Loading…</div>}
          {argsQ.error && (
            <div className="err pad">
              <pre>{String(argsQ.error)}</pre>
            </div>
          )}

          {argsQ.data && argsQ.data.length === 0 && (
            <h4 className="dyn-heading">
              Dynamic arguments
              <span className="muted"> (this config declares none)</span>
            </h4>
          )}

          {argsQ.data && argsQ.data.length > 0 && seeded && (
            // Wait for seeding before mounting the form so the
            // initial-open state DynArgGroupNode latches on first render
            // reflects cached overrides rather than a transient empty
            // map. See SubmitModal for the same gate.
            <>
              <h4 className="dyn-heading">Dynamic arguments</h4>
              <DynamicArgsForm
                schema={argsQ.data}
                values={values}
                onChange={(dest, v) =>
                  setValues((prev) => ({ ...prev, [dest]: v }))
                }
                enforceRequired
              />
            </>
          )}

          {(saveMut.error || clearMut.error) && (
            <div className="err pad">
              <pre>{String(saveMut.error ?? clearMut.error)}</pre>
            </div>
          )}
        </div>

        <footer className="modal-footer">
          <div className="btn-row">
            <button
              className="destructive"
              onClick={handleReset}
              disabled={busy}
              title="Remove all cached overrides for this config"
            >
              Reset to defaults
            </button>
          </div>
          <div className="btn-row">
            <button className="secondary" onClick={onClose} disabled={busy}>
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={busy || argsQ.isLoading}
            >
              {saveMut.isPending ? "Saving…" : "Save"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
