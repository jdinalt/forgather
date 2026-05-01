import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, ConfigInfo, ProjectInfo } from "../api";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import {
  coerceArgs,
  DynamicArgsForm,
  listMissingRequired,
  listOutOfBounds,
} from "./DynamicArgsForm";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

export function SubmitModal({ project, config, onClose, onSubmitted }: Props) {
  const qc = useQueryClient();
  const argsQ = useQuery({
    queryKey: ["dynamic-args", project.project_dir, config.name],
    queryFn: () => api.dynamicArgs(project.project_dir, config.name),
  });
  const gpusQ = useQuery({
    queryKey: ["gpus-once"],
    queryFn: api.listGpus,
  });
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });
  // Pulls nproc_per_node from the config's meta block. Cheap enough (same
  // materialization the Clean Output modal does) to run on open; we use it
  // to pick a sensible default for "GPUs" and to warn on mismatch.
  const outputQ = useQuery({
    queryKey: ["output-dir", project.project_dir, config.name],
    queryFn: () => api.configOutputDir(project.project_dir, config.name),
  });

  // Fetch cached overrides to pre-fill the form.
  const overridesQ = useQuery({
    queryKey: ["overrides", project.project_dir, config.name],
    queryFn: () => api.getOverrides(project.project_dir, config.name),
  });

  // Map of dest -> current value; strings only, coerced on submit
  const [values, setValues] = useState<Record<string, string>>({});
  const [requestedGpus, setRequestedGpus] = useState<number>(1);
  const [gpusTouched, setGpusTouched] = useState<boolean>(false);
  const [priority, setPriority] = useState<number>(0);
  // Track whether we've already seeded the form from the cache so we
  // don't overwrite edits the user has already made.
  const [overrideSeeded, setOverrideSeeded] = useState<boolean>(false);

  const maxGpus = Math.max(1, gpusQ.data?.length ?? 1);
  const idleGpuCount = useMemo(() => {
    if (!gpusQ.data) return null;
    return gpusQ.data.filter((g) => g.processes.length === 0).length;
  }, [gpusQ.data]);

  // Classify nproc_per_node. A positive integer means "fixed worker
  // count" — in that case we seed requestedGpus to match (torchrun will
  // spawn exactly that many workers regardless of visible GPUs, so
  // reserving fewer GPUs causes oversubscription and reserving more wastes
  // them). A string ("gpu" / "cpu" / "auto") means "auto-detect from
  // CUDA_VISIBLE_DEVICES" — the reservation count is load-bearing but
  // the user gets to choose any value.
  const nproc = outputQ.data?.nproc_per_node ?? null;
  const fixedWorkerCount =
    typeof nproc === "number" && Number.isInteger(nproc) && nproc > 0
      ? nproc
      : null;
  const gpuMismatch =
    fixedWorkerCount !== null && fixedWorkerCount !== requestedGpus;

  // Seed the GPU count once the config info arrives, unless the user has
  // already edited the field.
  useEffect(() => {
    if (gpusTouched) return;
    if (fixedWorkerCount !== null) {
      const clamped = Math.max(1, Math.min(maxGpus, fixedWorkerCount));
      setRequestedGpus(clamped);
    }
  }, [fixedWorkerCount, gpusTouched, maxGpus]);

  // Seed form values from cache once both schema and overrides have loaded.
  // Only seed entries whose dest exists in the current schema; silently
  // drop stale cached keys that are no longer in the schema.
  useEffect(() => {
    if (overrideSeeded) return;
    if (!argsQ.data || !overridesQ.data) return;
    const cached = overridesQ.data.values;
    if (Object.keys(cached).length === 0) {
      setOverrideSeeded(true);
      return;
    }
    const schemaDests = new Set(argsQ.data.map((a) => a.dest));
    const seed: Record<string, string> = {};
    for (const [k, v] of Object.entries(cached)) {
      if (schemaDests.has(k) && v != null) {
        seed[k] = String(v);
      }
    }
    if (Object.keys(seed).length > 0) {
      setValues((prev) => ({ ...seed, ...prev }));
    }
    setOverrideSeeded(true);
  }, [argsQ.data, overridesQ.data, overrideSeeded]);

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  // Required-arg enforcement: block Submit until every ``required: true``
  // field has a value. Recomputed every render so the button state tracks
  // the form live. The server enforces the same invariant — this is just
  // a usability layer.
  const missingRequired = useMemo(
    () => (argsQ.data ? listMissingRequired(argsQ.data, values) : []),
    [argsQ.data, values],
  );
  // Numeric bounds: same submit-gating idea as required.
  const outOfBounds = useMemo(
    () => (argsQ.data ? listOutOfBounds(argsQ.data, values) : []),
    [argsQ.data, values],
  );
  const submitBlockedReason: string | undefined =
    missingRequired.length > 0
      ? `Required arg(s) missing: ${missingRequired.map((a) => a.cli_name).join(", ")}`
      : outOfBounds.length > 0
        ? `Out-of-range value(s): ${outOfBounds.map((a) => a.cli_name).join(", ")}`
        : undefined;

  // Mirrors the OverridesModal Reset button: drop server-side cached
  // overrides for this config and zero out the in-form values so the
  // next submit goes out with template defaults. Stays open so the user
  // can review and submit (or tweak) without re-opening the modal.
  const clearOverridesMut = useMutation({
    mutationFn: () => api.clearOverrides(project.project_dir, config.name),
    onSuccess: () => {
      setValues({});
      setPriority(0);
      setGpusTouched(false);
      setRequestedGpus(fixedWorkerCount !== null ? Math.max(1, Math.min(maxGpus, fixedWorkerCount)) : 1);
      qc.invalidateQueries({
        queryKey: ["overrides", project.project_dir, config.name],
      });
      qc.invalidateQueries({
        queryKey: ["pp", project.project_dir, config.name],
      });
      qc.invalidateQueries({
        queryKey: ["output-dir", project.project_dir, config.name],
      });
    },
  });

  const handleReset = () => {
    if (!confirm("Clear all overrides for this config and reset the form?")) {
      return;
    }
    clearOverridesMut.mutate();
  };

  const submit = () => {
    const schema = argsQ.data ?? [];
    const dyn = coerceArgs(values, schema);
    enqueue.mutate({
      project_dir: project.project_dir,
      config: config.name,
      dynamic_args: dyn,
      requested_gpus: requestedGpus,
      priority,
    });
    // Best-effort: save overrides after enqueue so next open is pre-filled.
    // Don't block the submit on the result.
    api
      .setOverrides(project.project_dir, config.name, dyn)
      .then(() => {
        qc.invalidateQueries({
          queryKey: ["overrides", project.project_dir, config.name],
        });
        qc.invalidateQueries({
          queryKey: ["pp", project.project_dir, config.name],
        });
        qc.invalidateQueries({
          queryKey: ["output-dir", project.project_dir, config.name],
        });
      })
      .catch(() => {
        // best-effort; ignore failures
      });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Submit training job"
      >
        <header className="modal-header">
          <h3>Submit training job</h3>
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

          <div className="submit-row">
            <label>
              GPUs
              <input
                type="number"
                min={1}
                max={maxGpus}
                value={requestedGpus}
                onChange={(e) => {
                  setGpusTouched(true);
                  setRequestedGpus(
                    Math.max(1, Math.min(maxGpus, Number(e.target.value) || 1)),
                  );
                }}
              />
              {idleGpuCount !== null && (
                <span className="muted">
                  ({idleGpuCount} idle of {maxGpus})
                </span>
              )}
            </label>
            <label>
              Priority
              <input
                type="number"
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value) || 0)}
              />
              <span className="muted">higher runs sooner</span>
            </label>
          </div>

          <div className="submit-help muted">
            <strong>GPUs</strong> is how many CUDA devices the scheduler
            reserves for this run; the chosen indices become{" "}
            <code>CUDA_VISIBLE_DEVICES</code>. This config declares{" "}
            <code>nproc_per_node = {formatNproc(nproc)}</code>
            {nproc === "gpu" && (
              <>
                {" "}
                — torchrun will spawn one worker per visible GPU, so the
                number you pick here is also the worker count.
              </>
            )}
            {fixedWorkerCount !== null && (
              <>
                {" "}
                — torchrun will spawn exactly {fixedWorkerCount} worker(s)
                regardless of how many GPUs are visible. Picking a
                different number means the GPUs won't match the workers.
              </>
            )}
            {nproc !== null && typeof nproc === "string" && nproc !== "gpu" && (
              <> — torchrun will size workers from its own auto-detect.</>
            )}
          </div>

          {gpuMismatch && (
            <div className="notice notice-warn">
              This config has a fixed <code>nproc_per_node</code> of{" "}
              <strong>{fixedWorkerCount}</strong> but you're reserving{" "}
              <strong>{requestedGpus}</strong> GPU
              {requestedGpus === 1 ? "" : "s"}. The worker count won't
              match the reservation. Submit anyway only if you know what
              you're doing.
            </div>
          )}

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. The job will
              enqueue but won't start until the scheduler is enabled on the
              Queue tab.
            </div>
          )}

          <h4 className="dyn-heading">
            Dynamic arguments
            {argsQ.data && argsQ.data.length === 0 && (
              <span className="muted"> (this config declares none)</span>
            )}
          </h4>

          {argsQ.isLoading && <div className="muted pad">Loading…</div>}
          {argsQ.error && (
            <div className="err pad">
              <pre>{String(argsQ.error)}</pre>
            </div>
          )}
          {argsQ.data && argsQ.data.length > 0 && overrideSeeded && (
            // Seeding waits for both schema and cached overrides to land
            // so the form mounts with its true initial values. That
            // matters because DynArgGroupNode captures the initial
            // expansion state on first render — if we mount before
            // seeding, a required arg whose value was already cached
            // would still look "missing" briefly and force the group
            // open every time the modal reopens.
            <DynamicArgsForm
              schema={argsQ.data}
              values={values}
              onChange={(dest, v) =>
                setValues((prev) => ({ ...prev, [dest]: v }))
              }
              enforceRequired
            />
          )}
        </div>

        <footer className="modal-footer">
          <div className="muted current-path">
            {enqueue.error ? String(enqueue.error) : ""}
          </div>
          <div className="btn-row">
            <AutoWatchTtyToggle />
            <button
              className="secondary"
              onClick={handleReset}
              disabled={clearOverridesMut.isPending || enqueue.isPending}
              title="Drop saved overrides for this config and reset the form"
            >
              {clearOverridesMut.isPending ? "Resetting…" : "Reset to defaults"}
            </button>
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={
                enqueue.isPending ||
                argsQ.isLoading ||
                missingRequired.length > 0 ||
                outOfBounds.length > 0
              }
              title={submitBlockedReason}
            >
              {enqueue.isPending ? "Submitting…" : "Submit"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

function formatNproc(v: number | string | null): string {
  if (v === null) return "(unknown)";
  if (typeof v === "string") return `"${v}"`;
  return String(v);
}
