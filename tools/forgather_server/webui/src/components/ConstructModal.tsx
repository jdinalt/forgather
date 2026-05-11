import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, ConfigInfo, ProjectInfo } from "../api";
import { useDatasetSource } from "../dataset-source";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import {
  coerceArgs,
  DynamicArgsForm,
  listMissingRequired,
  listOutOfBounds,
} from "./DynamicArgsForm";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

/** Submit modal for ``forgather construct``.
 *
 *  Construct is the generic "materialize this target and print the repr"
 *  diagnostic — it also drives side-effecting targets (e.g. a tokenizer
 *  target that trains + saves the tokenizer on first construction). The
 *  modal lets the user pick the target (defaulting to ``main``), set
 *  dynamic args, optionally route dataset loads through a dataset
 *  server, and reserve N GPUs in case the target needs CUDA storage.
 *  The job runs in its own process — never inside the server — so
 *  long-running constructs don't block the API loop. */
export function ConstructModal({ project, config, onClose, onSubmitted }: Props) {
  const qc = useQueryClient();

  const argsQ = useQuery({
    queryKey: ["dynamic-args", project.project_dir, config.name],
    queryFn: () => api.dynamicArgs(project.project_dir, config.name),
  });
  const targetsQ = useQuery({
    queryKey: ["code-targets", project.project_dir, config.name],
    queryFn: () => api.configCodeTargets(project.project_dir, config.name),
  });
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });
  const overridesQ = useQuery({
    queryKey: ["overrides", project.project_dir, config.name],
    queryFn: () => api.getOverrides(project.project_dir, config.name),
  });

  const [values, setValues] = useState<Record<string, string>>({});
  const [overrideSeeded, setOverrideSeeded] = useState<boolean>(false);

  // Dataset-source selector — shared hook handles state + offline-
  // fallback seeding. Construct targets that hit ``fast_load_iterable_dataset``
  // pick this up via the FORGATHER_DATASET_SERVER env vars the server
  // merges into job_params.extra_env.
  const { source: datasetSource, selector: datasetSourceSelector } =
    useDatasetSource({
      ready: !!overridesQ.data,
      initial: overridesQ.data?.dataset_source ?? null,
    });

  const [target, setTarget] = useState<string>("main");
  const [call, setCall] = useState<boolean>(false);
  const [requestedGpus, setRequestedGpus] = useState<number>(0);
  const [priority, setPriority] = useState<number>(0);

  // When the targets list arrives, pick ``main`` if present, otherwise
  // the first target. The construct CLI defaults to ``main`` too, so
  // this matches the CLI's behaviour when the config doesn't expose it.
  useEffect(() => {
    if (!targetsQ.data || targetsQ.data.length === 0) return;
    if (targetsQ.data.includes(target)) return;
    setTarget(targetsQ.data.includes("main") ? "main" : targetsQ.data[0]);
  }, [targetsQ.data, target]);

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

  const missingRequired = useMemo(
    () => (argsQ.data ? listMissingRequired(argsQ.data, values) : []),
    [argsQ.data, values],
  );
  const outOfBounds = useMemo(
    () => (argsQ.data ? listOutOfBounds(argsQ.data, values) : []),
    [argsQ.data, values],
  );

  const submitBlockedReason: string | undefined =
    missingRequired.length > 0
      ? `Required arg(s) missing: ${missingRequired.map((a) => a.cli_name).join(", ")}`
      : outOfBounds.length > 0
        ? `Out-of-range value(s): ${outOfBounds.map((a) => a.cli_name).join(", ")}`
        : !target
          ? "Pick a target"
          : undefined;

  const submit = () => {
    const schema = argsQ.data ?? [];
    const dyn = coerceArgs(values, schema);
    enqueue.mutate({
      project_dir: project.project_dir,
      config: config.name,
      dynamic_args: dyn,
      requested_gpus: requestedGpus,
      priority,
      job_type: "construct",
      job_params: {
        target,
        call,
      },
      dataset_source: datasetSource,
    });
    api
      .setOverrides(
        project.project_dir,
        config.name,
        dyn,
        null,
        null,
        datasetSource,
      )
      .then(() => {
        qc.invalidateQueries({
          queryKey: ["overrides", project.project_dir, config.name],
        });
      })
      .catch(() => {});
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Submit construct job"
      >
        <header className="modal-header">
          <h3>Construct…</h3>
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

          {datasetSourceSelector}

          <div className="submit-row">
            <label
              style={{ flex: 1 }}
              title="Top-level configuration key to materialize. 'main' is the default; pick another to construct (and print the repr of) a specific sub-target — useful for diagnostics or for targets that have side effects (e.g. tokenizer training)."
            >
              Target
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                disabled={targetsQ.isLoading}
              >
                {targetsQ.data && targetsQ.data.length === 0 && (
                  <option value="">(no targets — config parse failed?)</option>
                )}
                {targetsQ.data?.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label title="GPUs to reserve for this job. 0 is fine for purely meta-device / CPU targets; bump it up only if the target allocates real CUDA storage during construction.">
              GPUs
              <input
                type="number"
                min={0}
                step={1}
                value={requestedGpus}
                onChange={(e) =>
                  setRequestedGpus(Math.max(0, Number(e.target.value) || 0))
                }
              />
            </label>
            <label title="Higher priority dispatches sooner once a slot is free.">
              Priority
              <input
                type="number"
                step={1}
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value) || 0)}
              />
              <span className="muted">higher runs sooner</span>
            </label>
          </div>

          <div className="submit-row">
            <label title="After materializing the target, call it as a zero-arg callable and print the repr of the return value. Use with !partial / !factory / !singleton targets that produce a callable.">
              <input
                type="checkbox"
                checked={call}
                onChange={(e) => setCall(e.target.checked)}
              />
              --call (invoke the materialized object)
            </label>
          </div>

          {targetsQ.error && (
            <div className="err pad">
              <pre>Could not list targets: {String(targetsQ.error)}</pre>
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
          {argsQ.data && argsQ.data.length > 0 && (
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
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={
                enqueue.isPending ||
                argsQ.isLoading ||
                targetsQ.isLoading ||
                !target ||
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
    </ModalBackdrop>
  );
}
