import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import {
  api,
  ClusterJobSubmitRequest,
  ClusterMembersResponse,
  ConfigInfo,
  ProjectInfo,
} from "../api";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import {
  coerceArgs,
  DynamicArgsForm,
  listMissingRequired,
  listOutOfBounds,
} from "./DynamicArgsForm";
import { ModalBackdrop } from "./ModalBackdrop";
import {
  emptyMultiNodeState,
  MultiNodePanelState,
  multiNodeStateFromOverrides,
  multiNodeStateToOverrides,
  MultiNodeSubmitPanel,
} from "./MultiNodeSubmitPanel";

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

  // Cluster membership — drives the multi-node panel inside this
  // modal. The query returns null cluster_name when the server isn't
  // running in cluster mode, in which case we render the modal in
  // its single-node form.
  const membersQ = useQuery<ClusterMembersResponse>({
    queryKey: ["cluster", "members"],
    queryFn: api.getClusterMembers,
    refetchInterval: 5000,
  });
  const clusterActive = !!membersQ.data?.cluster_name;
  // Per-node GPU snapshot for the multi-node panel: we cap each
  // peer's GPUs spinner by that node's actual hardware and show
  // idle counts. The 5s refresh keeps "(N idle of M)" approximately
  // live without saturating the master proxy.
  const clusterGpusQ = useQuery({
    queryKey: ["cluster", "gpus"],
    queryFn: api.getClusterGpus,
    refetchInterval: 5000,
    enabled: clusterActive,
  });

  // Map of dest -> current value; strings only, coerced on submit
  const [values, setValues] = useState<Record<string, string>>({});
  const [requestedGpus, setRequestedGpus] = useState<number>(1);
  const [gpusTouched, setGpusTouched] = useState<boolean>(false);
  const [priority, setPriority] = useState<number>(0);
  // Track whether we've already seeded the form from the cache so we
  // don't overwrite edits the user has already made.
  const [overrideSeeded, setOverrideSeeded] = useState<boolean>(false);

  // Multi-node panel state. Only consulted when ``clusterActive`` and
  // the user has more than the local node selected — otherwise the
  // submit takes the regular single-node path.
  const [mnState, setMnState] = useState<MultiNodePanelState>(emptyMultiNodeState);
  const [mnSeeded, setMnSeeded] = useState<boolean>(false);

  const maxGpus = Math.max(1, gpusQ.data?.length ?? 1);
  const idleGpuCount = useMemo(() => {
    if (!gpusQ.data) return null;
    // Mirror the scheduler's dispatch rule: a GPU is available iff it
    // is not excluded (CUDA_VISIBLE_DEVICES) and not runtime-disabled.
    // External processes — desktop compositors, unrelated CUDA work —
    // don't gate dispatch.
    return gpusQ.data.filter((g) => !g.excluded && !g.disabled).length;
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

  // Seed the GPU count from (in priority order):
  //   1. user edits in this session (gpusTouched)
  //   2. cached overrides for this (project, config)
  //   3. config's nproc_per_node, when it pins a fixed worker count
  //   4. 1
  // The cache wins over fixedWorkerCount because the user explicitly chose
  // a value last time; if they want to fall back to the config default they
  // can hit "Reset to defaults".
  useEffect(() => {
    if (gpusTouched) return;
    const cached = overridesQ.data?.requested_gpus;
    if (typeof cached === "number" && cached >= 1) {
      setRequestedGpus(Math.max(1, Math.min(maxGpus, cached)));
      return;
    }
    if (fixedWorkerCount !== null) {
      setRequestedGpus(Math.max(1, Math.min(maxGpus, fixedWorkerCount)));
    }
  }, [fixedWorkerCount, gpusTouched, maxGpus, overridesQ.data?.requested_gpus]);

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

  // Seed multi-node panel state from (in priority order):
  //   1. cached multinode overrides for this (project, config)
  //   2. cluster active + no cache → just-this-node default, with
  //      master designated rdzv host
  //   3. cluster inactive → empty state, panel hidden
  // Re-seeding is gated by ``mnSeeded`` so subsequent renders don't
  // clobber the operator's in-flight edits.
  useEffect(() => {
    if (mnSeeded) return;
    if (!membersQ.data) return;
    if (!overridesQ.data) return;
    const cached = overridesQ.data.multinode;
    if (cached) {
      setMnState(multiNodeStateFromOverrides(cached));
    } else if (clusterActive) {
      const selfId = membersQ.data.self_node_id;
      const masterId = membersQ.data.master_node_id;
      // Default = local node only, "Use" pre-checked. The user
      // explicitly opts other nodes in; we never auto-tick remote
      // peers because the multi-node submit triggers a fanout the
      // user might not expect.
      setMnState({
        rdzvPort: 29400,
        selected: selfId ? new Set([selfId]) : new Set(),
        perNodeNproc: {},
        perNodeIface: {},
        rdzvNodeId: selfId ?? masterId ?? null,
        allowMismatch: false,
      });
    }
    setMnSeeded(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [membersQ.data, overridesQ.data, clusterActive]);

  // True when the operator has opted in more than just the local
  // node. Drives whether Submit goes via the cluster fanout or the
  // existing single-node enqueue. A single-node selection that
  // happens to match the local node is intentionally treated as
  // single-node — there's no value in spinning up a one-rank
  // rendezvous on a single host.
  const useClusterFanout = useMemo(() => {
    if (!clusterActive) return false;
    if (mnState.selected.size === 0) return false;
    if (mnState.selected.size > 1) return true;
    // Exactly one node selected: cluster path only when it's
    // *not* the local node (e.g. the operator wants to remote-launch).
    const onlyId = Array.from(mnState.selected)[0];
    return onlyId !== membersQ.data?.self_node_id;
  }, [clusterActive, mnState.selected, membersQ.data?.self_node_id]);

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const submitCluster = useMutation({
    mutationFn: (req: ClusterJobSubmitRequest) => api.submitClusterJob(req),
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["cluster", "jobs"] });
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(resp.cluster_job.cluster_job_id);
      if (resp.warnings.length > 0) {
        // Server didn't block — surface what it did notice so the
        // operator gets one last chance to spot a divergence.
        alert("Submitted with warnings:\n\n" + resp.warnings.join("\n"));
      }
      onClose();
    },
  });

  const submitting = enqueue.isPending || submitCluster.isPending;
  const submitError = enqueue.error || submitCluster.error;

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
        : clusterActive && mnSeeded && mnState.selected.size === 0
          ? "Pick at least one node in the multi-node panel"
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
      // Also reset cluster panel state — "Reset to defaults" now
      // means "drop everything we cached for this config", including
      // the multi-node selection. Re-seeded by the seeding effect on
      // the next render.
      setMnState(emptyMultiNodeState());
      setMnSeeded(false);
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
    const members = membersQ.data?.members ?? [];

    // Persist last-used settings *before* submitting so a failed
    // submit still re-opens with the same form state. The cluster
    // panel state only gets persisted when cluster is active (no
    // point caching mn settings the user can't see in the next open).
    const mnPayload = clusterActive
      ? multiNodeStateToOverrides(mnState, members)
      : null;
    api
      .setOverrides(
        project.project_dir,
        config.name,
        dyn,
        requestedGpus,
        mnPayload,
      )
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

    if (useClusterFanout) {
      // Cluster fanout path. The master figures out rdzv args + the
      // per-peer NCCL/Gloo/TP iface (auto-derives from the peer's
      // advertised address when the operator left iface on "auto").
      // The same dynamic_args + priority ride along to every peer.
      const orderedSelected = members
        .filter((m) => mnState.selected.has(m.node_id))
        .map((m) => m.node_id);
      const req: ClusterJobSubmitRequest = {
        project_dir: project.project_dir,
        config: config.name,
        dynamic_args: dyn,
        priority,
        members: orderedSelected.map((id) => ({
          node_id: id,
          nproc_per_node: Math.max(1, mnState.perNodeNproc[id] ?? 1),
          nccl_socket_ifname:
            (mnState.perNodeIface[id] || "").trim() || null,
        })),
        rdzv_node_id: mnState.rdzvNodeId ?? undefined,
        rdzv_port: mnState.rdzvPort,
        allow_version_mismatch: mnState.allowMismatch,
      };
      submitCluster.mutate(req);
      return;
    }

    // Single-node enqueue. When cluster is active but the operator
    // only selected the local node, the GPUs spinner is hidden — the
    // panel's local-node nproc is the only knob, and it stands in
    // for requested_gpus. Falls back to the spinner value (or 1)
    // when cluster mode is off.
    let gpus = requestedGpus;
    if (clusterActive) {
      const selfId = membersQ.data?.self_node_id ?? null;
      const localNproc = selfId ? mnState.perNodeNproc[selfId] : undefined;
      if (typeof localNproc === "number" && localNproc >= 1) {
        gpus = localNproc;
      }
    }
    enqueue.mutate({
      project_dir: project.project_dir,
      config: config.name,
      dynamic_args: dyn,
      requested_gpus: gpus,
      priority,
    });
  };

  return (
    <ModalBackdrop onClose={onClose}>
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

          {/* Single-node controls. Hidden when the server is in
              cluster mode — the multi-node panel below owns the
              equivalent knobs (per-node nproc + iface) and showing
              both at once produced the duplicate UI the user
              flagged. Priority still belongs here because it
              applies to both submit paths, so it follows in its
              own always-visible row. */}
          {!clusterActive && (
            <>
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
                        Math.max(
                          1,
                          Math.min(maxGpus, Number(e.target.value) || 1),
                        ),
                      );
                    }}
                  />
                  {idleGpuCount !== null && (
                    <span className="muted">
                      ({idleGpuCount} idle of {maxGpus})
                    </span>
                  )}
                </label>
              </div>

              <div className="submit-help muted">
                <strong>GPUs</strong> is how many CUDA devices the
                scheduler reserves for this run; the chosen indices
                become <code>CUDA_VISIBLE_DEVICES</code>. This config
                declares{" "}
                <code>nproc_per_node = {formatNproc(nproc)}</code>
                {nproc === "gpu" && (
                  <>
                    {" "}
                    — torchrun will spawn one worker per visible GPU, so
                    the number you pick here is also the worker count.
                  </>
                )}
                {fixedWorkerCount !== null && (
                  <>
                    {" "}
                    — torchrun will spawn exactly {fixedWorkerCount}{" "}
                    worker(s) regardless of how many GPUs are visible.
                    Picking a different number means the GPUs won't
                    match the workers.
                  </>
                )}
                {nproc !== null &&
                  typeof nproc === "string" &&
                  nproc !== "gpu" && (
                    <> — torchrun will size workers from its own auto-detect.</>
                  )}
              </div>

              {gpuMismatch && (
                <div className="notice notice-warn">
                  This config has a fixed <code>nproc_per_node</code> of{" "}
                  <strong>{fixedWorkerCount}</strong> but you're reserving{" "}
                  <strong>{requestedGpus}</strong> GPU
                  {requestedGpus === 1 ? "" : "s"}. The worker count
                  won't match the reservation. Submit anyway only if you
                  know what you're doing.
                </div>
              )}
            </>
          )}

          <div className="submit-row">
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

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. The job will
              enqueue but won't start until the scheduler is enabled on the
              Queue tab.
            </div>
          )}

          {clusterActive && membersQ.data && mnSeeded && (
            // Collapsible section. Default open so the panel is
            // discoverable on first use; the operator can collapse
            // it on a small viewport so the dialog fits without
            // scrolling. When both this and the Dynamic args
            // sections are open, the CSS clamps each to half the
            // available body height and the inner table scrolls
            // independently.
            <details className="submit-section" open>
              <summary>
                <h4 className="dyn-heading">
                  Multi-node{" "}
                  <span className="muted">
                    (cluster <code>{membersQ.data.cluster_name}</code> —{" "}
                    {membersQ.data.members.length} member
                    {membersQ.data.members.length === 1 ? "" : "s"})
                  </span>
                </h4>
              </summary>
              <MultiNodeSubmitPanel
                members={membersQ.data}
                clusterGpus={clusterGpusQ.data}
                state={mnState}
                onChange={setMnState}
                defaultGpus={requestedGpus}
              />
            </details>
          )}

          <details className="submit-section" open>
            <summary>
              <h4 className="dyn-heading">
                Dynamic arguments
                {argsQ.data && argsQ.data.length === 0 && (
                  <span className="muted"> (this config declares none)</span>
                )}
              </h4>
            </summary>

            {argsQ.isLoading && <div className="muted pad">Loading…</div>}
            {argsQ.error && (
              <div className="err pad">
                <pre>{String(argsQ.error)}</pre>
              </div>
            )}
            {argsQ.data && argsQ.data.length > 0 && overrideSeeded && (
              // Seeding waits for both schema and cached overrides to
              // land so the form mounts with its true initial values.
              // That matters because DynArgGroupNode captures the
              // initial expansion state on first render — if we mount
              // before seeding, a required arg whose value was already
              // cached would still look "missing" briefly and force the
              // group open every time the modal reopens.
              <DynamicArgsForm
                schema={argsQ.data}
                values={values}
                onChange={(dest, v) =>
                  setValues((prev) => ({ ...prev, [dest]: v }))
                }
                enforceRequired
              />
            )}
          </details>
        </div>

        <footer className="modal-footer">
          <div className="muted current-path">
            {submitError ? String(submitError) : ""}
          </div>
          <div className="btn-row">
            <AutoWatchTtyToggle />
            <button
              className="secondary"
              onClick={handleReset}
              disabled={clearOverridesMut.isPending || submitting}
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
                submitting ||
                argsQ.isLoading ||
                missingRequired.length > 0 ||
                outOfBounds.length > 0 ||
                (clusterActive && mnSeeded && mnState.selected.size === 0)
              }
              title={submitBlockedReason}
            >
              {submitting
                ? "Submitting…"
                : useClusterFanout
                  ? `Submit to ${mnState.selected.size} nodes`
                  : "Submit"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}

function formatNproc(v: number | string | null): string {
  if (v === null) return "(unknown)";
  if (typeof v === "string") return `"${v}"`;
  return String(v);
}
