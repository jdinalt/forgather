import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import {
  api,
  ClusterJobSubmitRequest,
  ClusterMembersResponse,
  ConfigInfo,
  DiLoCoInfo,
  DiLoCoServer,
  ProjectInfo,
} from "../api";
import { useDatasetSource } from "../dataset-source";
import { persistGet, persistSet } from "../persist";
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
  // Optional override for torchrun's --nproc-per-node, sent through
  // job_params.nproc. Empty string = use the config's nproc_per_node
  // (with the launcher's "gpu"->1 fallback when 0 GPUs are reserved).
  // Free-form text so operators can type "4" or "auto" / "cpu" /
  // "gpu" -- torchrun accepts all three sentinels.
  const [nprocOverride, setNprocOverride] = useState<string>("");
  const [priority, setPriority] = useState<number>(0);
  // Track whether we've already seeded the form from the cache so we
  // don't overwrite edits the user has already made.
  const [overrideSeeded, setOverrideSeeded] = useState<boolean>(false);

  // Multi-node panel state. Only consulted when ``clusterActive`` and
  // the user has more than the local node selected — otherwise the
  // submit takes the regular single-node path.
  const [mnState, setMnState] = useState<MultiNodePanelState>(emptyMultiNodeState);
  const [mnSeeded, setMnSeeded] = useState<boolean>(false);

  // DiLoCo opt-in: when the operator picks a server from the radio
  // group, this worker joins it. ``selectedDiLoCoBase`` is the chosen
  // server's base_url ("" means "None — don't join"). The dependent
  // fields are only consulted when a non-empty base is selected.
  //
  // Persisted to localStorage per (project, config) so a second worker
  // submitted from the same modal slot retains the prior DiLoCo
  // selection — multi-worker setups otherwise required the operator
  // to re-check the box for every worker, which was easy to forget.
  // ``worker_id`` is intentionally NOT persisted (would cause server-
  // side duplicate-id collisions on the second submit); the rest are.
  const dilocoStorageKey =
    `forgather-submit-diloco/${project.project_dir}/${config.name}`;
  const dilocoServersQ = useQuery({
    queryKey: ["diloco", "servers"],
    queryFn: api.listDiLoCoServers,
    // Refresh every 10s so a server that just came up shows up here
    // without forcing the operator to reopen the modal.
    refetchInterval: 10_000,
  });
  // Track when persisted state existed but couldn't be restored, so we
  // can surface a warning instead of silently reverting to defaults
  // (which was the failure mode that made operators submit "vanilla
  // finetune" jobs thinking they were still configured for DiLoCo).
  const [dilocoPersistError, setDilocoPersistError] = useState<string | null>(
    null,
  );
  const persistedDiLoCo = useMemo<DiLoCoPersisted>(() => {
    const raw = persistGet(dilocoStorageKey);
    if (!raw) return DEFAULT_DILOCO_PERSISTED;
    try {
      return {
        ...DEFAULT_DILOCO_PERSISTED,
        ...(JSON.parse(raw) as Partial<DiLoCoPersisted>),
      };
    } catch (err) {
      setDilocoPersistError(
        `Saved DiLoCo settings for this config couldn't be parsed (${
          (err as Error).message
        }). Re-pick the server below.`,
      );
      return DEFAULT_DILOCO_PERSISTED;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dilocoStorageKey]);
  const [selectedDiLoCoBase, setSelectedDiLoCoBase] = useState<string>(
    persistedDiLoCo.base,
  );
  // Per-knob form state for the dependent fields. Strings (free form)
  // so empty == "use config / env default"; coerced on submit.
  const [diSyncEvery, setDiSyncEvery] = useState<string>(persistedDiLoCo.syncEvery);
  const [diNumFragments, setDiNumFragments] = useState<string>(
    persistedDiLoCo.numFragments,
  );
  const [diDylu, setDiDylu] = useState<boolean>(persistedDiLoCo.dylu);
  const [diBf16, setDiBf16] = useState<boolean>(persistedDiLoCo.bf16Comm);
  const [diHeartbeat, setDiHeartbeat] = useState<string>(
    persistedDiLoCo.heartbeatInterval,
  );
  // worker_id stays per-submit only — auto-generation is the desired
  // default and a stale value would collide with the prior worker's
  // registration.
  const [diWorkerId, setDiWorkerId] = useState<string>("");
  useEffect(() => {
    const cur: DiLoCoPersisted = {
      base: selectedDiLoCoBase,
      syncEvery: diSyncEvery,
      numFragments: diNumFragments,
      dylu: diDylu,
      bf16Comm: diBf16,
      heartbeatInterval: diHeartbeat,
    };
    persistSet(dilocoStorageKey, JSON.stringify(cur));
  }, [
    dilocoStorageKey,
    selectedDiLoCoBase,
    diSyncEvery,
    diNumFragments,
    diDylu,
    diBf16,
    diHeartbeat,
  ]);
  // If the persisted base isn't currently in the server list (server
  // went offline, was renamed, etc.) fall back to "None" AND surface a
  // warning. The silent fallback was the failure mode that produced
  // "vanilla finetune that the operator thought was DiLoCo" — the
  // warning makes the desync visible so the operator can re-pick.
  useEffect(() => {
    if (!selectedDiLoCoBase) return;
    const servers = dilocoServersQ.data;
    if (!servers) return; // still loading; don't clobber
    if (!servers.some((s) => s.base_url === selectedDiLoCoBase)) {
      setDilocoPersistError(
        `Previously-selected DiLoCo server ${selectedDiLoCoBase} is no longer ` +
          `in the server list. Re-pick below or DiLoCo will be off for this submit.`,
      );
      setSelectedDiLoCoBase("");
    }
  }, [dilocoServersQ.data, selectedDiLoCoBase]);
  // Clear the persistence warning once the operator makes a fresh
  // selection — they've acknowledged it by acting.
  useEffect(() => {
    if (selectedDiLoCoBase) setDilocoPersistError(null);
  }, [selectedDiLoCoBase]);
  // /info for the selected server — used to seed sensible defaults
  // (sync_every from dylu_base_sync_every, dylu requirement, etc.)
  // and to flag obvious mismatches. Disabled when no server picked.
  const dilocoInfoQ = useQuery({
    queryKey: ["diloco", "info", selectedDiLoCoBase],
    queryFn: () => api.diLoCoServerInfo(selectedDiLoCoBase),
    enabled: !!selectedDiLoCoBase,
    staleTime: 60_000,
  });
  // Seed defaults whenever /info loads for a fresh selection. Operator
  // edits aren't overwritten — we only seed empty fields.
  useEffect(() => {
    const info: DiLoCoInfo | undefined = dilocoInfoQ.data;
    if (!selectedDiLoCoBase || !info) return;
    const exp = info.expected_client_settings ?? {};
    if (exp.sync_every != null && diSyncEvery === "") {
      setDiSyncEvery(String(exp.sync_every));
    }
    if (typeof exp.dylu === "boolean") {
      // A DyLU server requires the worker to opt in; otherwise the
      // server's per-worker recommendations are ignored.
      setDiDylu(exp.dylu);
    }
    if (typeof exp.bf16_comm === "boolean") {
      setDiBf16(exp.bf16_comm);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dilocoInfoQ.data, selectedDiLoCoBase]);

  // Dataset-source selector — state + queries + seeding live in the
  // shared hook. ``null`` = local (in-process loader). The hook waits
  // for overrides to load (``ready``) before applying its offline
  // fallback rule, so a missing or unreachable cached choice snaps
  // back to local before the user sees the dropdown.
  const {
    source: datasetSource,
    setSource: setDatasetSource,
    selector: datasetSourceSelector,
  } = useDatasetSource({
    ready: !!overridesQ.data,
    initial: overridesQ.data?.dataset_source ?? null,
  });

  // No clamp to >=1: zero-GPU training dispatches go through (the
  // scheduler routes them past the placement search, and
  // launcher.build_command falls back from nproc_per_node='gpu' to 1
  // when no GPU is reserved -- mirrors the train CLI behaviour).
  // Useful for CPU debugging on hosts with no visible CUDA device.
  const maxGpus = gpusQ.data?.length ?? 0;
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
    if (typeof cached === "number" && cached >= 0) {
      setRequestedGpus(Math.max(0, Math.min(maxGpus, cached)));
      return;
    }
    if (fixedWorkerCount !== null) {
      setRequestedGpus(Math.max(0, Math.min(maxGpus, fixedWorkerCount)));
    }
  }, [fixedWorkerCount, gpusTouched, maxGpus, overridesQ.data?.requested_gpus]);

  // Clamp requestedGpus into [0, maxGpus] whenever the GPU list
  // resolves. Independent of the seed-from-cache logic above so it
  // also catches the "no cached override, no fixed worker count,
  // initial useState(1) on a 0-GPU host" case -- without this the
  // form ships requested_gpus=1 to the scheduler on hosts that have
  // no GPU at all. Idempotent w.r.t. the seed effect; ordering
  // between the two doesn't matter.
  useEffect(() => {
    if (gpusQ.data === undefined) return;
    setRequestedGpus((cur) => Math.max(0, Math.min(maxGpus, cur)));
  }, [gpusQ.data, maxGpus]);

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
  //   3. cluster inactive → don't mark seeded, so a late-discovered
  //      cluster can still trigger seeding when it appears
  //
  // The flag must be set *only when we actually applied a seed*. An
  // earlier version flipped ``mnSeeded`` unconditionally on first
  // render, which left the panel empty if the cluster query
  // resolved before clusterActive became true (modal opens during
  // mDNS discovery, master appears a beat later).
  useEffect(() => {
    if (mnSeeded) return;
    if (!membersQ.data) return;
    if (!overridesQ.data) return;
    const cached = overridesQ.data.multinode;
    if (cached) {
      setMnState(multiNodeStateFromOverrides(cached));
      setMnSeeded(true);
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
      setMnSeeded(true);
    }
    // Standalone-mode case: leave mnSeeded=false. The panel is
    // hidden when !clusterActive, and if cluster mode comes up
    // mid-modal we want this effect to run again with clusterActive
    // = true and seed properly.
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
  // Detect a multi-peer selection that would silently fall through
  // to the single-node enqueue path because cluster mode is no
  // longer active (e.g. the master went away mid-edit, mDNS lost
  // the cluster). Submit must block rather than route a "submit to
  // 3 nodes" click into a one-rank single-host job.
  const peerSelectedButClusterDown =
    !clusterActive &&
    mnState.selected.size > 0 &&
    membersQ.data !== undefined &&
    !(
      mnState.selected.size === 1 &&
      Array.from(mnState.selected)[0] === membersQ.data?.self_node_id
    );

  const submitBlockedReason: string | undefined =
    missingRequired.length > 0
      ? `Required arg(s) missing: ${missingRequired.map((a) => a.cli_name).join(", ")}`
      : outOfBounds.length > 0
        ? `Out-of-range value(s): ${outOfBounds.map((a) => a.cli_name).join(", ")}`
        : clusterActive && mnSeeded && mnState.selected.size === 0
          ? "Pick at least one node in the multi-node panel"
          : peerSelectedButClusterDown
            ? "Cluster is no longer active — close and reopen this dialog"
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
      setRequestedGpus(fixedWorkerCount !== null ? Math.max(0, Math.min(maxGpus, fixedWorkerCount)) : 1);
      setNprocOverride("");
      // Also reset cluster panel state — "Reset to defaults" now
      // means "drop everything we cached for this config", including
      // the multi-node selection. Re-seeded by the seeding effect on
      // the next render.
      setMnState(emptyMultiNodeState());
      setMnSeeded(false);
      // And the dataset-source dropdown — without this the live in-
      // form value survives the reset and the next submit writes it
      // straight back into overrides, defeating the clear.
      setDatasetSource(null);
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
        datasetSource,
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
        dataset_source: datasetSource,
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
    const job_params: Record<string, unknown> = {};
    const trimmedNproc = nprocOverride.trim();
    if (trimmedNproc) job_params.nproc = trimmedNproc;
    // DiLoCo opt-in: hand the chosen server + dependent settings to
    // the scheduler, which translates them into DILOCO_* env vars on
    // the spawned process. Only attached when the operator actually
    // picked a server in the radio group ("" = None).
    const diloco = buildDiLoCoPayload({
      base: selectedDiLoCoBase,
      syncEvery: diSyncEvery,
      numFragments: diNumFragments,
      dylu: diDylu,
      bf16Comm: diBf16,
      heartbeatInterval: diHeartbeat,
      workerId: diWorkerId,
    });
    if (diloco) job_params.diloco = diloco;
    enqueue.mutate({
      project_dir: project.project_dir,
      config: config.name,
      dynamic_args: dyn,
      requested_gpus: gpus,
      priority,
      dataset_source: datasetSource,
      ...(Object.keys(job_params).length > 0 ? { job_params } : {}),
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

          {datasetSourceSelector}


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
                    min={0}
                    max={maxGpus}
                    value={requestedGpus}
                    onChange={(e) => {
                      setGpusTouched(true);
                      const raw = Number(e.target.value);
                      const n = Number.isFinite(raw) ? raw : 0;
                      setRequestedGpus(Math.max(0, Math.min(maxGpus, n)));
                    }}
                  />
                  {idleGpuCount !== null && (
                    <span className="muted">
                      ({idleGpuCount} idle of {maxGpus})
                      {maxGpus === 0 && " — CPU only"}
                    </span>
                  )}
                  {requestedGpus === 0 && maxGpus > 0 && (
                    <span className="muted">
                      0 = run on CPU (nproc_per_node='gpu' falls back to 1)
                    </span>
                  )}
                </label>
                <label>
                  nproc
                  <input
                    type="text"
                    value={nprocOverride}
                    onChange={(e) => setNprocOverride(e.target.value)}
                    placeholder={formatNproc(nproc)}
                    style={{ width: "6em" }}
                    title={
                      "Override torchrun's --nproc-per-node for this " +
                      "submit. Blank = use the config's nproc_per_node " +
                      "(shown as placeholder). Accepts an integer or " +
                      "torchrun's 'gpu' / 'cpu' / 'auto' sentinels. " +
                      "Useful for CPU debugging when the config declares " +
                      "'gpu' but you want, e.g., 4 worker processes."
                    }
                  />
                  <span className="muted">override</span>
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

          {dilocoServersQ.data && dilocoServersQ.data.length > 0 && (
            <DiLoCoPicker
              servers={dilocoServersQ.data}
              selectedBase={selectedDiLoCoBase}
              onSelectBase={setSelectedDiLoCoBase}
              syncEvery={diSyncEvery}
              setSyncEvery={setDiSyncEvery}
              numFragments={diNumFragments}
              setNumFragments={setDiNumFragments}
              dylu={diDylu}
              setDylu={setDiDylu}
              bf16Comm={diBf16}
              setBf16Comm={setDiBf16}
              heartbeatInterval={diHeartbeat}
              setHeartbeatInterval={setDiHeartbeat}
              workerId={diWorkerId}
              setWorkerId={setDiWorkerId}
              persistError={dilocoPersistError}
              infoLoading={dilocoInfoQ.isLoading}
              infoError={dilocoInfoQ.error}
              info={dilocoInfoQ.data ?? null}
              clusterFanout={useClusterFanout}
            />
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
                (clusterActive && mnSeeded && mnState.selected.size === 0) ||
                peerSelectedButClusterDown
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

// ---------------------------------------------------------------------------
// DiLoCo
// ---------------------------------------------------------------------------

interface DiLoCoPersisted {
  base: string;
  syncEvery: string;
  numFragments: string;
  dylu: boolean;
  bf16Comm: boolean;
  heartbeatInterval: string;
}

const DEFAULT_DILOCO_PERSISTED: DiLoCoPersisted = {
  base: "",
  syncEvery: "",
  numFragments: "",
  dylu: false,
  bf16Comm: true,
  heartbeatInterval: "",
};

interface DiLoCoPickerProps {
  servers: DiLoCoServer[];
  selectedBase: string;
  onSelectBase: (base: string) => void;
  syncEvery: string;
  setSyncEvery: (v: string) => void;
  numFragments: string;
  setNumFragments: (v: string) => void;
  dylu: boolean;
  setDylu: (v: boolean) => void;
  bf16Comm: boolean;
  setBf16Comm: (v: boolean) => void;
  heartbeatInterval: string;
  setHeartbeatInterval: (v: string) => void;
  workerId: string;
  setWorkerId: (v: string) => void;
  persistError: string | null;
  infoLoading: boolean;
  infoError: unknown;
  info: DiLoCoInfo | null;
  clusterFanout: boolean;
}

/** Radio picker over the unified DiLoCo server list, with the
 *  dependent worker settings revealed only when a server is selected.
 *  "None" is always present and is the default — operators who don't
 *  want DiLoCo never see clutter beyond a single extra row. */
function DiLoCoPicker(props: DiLoCoPickerProps) {
  const {
    servers,
    selectedBase,
    onSelectBase,
    syncEvery,
    setSyncEvery,
    numFragments,
    setNumFragments,
    dylu,
    setDylu,
    bf16Comm,
    setBf16Comm,
    heartbeatInterval,
    setHeartbeatInterval,
    workerId,
    setWorkerId,
    persistError,
    infoLoading,
    infoError,
    info,
    clusterFanout,
  } = props;

  // Multi-node fanout + DiLoCo together needs per-peer worker IDs and
  // dataset sharding that isn't wired yet. Hide the picker (preserving
  // any in-flight None selection) so the operator can't accidentally
  // mis-submit. Standalone (no cluster) and single-node-within-cluster
  // are both fine.
  if (clusterFanout) {
    return (
      <details className="submit-section">
        <summary>
          <h4 className="dyn-heading">
            DiLoCo{" "}
            <span className="muted">
              — disabled in cluster fanout submits (per-peer worker IDs
              not yet wired)
            </span>
          </h4>
        </summary>
      </details>
    );
  }

  return (
    <details
      className="submit-section"
      open={!!selectedBase || !!persistError}
    >
      <summary>
        <h4 className="dyn-heading">
          DiLoCo{" "}
          {!selectedBase && !persistError && (
            <span className="muted">— none (vanilla training)</span>
          )}
          {selectedBase && (
            <span className="muted">— join {selectedBase}</span>
          )}
          {!selectedBase && persistError && (
            <span style={{ color: "tomato" }}>
              ⚠ previous selection couldn't be restored — re-pick below
            </span>
          )}
        </h4>
      </summary>

      <div style={{ padding: "4px 8px 8px 8px" }}>
        {persistError && (
          <div
            role="alert"
            style={{
              padding: "6px 8px",
              marginBottom: 8,
              border: "1px solid tomato",
              borderRadius: 4,
              color: "tomato",
              fontSize: "smaller",
            }}
          >
            {persistError}
          </div>
        )}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 4,
            marginBottom: 8,
          }}
        >
          <label>
            <input
              type="radio"
              name="diloco-server"
              checked={selectedBase === ""}
              onChange={() => onSelectBase("")}
            />{" "}
            <strong>None</strong>{" "}
            <span className="muted">— run as a regular training job</span>
          </label>
          {servers.map((s) => (
            <label key={s.id}>
              <input
                type="radio"
                name="diloco-server"
                checked={selectedBase === s.base_url}
                onChange={() => onSelectBase(s.base_url)}
              />{" "}
              <strong>{s.label}</strong>{" "}
              <span className="muted">
                — {s.base_url}
                {s.source === "registered" && " (external)"}
                {s.source === "local" && !s.alive && " (not running)"}
              </span>
            </label>
          ))}
        </div>

        {selectedBase && (
          <>
            {infoLoading && (
              <div className="muted">Loading server info…</div>
            )}
            {!!infoError && (
              <div className="muted" style={{ color: "tomato" }}>
                Could not fetch /info: {(infoError as Error).message}
              </div>
            )}
            {info && (
              <div className="muted" style={{ marginBottom: 6 }}>
                Server mode: <strong>{info.mode ?? "—"}</strong>
                {info.num_parameters !== undefined && (
                  <> · {info.num_parameters.toLocaleString()} params</>
                )}
                {info.dylu_enabled && (
                  <>
                    {" "}
                    · DyLU base sync_every={" "}
                    <strong>{info.dylu_base_sync_every}</strong>
                  </>
                )}
              </div>
            )}

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 8,
              }}
            >
              <label>
                sync_every
                <input
                  type="number"
                  min={1}
                  value={syncEvery}
                  onChange={(e) => setSyncEvery(e.target.value)}
                  placeholder={
                    info?.expected_client_settings?.sync_every != null
                      ? String(info.expected_client_settings.sync_every)
                      : "callback default (500)"
                  }
                  style={{ width: "100%" }}
                />
              </label>
              <label>
                num_fragments
                <input
                  type="number"
                  min={1}
                  value={numFragments}
                  onChange={(e) => setNumFragments(e.target.value)}
                  placeholder="1 (no streaming)"
                  style={{ width: "100%" }}
                />
              </label>
              <label>
                heartbeat_interval (s)
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={heartbeatInterval}
                  onChange={(e) => setHeartbeatInterval(e.target.value)}
                  placeholder="30"
                  style={{ width: "100%" }}
                />
              </label>
              <label>
                worker_id (optional)
                <input
                  type="text"
                  value={workerId}
                  onChange={(e) => setWorkerId(e.target.value)}
                  placeholder="auto"
                  style={{ width: "100%" }}
                />
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={dylu}
                  onChange={(e) => setDylu(e.target.checked)}
                />{" "}
                Enable DyLU{" "}
                {info?.dylu_enabled && (
                  <span className="muted">
                    (server requires this; the dependent fields above are
                    overridden by the server's heartbeat response)
                  </span>
                )}
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={bf16Comm}
                  onChange={(e) => setBf16Comm(e.target.checked)}
                />{" "}
                bf16 pseudo-gradient communication
              </label>
            </div>
          </>
        )}
      </div>
    </details>
  );
}

interface DiLoCoFormSnapshot {
  base: string;
  syncEvery: string;
  numFragments: string;
  dylu: boolean;
  bf16Comm: boolean;
  heartbeatInterval: string;
  workerId: string;
}

/** Construct the ``job_params.diloco`` payload from the form snapshot.
 *  Returns null when the operator picked "None" — callers should skip
 *  ``job_params.diloco`` entirely in that case. */
function buildDiLoCoPayload(
  s: DiLoCoFormSnapshot,
): Record<string, unknown> | null {
  if (!s.base) return null;
  // The base URL is what the proxy + UI use; the DiLoCoCallback
  // expects ``host:port``. Strip scheme + trailing slash here so the
  // callback can use the value verbatim.
  const serverAddr = s.base.replace(/^https?:\/\//, "").replace(/\/$/, "");
  const payload: Record<string, unknown> = {
    server_addr: serverAddr,
    dylu: s.dylu,
    bf16_comm: s.bf16Comm,
  };
  const sync = s.syncEvery.trim();
  if (sync) {
    const n = Number(sync);
    if (Number.isFinite(n)) payload.sync_every = Math.max(1, Math.floor(n));
  }
  const frags = s.numFragments.trim();
  if (frags) {
    const n = Number(frags);
    if (Number.isFinite(n)) payload.num_fragments = Math.max(1, Math.floor(n));
  }
  const hb = s.heartbeatInterval.trim();
  if (hb) {
    const n = Number(hb);
    if (Number.isFinite(n) && n >= 0) payload.heartbeat_interval = n;
  }
  const wid = s.workerId.trim();
  if (wid) payload.worker_id = wid;
  return payload;
}

