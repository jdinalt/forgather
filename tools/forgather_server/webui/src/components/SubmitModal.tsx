import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CSSProperties,
  Dispatch,
  SetStateAction,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  api,
  ClusterJobSubmitRequest,
  ClusterMembersResponse,
  ConfigInfo,
  DiLoCoInfo,
  DiLoCoServer,
  EnqueueRequest,
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
  //
  // The parser returns both the value AND the optional error so we
  // don't have to call setState inside useMemo (which would trigger
  // React's "Cannot update a component while rendering" warning).
  // ``error`` is null on the happy path; the matching useEffect
  // below lifts it into component state.
  const persistedDiLoCoParse = useMemo<{
    value: DiLoCoPersisted;
    error: string | null;
  }>(() => {
    const raw = persistGet(dilocoStorageKey);
    if (!raw) return { value: DEFAULT_DILOCO_PERSISTED, error: null };
    try {
      return {
        value: {
          ...DEFAULT_DILOCO_PERSISTED,
          ...(JSON.parse(raw) as Partial<DiLoCoPersisted>),
        },
        error: null,
      };
    } catch (err) {
      return {
        value: DEFAULT_DILOCO_PERSISTED,
        error: `Saved DiLoCo settings for this config couldn't be parsed (${
          (err as Error).message
        }). Re-pick the server below.`,
      };
    }
  }, [dilocoStorageKey]);
  const persistedDiLoCo = persistedDiLoCoParse.value;
  const [dilocoPersistError, setDilocoPersistError] = useState<string | null>(
    null,
  );
  useEffect(() => {
    if (persistedDiLoCoParse.error) {
      setDilocoPersistError(persistedDiLoCoParse.error);
    }
  }, [persistedDiLoCoParse.error]);
  const [selectedDiLoCoBase, setSelectedDiLoCoBase] = useState<string>(
    persistedDiLoCo.base,
  );
  // sync_every / num_fragments / dylu / bf16_comm are server-authoritative
  // (the worker reads them from /info); they have no form state here. Only
  // client-local knobs remain. Strings (free form) so empty == "use config
  // / env default"; coerced on submit.
  const [diHeartbeat, setDiHeartbeat] = useState<string>(
    persistedDiLoCo.heartbeatInterval,
  );
  // Sync backend for the worker pool (issue #154). Not persisted: it's a
  // per-submit choice and http is the safe default. Single-host only, so it's
  // ignored on the cluster-fanout path (which the backend rejects anyway).
  const [diBackend, setDiBackend] = useState<
    "http" | "shared_memory" | "collective"
  >("http");
  // Collective backend: the replica count (one torchrun job of N replicas =
  // nproc = GPU reservation). Only used when diBackend === "collective".
  const [diReplicate, setDiReplicate] = useState<number>(2);
  // Multi-node + DiLoCo composition: the bundle becomes one DiLoCo
  // worker group, all ranks sharing this base worker_id (the PP
  // callback appends ``_pp<rank>`` itself). Empty == let the master
  // auto-mint a memorable name. Not persisted because reusing a base
  // across runs would collide with the prior group's checkpoint.
  // Only consulted when ``useClusterFanout`` is true; the picker
  // hides the input in non-cluster mode (the worker pool covers
  // that path).
  const [composeWorkerIdBase, setComposeWorkerIdBase] = useState<string>("");
  // Worker pool (batch submit). A DiLoCo run is often N identical workers
  // differing only by worker_id, so instead of a single worker_id field we
  // maintain a pool: stopped workers the server knows about (toggled on to
  // resume from their checkpoint) plus freshly-added new names. On Submit one
  // job is spawned per pool member. Empty pool == one auto-named worker (the
  // pre-pool behavior). Per-submit only — not persisted, since reusing a
  // worker_id collides with the prior worker's registration.
  //
  // Stopped workers default to ON: they're usually stopped because the server
  // was restarted, and the normal intent is to bring every one back. So we
  // track the *exceptions* — the base worker-ids the operator toggled OFF —
  // rather than the enabled set; this also sidesteps async-roster seeding (a
  // worker is enabled the instant it appears unless explicitly disabled).
  // ``newWorkers`` is the ordered list of added/generated names.
  const [disabledStopped, setDisabledStopped] = useState<Set<string>>(
    () => new Set(),
  );
  const [newWorkers, setNewWorkers] = useState<string[]>([]);
  // Switching servers (or to "None") invalidates the pool — the ids belong to
  // a specific server's roster — so reset it.
  useEffect(() => {
    setDisabledStopped(new Set());
    setNewWorkers([]);
  }, [selectedDiLoCoBase]);
  // Roster of workers the selected server has ever seen (issue #103). Lifted
  // here (not in the picker) because Submit needs to know which stopped
  // workers are toggled on to spawn them.
  const knownWorkersQ = useQuery({
    queryKey: ["diloco", "known-workers", selectedDiLoCoBase],
    queryFn: () => api.diLoCoKnownWorkers(selectedDiLoCoBase),
    enabled: !!selectedDiLoCoBase,
    staleTime: 10_000,
  });
  const resumableWorkers = useMemo(() => {
    // Dedupe to the base worker-id the operator actually passes as
    // --diloco-worker-id; pipeline ranks register as ``<base>_pp<N>`` and
    // share one local output_dir, so the base is the resumable identity.
    const seen = new Map<string, string | null | undefined>();
    for (const w of knownWorkersQ.data?.workers ?? []) {
      if (w.running) continue;
      const base = w.worker_id.replace(/_pp\d+$/, "");
      if (!seen.has(base)) seen.set(base, w.output_dir);
    }
    return [...seen.entries()].map(([name, output_dir]) => ({
      name,
      output_dir,
    }));
  }, [knownWorkersQ.data]);
  // The concrete set of worker_ids Submit will spawn: enabled stopped
  // workers + every new worker, deduped, original order preserved.
  const poolWorkerIds = useMemo(() => {
    const ids: string[] = [];
    for (const w of resumableWorkers) {
      if (!disabledStopped.has(w.name)) ids.push(w.name);
    }
    for (const n of newWorkers) ids.push(n);
    return [...new Set(ids)];
  }, [resumableWorkers, disabledStopped, newWorkers]);
  useEffect(() => {
    const cur: DiLoCoPersisted = {
      base: selectedDiLoCoBase,
      heartbeatInterval: diHeartbeat,
    };
    persistSet(dilocoStorageKey, JSON.stringify(cur));
  }, [dilocoStorageKey, selectedDiLoCoBase, diHeartbeat]);
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
  // Under DiLoCo the worker no longer takes a model path: it fetches the
  // model definition (config + custom code + tokenizer) from the server's
  // /model_def endpoint, builds the model empty on meta, and pulls weights
  // via the parameter sync (issue #53). So there is nothing to seed into
  // --model-id-or-path here — the field is ignored on DiLoCo submissions
  // (its help text says so) and the server-side fingerprint check stays as
  // defense-in-depth. sync_every / dylu / bf16_comm / num_fragments are
  // server-authoritative too; the picker shows the server's values
  // read-only rather than seeding the form.

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

  // Batch enqueue: spawn one job per request, sequentially (deterministic
  // ordering; each queue_id is minted server-side). Handles the common case
  // of a single request too, so the single-node path always routes here.
  const enqueue = useMutation({
    mutationFn: async (reqs: EnqueueRequest[]) => {
      const items = [];
      for (const req of reqs) {
        items.push(await api.enqueue(req));
      }
      return items;
    },
    onSuccess: (items) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      if (items[0]) onSubmitted?.(items[0].queue_id);
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
      // DiLoCo composition: if the picker selected a server, fold the
      // composition block into the submit so every per-rank job joins
      // one DiLoCo worker group. The master resolves the base
      // worker_id (auto-mints when blank) and the bearer token.
      if (selectedDiLoCoBase) {
        const trimmedBase = composeWorkerIdBase.trim();
        const hb = diHeartbeat.trim();
        const hbNum = hb === "" ? null : Number(hb);
        req.diloco = {
          server_addr: selectedDiLoCoBase,
          worker_id: trimmedBase || null,
          heartbeat_interval:
            hbNum !== null && Number.isFinite(hbNum) ? hbNum : null,
        };
      }
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
    const trimmedNproc = nprocOverride.trim();

    // Collective backend: ONE torchrun job of N replicas (not the N-job worker
    // pool). nproc + requested_gpus = the replicate degree; the scheduler
    // derives DILOCO_BACKEND=collective + DILOCO_REPLICATE. Mirrors the CLI's
    // `forgather submit --backend collective --diloco-replicate N`.
    if (diBackend === "collective" && selectedDiLoCoBase) {
      const diloco = buildDiLoCoPayload({
        base: selectedDiLoCoBase,
        heartbeatInterval: diHeartbeat,
        workerId: "",
        backend: "collective",
        replicate: diReplicate,
      });
      const req: EnqueueRequest = {
        project_dir: project.project_dir,
        config: config.name,
        dynamic_args: dyn,
        requested_gpus: diReplicate,
        priority,
        dataset_source: datasetSource,
        job_params: { nproc: String(diReplicate), ...(diloco ? { diloco } : {}) },
      };
      enqueue.mutate([req]);
      return;
    }

    // DiLoCo batch: one job per pool member (enabled stopped + new). With
    // no pool members (or no DiLoCo server) we spawn a single auto-named
    // job — the pre-pool behavior.
    const groupWorkerIds: Array<string | null> =
      selectedDiLoCoBase && poolWorkerIds.length > 0 ? poolWorkerIds : [null];
    // Shared-memory backend: mint one group id for this submit so every
    // co-located worker shares it (and the size = the batch count); the
    // scheduler derives a single region dir + group size. Mirrors the CLI.
    const shmGroup =
      diBackend === "shared_memory"
        ? {
            shmGroupId: randomGroupId(),
            shmGroupSize: groupWorkerIds.length,
          }
        : null;
    // Build one enqueue request for a given worker_id (null == let the
    // scheduler auto-assign, i.e. fall back to the queue_id).
    const buildRequest = (workerId: string | null): EnqueueRequest => {
      const job_params: Record<string, unknown> = {};
      if (trimmedNproc) job_params.nproc = trimmedNproc;
      // DiLoCo opt-in: hand the chosen server + dependent settings to
      // the scheduler, which translates them into DILOCO_* env vars on
      // the spawned process. Only attached when the operator actually
      // picked a server in the radio group ("" = None).
      const diloco = buildDiLoCoPayload({
        base: selectedDiLoCoBase,
        heartbeatInterval: diHeartbeat,
        workerId: workerId ?? "",
        backend: diBackend,
        shmGroupId: shmGroup?.shmGroupId,
        shmGroupSize: shmGroup?.shmGroupSize,
      });
      if (diloco) job_params.diloco = diloco;
      return {
        project_dir: project.project_dir,
        config: config.name,
        dynamic_args: dyn,
        requested_gpus: gpus,
        priority,
        dataset_source: datasetSource,
        ...(Object.keys(job_params).length > 0 ? { job_params } : {}),
      };
    };

    const reqs: EnqueueRequest[] = groupWorkerIds.map((wid) =>
      buildRequest(wid),
    );
    enqueue.mutate(reqs);
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

          {/* Render the picker whenever the server list has resolved
              (even if it's empty) so the persistence-warning banner
              has somewhere to surface. Hiding the picker on an empty
              list defeated the whole point of the warning. */}
          {dilocoServersQ.data && (
            <DiLoCoPicker
              servers={dilocoServersQ.data}
              selectedBase={selectedDiLoCoBase}
              onSelectBase={setSelectedDiLoCoBase}
              heartbeatInterval={diHeartbeat}
              setHeartbeatInterval={setDiHeartbeat}
              backend={diBackend}
              setBackend={setDiBackend}
              replicate={diReplicate}
              setReplicate={setDiReplicate}
              resumableWorkers={resumableWorkers}
              knownWorkersLoading={knownWorkersQ.isLoading}
              disabledStopped={disabledStopped}
              setDisabledStopped={setDisabledStopped}
              newWorkers={newWorkers}
              setNewWorkers={setNewWorkers}
              poolWorkerIds={poolWorkerIds}
              persistError={dilocoPersistError}
              infoLoading={dilocoInfoQ.isLoading}
              infoError={dilocoInfoQ.error}
              info={dilocoInfoQ.data ?? null}
              clusterFanout={useClusterFanout}
              composeWorkerIdBase={composeWorkerIdBase}
              setComposeWorkerIdBase={setComposeWorkerIdBase}
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
                  : selectedDiLoCoBase && diBackend === "collective"
                    ? `Submit collective (${diReplicate} replica${
                        diReplicate === 1 ? "" : "s"
                      })`
                    : selectedDiLoCoBase && poolWorkerIds.length > 0
                      ? `Submit ${poolWorkerIds.length} worker${
                          poolWorkerIds.length === 1 ? "" : "s"
                        }`
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
  heartbeatInterval: string;
}

const DEFAULT_DILOCO_PERSISTED: DiLoCoPersisted = {
  base: "",
  heartbeatInterval: "",
};

interface ResumableWorker {
  name: string;
  output_dir?: string | null;
}

interface DiLoCoPickerProps {
  servers: DiLoCoServer[];
  selectedBase: string;
  onSelectBase: (base: string) => void;
  heartbeatInterval: string;
  setHeartbeatInterval: (v: string) => void;
  // Sync backend for the worker pool (non-cluster only; single-host).
  backend: "http" | "shared_memory" | "collective";
  setBackend: (v: "http" | "shared_memory" | "collective") => void;
  replicate: number;
  setReplicate: (v: number) => void;
  // Worker pool (batch submit; non-cluster only).
  resumableWorkers: ResumableWorker[];
  knownWorkersLoading: boolean;
  disabledStopped: Set<string>;
  setDisabledStopped: Dispatch<SetStateAction<Set<string>>>;
  newWorkers: string[];
  setNewWorkers: Dispatch<SetStateAction<string[]>>;
  poolWorkerIds: string[];
  persistError: string | null;
  infoLoading: boolean;
  infoError: unknown;
  info: DiLoCoInfo | null;
  clusterFanout: boolean;
  // Multi-node DiLoCo composition: when ``clusterFanout`` is true and
  // a server is selected, the bundle becomes one DiLoCo group. The
  // operator can pin a base worker_id here or leave it blank to let
  // the master auto-mint one.
  composeWorkerIdBase: string;
  setComposeWorkerIdBase: (v: string) => void;
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
    heartbeatInterval,
    setHeartbeatInterval,
    backend,
    setBackend,
    replicate,
    setReplicate,
    resumableWorkers,
    knownWorkersLoading,
    disabledStopped,
    setDisabledStopped,
    newWorkers,
    setNewWorkers,
    poolWorkerIds,
    persistError,
    infoLoading,
    infoError,
    info,
    clusterFanout,
    composeWorkerIdBase,
    setComposeWorkerIdBase,
  } = props;

  // Local UI state for the pool's add/generate controls. Hooks must run
  // unconditionally, so they precede the cluster-fanout early return.
  const [addName, setAddName] = useState<string>("");
  const [genCount, setGenCount] = useState<string>("4");
  const [generating, setGenerating] = useState<boolean>(false);
  const [poolError, setPoolError] = useState<string | null>(null);

  // Every name already claimed in the pool — stopped roster ids + added new
  // ones. Used to reject duplicate adds and to keep generated batches disjoint.
  const claimedNames = useMemo(() => {
    const s = new Set<string>();
    for (const w of resumableWorkers) s.add(w.name);
    for (const n of newWorkers) s.add(n);
    return s;
  }, [resumableWorkers, newWorkers]);

  // Toggling tracks the disabled exceptions: a name in the set is OFF.
  const toggleStopped = (name: string) => {
    setPoolError(null);
    setDisabledStopped((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const addWorker = (raw: string) => {
    const name = raw.trim();
    if (!name) return;
    if (resumableWorkers.some((w) => w.name === name)) {
      setPoolError(
        `"${name}" is a stopped worker — toggle its chip above to resume it.`,
      );
      return;
    }
    if (newWorkers.includes(name)) {
      setPoolError(`"${name}" is already in the pool.`);
      return;
    }
    setNewWorkers((prev) => [...prev, name]);
    setAddName("");
    setPoolError(null);
  };

  const removeNewWorker = (name: string) => {
    setNewWorkers((prev) => prev.filter((n) => n !== name));
    setPoolError(null);
  };

  const generateBatch = async () => {
    const count = parseInt(genCount, 10);
    if (!Number.isFinite(count) || count < 1) {
      setPoolError("Enter a worker count of 1 or more.");
      return;
    }
    setGenerating(true);
    setPoolError(null);
    try {
      const names = await api.generateDiLoCoWorkerNames(count, [
        ...claimedNames,
      ]);
      setNewWorkers((prev) => [...prev, ...names]);
    } catch (e) {
      setPoolError((e as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  // Multi-node composition: the bundle joins the chosen server as ONE
  // logical DiLoCo worker group, all ranks sharing one base worker_id
  // (the PP callback appends ``_pp<rank>``). The worker pool doesn't
  // apply here — composition is one-bundle-one-group; K independent
  // groups is a follow-up (CLI ``--diloco-worker-count`` + non-overlap
  // member partitioning). Show the radio + heartbeat + optional base
  // worker_id, hide the pool.
  if (clusterFanout) {
    return (
      <details className="submit-section" open>
        <summary>
          <h4 className="dyn-heading">
            DiLoCo{" "}
            {!selectedBase && (
              <span className="muted">— none (vanilla PP submit)</span>
            )}
            {selectedBase && (
              <span className="muted">— compose with {selectedBase}</span>
            )}
          </h4>
        </summary>

        <div style={{ padding: "4px 8px 8px 8px" }}>
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
              <span className="muted">
                — submit as a plain multi-node training bundle
              </span>
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
              <div
                className="muted"
                style={{ fontSize: "smaller", marginBottom: 8 }}
              >
                The bundle becomes <strong>one DiLoCo worker group</strong>:
                every per-rank training job shares the base worker id and
                the PP callback appends <code>_pp&lt;rank&gt;</code>. The
                master forwards the server bearer to every peer.
              </div>
              <label style={{ display: "block", maxWidth: "20em" }}>
                base worker_id{" "}
                <span className="muted">(blank = auto-mint)</span>
                <input
                  type="text"
                  value={composeWorkerIdBase}
                  onChange={(e) => setComposeWorkerIdBase(e.target.value)}
                  placeholder="leave blank for memorable default"
                  style={{ width: "100%" }}
                />
              </label>
              <label
                style={{
                  display: "block",
                  maxWidth: "12em",
                  marginTop: 8,
                }}
              >
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
            </>
          )}
        </div>
      </details>
    );
  }

  return (
    // Expanded by default (like the model/dataset sections) so the
    // current DiLoCo state — including a reset-to-None when a prior
    // selection couldn't be restored — is always visible at a glance.
    // Collapsing it on "None" previously hid the reset (issue #95).
    // Still user-collapsible: this only sets the initial open state.
    <details className="submit-section" open>
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
              </div>
            )}

            {/* Server-authoritative settings (issue #95 follow-up). These
                must match across the group, so the server owns them and
                the worker reads them from /info — there is no operator
                knob. Shown read-only so the operator can see what the
                worker will use. */}
            {info && (
              <fieldset
                disabled
                style={{
                  border: "1px solid var(--border, #444)",
                  borderRadius: 4,
                  padding: "6px 8px",
                  marginBottom: 8,
                }}
              >
                <legend className="muted" style={{ fontSize: "smaller" }}>
                  Managed by server (read-only)
                </legend>
                <div
                  className="muted"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 4,
                    fontSize: "smaller",
                  }}
                >
                  <span>
                    sync_every:{" "}
                    <strong>
                      {info.expected_client_settings?.sync_every ?? "—"}
                    </strong>
                  </span>
                  <span>
                    num_fragments:{" "}
                    <strong>
                      {info.expected_client_settings?.num_fragments_default ??
                        1}
                    </strong>
                  </span>
                  <span>
                    dylu:{" "}
                    <strong>
                      {info.expected_client_settings?.dylu ? "on" : "off"}
                    </strong>
                  </span>
                  <span>
                    {/* Upload leg (worker → server pseudo-grads). Prefer
                        the four-knob format when the server advertises
                        it; fall back to the legacy ``bf16_comm`` so old
                        servers still render. */}
                    upload:{" "}
                    <strong>
                      {(() => {
                        const ecs = info.expected_client_settings;
                        const dt =
                          ecs?.upload_dtype ??
                          (ecs?.bf16_comm === false ? "fp32" : "bf16");
                        const sr = ecs?.upload_sr ? " + SR" : "";
                        return `${dt}${sr}`;
                      })()}
                    </strong>
                  </span>
                  <span>
                    download:{" "}
                    <strong>
                      {(() => {
                        const ecs = info.expected_client_settings;
                        const dt = ecs?.download_dtype ?? "fp32";
                        const sr = ecs?.download_sr ? " + SR" : "";
                        return `${dt}${sr}`;
                      })()}
                    </strong>
                  </span>
                </div>
              </fieldset>
            )}

            <label style={{ display: "block", maxWidth: "12em" }}>
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

            <label style={{ display: "block", maxWidth: "16em" }}>
              sync backend
              <select
                value={backend}
                onChange={(e) =>
                  setBackend(
                    e.target.value as "http" | "shared_memory" | "collective",
                  )
                }
                style={{ width: "100%" }}
              >
                <option value="http">http (param server)</option>
                <option value="shared_memory">shared_memory (single-host)</option>
                <option value="collective">collective (single-host)</option>
              </select>
              {backend === "shared_memory" && (
                <span className="muted" style={{ fontSize: 11 }}>
                  Co-located workers share a CPU master region (no on-wire sync).
                  Single-host: every worker in this pool runs on one machine.
                </span>
              )}
              {backend === "collective" && (
                <span className="muted" style={{ fontSize: 11 }}>
                  N replicas run as one torchrun job that all-reduce
                  pseudo-gradients (single-host). One job, sized below — the
                  worker pool doesn't apply.
                </span>
              )}
            </label>

            {backend === "collective" ? (
              <label style={{ display: "block", maxWidth: "16em" }}>
                replicate degree (replicas)
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={replicate}
                  onChange={(e) =>
                    setReplicate(Math.max(1, Number(e.target.value) || 1))
                  }
                  style={{ width: "100%" }}
                />
                <span className="muted" style={{ fontSize: 11 }}>
                  = nproc_per_node and the GPU reservation for the one job.
                </span>
              </label>
            ) : (
              <WorkerPool
              resumableWorkers={resumableWorkers}
              knownWorkersLoading={knownWorkersLoading}
              disabledStopped={disabledStopped}
              newWorkers={newWorkers}
              poolWorkerIds={poolWorkerIds}
              addName={addName}
              setAddName={setAddName}
              genCount={genCount}
              setGenCount={setGenCount}
              generating={generating}
              poolError={poolError}
                onToggleStopped={toggleStopped}
                onAddWorker={addWorker}
                onRemoveNewWorker={removeNewWorker}
                onGenerateBatch={generateBatch}
              />
            )}
          </>
        )}
      </div>
    </details>
  );
}

interface WorkerPoolProps {
  resumableWorkers: ResumableWorker[];
  knownWorkersLoading: boolean;
  disabledStopped: Set<string>;
  newWorkers: string[];
  poolWorkerIds: string[];
  addName: string;
  setAddName: (v: string) => void;
  genCount: string;
  setGenCount: (v: string) => void;
  generating: boolean;
  poolError: string | null;
  onToggleStopped: (name: string) => void;
  onAddWorker: (name: string) => void;
  onRemoveNewWorker: (name: string) => void;
  onGenerateBatch: () => void;
}

const CHIP_BASE: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  fontSize: "smaller",
  padding: "2px 8px",
  borderRadius: 10,
  background: "transparent",
};

/** The DiLoCo worker pool: a row of chips for stopped workers (toggle to
 *  resume from checkpoint) and new workers (added manually or generated in
 *  a batch; removable), plus the add/generate controls. On Submit one job is
 *  spawned per enabled chip; an empty pool spawns a single auto-named worker. */
function WorkerPool(props: WorkerPoolProps) {
  const {
    resumableWorkers,
    knownWorkersLoading,
    disabledStopped,
    newWorkers,
    poolWorkerIds,
    addName,
    setAddName,
    genCount,
    setGenCount,
    generating,
    poolError,
    onToggleStopped,
    onAddWorker,
    onRemoveNewWorker,
    onGenerateBatch,
  } = props;

  const count = poolWorkerIds.length;

  return (
    <div style={{ marginTop: 10 }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <strong style={{ fontSize: "smaller" }}>Worker pool</strong>
        <span className="muted" style={{ fontSize: "smaller" }}>
          {count > 0
            ? `${count} worker${count === 1 ? "" : "s"} will be spawned`
            : "empty — one auto-named worker will be spawned"}
        </span>
      </div>

      {/* Stopped workers — toggle on to relaunch under the old id and resume
          from that worker's checkpoint. */}
      <div style={{ marginTop: 6 }}>
        <div className="muted" style={{ fontSize: "smaller" }}>
          Stopped workers{" "}
          <span style={{ opacity: 0.7 }}>
            (resumed from checkpoint by default — toggle off to skip)
          </span>
        </div>
        {knownWorkersLoading && (
          <div className="muted" style={{ fontSize: "smaller", marginTop: 4 }}>
            Loading roster…
          </div>
        )}
        {!knownWorkersLoading && resumableWorkers.length === 0 && (
          <div className="muted" style={{ fontSize: "smaller", marginTop: 4 }}>
            None — the server has no stopped workers to resume.
          </div>
        )}
        {resumableWorkers.length > 0 && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
              marginTop: 4,
            }}
          >
            {resumableWorkers.map((w) => {
              const on = !disabledStopped.has(w.name);
              const dir = (w.output_dir ?? "").replace(/\/+$/, "");
              const leaf = dir.split("/").pop() || "";
              return (
                <button
                  type="button"
                  key={w.name}
                  aria-pressed={on}
                  onClick={() => onToggleStopped(w.name)}
                  title={w.output_dir ?? undefined}
                  style={{
                    ...CHIP_BASE,
                    cursor: "pointer",
                    opacity: on ? 1 : 0.55,
                    border: on
                      ? "1px solid var(--accent, #7aa2f7)"
                      : "1px dashed var(--border, #3b4261)",
                  }}
                >
                  <span aria-hidden style={{ fontSize: "0.85em" }}>
                    {on ? "☑" : "☐"}
                  </span>
                  {w.name}
                  {leaf && leaf !== w.name && (
                    <span className="muted"> · {leaf}</span>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* New workers — added manually or generated; removable. */}
      {newWorkers.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="muted" style={{ fontSize: "smaller" }}>
            New workers
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
              marginTop: 4,
            }}
          >
            {newWorkers.map((name) => (
              <span
                key={name}
                style={{
                  ...CHIP_BASE,
                  border: "1px solid var(--accent, #7aa2f7)",
                }}
              >
                {name}
                <button
                  type="button"
                  aria-label={`Remove ${name}`}
                  title={`Remove ${name}`}
                  onClick={() => onRemoveNewWorker(name)}
                  style={{
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    padding: 0,
                    lineHeight: 1,
                    fontSize: "1.1em",
                    color: "inherit",
                  }}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Add / generate controls. */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "flex-end",
          gap: 8,
          marginTop: 10,
        }}
      >
        <label style={{ flex: "1 1 12em" }}>
          <span className="muted" style={{ fontSize: "smaller" }}>
            Add a worker
          </span>
          <div style={{ display: "flex", gap: 4, marginTop: 2 }}>
            <input
              type="text"
              value={addName}
              onChange={(e) => setAddName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  onAddWorker(addName);
                }
              }}
              placeholder="worker name"
              style={{ flex: 1, minWidth: 0 }}
            />
            <button
              type="button"
              className="secondary"
              onClick={() => onAddWorker(addName)}
              disabled={!addName.trim()}
            >
              Add
            </button>
          </div>
        </label>

        <label>
          <span className="muted" style={{ fontSize: "smaller" }}>
            Generate
          </span>
          <div style={{ display: "flex", gap: 4, marginTop: 2 }}>
            <input
              type="number"
              min={1}
              step={1}
              value={genCount}
              onChange={(e) => setGenCount(e.target.value)}
              style={{ width: "4.5em" }}
              title="How many random worker names to generate at once"
            />
            <button
              type="button"
              className="secondary"
              onClick={onGenerateBatch}
              disabled={generating}
            >
              {generating ? "Generating…" : "Generate"}
            </button>
          </div>
        </label>
      </div>

      {poolError && (
        <div
          role="alert"
          className="muted"
          style={{ color: "tomato", fontSize: "smaller", marginTop: 6 }}
        >
          {poolError}
        </div>
      )}
    </div>
  );
}

interface DiLoCoFormSnapshot {
  base: string;
  heartbeatInterval: string;
  workerId: string;
  // Sync backend (issue #154). "shared_memory" has the co-located workers
  // share a CPU master region instead of syncing through the param server.
  // ``shmGroupId`` is uniform across the workers of one submit; ``shmGroupSize``
  // is the worker count. "collective" runs N replicas as one torchrun job that
  // all-reduce (``replicate`` = the replica count). Absent / "http" → the
  // default param-server path.
  backend?: "http" | "shared_memory" | "collective";
  shmGroupId?: string;
  shmGroupSize?: number;
  replicate?: number;
}

/** 16 hex chars, mirroring the CLI's ``uuid.uuid4().hex[:16]`` shared-memory
 *  group id. Uses ``getRandomValues`` rather than ``crypto.randomUUID`` so it
 *  also works over plain HTTP from a non-localhost origin (a LAN peer) —
 *  ``randomUUID`` is secure-context-only and would throw there. */
function randomGroupId(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Construct the ``job_params.diloco`` payload from the form snapshot.
 *  Returns null when the operator picked "None" — callers should skip
 *  ``job_params.diloco`` entirely in that case.
 *
 *  sync_every / num_fragments / dylu / bf16_comm are intentionally absent:
 *  they are server-authoritative and the worker reads them from /info, so
 *  the submission never carries them. */
function buildDiLoCoPayload(
  s: DiLoCoFormSnapshot,
): Record<string, unknown> | null {
  if (!s.base) return null;
  // Pass the full URL with scheme through — the worker's
  // DiLoCoClient routes ``http://`` cleartext and ``https://``
  // through urllib_ssl_context, but cannot tell which the upstream
  // wants from a bare ``host:port``. Stripping the scheme (as we
  // did pre-#90) caused the worker to dial HTTP at a TLS-wrapped
  // server, which slams the connection with RST. Only the trailing
  // slash gets trimmed.
  const serverAddr = s.base.replace(/\/$/, "");
  const payload: Record<string, unknown> = {
    server_addr: serverAddr,
  };
  const hb = s.heartbeatInterval.trim();
  if (hb) {
    const n = Number(hb);
    if (Number.isFinite(n) && n >= 0) payload.heartbeat_interval = n;
  }
  const wid = s.workerId.trim();
  if (wid) payload.worker_id = wid;
  // Shared-memory backend: declare the structured intent (backend + uniform
  // group id + size); the scheduler derives DILOCO_SHM_* env, mirroring the
  // CLI's ``--backend shared_memory``. Omitted for the default http backend.
  if (s.backend === "shared_memory") {
    payload.backend = "shared_memory";
    if (s.shmGroupId) payload.shm_group_id = s.shmGroupId;
    if (s.shmGroupSize) payload.shm_group_size = s.shmGroupSize;
  } else if (s.backend === "collective") {
    // Collective backend: the scheduler derives DILOCO_BACKEND=collective +
    // DILOCO_REPLICATE; the torchrun world size rides job_params.nproc (set by
    // the caller). Mirrors the CLI's ``--backend collective --diloco-replicate``.
    payload.backend = "collective";
    if (s.replicate) payload.diloco_replicate = s.replicate;
  }
  return payload;
}

