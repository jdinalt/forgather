import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ControlAction, Job } from "../api";
import { persistGet, persistSet } from "../persist";
import { ContextMenu } from "./ContextMenu";
import { TtyViewer } from "./TtyViewer";

interface JobMenuTarget {
  job: Job;
  x: number;
  y: number;
}

/** Per-job-type styling for the row chip. ``label`` is the short
 *  user-facing tag; ``className`` picks one of the ``.type-*`` styles
 *  in ``styles.css``. ``training`` is also the fallback for unknown
 *  values arriving from older server versions. */
const JOB_TYPE_CHIPS: Record<Job["job_type"], { label: string; className: string }> = {
  training: { label: "train", className: "type-train" },
  eval: { label: "eval", className: "type-eval" },
  inference: { label: "serve", className: "type-inference" },
  tensorboard: { label: "tb", className: "type-tensorboard" },
  mkdocs: { label: "docs", className: "type-mkdocs" },
  convert: { label: "convert", className: "type-convert" },
  finalize: { label: "finalize", className: "type-finalize" },
  update: { label: "update", className: "type-update" },
  model: { label: "model", className: "type-model" },
  dataset: { label: "dataset", className: "type-dataset" },
};

// Split-pane bounds: keep both panes big enough to be useful.
const MIN_SPLIT_PCT = 15;
const MAX_SPLIT_PCT = 85;
const DEFAULT_SPLIT_PCT = 45;
const SPLIT_STORAGE_KEY = "forgather-jobs-split-pct";

function loadStoredSplit(): number {
  const v = persistGet(SPLIT_STORAGE_KEY);
  if (v == null) return DEFAULT_SPLIT_PCT;
  const n = parseFloat(v);
  if (Number.isFinite(n) && n >= MIN_SPLIT_PCT && n <= MAX_SPLIT_PCT) return n;
  return DEFAULT_SPLIT_PCT;
}

interface Props {
  /** When set, the panel auto-selects the matching job and opens the TTY
   *  pane the moment that job shows up alive in the polled list. Set by
   *  App after a submit modal closes with the "Watch TTY on start"
   *  toggle on. The panel calls ``onAutoWatchConsumed`` once it has
   *  fired so the trigger is one-shot. */
  autoWatchJobId?: string | null;
  onAutoWatchConsumed?: () => void;
}

/** Unified jobs view: JobRecords we launched + TrainerControlClient endpoints
 *  discovered elsewhere. Everything converges here once it's out of the
 *  queue (source="record" or "merged") or was started outside the server
 *  entirely (source="endpoint"). */
export function JobsPanel({ autoWatchJobId, onAutoWatchConsumed }: Props = {}) {
  const [includeDead, setIncludeDead] = useState(false);
  const [showTty, setShowTty] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [splitPct, setSplitPct] = useState<number>(() => loadStoredSplit());
  const [menuTarget, setMenuTarget] = useState<JobMenuTarget | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const qc = useQueryClient();

  const onJobContextRequest = useCallback(
    (job: Job, e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setMenuTarget({ job, x: e.clientX, y: e.clientY });
    },
    [],
  );

  // Pointer capture + manual geometry math is simpler than pulling in a
  // dedicated splitter library for one axis of resize. setPointerCapture
  // keeps the drag live even when the cursor leaves the 6 px handle.
  const onHandlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
      draggingRef.current = true;
      document.body.style.cursor = "row-resize";
      document.body.style.userSelect = "none";
    },
    [],
  );

  const onHandlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      const panel = panelRef.current;
      if (!panel) return;
      const rect = panel.getBoundingClientRect();
      // Measure against the panel's content area (header is inside the
      // same flex container, so its height is part of the travel — we
      // clamp instead of trying to subtract it exactly).
      const pct = ((e.clientY - rect.top) / rect.height) * 100;
      const clamped = Math.max(
        MIN_SPLIT_PCT,
        Math.min(MAX_SPLIT_PCT, pct),
      );
      setSplitPct(clamped);
    },
    [],
  );

  const onHandlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      try {
        (e.currentTarget as Element).releasePointerCapture(e.pointerId);
      } catch {
        // Capture may already be released if the pointer was cancelled.
      }
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      persistSet(SPLIT_STORAGE_KEY, String(splitPct));
    },
    [splitPct],
  );

  const jobsQ = useQuery({
    queryKey: ["jobs", includeDead],
    queryFn: () => api.listJobs(includeDead),
    refetchInterval: 5000,
  });

  const jobs = jobsQ.data ?? [];
  const alive = jobs.filter((j) => j.alive);

  // Auto-watch handoff from App: once the just-submitted job shows up alive
  // in the polled list, select it and reveal the TTY pane. One-shot — we
  // notify App to clear the trigger as soon as we've consumed it so a later
  // refresh of the same id won't re-open the TTY against the user's intent.
  useEffect(() => {
    if (!autoWatchJobId) return;
    const target = jobs.find((j) => j.id === autoWatchJobId);
    if (!target || !target.alive) return;
    setSelectedId(autoWatchJobId);
    setShowTty(true);
    onAutoWatchConsumed?.();
  }, [autoWatchJobId, jobs, onAutoWatchConsumed]);

  // Detect alive→dead transitions and invalidate caches scoped to the
  // finished job so panels (project tree, context menus, run/checkpoint/
  // eval lists) reflect newly-written artifacts without requiring the
  // user to hit the global Refresh button. Tracks last-seen alive jobs in
  // a ref so the effect only fires on transitions, not every poll.
  const prevAliveRef = useRef<Map<string, Job>>(new Map());
  useEffect(() => {
    if (jobsQ.data === undefined) return;
    const aliveNow = new Set(jobs.filter((j) => j.alive).map((j) => j.id));
    const finished: Job[] = [];
    for (const [id, j] of prevAliveRef.current) {
      if (!aliveNow.has(id)) finished.push(j);
    }
    if (finished.length > 0) {
      let invalidatedAny = false;
      for (const j of finished) {
        if (j.project_dir) {
          qc.invalidateQueries({
            queryKey: ["project-models", j.project_dir],
          });
          qc.invalidateQueries({
            queryKey: ["project-templates", j.project_dir],
          });
          if (j.config) {
            qc.invalidateQueries({
              queryKey: ["config-meta", j.project_dir, j.config],
            });
          }
          invalidatedAny = true;
        }
        if (j.output_dir) {
          qc.invalidateQueries({ queryKey: ["model-runs", j.output_dir] });
          qc.invalidateQueries({
            queryKey: ["model-checkpoints", j.output_dir],
          });
          qc.invalidateQueries({
            queryKey: ["model-evaluations", j.output_dir],
          });
          invalidatedAny = true;
        }
      }
      // Catch jobs that produce new projects/models on disk (dataset, model,
      // finalize, convert) where the unscoped projects list itself may need
      // a re-read. Cheap relative to the surprise of stale tree state.
      if (invalidatedAny) {
        qc.invalidateQueries({ queryKey: ["projects"] });
      }
    }
    const next = new Map<string, Job>();
    for (const j of jobs) if (j.alive) next.set(j.id, j);
    prevAliveRef.current = next;
  }, [jobs, jobsQ.data, qc]);

  const statusQs = useQueries({
    queries: alive.map((j) => ({
      queryKey: ["job-status", j.id, j.job_id],
      queryFn: () => api.jobStatus(j.id),
      refetchInterval: 5000,
      staleTime: 2000,
      // Pre-correlation records have no job_id; the /status endpoint
      // returns 409 for them until correlation completes.
      enabled: j.job_id !== null || j.source === "endpoint",
    })),
  });
  const statusByJobId = new Map<string, Record<string, unknown>>();
  alive.forEach((j, i) => {
    const d = statusQs[i].data;
    if (d) statusByJobId.set(j.id, d);
  });

  const control = useMutation({
    mutationFn: ({ id, action }: { id: string; action: ControlAction }) =>
      api.jobControl(id, action),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["job-status"] });
    },
  });

  const removeJob = useMutation({
    mutationFn: (id: string) => api.removeJob(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });

  // Count of terminal records we own — ``source === "endpoint"`` entries
  // aren't records we can delete. The button is disabled when there's
  // nothing to clean up so the user doesn't hit a no-op confirm dialog.
  const completedOursCount = jobs.filter(
    (j) => j.source !== "endpoint" && !j.alive,
  ).length;

  const cleanup = useMutation({
    mutationFn: () => api.cleanupJobs(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });

  // Keep selection valid across list refreshes — match by id.
  const selectedJob = jobs.find((j) => j.id === selectedId) ?? null;

  if (jobsQ.isLoading) {
    return <div className="pane-state muted">Loading jobs…</div>;
  }
  if (jobsQ.error) {
    return (
      <div className="pane-state err">
        <pre>{String(jobsQ.error)}</pre>
      </div>
    );
  }

  const panelClass = "jobs-panel" + (showTty ? " split" : "");

  return (
    <div className={panelClass} ref={panelRef}>
      <header className="jobs-panel-header">
        <span className="muted">
          {jobs.length} job{jobs.length === 1 ? "" : "s"} · {alive.length}{" "}
          running
        </span>
        <div className="jobs-header-controls">
          <button
            className="secondary"
            disabled={completedOursCount === 0 || cleanup.isPending}
            onClick={() => {
              if (completedOursCount === 0) return;
              if (
                confirm(
                  `Remove ${completedOursCount} completed job record${
                    completedOursCount === 1 ? "" : "s"
                  }? Only records for jobs we launched are affected; external endpoints are untouched.`,
                )
              ) {
                cleanup.mutate();
              }
            }}
            title={
              completedOursCount === 0
                ? "No completed jobs to clean up"
                : `Remove ${completedOursCount} completed job record(s)`
            }
          >
            🧹 Clean completed ({completedOursCount})
          </button>
          <button
            className={"secondary jobs-tty-toggle" + (showTty ? " active" : "")}
            onClick={() => setShowTty((v) => !v)}
            title={showTty ? "Hide TTY pane" : "Show TTY pane"}
          >
            {showTty ? "⊟ Hide TTY" : "⊞ Show TTY"}
          </button>
          <label className="hidden-toggle">
            <input
              type="checkbox"
              checked={includeDead}
              onChange={(e) => setIncludeDead(e.target.checked)}
            />
            Include dead endpoint files
          </label>
        </div>
      </header>

      <div
        className="jobs-list-pane"
        style={showTty ? { flex: `0 0 ${splitPct}%` } : undefined}
      >
        {jobs.length === 0 && (
          <div className="pane-state muted">
            No jobs yet. Submit one from a config's ▶ button or launch
            training via the CLI — either way it'll show up here.
          </div>
        )}
        <div className="jobs-list">
          {jobs.map((j) => (
            <JobCard
              key={j.id}
              job={j}
              status={statusByJobId.get(j.id) ?? null}
              selected={j.id === selectedId}
              onSelect={() => setSelectedId(j.id)}
              onContextRequest={(e) => onJobContextRequest(j, e)}
              onControl={(action) => control.mutate({ id: j.id, action })}
              onRemove={() => {
                if (confirm(`Remove ${j.id} from the record list?`)) {
                  removeJob.mutate(j.id);
                }
              }}
              controlPending={
                control.isPending && control.variables?.id === j.id
              }
            />
          ))}
        </div>
      </div>

      {showTty && (
        <div
          className="jobs-split-handle"
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize TTY pane"
          title="Drag to resize"
          onPointerDown={onHandlePointerDown}
          onPointerMove={onHandlePointerMove}
          onPointerUp={onHandlePointerUp}
          onPointerCancel={onHandlePointerUp}
          onDoubleClick={() => setSplitPct(DEFAULT_SPLIT_PCT)}
        />
      )}

      {showTty && (
        <div className="jobs-tty-pane-wrapper">
          {selectedJob ? (
            <TtyViewer job={selectedJob} />
          ) : (
            <div className="jobs-tty-pane jobs-tty-placeholder">
              <div className="tty-pane-meta muted">
                Select a job above to view its TTY output.
              </div>
              <pre className="tty-pre tty-pre-placeholder">
                (no job selected)
              </pre>
            </div>
          )}
        </div>
      )}

      {menuTarget && (
        <ContextMenu
          x={menuTarget.x}
          y={menuTarget.y}
          onClose={() => setMenuTarget(null)}
        >
          <JobContextMenuItems
            job={menuTarget.job}
            onChoose={(action) => {
              control.mutate({ id: menuTarget.job.id, action });
              setMenuTarget(null);
            }}
          />
        </ContextMenu>
      )}
    </div>
  );
}

/** Right-click menu for a job card. Force-kill is the headline use case
 *  — for hung torchrun groups that don't respond to SIGTERM. Only offered
 *  for our own server-launched jobs (we can SIGKILL the process group);
 *  for externally-discovered endpoints there's no local handle to signal. */
function JobContextMenuItems({
  job,
  onChoose,
}: {
  job: Job;
  onChoose: (action: ControlAction) => void;
}) {
  const isOurs = job.source !== "endpoint";
  const isActive = job.alive;
  return (
    <>
      <div className="context-menu-header muted">
        {job.config ?? job.id}
        <span className="context-menu-class">{job.source}</span>
      </div>
      {isOurs && isActive && (
        <button
          className="context-menu-destructive"
          onClick={() => {
            if (
              confirm(
                `Force-kill ${job.id}?\n\nSends SIGKILL to the entire ` +
                  `process group. Use this only when graceful Stop / Abort ` +
                  `(SIGTERM) has failed — the trainer gets no chance to ` +
                  `flush state, save checkpoints, or clean up.`,
              )
            ) {
              onChoose("force-kill");
            }
          }}
        >
          ☠ Force kill (SIGKILL)
        </button>
      )}
      {(!isOurs || !isActive) && (
        <div className="context-menu-empty muted">
          No actions for this job.
        </div>
      )}
    </>
  );
}

function JobCard({
  job,
  status,
  selected,
  onSelect,
  onContextRequest,
  onControl,
  onRemove,
  controlPending,
}: {
  job: Job;
  status: Record<string, unknown> | null;
  selected: boolean;
  onSelect: () => void;
  onContextRequest: (e: React.MouseEvent) => void;
  onControl: (action: ControlAction) => void;
  onRemove: () => void;
  controlPending: boolean;
}) {
  const startedSec = job.started_at ?? job.submitted_at ?? null;
  const started = startedSec ? new Date(startedSec * 1000).toLocaleString() : "—";
  const uptimeSec =
    startedSec && job.finished_at
      ? job.finished_at - startedSec
      : startedSec
        ? Date.now() / 1000 - startedSec
        : 0;

  const statusClass = "status-" + job.status;
  const isTerminal = !job.alive;
  const isOurs = job.source !== "endpoint";
  const isEval = job.job_type === "eval";
  const isInference = job.job_type === "inference";
  const isTensorBoard = job.job_type === "tensorboard";
  const isMkDocs = job.job_type === "mkdocs";
  const isConvert = job.job_type === "convert";
  const isFinalize = job.job_type === "finalize";
  const chip = JOB_TYPE_CHIPS[job.job_type] ?? JOB_TYPE_CHIPS.training;
  // Only actual training jobs get the trainer-progress UI / control
  // protocol affordances. Everything else (model, dataset, eval, …) is
  // fire-and-forget.
  const isTraining = job.job_type === "training";
  // Trainer-protocol controls require a correlated job_id and the job
  // alive. Eval / inference never correlate (no endpoint.json), so
  // these stay hidden and only kill / force-kill apply.
  const canControl = !!job.job_id && job.alive && isTraining;

  // Header identifier: prefer a config name, fall back to job_id / queue_id
  const headerTitle = job.config ?? job.job_id ?? job.queue_id ?? job.id;
  const dyn =
    isTraining && job.dynamic_args ? Object.entries(job.dynamic_args) : [];

  // Inference jobs run a local HTTP server on a user-chosen port.
  // Synthesize the URL for a clickable link so the user can jump straight
  // to the OpenAPI root rather than copy-pasting host:port.
  const inferenceHost =
    isInference && typeof job.job_params?.host === "string"
      ? (job.job_params.host as string)
      : null;
  const inferencePort =
    isInference && typeof job.job_params?.port === "number"
      ? (job.job_params.port as number)
      : null;
  const inferenceUrl =
    inferenceHost && inferencePort
      ? `http://${inferenceHost}:${inferencePort}`
      : null;

  // TensorBoard is the same idea: a local web server. Host isn't always
  // set on the params; bind_all implies 0.0.0.0 → browser-friendly at
  // localhost, otherwise fall back to the passed host or 127.0.0.1.
  const tbPort =
    isTensorBoard && typeof job.job_params?.port === "number"
      ? (job.job_params.port as number)
      : null;
  const tbBindAll = isTensorBoard && Boolean(job.job_params?.bind_all);
  const tbHost =
    isTensorBoard && typeof job.job_params?.host === "string"
      ? (job.job_params.host as string)
      : null;
  const tbUrl = tbPort
    ? `http://${tbBindAll ? "localhost" : tbHost ?? "localhost"}:${tbPort}`
    : null;

  // MkDocs serve runs a local HTTP dev server. host:port pair is folded
  // into ``--dev-addr``; render the URL the same way TB does so the user
  // can jump straight to the rendered docs.
  const mkPort =
    isMkDocs && typeof job.job_params?.port === "number"
      ? (job.job_params.port as number)
      : null;
  const mkHost =
    isMkDocs && typeof job.job_params?.host === "string"
      ? (job.job_params.host as string)
      : null;
  const mkUrl = mkPort
    ? `http://${mkHost ?? "127.0.0.1"}:${mkPort}`
    : null;

  const cardClass =
    "job-card " +
    statusClass +
    (job.alive ? " alive" : " dead") +
    (selected ? " selected" : "");

  return (
    <div
      className={cardClass}
      onClick={onSelect}
      onContextMenu={onContextRequest}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSelect();
      }}
    >
      <div className="job-row-main">
        <span className={"queue-status " + statusClass}>
          {job.status.toUpperCase()}
        </span>
        <span className={"job-type-chip " + chip.className}>{chip.label}</span>
        <span className="queue-config">{headerTitle}</span>
        <span className="muted queue-meta">
          {job.gpu_indices && job.gpu_indices.length > 0 && (
            <>GPUs [{job.gpu_indices.join(",")}]</>
          )}
          {job.pid != null && ` · pid ${job.pid}`}
          {job.host && ` · ${job.host}${job.port ? `:${job.port}` : ""}`}
          {" · "}
          {job.source}
        </span>
      </div>
      <div className="job-row-meta muted">
        started {started} ({fmtUptime(uptimeSec)})
        {job.exit_code !== null && job.exit_code !== undefined &&
          ` · exit ${job.exit_code}`}
      </div>
      {job.project_dir && isTraining && (
        <div className="queue-row-meta muted">{job.project_dir}</div>
      )}
      {isEval && job.job_params && (
        <div className="queue-dirs muted">
          <div>
            <span>eval:</span>{" "}
            <code>{String(job.job_params.eval_template ?? "")}</code>
          </div>
          <div>
            <span>model:</span>{" "}
            <code>{String(job.job_params.model_path ?? "")}</code>
          </div>
          {job.job_params.checkpoint_path ? (
            <div>
              <span>ckpt:</span>{" "}
              <code>{String(job.job_params.checkpoint_path)}</code>
            </div>
          ) : (
            <div>
              <span>ckpt:</span> <em>model default</em>
            </div>
          )}
        </div>
      )}
      {isInference && job.job_params && (
        <div className="queue-dirs muted">
          <div>
            <span>url:</span>{" "}
            {job.alive && inferenceUrl ? (
              <a
                href={inferenceUrl}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                {inferenceUrl}
              </a>
            ) : (
              <code>{inferenceUrl ?? "—"}</code>
            )}
          </div>
          <div>
            <span>model:</span>{" "}
            <code>{String(job.job_params.model_path ?? "")}</code>
          </div>
          {job.job_params.checkpoint_path ? (
            <div>
              <span>ckpt:</span>{" "}
              <code>{String(job.job_params.checkpoint_path)}</code>
            </div>
          ) : job.job_params.from_checkpoint ? (
            <div>
              <span>ckpt:</span> <em>latest</em>
            </div>
          ) : (
            <div>
              <span>ckpt:</span> <em>from_pretrained</em>
            </div>
          )}
        </div>
      )}
      {isTensorBoard && job.job_params && (
        <div className="queue-dirs muted">
          <div>
            <span>url:</span>{" "}
            {job.alive && tbUrl ? (
              <a
                href={tbUrl}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                {tbUrl}
              </a>
            ) : (
              <code>{tbUrl ?? "—"}</code>
            )}
          </div>
          <div>
            <span>logdir:</span>{" "}
            <code>{String(job.job_params.logdir ?? "")}</code>
          </div>
          {typeof job.job_params.window_title === "string" &&
            job.job_params.window_title && (
              <div>
                <span>title:</span>{" "}
                <code>{job.job_params.window_title}</code>
              </div>
            )}
        </div>
      )}
      {isMkDocs && job.job_params && (
        <div className="queue-dirs muted">
          <div>
            <span>url:</span>{" "}
            {job.alive && mkUrl ? (
              <a
                href={mkUrl}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                {mkUrl}
              </a>
            ) : (
              <code>{mkUrl ?? "—"}</code>
            )}
          </div>
          <div>
            <span>config:</span>{" "}
            <code>{String(job.job_params.config_file ?? "")}</code>
          </div>
        </div>
      )}
      {(job.logs_dir || job.output_dir) && (
        <div className="queue-dirs muted">
          {job.logs_dir && (
            <div>
              <span>logs_dir:</span> <code>{job.logs_dir}</code>
            </div>
          )}
          {job.output_dir && (
            <div>
              <span>output_dir:</span> <code>{job.output_dir}</code>
            </div>
          )}
        </div>
      )}
      {dyn.length > 0 && (
        <div className="queue-dyn">
          {dyn.map(([k, v]) => (
            <span key={k} className="stat-pill">
              <span className="muted">{k}</span> {String(v)}
            </span>
          ))}
        </div>
      )}
      {status && <JobStatusBlock status={status} />}
      {job.error && <div className="err queue-error">{job.error}</div>}

      <div
        className="job-actions"
        onClick={(e) => e.stopPropagation()}
      >
        {canControl && (
          <>
            <button
              className="secondary"
              disabled={controlPending}
              onClick={() => onControl("save")}
            >
              Save checkpoint
            </button>
            <button
              className="secondary"
              disabled={controlPending}
              onClick={() => onControl("save-stop")}
            >
              Save &amp; stop
            </button>
            <button
              className="secondary"
              disabled={controlPending}
              onClick={() => onControl("stop")}
            >
              Graceful stop
            </button>
            <button
              className="destructive"
              disabled={controlPending}
              onClick={() => {
                if (confirm(`Abort ${job.id}? Training state is NOT saved.`)) {
                  onControl("abort");
                }
              }}
            >
              Abort
            </button>
          </>
        )}
        {isOurs && job.alive && !canControl && (
          <button
            className="destructive"
            disabled={controlPending}
            onClick={() => {
              const prompt = isEval
                ? `Kill ${job.id}? Ends the evaluation subprocess.`
                : isInference
                  ? `Stop inference server ${job.id}? The HTTP endpoint will drop.`
                  : isTensorBoard
                    ? `Stop TensorBoard ${job.id}? The viewer at :${tbPort} will drop.`
                    : isMkDocs
                      ? `Stop MkDocs ${job.id}? The docs server at :${mkPort} will drop.`
                      : isConvert
                        ? `Kill ${job.id}? Ends the convert subprocess; partial output may be left at the destination.`
                        : isFinalize
                          ? `Kill ${job.id}? Ends the finalize subprocess; partial output may be left at the destination.`
                          : `Kill ${job.id}? (pre-correlation hard kill)`;
              if (confirm(prompt)) {
                onControl("kill");
              }
            }}
          >
            {isInference || isTensorBoard || isMkDocs ? "Stop server" : "Kill"}
          </button>
        )}
        {isOurs && isTerminal && (
          <button className="secondary" onClick={onRemove}>
            Remove record
          </button>
        )}
      </div>
    </div>
  );
}

function JobStatusBlock({ status }: { status: Record<string, unknown> }) {
  const known: Array<[string, string]> = [];
  // Display order matters — keep the most-glanced metrics first. Pickers
  // return null when the field is missing or wrong-shaped so the pill is
  // skipped silently.
  const pickers: [string, (v: unknown) => string | null][] = [
    ["loss", (v) => (typeof v === "number" ? v.toFixed(4) : null)],
    ["lr", (v) => (typeof v === "number" ? fmtLr(v) : null)],
    ["grad_norm", (v) => (typeof v === "number" ? v.toFixed(3) : null)],
    ["epoch", (v) => (typeof v === "number" ? v.toFixed(3) : null)],
    ["tok/s", (v) => (typeof v === "number" ? fmtCount(v) : null)],
    ["tokens", (v) => (typeof v === "number" ? fmtCount(v) : null)],
    ["peak_mem", (v) => fmtPeakMem(v)],
  ];
  // Map display labels back to the actual status keys.
  const keyAliases: Record<string, string> = {
    lr: "learning_rate",
    "tok/s": "tok_per_sec",
  };
  for (const [label, f] of pickers) {
    const statusKey = keyAliases[label] ?? label;
    const v = f(status[statusKey]);
    if (v !== null) known.push([label, v]);
  }

  // Progress bar derived from `global_step / max_steps`. Some trainers
  // sentinel max_steps as -1 (no fixed budget) — only render a bar when
  // both values are positive numbers, otherwise just leave the pills.
  const step = numOrNull(status["global_step"]);
  const max = numOrNull(status["max_steps"]);
  const showProgress = step !== null && max !== null && max > 0;
  const pct = showProgress
    ? Math.max(0, Math.min(100, ((step as number) / (max as number)) * 100))
    : 0;

  return (
    <div className="job-status-block">
      {showProgress && (
        <div className="job-progress" title={`${step} / ${max} steps`}>
          <div className="job-progress-track">
            <div
              className="job-progress-fill"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="job-progress-label">
            {step}/{max} ({pct.toFixed(1)}%)
          </span>
        </div>
      )}
      {known.length > 0 && (
        <div className="job-status-row">
          {known.map(([k, v]) => (
            <span key={k} className="stat-pill">
              <span className="muted">{k}</span> {v}
            </span>
          ))}
        </div>
      )}
      <details className="job-status-raw">
        <summary className="muted">raw status</summary>
        <pre>{JSON.stringify(status, null, 2)}</pre>
      </details>
    </div>
  );
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Learning rate: decimal for the typical 1e-5..1e-2 range, scientific
 *  outside it. ML LRs are usually written as decimals (8e-4 → 0.0008). */
function fmtLr(v: number): string {
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs < 1e-5 || abs >= 100) return v.toExponential(2);
  return v.toPrecision(4).replace(/\.?0+$/, "");
}

/** Whole-number counts with thousands separators. */
function fmtCount(v: number): string {
  return Math.round(v).toLocaleString();
}

/** Peak GPU memory: trainer reports a list of bytes, one entry per
 *  assigned device. Render each entry individually (comma-separated),
 *  picking the unit once from the largest value so comparisons stay
 *  visually aligned. A bare scalar (older records) renders as a single
 *  value. Summing across devices would be misleading for DDP where each
 *  rank holds roughly the same footprint. */
function fmtPeakMem(v: unknown): string | null {
  const values: number[] = [];
  if (typeof v === "number") {
    values.push(v);
  } else if (Array.isArray(v)) {
    for (const x of v) if (typeof x === "number") values.push(x);
  } else {
    return null;
  }
  if (values.length === 0) return null;
  const max = Math.max(...values);
  if (!Number.isFinite(max) || max <= 0) return null;

  // Unit chosen by the largest value so every entry renders in the same
  // unit (otherwise a near-GiB value and a smaller MiB value would make
  // the list hard to scan).
  const useGiB = max / 1024 ** 3 >= 1;
  const fmt = (n: number) =>
    useGiB
      ? (n / 1024 ** 3).toFixed(2)
      : (n / 1024 ** 2).toFixed(0);
  const unit = useGiB ? "GiB" : "MiB";
  return `${values.map(fmt).join(", ")} ${unit}`;
}

function fmtUptime(seconds: number): string {
  if (seconds < 0) return "just now";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) {
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  }
  const d = Math.floor(seconds / 86400);
  const h = Math.round((seconds % 86400) / 3600);
  return `${d}d ${h}h`;
}
