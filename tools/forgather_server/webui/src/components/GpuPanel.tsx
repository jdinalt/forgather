import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, GpuInfo, GpuPolicy, Job, RUNNING_JOB_STATUSES } from "../api";
import { ContextMenu } from "./ContextMenu";

interface GpuMenuTarget {
  gpu: GpuInfo;
  x: number;
  y: number;
}

/** Subscribe to the /api/gpus/stream WebSocket. Falls back to a single REST
 *  fetch on connect failure so the panel still shows something useful.
 *
 *  Auto-reconnects with exponential backoff (500ms → 10s cap, reset to 500ms
 *  on a healthy message) so transient drops — laptop sleep, SSH tunnel reset,
 *  brief NVML hangs — heal on their own without needing a manual reload. The
 *  ``stale`` flag goes true the moment the socket closes and clears on the
 *  next successful message, so the user sees an honest live/stale indicator
 *  during the gap.
 *
 *  In Vite dev mode the WebSocket target is the dev server (5173); the proxy
 *  config in vite.config.ts forwards it to the backend on 8765. In production
 *  (backend serves the SPA directly) the origin already matches. */
function useGpuStream(): {
  data: GpuInfo[] | null;
  error: string | null;
  stale: boolean;
} {
  const [data, setData] = useState<GpuInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let backoffMs = 500;
    const BACKOFF_CAP = 10_000;

    // Prime with a REST call so the first paint has data before the WS opens.
    api.listGpus().then(
      (d) => !cancelled && setData(d),
      (e) => !cancelled && setError(String(e)),
    );

    const connect = () => {
      if (cancelled) return;
      ws = new WebSocket(api.gpuStreamUrl());
      ws.onmessage = (ev) => {
        try {
          const parsed: GpuInfo[] = JSON.parse(ev.data);
          setData(parsed);
          setError(null);
          setStale(false);
          backoffMs = 500;
        } catch (e) {
          setError(String(e));
        }
      };
      ws.onerror = () => {
        setStale(true);
      };
      ws.onclose = () => {
        if (cancelled) return;
        setStale(true);
        reconnectTimer = setTimeout(connect, backoffMs);
        backoffMs = Math.min(backoffMs * 2, BACKOFF_CAP);
      };
    };
    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  return { data, error, stale };
}

export function GpuPanel() {
  const { data, error, stale } = useGpuStream();
  const qc = useQueryClient();
  const [menuTarget, setMenuTarget] = useState<GpuMenuTarget | null>(null);

  // Job listing is cheap; keep it refreshing so PID→job attribution on the
  // cards stays in sync if a job starts or finishes while the panel is open.
  const jobsQ = useQuery({
    queryKey: ["jobs", false],
    queryFn: () => api.listJobs(false),
    refetchInterval: 5000,
  });
  const jobByPid = new Map<number, Job>();
  const reservedGpus = new Set<number>();
  for (const j of jobsQ.data ?? []) {
    if (j.pid != null) jobByPid.set(j.pid, j);
    // Mirror scheduler._reserved_gpu_set: a GPU is reserved by Forgather
    // whenever a JobRecord with status in RUNNING_JOB_STATUSES lists it
    // under gpu_indices. Single source of truth in api.ts.
    if (RUNNING_JOB_STATUSES.has(j.status) && j.gpu_indices) {
      for (const idx of j.gpu_indices) reservedGpus.add(idx);
    }
  }

  const killGpu = useMutation({
    mutationFn: (gpuIndex: number) => api.killGpuProcesses(gpuIndex),
    onSuccess: (resp) => {
      // `failed` is the count of NVML-reported PIDs we couldn't signal — inside
      // a container those are out-of-container holders (host-namespace PIDs),
      // which is expected and not a failure of the in-container reap.
      const failedNote =
        resp.failed.length > 0
          ? ` (${resp.failed.length} out-of-container PID(s) not reachable)`
          : "";
      // Cheap visual confirmation so the operator sees it landed; the GPU
      // stream will refresh independently within a couple of seconds.
      alert(
        `GPU ${resp.gpu_index}: SIGKILL sent to ${resp.killed.length} ` +
          `process(es)${failedNote}.`,
      );
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (e) => alert(`Kill failed: ${String(e)}`),
  });

  const setPolicy = useMutation({
    mutationFn: ({
      index,
      policy,
    }: {
      index: number;
      policy: { disabled?: boolean; min_priority?: number };
    }) => api.setGpuPolicy(index, policy),
    onError: (e) => alert(`Policy update failed: ${String(e)}`),
    // The WebSocket stream will carry the updated fields within ~2 s.
  });

  if (error && !data) {
    return (
      <div className="pane-state err">
        <pre>{error}</pre>
      </div>
    );
  }
  if (!data) {
    return <div className="pane-state muted">Loading GPUs…</div>;
  }
  if (data.length === 0) {
    return (
      <div className="pane-state muted">
        No GPUs detected. pynvml init failed and torch.cuda reports no devices.
      </div>
    );
  }
  return (
    <div className="gpu-panel">
      <header className="gpu-panel-header">
        <span className="muted">
          {data.length} GPU{data.length === 1 ? "" : "s"} ·{" "}
          {stale ? "stream disconnected — showing last snapshot" : "live"}
        </span>
      </header>
      <div className="gpu-grid">
        {data.map((g) => (
          <GpuCard
            key={g.index}
            g={g}
            jobByPid={jobByPid}
            reserved={reservedGpus.has(g.index)}
            onContextRequest={(e) => {
              e.preventDefault();
              setMenuTarget({ gpu: g, x: e.clientX, y: e.clientY });
            }}
            onToggleDisabled={() =>
              setPolicy.mutate({ index: g.index, policy: { disabled: !g.disabled } })
            }
          />
        ))}
      </div>

      {menuTarget && (
        <ContextMenu
          x={menuTarget.x}
          y={menuTarget.y}
          onClose={() => setMenuTarget(null)}
        >
          <GpuContextMenuItems
            gpu={menuTarget.gpu}
            jobByPid={jobByPid}
            onKill={() => {
              killGpu.mutate(menuTarget.gpu.index);
              setMenuTarget(null);
            }}
            onSetPolicy={(policy: Partial<GpuPolicy>) => {
              setPolicy.mutate({ index: menuTarget.gpu.index, policy });
              setMenuTarget(null);
            }}
          />
        </ContextMenu>
      )}
    </div>
  );
}

/** Right-click menu for a GPU card. */
function GpuContextMenuItems({
  gpu,
  jobByPid,
  onKill,
  onSetPolicy,
}: {
  gpu: GpuInfo;
  jobByPid: Map<number, Job>;
  onKill: () => void;
  onSetPolicy: (policy: Partial<GpuPolicy>) => void;
}) {
  const procCount = gpu.processes.length;
  return (
    <>
      <div className="context-menu-header muted">
        GPU {gpu.index} · {gpu.name}
        {gpu.excluded && (
          <span className="context-menu-class">excluded</span>
        )}
      </div>

      <button
        className="context-menu-item"
        onClick={() => onSetPolicy({ disabled: !gpu.disabled })}
      >
        {gpu.disabled ? "Enable GPU (allow scheduling)" : "Disable GPU (block scheduling)"}
      </button>

      <button
        className="context-menu-item"
        onClick={() => {
          const raw = prompt(
            `Minimum priority gate for GPU ${gpu.index}.\n` +
              `Jobs with priority < this value will not be assigned here.\n` +
              `(0 = no restriction; negative values are allowed)`,
            String(gpu.min_priority),
          );
          if (raw === null) return; // user cancelled
          const n = Number(raw);
          if (!Number.isInteger(n)) {
            alert("Priority must be an integer.");
            return;
          }
          onSetPolicy({ min_priority: n });
        }}
      >
        Set minimum priority… (current: {gpu.min_priority})
      </button>

      {gpu.min_priority !== 0 && (
        <button
          className="context-menu-item"
          onClick={() => onSetPolicy({ min_priority: 0 })}
        >
          Clear priority gate (reset to 0)
        </button>
      )}

      {procCount === 0 ? (
        <div className="context-menu-empty muted">
          No processes to kill on this GPU.
        </div>
      ) : (
        <button
          className="context-menu-destructive"
          onClick={() => {
            const summary = gpu.processes
              .map((p) => {
                const job = jobByPid.get(p.pid);
                return `  pid ${p.pid}` + (job ? ` (${job.config ?? job.id})` : "");
              })
              .join("\n");
            if (
              confirm(
                `Kill ALL ${procCount} process(es) on GPU ${gpu.index}?\n\n` +
                  `${summary}\n\nSends SIGKILL — no chance to clean up. ` +
                  `Hits processes the server didn't launch too.`,
              )
            ) {
              onKill();
            }
          }}
        >
          ☠ Kill all {procCount} process{procCount === 1 ? "" : "es"} (SIGKILL)
        </button>
      )}
    </>
  );
}

/** Single GPU card. Exported so the cluster Nodes view can reuse the
 *  exact card layout per peer node (read-only — peer mutations are
 *  not routed in Phase 1). When ``onContextRequest`` and
 *  ``onToggleDisabled`` are omitted the card renders without click
 *  affordances, which matches the read-only peer use case. */
export function GpuCard({
  g,
  jobByPid,
  reserved = false,
  onContextRequest,
  onToggleDisabled,
}: {
  g: GpuInfo;
  jobByPid: Map<number, Job>;
  /** True when a Forgather-dispatched job currently lists this GPU under
   *  its ``gpu_indices`` (status in RUNNING_STATUSES). Mirrors
   *  scheduler._reserved_gpu_set on the backend. */
  reserved?: boolean;
  onContextRequest?: (e: React.MouseEvent) => void;
  onToggleDisabled?: () => void;
}) {
  const interactive = !!onToggleDisabled;
  const memPct = g.total_mem_bytes
    ? (g.used_mem_bytes / g.total_mem_bytes) * 100
    : 0;
  // The card color mirrors the scheduler's dispatch rule
  // (scheduler._idle_gpu_indices minus _reserved_gpu_set): a GPU shows
  // "idle" iff it isn't excluded via CUDA_VISIBLE_DEVICES, isn't user-
  // disabled, AND isn't currently reserved by a running Forgather job.
  // External processes — desktop compositors, unrelated CUDA work — do
  // *not* gate dispatch and do *not* mark the card busy on their own;
  // they show up in the process list and the util/memory bars.
  const idle = !g.excluded && !g.disabled && !reserved;
  // excluded trumps disabled visually
  const cardClass =
    "gpu-card" +
    (g.excluded ? " excluded" : g.disabled ? " disabled" : idle ? " idle" : " busy");

  const disabledTitle = g.disabled
    ? "Click to enable GPU (allow scheduling)"
    : "Click to disable GPU (block scheduling)";

  return (
    <div
      className={cardClass + (interactive ? "" : " readonly")}
      onContextMenu={onContextRequest}
      onClick={
        interactive
          ? (e) => {
              if (e.button !== 0) return;
              onToggleDisabled!();
            }
          : undefined
      }
      title={interactive && !g.excluded ? disabledTitle : undefined}
      style={
        interactive
          ? { cursor: g.excluded ? undefined : "pointer" }
          : undefined
      }
    >
      <div className="gpu-header">
        <span className="gpu-idx">GPU{g.index}</span>
        <span className="gpu-name">{g.name}</span>
        {g.excluded && (
          <span
            className="badge excluded-badge"
            title="Excluded from scheduling by CUDA_VISIBLE_DEVICES at server start"
          >
            EXCLUDED
          </span>
        )}
        {!g.excluded && g.disabled && (
          <span
            className="badge disabled-badge"
            title="Runtime-disabled — click card to re-enable"
          >
            DISABLED
          </span>
        )}
        {g.min_priority !== 0 && (
          <span
            className="badge priority-badge"
            title={`Only jobs with priority >= ${g.min_priority} may use this GPU`}
          >
            {">="}{g.min_priority}
          </span>
        )}
        {g.source !== "nvml" && (
          <span className="badge" title="NVML unavailable — limited info">
            {g.source}
          </span>
        )}
      </div>

      <Bar
        label={g.unified_memory ? "memory (shared)" : "memory"}
        pct={memPct}
        right={`${fmtMiB(g.used_mem_bytes)} / ${fmtMiB(g.total_mem_bytes)}`}
      />
      {g.util_pct !== null && (
        <Bar label="util" pct={g.util_pct} right={`${g.util_pct}%`} />
      )}

      <div className="gpu-row">
        {g.temp_c !== null && <span title="Temperature">🌡️ {g.temp_c}°C</span>}
        {g.power_w !== null && <span title="Power draw">⚡ {Math.round(g.power_w)}W</span>}
        {g.fan_pct !== null && <span title="Fan speed">🌀 {g.fan_pct}%</span>}
        <span className={"gpu-status " + (idle ? "idle" : "busy")}>
          {idle ? "idle" : "busy"}
        </span>
      </div>

      {g.processes.length > 0 && (
        <div
          className="gpu-procs"
          onClick={(e) => e.stopPropagation()}
        >
          <span className="muted">processes:</span>
          {g.processes.map((p) => {
            const job = jobByPid.get(p.pid);
            const chipClass = job ? "proc-chip job" : "proc-chip";
            const label = job ? (job.config ?? job.id) : `${p.pid}`;
            const title = job
              ? `${job.id}\npid ${p.pid} · ${fmtMiB(p.used_mem_bytes)}`
              : `${fmtMiB(p.used_mem_bytes)}`;
            return (
              <span key={p.pid} className={chipClass} title={title}>
                {label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Bar({ label, pct, right }: { label: string; pct: number; right: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${clamped}%` }} />
      </div>
      <span className="bar-right">{right}</span>
    </div>
  );
}

function fmtMiB(bytes: number): string {
  const mib = bytes / (1024 * 1024);
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${Math.round(mib)} MiB`;
}
