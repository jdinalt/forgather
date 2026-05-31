import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, QueueItem } from "../api";

/** Queued (not-yet-dispatched) jobs plus a compact scheduler status line.
 *  Rendered at the top of the Jobs view's list pane (see JobsPanel) so
 *  queued and running work read as one continuous list — queued items sit
 *  above the jobs the scheduler has already dispatched. The verbose
 *  empty-state copy the old standalone Queue view carried is dropped: when
 *  nothing is queued the section collapses to just the scheduler line, and
 *  the jobs list below has its own empty-state. */
export function QueueSection() {
  const qc = useQueryClient();
  const listQ = useQuery({
    queryKey: ["queue"],
    queryFn: api.listQueue,
    refetchInterval: 2000,
  });
  // Scheduler toggle lives in the app sidebar header — this section only
  // reports the current state.
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
    refetchInterval: 3000,
  });

  const cancel = useMutation({
    mutationFn: api.abortQueueItem,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const items = listQ.data ?? [];
  const schedEnabled = !!schedQ.data?.enabled;

  return (
    <div className="queue-section">
      <header className="queue-section-header">
        <div className="sched-toggle">
          <span className={"sched-light " + (schedEnabled ? "on" : "off")} />
          <strong>Scheduler</strong>
          <span className={"sched-state " + (schedEnabled ? "on" : "off")}>
            {schedEnabled ? "running" : "paused"}
          </span>
          {schedQ.data && (
            <span className="muted">
              {schedQ.data.running_count} running · tick{" "}
              {schedQ.data.tick_count}
            </span>
          )}
        </div>
        <span className="muted">{items.length} queued</span>
      </header>

      {listQ.error && (
        <div className="pane-state err">
          <pre>{String(listQ.error)}</pre>
        </div>
      )}

      {items.length > 0 && (
        <div className="queue-list">
          {items.map((it) => (
            <QueueCard
              key={it.queue_id}
              item={it}
              onCancel={() => {
                if (confirm(`Cancel queued ${it.queue_id}?`)) {
                  cancel.mutate(it.queue_id);
                }
              }}
              cancelPending={
                cancel.isPending && cancel.variables === it.queue_id
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

function QueueCard({
  item,
  onCancel,
  cancelPending,
}: {
  item: QueueItem;
  onCancel: () => void;
  cancelPending: boolean;
}) {
  const submitted = new Date(item.submitted_at * 1000).toLocaleString();
  const dyn = Object.entries(item.dynamic_args);
  return (
    <div className="queue-card status-queued">
      <div className="queue-row-main">
        <span className="queue-status status-queued">QUEUED</span>
        <span className="queue-config">{item.config}</span>
        <span className="muted queue-meta">
          GPUs {item.requested_gpus} · pri {item.priority}
        </span>
      </div>
      <div className="queue-row-meta muted">
        {item.project_dir} · submitted {submitted}
      </div>
      {dyn.length > 0 && (
        <div className="queue-dyn">
          {dyn.map(([k, v]) => (
            <span key={k} className="stat-pill">
              <span className="muted">{k}</span> {String(v)}
            </span>
          ))}
        </div>
      )}
      <div className="job-actions">
        <button
          className="destructive"
          onClick={onCancel}
          disabled={cancelPending}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
