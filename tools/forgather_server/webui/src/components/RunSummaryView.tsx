import { useQuery } from "@tanstack/react-query";
import { api, RunSummary } from "../api";

const SUMMARY_KEY_ORDER = [
  "total_steps",
  "final_epoch",
  "final_loss",
  "best_loss",
  "best_loss_step",
  "avg_loss",
  "min_loss",
  "final_eval_loss",
  "best_eval_loss",
  "best_eval_loss_step",
  "avg_grad_norm",
  "max_grad_norm_value",
  "max_grad_norm_step",
  "initial_lr",
  "final_lr",
  "train_runtime",
  "train_samples",
  "train_samples_per_second",
  "train_steps_per_second",
  "effective_batch_size",
];

function buildSummaryRows(summary: Record<string, unknown>): [string, string][] {
  const rows: [string, string][] = [];
  const seen = new Set<string>();
  for (const key of SUMMARY_KEY_ORDER) {
    if (key in summary) {
      rows.push([key, formatSummaryValue(summary[key])]);
      seen.add(key);
    }
  }
  for (const key of Object.keys(summary)) {
    if (seen.has(key) || key === "run_name" || key === "log_path") continue;
    rows.push([key, formatSummaryValue(summary[key])]);
  }
  return rows;
}

export function formatSummaryValue(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(4);
  }
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

export function RunSummaryView({ run_dir }: { run_dir: string }) {
  const q = useQuery({
    queryKey: ["run-summary", run_dir],
    queryFn: () => api.runSummary(run_dir),
  });
  if (q.isLoading) return <div className="pane-state">Loading summary…</div>;
  if (q.error)
    return (
      <div className="pane-state err">
        <pre>{String(q.error)}</pre>
      </div>
    );
  const data = q.data as RunSummary;
  const rows = buildSummaryRows(data.summary);
  return (
    <div className="run-summary">
      <div className="summary-head">
        <h3>
          {String(data.summary.run_name ?? run_dir.split("/").pop() ?? "run")}
        </h3>
        <div className="summary-paths muted">
          <div>
            <span>log:</span>
            <code>{data.log_path ?? "—"}</code>
          </div>
          <div>
            <span>config:</span>
            <code>{data.config_path ?? "—"}</code>
          </div>
          <div>
            <span>pp:</span>
            <code>{data.pp_path ?? "—"}</code>
          </div>
        </div>
      </div>
      {rows.length === 0 ? (
        <div className="pane-state muted">No summary data available.</div>
      ) : (
        <table className="summary-table">
          <tbody>
            {rows.map(([key, value]) => (
              <tr key={key}>
                <th>{key}</th>
                <td>{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
