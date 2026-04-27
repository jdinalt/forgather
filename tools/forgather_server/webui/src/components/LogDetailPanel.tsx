import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, ConfigInfo, ProjectInfo } from "../api";
import { RunSummaryView } from "./RunSummaryView";

type LogTab = "tty" | "summary";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  run_dir: string;
  run_id: string;
  onBack: (project: ProjectInfo, config: ConfigInfo) => void;
}

export function LogDetailPanel({ project, config, run_dir, run_id, onBack }: Props) {
  const [logTab, setLogTab] = useState<LogTab>("tty");

  return (
    <div className="viewer">
      <header className="viewer-header">
        <div className="detail-panel-head">
          <button
            className="secondary"
            onClick={() => onBack(project, config)}
            title="Back to config"
          >
            ← {config.name}
          </button>
          <span className="muted detail-run-id">{run_id}</span>
          <code className="muted detail-path">{run_dir}</code>
        </div>
        <nav className="tabs">
          <button
            className={logTab === "tty" ? "active" : ""}
            onClick={() => setLogTab("tty")}
          >
            TTY
          </button>
          <button
            className={logTab === "summary" ? "active" : ""}
            onClick={() => setLogTab("summary")}
          >
            Summary
          </button>
        </nav>
      </header>
      {logTab === "tty" && <RunTtyView run_dir={run_dir} />}
      {logTab === "summary" && <RunSummaryView run_dir={run_dir} />}
    </div>
  );
}

function RunTtyView({ run_dir }: { run_dir: string }) {
  const q = useQuery({
    queryKey: ["run-tty", run_dir],
    queryFn: () => api.runTty(run_dir),
  });
  if (q.isLoading) return <div className="pane-state">Loading TTY output…</div>;
  if (q.error)
    return (
      <div className="pane-state err">
        <pre>{String(q.error)}</pre>
      </div>
    );
  return <pre className="tty-pre tty-static">{q.data ?? ""}</pre>;
}
