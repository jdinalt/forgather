import { useState } from "react";
import { CheckpointEntry, ConfigInfo, ProjectInfo } from "../api";
import { EvalModal } from "./EvalModal";
import { InferenceModal } from "./InferenceModal";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  checkpoint: CheckpointEntry;
  onBack: (project: ProjectInfo, config: ConfigInfo) => void;
}

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function basename(p: string): string {
  const clean = p.replace(/\/+$/, "");
  const i = clean.lastIndexOf("/");
  return i < 0 ? clean : clean.slice(i + 1);
}

export function CheckpointDetailPanel({
  project,
  config,
  output_dir,
  checkpoint,
  onBack,
}: Props) {
  const [evaluating, setEvaluating] = useState(false);
  const [serving, setServing] = useState(false);

  const modelName = basename(output_dir);

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
          <span className="muted detail-run-id">
            checkpoint-{checkpoint.step}
          </span>
        </div>
        <div className="btn-row">
          <button
            className="clean-btn"
            onClick={() => setServing(true)}
            title="Start an inference server from this checkpoint"
          >
            🔮 Serve Inference…
          </button>
          <button
            className="clean-btn"
            onClick={() => setEvaluating(true)}
            title="Evaluate this checkpoint"
          >
            📐 Evaluate…
          </button>
        </div>
      </header>

      <div className="run-summary">
        <table className="summary-table">
          <tbody>
            <tr>
              <th>step</th>
              <td>{checkpoint.step}</td>
            </tr>
            <tr>
              <th>size</th>
              <td>{formatBytes(checkpoint.size_bytes)}</td>
            </tr>
            <tr>
              <th>world_size</th>
              <td>{checkpoint.world_size ?? "—"}</td>
            </tr>
            <tr>
              <th>saved</th>
              <td>
                {checkpoint.timestamp ??
                  (checkpoint.manifest_present ? "—" : "no manifest")}
              </td>
            </tr>
            <tr>
              <th>path</th>
              <td>
                <code className="path-code">{checkpoint.checkpoint_dir}</code>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {evaluating && (
        <EvalModal
          modelOutputDir={output_dir}
          modelName={modelName}
          checkpointPath={checkpoint.checkpoint_dir}
          projectDir={project.project_dir}
          onClose={() => setEvaluating(false)}
        />
      )}
      {serving && (
        <InferenceModal
          modelOutputDir={output_dir}
          modelName={modelName}
          checkpointPath={checkpoint.checkpoint_dir}
          projectDir={project.project_dir}
          onClose={() => setServing(false)}
        />
      )}
    </div>
  );
}
