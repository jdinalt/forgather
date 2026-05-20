import { ConfigInfo, EvalEntry, ProjectInfo } from "../api";
import { EvalResultTable } from "./EvalResultTable";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  evaluation: EvalEntry;
  onBack: (project: ProjectInfo, config: ConfigInfo) => void;
}

export function EvalDetailPanel({
  project,
  config,
  evaluation,
  onBack,
}: Props) {
  const result = evaluation.result;

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
            {result?.config_name ?? evaluation.eval_id}
          </span>
          {result?.timestamp && (
            <span className="muted detail-path">{result.timestamp}</span>
          )}
        </div>
        {result && (
          <div className="btn-row">
            {result.eval_loss != null && (
              <span className="muted">
                loss {result.eval_loss.toFixed(4)}
              </span>
            )}
            {result.perplexity != null && (
              <span className="muted">
                ppl {result.perplexity.toFixed(2)}
              </span>
            )}
            {result.bpb != null && (
              <span className="muted">
                bpb {result.bpb.toFixed(4)}
              </span>
            )}
          </div>
        )}
      </header>

      <div className="evaluations-list">
        {evaluation.parse_error && (
          <div className="pane-state err">
            <pre>{evaluation.parse_error}</pre>
          </div>
        )}
        {result ? (
          <EvalResultTable result={result} />
        ) : (
          !evaluation.parse_error && (
            <div className="pane-state muted">No result data available.</div>
          )
        )}
      </div>
    </div>
  );
}
