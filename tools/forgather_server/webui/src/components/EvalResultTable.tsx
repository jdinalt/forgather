import { EvalResultData } from "../api";
import { formatSummaryValue } from "./RunSummaryView";

const EVAL_RESULT_KEY_ORDER = [
  "eval_name",
  "config_name",
  "description",
  "timestamp",
  "eval_loss",
  "perplexity",
  "wall_time_s",
  "model_path",
  "checkpoint_path",
  "dataset_proj",
  "dataset_config",
  "dataset_target",
  "batch_size",
  "max_length",
  "stride",
  "dtype",
  "attn_implementation",
  "trainer",
  "world_size",
];

export function EvalResultTable({ result }: { result: EvalResultData }) {
  return (
    <table className="summary-table eval-result-table">
      <tbody>
        {EVAL_RESULT_KEY_ORDER.map((key) => {
          const value = (result as unknown as Record<string, unknown>)[key];
          return (
            <tr key={key}>
              <th>{key}</th>
              <td>{formatSummaryValue(value)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
