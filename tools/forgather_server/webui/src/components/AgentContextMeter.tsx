/** Compact context-occupancy meter for the agent header: how much of the
 *  model's context window the conversation is using (prompt + output vs
 *  max_model_len). Shows token counts only when the window is unknown
 *  (e.g. Claude). */

import { AgentUsage } from "../useAgent";

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "K";
  return String(n);
}

export function AgentContextMeter({ usage }: { usage: AgentUsage | null }) {
  if (!usage) return null;
  const used = (usage.inputTokens || 0) + (usage.outputTokens || 0);
  if (!used) return null;
  const win = usage.contextWindow;
  const pct = win ? Math.min(100, Math.round((used / win) * 100)) : null;
  const cls = pct == null ? "" : pct >= 90 ? " hot" : pct >= 70 ? " warn" : "";
  const title =
    `prompt ${usage.inputTokens} · output ${usage.outputTokens}` +
    (win ? ` · context ${win}` : " · context window unknown");
  return (
    <span className="agent-ctx-meter" title={title}>
      {win != null && (
        <span className="agent-ctx-bar">
          <span className={"agent-ctx-fill" + cls} style={{ width: `${pct}%` }} />
        </span>
      )}
      <span className="agent-ctx-text">
        {fmtTokens(used)}
        {win ? ` / ${fmtTokens(win)}` : ""}
        {pct != null ? ` (${pct}%)` : ""}
      </span>
    </span>
  );
}
