/** Compact token meters for the agent header.
 *
 *  Two distinct numbers, often confused:
 *  - **context occupancy** — how much of the model's context window the current
 *    conversation fills (latest request's prompt + output vs max_model_len).
 *  - **session billed** — the cumulative tokens billed across every API
 *    round-trip. An agentic loop re-sends the prefix each turn, so this is
 *    typically many times the occupancy and is what reconciles with the
 *    provider's billing dashboard. */

import { AgentSessionCost, AgentUsage } from "../useAgent";

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "K";
  return String(n);
}

export function AgentContextMeter({
  usage,
  sessionCost,
}: {
  usage: AgentUsage | null;
  sessionCost?: AgentSessionCost | null;
}) {
  if (!usage && !sessionCost) return null;

  // --- context occupancy (latest request) ---
  const used = usage ? (usage.inputTokens || 0) + (usage.outputTokens || 0) : 0;
  const win = usage?.contextWindow ?? null;
  const pct = win && used ? Math.min(100, Math.round((used / win) * 100)) : null;
  const cls = pct == null ? "" : pct >= 90 ? " hot" : pct >= 70 ? " warn" : "";
  const ctxTitle = usage
    ? `Context occupancy (latest request).\n` +
      `prompt ${usage.inputTokens} · output ${usage.outputTokens}` +
      (usage.cacheReadTokens ? ` · cache read ${usage.cacheReadTokens}` : "") +
      (win ? ` · window ${win}` : " · window unknown")
    : "";

  // --- cumulative billed (whole session) ---
  let billed: number | null = null;
  let billedTitle = "";
  if (sessionCost) {
    const billedIn =
      sessionCost.inputTokens +
      sessionCost.cacheReadTokens +
      sessionCost.cacheCreationTokens;
    billed = billedIn + sessionCost.outputTokens;
    const cacheable = billedIn || 1;
    const hitPct = Math.round((sessionCost.cacheReadTokens / cacheable) * 100);
    billedTitle =
      `Cumulative tokens billed this session over ${sessionCost.requests} ` +
      `request(s) — reconciles with the billing dashboard.\n` +
      `input (uncached) ${sessionCost.inputTokens} · ` +
      `cache read ${sessionCost.cacheReadTokens} · ` +
      `cache write ${sessionCost.cacheCreationTokens} · ` +
      `output ${sessionCost.outputTokens}\n` +
      `cache hit ${hitPct}% of billed input`;
  }

  return (
    <span className="agent-ctx-meter">
      {used > 0 && (
        <span className="agent-ctx-group" title={ctxTitle}>
          {win && pct != null && (
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
      )}
      {billed != null && billed > 0 && (
        <span className="agent-ctx-billed" title={billedTitle}>
          billed {fmtTokens(billed)}
        </span>
      )}
    </span>
  );
}
