/** Compact token meters for the agent header.
 *
 *  Two distinct numbers, often confused:
 *  - **context occupancy** — how much of the model's context window the current
 *    conversation fills (latest request's prompt + output vs max_model_len).
 *  - **session billed** — the cumulative tokens billed across every API
 *    round-trip. An agentic loop re-sends the prefix each turn, so this is
 *    typically many times the occupancy and is what reconciles with the
 *    provider's billing dashboard. */

import { AgentPricing } from "../agent-client";
import { AgentSessionCost, AgentUsage } from "../useAgent";

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "K";
  return String(n);
}

function fmtUsd(n: number): string {
  // Sub-cent runs are common; show enough digits to be meaningful.
  return "$" + n.toFixed(n < 1 ? 4 : 2);
}

export function AgentContextMeter({
  usage,
  sessionCost,
  pricing,
}: {
  usage: AgentUsage | null;
  sessionCost?: AgentSessionCost | null;
  pricing?: AgentPricing | null;
}) {
  if (!usage && !sessionCost) return null;

  // --- context occupancy (latest request) ---
  const used = usage ? (usage.inputTokens || 0) + (usage.outputTokens || 0) : 0;
  const win = usage?.contextWindow ?? null;
  const pct = win && used ? Math.min(100, Math.round((used / win) * 100)) : null;
  const cls = pct == null ? "" : pct >= 90 ? " hot" : pct >= 70 ? " warn" : "";
  const ctxTitle = usage
    ? `Context occupancy — LATEST request only.\n` +
      `fresh input ${usage.inputTokens} · output ${usage.outputTokens}` +
      (usage.cacheReadTokens ? ` · cache read ${usage.cacheReadTokens}` : "") +
      (usage.cacheCreationTokens ? ` · cache write ${usage.cacheCreationTokens}` : "") +
      (win ? ` · window ${win}` : " · window unknown")
    : "";

  // --- cumulative billed (whole session) ---
  let billed: number | null = null;
  let billedTitle = "";
  let hitPct: number | null = null;
  if (sessionCost) {
    const billedIn =
      sessionCost.inputTokens +
      sessionCost.cacheReadTokens +
      sessionCost.cacheCreationTokens;
    billed = billedIn + sessionCost.outputTokens;
    const cacheable = billedIn || 1;
    hitPct = Math.round((sessionCost.cacheReadTokens / cacheable) * 100);
    billedTitle =
      `Cumulative tokens billed this session over ${sessionCost.requests} ` +
      `request(s) — the loop re-sends the prefix each round-trip, so this sum is ` +
      `many times the latest-request occupancy and is what reconciles with the ` +
      `billing dashboard (which counts these four categories separately).\n` +
      `fresh input ${sessionCost.inputTokens} (1x) · ` +
      `cache read ${sessionCost.cacheReadTokens} (~0.1x) · ` +
      `cache write ${sessionCost.cacheCreationTokens} (~1.25x) · ` +
      `output ${sessionCost.outputTokens}\n` +
      `cache hit ${hitPct}% of billed input — high means the prefix is being ` +
      `reused (caching works); low means it is being re-created.`;
  }

  // --- estimated cost (cumulative tokens x per-Mtok rates) ---
  let estUsd: number | null = null;
  if (sessionCost && pricing) {
    const per = 1_000_000;
    const cIn = (sessionCost.inputTokens / per) * pricing.input;
    const cRead = (sessionCost.cacheReadTokens / per) * pricing.cache_read;
    const cWrite = (sessionCost.cacheCreationTokens / per) * pricing.cache_write;
    const cOut = (sessionCost.outputTokens / per) * pricing.output;
    estUsd = cIn + cRead + cWrite + cOut;
    billedTitle +=
      `\nestimated cost ${fmtUsd(estUsd)} ` +
      `(input ${fmtUsd(cIn)} · cache read ${fmtUsd(cRead)} · ` +
      `cache write ${fmtUsd(cWrite)} · output ${fmtUsd(cOut)}) — ESTIMATE from a ` +
      `built-in price table; the billing dashboard is authoritative.`;
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
          {hitPct != null ? ` · ${hitPct}% cached` : ""}
          {estUsd != null ? ` · ~${fmtUsd(estUsd)}` : ""}
        </span>
      )}
    </span>
  );
}
