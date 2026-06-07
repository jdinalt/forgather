/** One proposed action (propose/confirm) awaiting — or past — a decision.
 *
 *  Full view: a Monaco DiffEditor (before vs after). Compact (sidebar):
 *  a one-line change summary plus a "Review in Agent view" affordance, so
 *  a large diff never tries to render inside the narrow sidebar.
 */

import { DiffEditor } from "@monaco-editor/react";

import { ActionCard } from "../agent-client";

export type ActionStatus = "pending" | "approved" | "rejected" | "error";

interface Props {
  card: ActionCard;
  status: ActionStatus;
  result?: string;
  compact?: boolean;
  busy?: boolean;
  onApprove: () => void;
  onReject: () => void;
  onOpenFull?: () => void;
}

/** Multiset line delta for the summary chip: lines in `after` not matched in
 *  `before` are additions, and vice versa. Two passes, each consuming the
 *  other side's counts. */
function lineDelta(before: string | null, after: string | null): { added: number; removed: number } {
  const b = (before ?? "").split("\n");
  const a = (after ?? "").split("\n");

  const bCount = new Map<string, number>();
  for (const l of b) bCount.set(l, (bCount.get(l) ?? 0) + 1);
  let added = 0;
  for (const l of a) {
    const c = bCount.get(l) ?? 0;
    if (c > 0) bCount.set(l, c - 1);
    else added += 1;
  }

  const aCount = new Map<string, number>();
  for (const l of a) aCount.set(l, (aCount.get(l) ?? 0) + 1);
  let removed = 0;
  for (const l of b) {
    const c = aCount.get(l) ?? 0;
    if (c > 0) aCount.set(l, c - 1);
    else removed += 1;
  }

  return { added, removed };
}

export function AgentActionCard({
  card,
  status,
  result,
  compact,
  busy,
  onApprove,
  onReject,
  onOpenFull,
}: Props) {
  const hasDiff = card.before != null || card.after != null;
  const { added, removed } = lineDelta(card.before, card.after);
  const isPending = status === "pending";
  // Non-file actions (e.g. create project/workspace) carry no diff — show the
  // planned details from ``extra`` instead of an empty diff editor.
  const extraEntries = Object.entries(card.extra ?? {}).filter(
    ([, v]) => v !== null && v !== undefined && v !== "",
  );

  return (
    <div className={"agent-action-card" + (compact ? " compact" : "")} data-status={status}>
      <div className="agent-action-head">
        <span className={"agent-risk-chip risk-" + card.risk}>{card.risk}</span>
        <span className="agent-action-title">{card.title}</span>
      </div>
      {card.summary && <div className="agent-action-summary">{card.summary}</div>}
      {card.path && <div className="agent-action-path">{card.path}</div>}

      {hasDiff && (
        <div className="agent-diff-stat">
          <span className="diff-added">+{added}</span> <span className="diff-removed">−{removed}</span>
        </div>
      )}

      {!hasDiff && extraEntries.length > 0 && (
        <dl className="agent-action-extra">
          {extraEntries.map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{String(v)}</dd>
            </div>
          ))}
        </dl>
      )}

      {hasDiff && !compact && isPending && (
        <div className="agent-diff-editor">
          <DiffEditor
            height="320px"
            language="yaml"
            original={card.before ?? ""}
            modified={card.after ?? ""}
            options={{
              readOnly: true,
              renderSideBySide: true,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              fontSize: 12,
            }}
          />
        </div>
      )}

      {!compact && card.pp_preview && (
        <details className="agent-pp-preview">
          <summary>Preprocessor output</summary>
          <pre>{card.pp_preview}</pre>
        </details>
      )}

      {isPending ? (
        <div className="agent-action-buttons">
          <button className="btn-approve" disabled={busy} onClick={onApprove}>
            Approve
          </button>
          <button className="btn-reject" disabled={busy} onClick={onReject}>
            Reject
          </button>
          {compact && onOpenFull && (
            <button className="btn-link" onClick={onOpenFull}>
              Review in Agent view →
            </button>
          )}
        </div>
      ) : (
        <div className={"agent-action-resolved status-" + status}>
          {status === "approved" && "Approved"}
          {status === "rejected" && "Rejected"}
          {status === "error" && "Failed"}
          {result && <div className="agent-action-result">{result}</div>}
        </div>
      )}
    </div>
  );
}
