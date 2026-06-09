/** One proposed action (propose/confirm) awaiting — or past — a decision.
 *
 *  Full view: a Monaco DiffEditor (before vs after). Compact (sidebar):
 *  a one-line change summary plus a "Review in Agent view" affordance, so
 *  a large diff never tries to render inside the narrow sidebar.
 */

import { useEffect, useRef, useState } from "react";

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
  onReject: (reason?: string) => void;
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

const MAX_ARG_VALUE_CHARS = 800;

/** Render one key's value. Objects/arrays pretty-print as JSON (a bare
 *  ``String(obj)`` yields a useless ``[object Object]``); an explicit empty
 *  string shows as ``""`` so a passed-but-blank arg stays visible. When
 *  ``truncate`` is set (the verbatim proposed-args block), an over-long value —
 *  e.g. a multi-thousand-char prompt passed to ``query_model`` — is clipped so
 *  one argument can't blow up the card. The curated ``extra`` block passes
 *  short, tool-chosen values and renders them in full. */
function ValueCell({ value, truncate }: { value: unknown; truncate?: boolean }) {
  const isObj = value !== null && typeof value === "object";
  const raw = isObj
    ? JSON.stringify(value, null, 2)
    : value === ""
      ? '""'
      : String(value);
  if (truncate && raw.length > MAX_ARG_VALUE_CHARS) {
    const hidden = raw.length - MAX_ARG_VALUE_CHARS;
    return (
      <dd>
        <pre className="agent-action-extra-json">
          {raw.slice(0, MAX_ARG_VALUE_CHARS)}
          {`\n… (+${hidden} more characters)`}
        </pre>
      </dd>
    );
  }
  return isObj ? (
    <dd>
      <pre className="agent-action-extra-json">{raw}</pre>
    </dd>
  ) : (
    <dd>{raw}</dd>
  );
}

/** Render an ordered list of key/value entries. Shared by the curated
 *  ``extra`` block and the verbatim ``proposed_args`` block (pass ``truncate``
 *  for the latter so a huge single argument is clipped). */
function KeyValueList({
  entries,
  truncate,
}: {
  entries: [string, unknown][];
  truncate?: boolean;
}) {
  return (
    <dl className="agent-action-extra">
      {entries.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <ValueCell value={v} truncate={truncate} />
        </div>
      ))}
    </dl>
  );
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
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  // The reject form (reason box + Confirm/Cancel) opens at the bottom of the
  // card, which on a tall card lands below the thread's clip edge — hidden
  // behind the composer with no hint to scroll. Pull it into view so the
  // controls are reachable the instant Reject is clicked.
  const rejectRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (rejecting) {
      rejectRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
    }
  }, [rejecting]);
  const hasDiff = card.before != null || card.after != null;
  const { added, removed } = lineDelta(card.before, card.after);
  const isPending = status === "pending";
  // Non-file actions (e.g. create project/workspace) carry no diff — show the
  // planned details from ``extra`` instead of an empty diff editor.
  const extraEntries = Object.entries(card.extra ?? {}).filter(
    ([, v]) => v !== null && v !== undefined && v !== "",
  );
  // The verbatim arguments the agent passed in this tool call. ``proposed_args``
  // is the raw tool-call dict, so it already contains ONLY the keys the agent
  // actually specified — keys it omitted (handing off to defaults) are absent,
  // so a tool with a large optional-arg surface never bloats this list. We do
  // NOT filter empties here (unlike the curated ``extra`` summary below): an
  // explicit null / "" the agent passed IS an input the user must see.
  const argsEntries = Object.entries(card.proposed_args ?? {});

  return (
    <div
      className={
        "agent-action-card" +
        (compact ? " compact" : "") +
        (hasDiff && !compact ? " has-diff" : "")
      }
      data-status={status}
    >
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

      {!hasDiff && extraEntries.length > 0 && <KeyValueList entries={extraEntries} />}

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
              // Reclaim per-side width for the actual text: drop the unused
              // glyph margin and keep the line-number column tight.
              glyphMargin: false,
              lineNumbersMinChars: 3,
              // The card width is now responsive (breaks out of the reading
              // band to fill the thread), so let Monaco re-measure on resize.
              automaticLayout: true,
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

      {argsEntries.length > 0 && (
        // Open by default for non-diff actions (where args ARE the change the
        // user is approving); collapsed for diff cards, where the diff already
        // shows the change and the args would just be noise.
        <details className="agent-action-args" open={!hasDiff}>
          <summary>Agent-proposed arguments ({argsEntries.length})</summary>
          <KeyValueList entries={argsEntries} truncate />
        </details>
      )}

      {isPending ? (
        rejecting ? (
          <div className="agent-action-reject" ref={rejectRef}>
            <textarea
              className="agent-reject-reason"
              placeholder="Optional: why? (the agent uses this to adapt — e.g. 'use config Y instead')"
              value={reason}
              autoFocus
              rows={2}
              disabled={busy}
              onChange={(e) => setReason(e.target.value)}
            />
            <div className="agent-action-buttons">
              <button
                className="btn-reject"
                disabled={busy}
                onClick={() => onReject(reason)}
              >
                Confirm reject
              </button>
              <button
                className="btn-link"
                disabled={busy}
                onClick={() => {
                  setRejecting(false);
                  setReason("");
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="agent-action-buttons">
            <button className="btn-approve" disabled={busy} onClick={onApprove}>
              Approve
            </button>
            <button
              className="btn-reject"
              disabled={busy}
              onClick={() => setRejecting(true)}
            >
              Reject
            </button>
            {compact && onOpenFull && (
              <button className="btn-link" onClick={onOpenFull}>
                Review in Agent view →
              </button>
            )}
          </div>
        )
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
