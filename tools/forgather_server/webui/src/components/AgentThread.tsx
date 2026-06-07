/** Renders the agent conversation: user/assistant text, tool activity,
 *  error notices, and action cards. Shared by the sidebar (compact) and the
 *  full Agent view. */

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { AgentController } from "../useAgent";
import { AgentActionCard } from "./AgentActionCard";

interface Props {
  agent: AgentController;
  compact?: boolean;
  onOpenFull?: () => void;
}

export function AgentThread({ agent, compact, onOpenFull }: Props) {
  const endRef = useRef<HTMLDivElement | null>(null);

  // Autoscroll to the latest content as the stream grows.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [agent.items]);

  return (
    <div className={"agent-thread" + (compact ? " compact" : "")}>
      {agent.items.length === 0 && (
        <div className="agent-empty muted">
          Ask about a project or config, search the docs, or request a change.
          Proposed changes are shown as a diff for you to approve.
        </div>
      )}
      {agent.items.map((it) => {
        if (it.type === "user") {
          return (
            <div key={it.id} className="agent-msg user">
              {it.text}
            </div>
          );
        }
        if (it.type === "assistant") {
          return (
            <div key={it.id} className="agent-msg assistant">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{it.text}</ReactMarkdown>
            </div>
          );
        }
        if (it.type === "tool") {
          return (
            <details key={it.id} className={"agent-tool" + (it.isError ? " error" : "")}>
              <summary>
                <span className="agent-tool-name">{it.name}</span>
                {it.isError && <span className="agent-tool-badge">error</span>}
              </summary>
              <pre className="agent-tool-input">{JSON.stringify(it.input, null, 2)}</pre>
              {it.content !== undefined && (
                <pre className="agent-tool-output">{it.content}</pre>
              )}
            </details>
          );
        }
        if (it.type === "action") {
          return (
            <AgentActionCard
              key={it.id}
              card={it.card}
              status={it.status}
              result={it.result}
              compact={compact}
              busy={agent.busy}
              onApprove={() => agent.decide(it.card.action_id, true)}
              onReject={() => agent.decide(it.card.action_id, false)}
              onOpenFull={onOpenFull}
            />
          );
        }
        return (
          <div key={it.id} className="agent-msg error">
            {it.message}
          </div>
        );
      })}
      {agent.busy && <div className="agent-typing muted">…</div>}
      <div ref={endRef} />
    </div>
  );
}
