/** Compact right-sidebar agent panel. The always-on conversational entry
 *  point; large diffs defer to the full Agent view via onOpenFull. Shares
 *  one AgentController with the full view. */

import { AgentController } from "../useAgent";
import { AgentComposer } from "./AgentComposer";
import { AgentThread } from "./AgentThread";

interface Props {
  agent: AgentController;
  onOpenFull: () => void;
  onOpenSettings: () => void;
  onCollapse: () => void;
}

export function AgentSidebarPanel({ agent, onOpenFull, onOpenSettings, onCollapse }: Props) {
  return (
    <div className="agent-sidebar-content">
      <header className="agent-sidebar-header">
        <span className="agent-sidebar-title">Agent</span>
        {agent.profiles.length > 0 ? (
          <select
            className="agent-profile-switch"
            value={agent.activeProfileId ?? ""}
            onChange={(e) => agent.activate(e.target.value)}
            title="Active profile"
          >
            {agent.profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        ) : (
          agent.status?.model && (
            <span className="agent-model-badge" title={agent.status.base_url || "Claude"}>
              {agent.status.model}
            </span>
          )
        )}
        <span className="agent-sidebar-spacer" />
        <button className="btn-icon" title="Agent profiles…" onClick={onOpenSettings}>
          ⚙
        </button>
        <button className="btn-icon" title="Open full Agent view" onClick={onOpenFull}>
          ⤢
        </button>
        <button className="btn-icon" title="New conversation" onClick={agent.reset}>
          ✎
        </button>
        <button className="btn-icon" title="Collapse" onClick={onCollapse}>
          ›
        </button>
      </header>
      <AgentThread agent={agent} compact onOpenFull={onOpenFull} />
      <AgentComposer agent={agent} />
    </div>
  );
}
