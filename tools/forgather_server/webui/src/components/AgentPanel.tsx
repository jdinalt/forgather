/** Full main-canvas Agent view. Same AgentController as the sidebar, with
 *  room for side-by-side diffs (Monaco) and longer history. */

import { AgentController } from "../useAgent";
import { AgentComposer } from "./AgentComposer";
import { AgentThread } from "./AgentThread";

export function AgentPanel({ agent }: { agent: AgentController }) {
  return (
    <div className="agent-full">
      <header className="agent-full-header">
        <span className="agent-full-title">Agent</span>
        {agent.status?.enabled ? (
          <span className="agent-model-badge" title={agent.status.base_url || "Claude"}>
            {agent.status.provider}/{agent.status.model}
          </span>
        ) : (
          <span className="agent-disabled-note muted">
            Not configured — set <code>agent.model</code> in the server config.
          </span>
        )}
        <span className="agent-full-spacer" />
        {agent.awaiting && <span className="agent-awaiting-badge">awaiting approval</span>}
        <button className="btn-secondary" onClick={agent.reset}>
          New conversation
        </button>
      </header>
      <div className="agent-full-body">
        <AgentThread agent={agent} />
      </div>
      <AgentComposer agent={agent} />
    </div>
  );
}
