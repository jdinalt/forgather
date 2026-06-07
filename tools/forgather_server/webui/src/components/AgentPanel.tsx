/** Full main-canvas Agent view. Same AgentController as the sidebar, with
 *  room for side-by-side diffs (Monaco) and longer history. */

import { AgentController } from "../useAgent";
import { AgentComposer } from "./AgentComposer";
import { AgentThread } from "./AgentThread";

export function AgentPanel({
  agent,
  onOpenSettings,
  onOpenDoc,
  repoRoot,
}: {
  agent: AgentController;
  onOpenSettings: () => void;
  onOpenDoc?: (absPath: string) => void;
  repoRoot?: string;
}) {
  return (
    <div className="agent-full">
      <header className="agent-full-header">
        <span className="agent-full-title">Agent</span>
        {agent.status?.enabled ? (
          <span className="agent-model-badge" title={agent.status.base_url || "Claude"}>
            {agent.status.provider}/{agent.status.model ?? "(auto)"}
          </span>
        ) : (
          <span className="agent-disabled-note muted">
            No profile configured — open profiles to add one.
          </span>
        )}
        {agent.profiles.length > 0 && (
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
        )}
        <span className="agent-full-spacer" />
        {agent.awaiting && <span className="agent-awaiting-badge">awaiting approval</span>}
        <button className="btn-secondary" onClick={onOpenSettings}>
          Profiles…
        </button>
        <button className="btn-secondary" onClick={agent.reset}>
          New conversation
        </button>
      </header>
      <div className="agent-full-body">
        <AgentThread agent={agent} onOpenDoc={onOpenDoc} repoRoot={repoRoot} />
      </div>
      <AgentComposer agent={agent} />
    </div>
  );
}
