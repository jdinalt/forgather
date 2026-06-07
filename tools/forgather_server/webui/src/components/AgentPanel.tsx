/** Full main-canvas Agent view. Same AgentController as the sidebar, with
 *  room for side-by-side diffs (Monaco) and longer history. */

import { useRef } from "react";

import { AgentController } from "../useAgent";
import { AgentComposer } from "./AgentComposer";
import { AgentContextMeter } from "./AgentContextMeter";
import { AgentThread } from "./AgentThread";

function downloadJson(obj: unknown, name: string) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

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
  const importRef = useRef<HTMLInputElement | null>(null);

  const onExport = async () => {
    const dump = await agent.dumpConversation();
    const ts = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .replace("T", "_")
      .slice(0, 19);
    downloadJson(dump, `forgather-agent-${ts}.json`);
  };

  const onImportFile = async (file: File) => {
    try {
      const data = JSON.parse(await file.text());
      await agent.loadConversation(data);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("failed to import conversation", e);
    }
  };

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
        <AgentContextMeter
          usage={agent.usage}
          sessionCost={agent.sessionCost}
          pricing={agent.status?.pricing}
        />
        <span className="agent-full-spacer" />
        {agent.awaiting && <span className="agent-awaiting-badge">awaiting approval</span>}
        <button className="btn-secondary" onClick={onExport} title="Download this conversation as JSON">
          Export
        </button>
        <button
          className="btn-secondary"
          onClick={() => importRef.current?.click()}
          title="Load a conversation from a JSON file"
        >
          Import
        </button>
        <input
          ref={importRef}
          type="file"
          accept="application/json,.json"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void onImportFile(f);
            e.target.value = ""; // allow re-importing the same file
          }}
        />
        <button className="btn-secondary" onClick={onOpenSettings}>
          Profiles…
        </button>
        <button className="btn-secondary" onClick={agent.newConversation}>
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
