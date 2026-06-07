/** Message input shared by the sidebar and full Agent view. Enter sends,
 *  Shift+Enter inserts a newline. Shows Stop while a turn is streaming. */

import { useState } from "react";

import { AgentController } from "../useAgent";

export function AgentComposer({ agent }: { agent: AgentController }) {
  const [text, setText] = useState("");
  const disabled = !agent.status?.enabled;

  const submit = () => {
    if (!text.trim() || agent.busy) return;
    agent.send(text);
    setText("");
  };

  return (
    <div className="agent-composer">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder={
          disabled
            ? "Agent not configured — set agent.model in server config"
            : "Message the Forgather agent…"
        }
        disabled={disabled}
        rows={3}
      />
      <div className="agent-composer-buttons">
        {agent.busy ? (
          <button className="btn-stop" onClick={agent.stop}>
            Stop
          </button>
        ) : (
          <button className="btn-send" onClick={submit} disabled={disabled || !text.trim()}>
            Send
          </button>
        )}
      </div>
    </div>
  );
}
