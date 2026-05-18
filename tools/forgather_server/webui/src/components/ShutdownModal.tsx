import { useState } from "react";
import { api } from "../api";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  onClose: () => void;
  // Called after the shutdown request was accepted by the server, so the
  // caller can swap the UI into a "server going down" state.
  onShutdownStarted: () => void;
  // Optional: target a specific cluster node. When omitted, defaults to
  // shutting down the local server. ``nodeLabel`` is shown in the modal
  // title/body so the operator can tell which node they're acting on.
  nodeId?: string;
  nodeLabel?: string;
}

// Two-button confirmation: shut the server down on its own, or first
// SIGTERM every running job (training, inference, dataset_server, …)
// and then shut down. Cancel just dismisses.
export function ShutdownModal({
  onClose,
  onShutdownStarted,
  nodeId,
  nodeLabel,
}: Props) {
  const [busy, setBusy] = useState<"plain" | "with-jobs" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async (stopJobs: boolean) => {
    setBusy(stopJobs ? "with-jobs" : "plain");
    setError(null);
    try {
      if (nodeId) {
        await api.shutdownNode(nodeId, { stopJobs });
      } else {
        await api.shutdownServer({ stopJobs });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(null);
      return;
    }
    onShutdownStarted();
  };

  const targetLabel = nodeLabel ?? "forgather server";
  const disconnectNotice = nodeId
    ? `The node will drop off the cluster once the process exits.`
    : `The webui will disconnect once the process exits.`;

  return (
    <ModalBackdrop onClose={busy ? () => {} : onClose}>
      <div
        className="modal shutdown-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label={`Shutdown ${targetLabel}`}
      >
        <header className="modal-header">
          <h3>Shutdown {targetLabel}</h3>
          <button
            className="tiny"
            onClick={onClose}
            disabled={!!busy}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        <div className="modal-body">
          <p>
            Choose how to shut down{" "}
            {nodeId ? <code>{targetLabel}</code> : "the server"}.{" "}
            {disconnectNotice}
          </p>
          <ul>
            <li>
              <b>Stop server only</b> leaves any running training,
              inference, and dataset servers attached to their own process
              groups. The next <code>forgather server</code> boot will
              reattach them.
            </li>
            <li>
              <b>Stop all jobs and shutdown</b> sends SIGTERM to every
              running job's process group before exiting.
              <div className="notice notice-warn">
                Warning: this will terminate any running training jobs.
                In-flight steps that haven't checkpointed will be lost.
              </div>
            </li>
          </ul>

          {error && (
            <div className="err pad">
              <pre>{error}</pre>
            </div>
          )}
        </div>

        <footer className="modal-footer">
          <div className="btn-row">
            <button
              className="secondary"
              onClick={onClose}
              disabled={!!busy}
            >
              Cancel
            </button>
          </div>
          <div className="btn-row">
            <button
              onClick={() => submit(false)}
              disabled={!!busy}
              title="Exit the server process; leave running jobs alone"
            >
              {busy === "plain" ? "Shutting down…" : "Stop server only"}
            </button>
            <button
              className="destructive"
              onClick={() => submit(true)}
              disabled={!!busy}
              title="SIGTERM every running job, then exit the server"
            >
              {busy === "with-jobs"
                ? "Stopping jobs…"
                : "Stop all jobs and shutdown"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
