import { useEffect, useRef, useState } from "react";
import { api, Job } from "../api";

interface Props {
  job: Job;
}

/** Inline TTY pane. Subscribes to /api/jobs/{id}/tty which sends the existing
 *  backlog then follows new bytes. Auto-scrolls to the bottom unless the user
 *  has scrolled up. Only works for server-launched jobs — externally-
 *  discovered endpoints have no captured TTY on disk.
 *
 *  Implementation note: the terminal output is appended imperatively to a
 *  bare ``<pre>`` via ``appendChild(document.createTextNode(...))``. Going
 *  through React state (e.g. a ``chunks`` array joined on each render)
 *  re-creates the single text node on every new byte, which wipes any
 *  browser Selection anchored in it — making it impossible to copy log
 *  lines from a running job. Imperative append-only never touches the
 *  earlier text nodes, so selection is preserved. */
export function TtyViewer({ job }: Props) {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoscroll, setAutoscroll] = useState(true);

  const preRef = useRef<HTMLPreElement>(null);
  // Mirror of `autoscroll` read from inside the long-lived onmessage
  // closure, so the setting can update without rebuilding the WebSocket.
  const autoscrollRef = useRef(true);
  useEffect(() => {
    autoscrollRef.current = autoscroll;
  }, [autoscroll]);

  useEffect(() => {
    // Switching jobs: clear the pane, reset state, open a new socket.
    setConnected(false);
    setError(null);
    setAutoscroll(true);
    autoscrollRef.current = true;
    if (preRef.current) preRef.current.textContent = "";

    if (!job.tty_log_path) return;

    const ws = new WebSocket(api.ttyStreamUrl(job.id, true));
    ws.binaryType = "arraybuffer";
    ws.onopen = () => setConnected(true);
    ws.onmessage = (ev) => {
      let text: string;
      if (typeof ev.data === "string") {
        text = ev.data;
      } else {
        text = new TextDecoder("utf-8", { fatal: false }).decode(
          new Uint8Array(ev.data as ArrayBuffer),
        );
      }
      // Server-side error frames arrive as small JSON objects rather than
      // raw bytes. Heuristic: a tiny string shaped like {"type":"error"...}.
      if (text.length < 200 && text.startsWith("{") && text.endsWith("}")) {
        try {
          const parsed = JSON.parse(text);
          if (parsed.type === "error") {
            setError(parsed.detail ?? "error from server");
            return;
          }
        } catch {
          // not JSON — fall through to the append path
        }
      }
      const pre = preRef.current;
      if (!pre) return;
      pre.appendChild(document.createTextNode(text));
      if (autoscrollRef.current) {
        pre.scrollTop = pre.scrollHeight;
      }
    };
    ws.onerror = () => setError("WebSocket error");
    ws.onclose = () => setConnected(false);
    return () => ws.close();
  }, [job.id, job.tty_log_path]);

  const onScroll = () => {
    const el = preRef.current;
    if (!el) return;
    // Within 32 px of the bottom counts as "at bottom" — tiny rubber-band
    // over/undershoots don't flip the follow-tail indicator.
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    if (atBottom !== autoscroll) setAutoscroll(atBottom);
  };

  if (!job.tty_log_path) {
    return (
      <div className="jobs-tty-pane">
        <div className="tty-pane-meta muted">
          No TTY log available — this job was not launched by the server.
        </div>
        <pre className="tty-pre tty-pre-placeholder">
          (no TTY stream for externally-discovered endpoints)
        </pre>
      </div>
    );
  }

  return (
    <div className="jobs-tty-pane">
      <div className="tty-pane-meta muted">
        <span>
          {connected ? "● live" : "○ disconnected"}
          {" · "}
          {autoscroll ? "following tail" : "paused (scroll to bottom to resume)"}
          {" · "}
          <span className="tty-pane-id">
            {job.config ?? "(unknown config)"} · {job.id}
          </span>
        </span>
        <span className="tty-pane-paths">
          {job.logs_dir && (
            <span>
              <span className="tty-path-label">logs:</span>
              <code>{job.logs_dir}</code>
            </span>
          )}
          {job.output_dir && (
            <span>
              <span className="tty-path-label">out:</span>
              <code>{job.output_dir}</code>
            </span>
          )}
          {job.tty_log_path && (
            <span>
              <span className="tty-path-label">tty:</span>
              <code>{job.tty_log_path}</code>
            </span>
          )}
        </span>
      </div>
      {error && (
        <div className="err tty-pane-error">
          <pre>{error}</pre>
        </div>
      )}
      {/*
        No React children for this <pre>: text is appended imperatively in
        the WebSocket onmessage handler above. The :empty::before rule in
        styles.css renders the data-placeholder string when nothing has
        arrived yet.
      */}
      <pre
        ref={preRef}
        className="tty-pre"
        data-placeholder={connected ? "(no output yet)" : "connecting…"}
        onScroll={onScroll}
      />
    </div>
  );
}
