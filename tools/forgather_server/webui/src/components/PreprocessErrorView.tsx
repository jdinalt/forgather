import { PreprocessErrorDetail } from "../api";

/** Renders a Jinja2 preprocessing error in a compiler-style block:
 *
 *      <template>:<line>: <message>
 *      <line-numbered source excerpt with caret on the offending line>
 *
 *  Used both by the **pp** panel (when /api/config/pp returns 400 with a
 *  PreprocessErrorDetail) and the **debug** panel (when /api/config/debug
 *  returns the same shape). */
export function PreprocessErrorView({ err }: { err: PreprocessErrorDetail }) {
  const loc = err.template ?? "<config>";
  const header = err.lineno != null ? `${loc}:${err.lineno}` : loc;
  return (
    <div className="pp-error">
      <div className="pp-error-header">
        <span className="pp-error-kind">PreprocessError</span>
        <code className="pp-error-loc">{header}</code>
      </div>
      <pre className="pp-error-message">{err.message}</pre>
      {err.source_context && (
        <pre className="pp-error-context">{err.source_context}</pre>
      )}
    </div>
  );
}
