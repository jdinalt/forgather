import { ConfigErrorDetail, ConfigErrorKind } from "../api";

const KIND_LABEL: Record<ConfigErrorKind, string> = {
  preprocess_error: "PreprocessError",
  yaml_error: "YAMLParseError",
  code_error: "CodeGenError",
};

/** Renders a structured Forgather config error in a compiler-style block:
 *
 *      <KIND>  <template>:<line>
 *      <message>
 *      <line-numbered source excerpt with caret on the offending line>
 *
 *  Used by every panel that consumes a Config endpoint and may surface a
 *  structured 400 (`pp`, `code`, `debug`). */
export function ConfigErrorView({ err }: { err: ConfigErrorDetail }) {
  const loc = err.template ?? "<config>";
  const header = err.lineno != null ? `${loc}:${err.lineno}` : loc;
  return (
    <div className="pp-error">
      <div className="pp-error-header">
        <span className="pp-error-kind">{KIND_LABEL[err.kind]}</span>
        <code className="pp-error-loc">{header}</code>
      </div>
      <pre className="pp-error-message">{err.message}</pre>
      {err.source_context && (
        <pre className="pp-error-context">{err.source_context}</pre>
      )}
    </div>
  );
}
