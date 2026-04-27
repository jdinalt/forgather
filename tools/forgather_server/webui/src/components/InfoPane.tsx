import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { api } from "../api";

interface Props {
  project_dir: string;
  enabled: boolean;
}

export function InfoPane({ project_dir, enabled }: Props) {
  const readmeQ = useQuery({
    queryKey: ["project-readme", project_dir],
    queryFn: () => api.projectReadme(project_dir),
    enabled,
    retry: false,
  });

  if (!enabled) return null;

  if (readmeQ.isLoading) {
    return <div className="pane-state">Loading...</div>;
  }

  if (readmeQ.isError) {
    const msg = String(readmeQ.error);
    if (msg.includes("404")) {
      return (
        <div className="pane-state muted">This project has no README.md.</div>
      );
    }
    return (
      <div className="pane-state err">
        <pre>{msg}</pre>
      </div>
    );
  }

  const components: Components = {
    img({ src, alt, ...rest }) {
      if (!src) return null;
      const isAbsolute =
        src.startsWith("http://") ||
        src.startsWith("https://") ||
        src.startsWith("//") ||
        src.startsWith("data:");
      const resolvedSrc = isAbsolute
        ? src
        : api.projectAssetUrl(
            project_dir,
            src.startsWith("./") ? src.slice(2) : src,
          );
      return <img src={resolvedSrc} alt={alt ?? ""} {...rest} />;
    },
    a({ href, children, ...rest }) {
      if (href && !href.startsWith("#")) {
        return (
          <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
            {children}
          </a>
        );
      }
      return (
        <a href={href} {...rest}>
          {children}
        </a>
      );
    },
  };

  // Single inner content wrapper: that's where max-width + centering
  // live. Everything inside flows at the wrapper's full width so headers,
  // paragraphs, hr, lists, tables all share the same left/right edges
  // (instead of each centering itself independently with its own
  // max-width, which produced the inconsistent-spacing rendering).
  return (
    <div className="info-pane">
      <div className="info-pane-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {readmeQ.data ?? ""}
        </ReactMarkdown>
      </div>
    </div>
  );
}
