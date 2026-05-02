import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { api } from "../api";

interface Props {
  project_dir: string;
  enabled: boolean;
  /** When set, links to local .md / .ipynb / README files are intercepted and
   *  routed to the Docs view instead of opening in a new tab. */
  onOpenDoc?: (path: string) => void;
  /** When set, links to source / config files (.yaml, .py, etc.) open in the
   *  editor instead of streaming through the asset endpoint as a download. */
  onEditFile?: (path: string) => void;
}

const EDITABLE_SUFFIXES = [
  ".yaml", ".yml", ".py", ".jinja", ".j2",
  ".txt", ".json", ".toml", ".cfg", ".ini",
  ".sh", ".env",
];

function isEditableSource(path: string): boolean {
  const lower = path.toLowerCase();
  return EDITABLE_SUFFIXES.some((s) => lower.endsWith(s));
}

function isExternalUrl(href: string): boolean {
  return (
    href.startsWith("http://") ||
    href.startsWith("https://") ||
    href.startsWith("mailto:") ||
    href.startsWith("//") ||
    href.startsWith("data:")
  );
}

function isDocLike(path: string): boolean {
  const lower = path.toLowerCase();
  return (
    lower.endsWith(".md") ||
    lower.endsWith(".markdown") ||
    lower.endsWith(".ipynb")
  );
}

function joinAndNormalize(base: string, rel: string): string {
  const parts = (base + "/" + rel).split("/");
  const out: string[] = [];
  for (const p of parts) {
    if (p === "" || p === ".") continue;
    if (p === "..") {
      if (out.length > 0) out.pop();
      continue;
    }
    out.push(p);
  }
  return "/" + out.join("/");
}

function resolveAbsolute(projectDir: string, href: string): string | null {
  if (!href || href.startsWith("#")) return null;
  if (isExternalUrl(href)) return null;
  let clean = href;
  const hash = clean.indexOf("#");
  if (hash >= 0) clean = clean.slice(0, hash);
  const q = clean.indexOf("?");
  if (q >= 0) clean = clean.slice(0, q);
  if (clean.startsWith("/")) return clean;
  return joinAndNormalize(projectDir, clean);
}

export function InfoPane({ project_dir, enabled, onOpenDoc, onEditFile }: Props) {
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
      // In-page anchors stay in-page.
      if (href && href.startsWith("#")) {
        return (
          <a href={href} {...rest}>
            {children}
          </a>
        );
      }
      if (href && !isExternalUrl(href)) {
        const abs = resolveAbsolute(project_dir, href);
        if (abs && isDocLike(abs) && onOpenDoc) {
          // Markdown / ipynb → Docs view.
          return (
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                onOpenDoc(abs);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
        if (abs && isEditableSource(abs) && onEditFile) {
          // Source / config file → open in the editor.
          return (
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                onEditFile(abs);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
        if (abs && onOpenDoc) {
          // Unknown extension — most likely a directory link
          // (GitHub-style "look in this folder"). Route to Docs;
          // the backend resolves a directory to its README.md.
          return (
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                onOpenDoc(abs);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
      }
      if (href) {
        return (
          <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
            {children}
          </a>
        );
      }
      return <a {...rest}>{children}</a>;
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
