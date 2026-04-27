import { FORGATHER_LANGUAGE_ID } from "./forgather-syntax";

function basenameOf(path: string): string {
  const slash = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return slash >= 0 ? path.slice(slash + 1) : path;
}

function extensionOf(path: string): string {
  const base = basenameOf(path);
  const dot = base.lastIndexOf(".");
  if (dot <= 0) return "";
  return base.slice(dot).toLowerCase();
}

/** Map a file path to a Monaco language id.
 *
 *  Every file in the sidebar Files tree is click-to-open; the backend
 *  refuses truly binary files via a null-byte / UTF-8 check, so the
 *  job here is to pick the best built-in tokenizer for whatever text
 *  we end up loading.
 *
 *  Strategy:
 *    1. Forgather's YAML+Jinja2 dialect wins for ``.yaml`` / ``.yml``
 *       / ``.jinja`` / ``.jinja2`` — that's the whole reason we
 *       registered a custom Monarch tokenizer.
 *    2. Filenames recognised by basename (``Dockerfile``,
 *       ``Containerfile``) get their language regardless of extension.
 *    3. Everything else routes by extension to a Monaco built-in.
 *       Monaco's ``basic-languages`` package ships ~80 tokenizers
 *       which are auto-registered on import, so this is mostly a
 *       lookup table — no per-language registration needed on our
 *       side.
 *    4. Unrecognised text falls back to ``"plaintext"`` (logs,
 *       READMEs without an extension, ``LICENSE``, ``Makefile``
 *       — the latter has no built-in tokenizer in Monaco). The file
 *       still opens; it just doesn't get syntax highlighting. */
export function languageFor(path: string): string {
  const base = basenameOf(path);

  // Basename-keyed special cases: extensionless or non-extension-driven
  // files where the conventional name is the whole identifier.
  switch (base) {
    case "Dockerfile":
    case "Containerfile":
      return "dockerfile";
  }

  const ext = extensionOf(path);
  switch (ext) {
    // Forgather custom YAML + Jinja2 dialect.
    case ".yaml":
    case ".yml":
    case ".jinja":
    case ".jinja2":
      return FORGATHER_LANGUAGE_ID;

    // Markdown / docs.
    case ".md":
    case ".markdown":
    case ".mdx":
      return "markdown";
    case ".rst":
      return "restructuredtext";

    // Plain text.
    case ".txt":
    case ".log":
      return "plaintext";

    // JSON family.
    case ".json":
    case ".jsonc":
    case ".json5":
      return "json";

    // Web frontend.
    case ".js":
    case ".mjs":
    case ".cjs":
    case ".jsx":
      return "javascript";
    case ".ts":
    case ".tsx":
    case ".cts":
    case ".mts":
      return "typescript";
    case ".html":
    case ".htm":
    case ".xhtml":
      return "html";
    case ".css":
      return "css";
    case ".scss":
      return "scss";
    case ".less":
      return "less";
    case ".xml":
    case ".xsd":
    case ".xsl":
    case ".xslt":
    case ".svg":
      return "xml";
    case ".graphql":
    case ".gql":
      return "graphql";
    case ".vue":
      return "html"; // closest built-in; full Vue tokenizer isn't bundled
    case ".pug":
      return "pug";
    case ".hbs":
    case ".handlebars":
      return "handlebars";
    case ".twig":
      return "twig";
    case ".liquid":
      return "liquid";
    case ".razor":
    case ".cshtml":
      return "razor";

    // Python.
    case ".py":
    case ".pyi":
    case ".pyw":
      return "python";

    // Systems languages.
    case ".c":
    case ".h":
      return "c";
    case ".cpp":
    case ".cc":
    case ".cxx":
    case ".hpp":
    case ".hh":
    case ".hxx":
    case ".ipp":
      return "cpp";
    case ".rs":
      return "rust";
    case ".go":
      return "go";
    case ".swift":
      return "swift";
    case ".m":
    case ".mm":
      return "objective-c";
    case ".dart":
      return "dart";

    // JVM languages.
    case ".java":
      return "java";
    case ".kt":
    case ".kts":
      return "kotlin";
    case ".scala":
    case ".sc":
      return "scala";
    case ".clj":
    case ".cljs":
    case ".cljc":
    case ".edn":
      return "clojure";

    // .NET.
    case ".cs":
    case ".csx":
      return "csharp";
    case ".fs":
    case ".fsi":
    case ".fsx":
      return "fsharp";
    case ".vb":
      return "vb";

    // Scripting.
    case ".sh":
    case ".bash":
    case ".zsh":
    case ".ksh":
    case ".fish":
      return "shell";
    case ".ps1":
    case ".psm1":
    case ".psd1":
      return "powershell";
    case ".bat":
    case ".cmd":
      return "bat";
    case ".rb":
    case ".rake":
    case ".gemspec":
      return "ruby";
    case ".pl":
    case ".pm":
      return "perl";
    case ".php":
    case ".phtml":
      return "php";
    case ".lua":
      return "lua";
    case ".tcl":
      return "tcl";
    case ".r":
      return "r";
    case ".jl":
      return "julia";
    case ".dart_test":
      return "dart";

    // Functional.
    case ".ex":
    case ".exs":
      return "elixir";
    case ".erl":
    case ".hrl":
      return "erlang";
    case ".scm":
    case ".ss":
      return "scheme";

    // Data / query.
    case ".sql":
    case ".pgsql":
      return "sql";
    case ".proto":
      return "protobuf";
    case ".sparql":
    case ".rq":
      return "sparql";

    // Config / build.
    case ".ini":
    case ".cfg":
    case ".conf":
    case ".properties":
    case ".toml": // Monaco has no toml; ini is the closest visual fit
      return "ini";
    case ".dockerfile":
      return "dockerfile";
    case ".tf":
    case ".tfvars":
    case ".hcl":
      return "hcl";
    case ".bicep":
      return "bicep";
    case ".sol":
      return "sol";
    case ".cmake":
      return "cmake";

    // Misc / niche.
    case ".pas":
    case ".pp":
      return "pascal";
    case ".lex":
    case ".y":
      return "plaintext"; // not bundled
    case ".asm":
    case ".s":
      return "mips"; // closest built-in; many flavours not covered
    case ".cypher":
    case ".cyp":
      return "cypher";
    case ".sb":
      return "sb";
    case ".st":
      return "st";
    case ".sv":
    case ".svh":
    case ".v":
    case ".vh":
      return "systemverilog";

    default:
      return "plaintext";
  }
}
