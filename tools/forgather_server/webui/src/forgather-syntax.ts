import type * as monacoT from "monaco-editor";

export const FORGATHER_LANGUAGE_ID = "forgather-yaml";

export function registerForgatherLanguage(monaco: typeof monacoT) {
  if (monaco.languages.getLanguages().some((l) => l.id === FORGATHER_LANGUAGE_ID)) {
    return;
  }

  monaco.languages.register({ id: FORGATHER_LANGUAGE_ID, extensions: [".yaml"] });

  monaco.languages.setLanguageConfiguration(FORGATHER_LANGUAGE_ID, {
    comments: { lineComment: "#" },
    brackets: [
      ["{", "}"],
      ["[", "]"],
    ],
    autoClosingPairs: [
      { open: "{", close: "}" },
      { open: "[", close: "]" },
      { open: '"', close: '"' },
      { open: "'", close: "'" },
    ],
  });

  // Monarch tokenizer for Forgather's YAML+Jinja2 line-statement dialect.
  monaco.languages.setMonarchTokensProvider(FORGATHER_LANGUAGE_ID, {
    defaultToken: "",
    tokenPostfix: ".forgather-yaml",

    yamlTags: [
      "!call",
      "!factory",
      "!partial",
      "!singleton",
      "!var",
      "!meta",
      "!tuple",
      "!list",
      "!dict",
      "!dlist",
      "!lambda",
    ],

    tokenizer: {
      root: [
        // Inline template markers: #--- template.name ---
        [/^#-{3,}\s+[^\s].*-{3,}\s*$/, "keyword.inline-template"],

        // Line comments (## = Jinja comment, # = YAML comment, but both render as comments)
        [/^\s*##.*$/, "comment.doc"],
        [/#.*$/, "comment"],

        // Line statements converted to {% ... %}: --, <<, >>
        [/^\s*(--|<<|>>)\s*.*$/, "keyword.control.jinja"],

        // Print statements: == expression, => expression
        [/^\s*(==|=>)\s*.*$/, "variable.jinja"],

        // TOML-style block open/close: [name]  [/name]  [name!]
        [/^\s*\[\/[^\]]+\]\s*$/, "keyword.block.end"],
        [/^\s*\[[A-Za-z_][^\]]*\]\s*$/, "keyword.block.begin"],

        // YAML tags (!call, !partial, etc.)
        [
          /![A-Za-z_][A-Za-z0-9_.:]*/,
          {
            cases: {
              "@yamlTags": "tag.forgather",
              "@default": "tag",
            },
          },
        ],

        // Anchors (&name) and aliases (*name)
        [/[&*][A-Za-z_][A-Za-z0-9_-]*/, "variable.predefined"],

        // Strings
        [/"([^"\\]|\\.)*"/, "string"],
        [/'([^'\\]|\\.)*'/, "string"],

        // Numbers
        [/\b-?\d+(\.\d+)?([eE][+-]?\d+)?\b/, "number"],

        // Booleans / null
        [/\b(true|false|null|True|False|None|yes|no)\b/, "constant.language"],

        // Mapping keys: foo:   foo_bar:
        [/^[\s-]*[A-Za-z_][A-Za-z0-9_-]*\s*:/, "type"],
      ],
    },
  });
}
