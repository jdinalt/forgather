# Serving the Forgather Docs with MkDocs

The Forgather repository ships an `mkdocs.yml` at the repo root that renders the
contents of `docs/` as a static site with the Material theme, search, and a
nav tree. This guide covers how to serve those docs locally with live-reload,
either from the webui's "Services -> MkDocs…" entry or from the CLI.

The bundled config (`mkdocs.yml`) sets `docs_dir: docs` and pins the Material
theme, the `search` / `mkdocs-jupyter` / `mkdocstrings` plugins, and a nav
tree covering Getting Started, Configuration, Training, Checkpointing,
Datasets, API Reference, Tutorials, and Development. None of that needs
editing to serve the docs -- pointing `mkdocs serve` at the file is enough.

## Why use this over the webui's built-in Docs view

The webui's Docs view renders `docs/` directly and works offline / on remote
servers without exposing an extra port. It is the right default for casual
browsing.

MkDocs is the prettier rendered version: full-text search, theme styling,
permalink anchors, embedded notebooks via `mkdocs-jupyter`, and rendered API
docs via `mkdocstrings`. The cost is that it runs as a separate HTTP server
on its own port -- launch it when you want the polished view or are
authoring documentation; the built-in view is fine the rest of the time.

## Launch from the webui

Sidebar menu: **Services -> MkDocs…** opens a modal that enqueues an
`mkdocs serve` job through the scheduler. The defaults are picked from the
modal's persisted state, with the following first-run values
(see `tools/forgather_server/webui/src/components/MkDocsModal.tsx`):

| Field | Default | Notes |
|-------|---------|-------|
| `mkdocs.yml` | bundled `<repo>/mkdocs.yml` | Backfilled from the server's "Forgather repo" quick path. |
| Host | `localhost` | Loopback. Some browsers only follow clickable links to `localhost`, not `127.0.0.1`. |
| Port | `8000` | MkDocs' own default. Each concurrent MkDocs job needs a distinct port. |
| Priority | `0` | No GPUs are reserved either way. |
| `--strict` | off | Treat warnings as errors. Useful while authoring; noisy when just serving. |
| live reload | on | Maps to `--no-livereload` on the CLI when unchecked. |
| `--dirty` | off | Only rebuild changed files (faster, but cross-file links may go stale). |
| Extra watch dirs | empty | Comma-separated list, forwarded as repeated `--watch <path>` flags. |

Submitting enqueues a `job_type: "mkdocs"` job with the chosen flags. The
running job lands in the Jobs view with a clickable URL on the card
(`http://<host>:<port>`) -- the same scheme the TensorBoard and Inference
job cards use. The job is CPU-only (`requested_gpus: 0`) and long-lived
until killed.

## Launch from the CLI

Same scheduler, no browser:

```bash
# Run mkdocs serve in the foreground (or background via shell).
forgather mkdocs -f mkdocs.yml

# Submit the same thing to the running forgather server's queue.
forgather mkdocs -f mkdocs.yml --enqueue --priority 0
```

See `tools/forgather_server/README.md` for the full enqueue / scheduler
reference. The argv that ends up on the spawned process is built by
`tools/forgather_server/mkdocs_ops.py:build_mkdocs_command` -- it folds
`host` and `port` into MkDocs' single `--dev-addr host:port` flag, and
passes `--strict`, `--no-livereload`, `--dirty`, and `--watch <path>` per
the modal toggles.

## Jobs view behaviour

- One row per `mkdocs serve` instance, labelled `mkdocs:<port>`.
- URL on the card is clickable while the job is alive.
- CPU-only: launches even when every GPU is in use.
- Long-lived: it will not exit on its own; kill it from the Jobs view (or
  `forgather job kill <id>`) when done.
- Concurrent instances are fine as long as their ports differ.

## Common gotchas

- **First request is slow.** MkDocs scans the entire `docs/` tree on
  startup -- which is a fan-out of symlinks pointing back into the repo,
  not a flat directory -- so the first page load can take a few seconds
  while the build completes. Subsequent edits are incremental.
- **Live-reload only watches `docs_dir`.** Edits to `docs/*.md` show up
  immediately. Docs that are symlinked from outside `docs/` (for
  example `docs/tools/dataset_server/README.md`, which points to
  `tools/dataset_server/README.md`) are still served, but MkDocs will not
  pick up edits to the real file unless you pass `--watch <real_path>`
  -- either via the modal's "Extra watch dirs" field or `-W` /
  `--watch` on the CLI.
- **Host defaults to loopback.** There is no `--bind_all` shortcut here;
  if you need LAN access set the host explicitly (e.g. `0.0.0.0` or a
  specific interface address). The standard loopback caveats apply --
  only do this on a trusted network.
- **`--strict` turns broken links into errors.** Handy while authoring
  to catch dangling refs; noisy if you only want to read the docs and
  the tree is mid-edit.
- **`--dirty` skips full rebuilds.** Faster reloads, but cross-file
  changes (nav, new pages, link targets) can render stale. Toggle it
  off if something looks wrong.
- **Port already in use.** If another MkDocs (or anything else) holds
  the port, the job exits immediately -- check its TTY log via the
  Jobs view, then resubmit with a free port.

## See also

- [Forgather server overview](../forgather-server.md) -- the broader
  server / scheduler / Jobs view architecture that hosts the MkDocs
  job.
- [TensorBoard](tensorboard.md) -- the other long-lived viewer
  spawned from the Services menu; same lifecycle and auth-gating
  model.
- [`docs/README.md`](../README.md) -- table of contents for the doc
  tree this site renders.
