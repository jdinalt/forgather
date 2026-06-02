# TensorBoard

TensorBoard is a pure viewer over the TensorFlow event files that the
Forgather trainer writes alongside `trainer_logs.json`. Forgather does not
ship a fork or a wrapper -- the server shells out to the stock
`tensorboard` CLI, schedules it as a job like any other, and proxies the
HTTP UI through the same auth gate as the rest of the webui.

## Event files

The trainer writes TF event files into each run directory:

```
output_models/<model>/runs/<run_id>/
├── trainer_logs.json        # native Forgather log (see logs-analysis.md)
├── events.out.tfevents.*    # TensorBoard event files
└── ...
```

Point TensorBoard's `--logdir` at:

- A specific run directory to view one run.
- A model's `output_models/<model>/` to compare runs of one model.
- A project's `output_models/` (or any common parent) to compare runs
  across configs.

## Launching from the webui

### Sidebar Services menu

The sidebar **Services** group has a **📊 TensorBoard…** entry.
It opens a modal that takes an arbitrary logdir.

| Field | Default | Notes |
|-------|---------|-------|
| Log directory | empty | Path picker. Usually a model's `output_dir` or a `runs/` root. |
| Port | `6006` | TensorBoard's own default. Distinct ports for concurrent instances. |
| Priority | `0` | Scheduler priority. No GPUs are reserved. |
| `--bind_all` | off | Listen on every interface (LAN-reachable). See [Bind and proxy](#bind-and-proxy). |
| Window title | derived | Shown in the TB browser tab. Defaults to the model / project basename. |

The **Advanced options** section adds:

- **Reload interval** -- seconds between scans of the logdir for new
  events. Blank uses TB's default.
- **`--reload_multifile`** -- re-scan all event files on each reload
  instead of only the newest. Useful for long-running multi-rank runs
  where multiple event files are being written concurrently.
- **Samples per plugin** -- per-plugin sample caps, e.g.
  `images=100,scalars=500`.
- **Host** -- override the bind address (ignored when `--bind_all` is
  on). Defaults to `127.0.0.1`.

The global modal persists its last-committed settings in localStorage
under `forgather-global-tensorboard-v1`; **Reset to defaults** clears
that.

### From a config in the Projects tree

Right-clicking a config in the Projects tree (or using the config's
toolbar button) launches TensorBoard with the logdir seeded to the
config's resolved `output_dir`. The window title is seeded to the
config name. This is the fastest path while a training job is running:
right-click the config, click **TensorBoard...**, submit.

## Launching from the CLI

```bash
forgather tb --enqueue [--port 6006] [--priority N] [--server URL]
```

This enqueues a TensorBoard job through the same scheduler as the webui
modal -- the resulting job shows up in the Jobs view and is controlled
the same way. The full server CLI surface is documented in
[`tools/forgather_server/README.md`](../forgather-server.md)
and [`guides/server-cli.md`](server-cli.md).

The `forgather tb` wrapper picks sensible defaults; for arbitrary TB
flags, run the stock `tensorboard` CLI directly outside the scheduler.

## Jobs view

A queued TensorBoard job appears on the Queue / Jobs tabs with a label
like `tensorboard:6006`. Once started, the job card surfaces:

- A **clickable URL** that opens the TB UI through the auth-gated
  reverse proxy.
- A **TTY** tab streaming the `tensorboard` process output.
- A **Kill** action.

TensorBoard is long-lived: it does not exit on its own. Kill the job
from the Jobs view (or `forgather job kill <id>`) when you're done.

## Bind and proxy

Default bind is loopback (`127.0.0.1`). Browser access goes through the
forgather server's auth-gated reverse proxy at:

```
/api/tb/<queue_id>/
```

The proxy strips the prefix on inbound requests; the server passes
`--path_prefix /api/tb/<queue_id>` to TB so that its internally
generated links and asset URLs resolve correctly under the proxy mount.
This is why the URL on the job card includes `/api/tb/<queue_id>/`
rather than pointing at the raw TB port. The webui handles this
plumbing automatically -- you should not normally need to set
`--path_prefix` yourself.

Enabling `--bind_all` in the modal makes TB listen on every interface
on the chosen port. **This bypasses the proxy gate** -- anyone who can
reach that port on the host can read your training metrics. Use it only
on trusted networks (or, more typically, leave it off and rely on the
proxy + the webui's login).

WebSockets are not proxied, so the realtime profile plugin is
unavailable through `/api/tb/<queue_id>/`. Set `--bind_all` and connect
directly to the upstream port if you need it.

If both `--bind_all` and a host override are passed, `--bind_all` wins
(matching the `tensorboard` CLI's own precedence).

## Common gotchas

- **Port collisions.** Each concurrent TB instance needs a distinct
  port on the host. SSH port-forwards keyed to `6006` are common, so
  the modal sticks with that default rather than rotating it -- pick a
  fresh port per submit if you already have one running.
- **No GPUs are reserved.** TensorBoard is CPU-only; the scheduler
  enqueues it with `requested_gpus=0`. It will not block training jobs
  on the GPU pool.
- **Event files appear late.** The trainer flushes events
  periodically; immediately after `forgather train` starts, the logdir
  may be empty for a few seconds.
- **Stale logdir.** TB caches its event-file scan; if you add new run
  directories under the same logdir while TB is running, set a smaller
  reload interval or restart the job.

## See also

- [Log Analysis](logs-analysis.md) -- the Forgather-native
  `forgather logs summary` / `forgather logs plot` CLI works against
  `trainer_logs.json` and complements TB for quick offline plots and
  cross-run summaries.
- [Forgather Server](../forgather-server.md) -- auth model behind the
  `/api/tb/` proxy and the rest of the webui.
- [Server CLI](server-cli.md) -- workflow walkthrough for `forgather
  tb --enqueue`, `forgather job`, `forgather submit`, and friends.
- [MkDocs](mkdocs.md) -- the other long-lived viewer spawned from
  the Services menu; same lifecycle and auth-gating model.
