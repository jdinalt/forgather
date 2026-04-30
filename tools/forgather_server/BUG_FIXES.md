# Server code-cleanup bug fixes

Bugs identified and fixed during the `feature/server_code_cleanup` sweep.
Sourced from three parallel bug-hunt audits across `scheduler.py`,
`launcher.py`, `queue_store.py`, `job_records.py`, the routes layer,
and the ops/discovery modules. Most are correctness or resource bugs;
a few are defense-in-depth tightenings.

Out of scope for this pass: the broad CSRF / "any localhost browser
tab can cross-origin POST" exposure. The README documents the threat
model as "single-user localhost prototype, no auth" and mitigation
would require either a CSRF token, an Origin check, or a UNIX-socket
bind — a larger design call.

## High severity

### TTY file descriptor leak in `_spawn_subprocess`

**File:** `launcher.py:114-122`

Each spawned job opened the TTY log with `open(tty_log_path, "wb")`
and never closed the parent's fd. `subprocess.Popen` inherits the fd
into the child (intended), but the parent kept its own copy. With
hundreds of jobs the process eventually hits its fd limit and starts
failing to open `queue.json` / `job_records.json` writes.

Fix: wrap the `open(...)` in a `with` block so Popen's dup-then-close
contract leaves only the child's fd. Also catches a separate race in
`os.getpgid(proc.pid)` — if the child crashes immediately on exec,
the leader pid is gone before we can read its pgid; fall back to
`pid` (which equals pgid since `start_new_session=True`).

### `kill_process_group` returned False without trying when leader pid was gone

**File:** `launcher.py:517-530`

If the torchrun leader exited but worker children were still alive in
the same process group (rare but real for distributed jobs), the
helper raised `ProcessLookupError` from `os.getpgid` and returned
False without attempting `killpg`. Workers leaked.

Fix: when `getpgid` fails, fall back to `pid` as the pgid (guaranteed
equal because we spawn with `start_new_session=True`). The killpg
itself still returns False if no group survives.

### Reap-vs-abort status race could clobber "aborted" with "failed"

**File:** `scheduler.py:131-183`, `job_records.py`

If an abort (`POST /api/jobs/{id}/control/kill`) raced the reap loop's
final `update_record` call, the reap path could overwrite the
`"aborted"` status set by the abort with `"failed"` (or `"done"`)
based on the exit code it observed.

Fix: added `job_records.update_if_not_terminal(...)` — a
compare-and-swap that re-reads the record under the file-write lock
and refuses the update if the record is already terminal. Reap calls
that helper instead of plain `update_record`.

### `tty_dump` read entire log into memory

**File:** `routes/jobs.py:325-336`

A long-running training job's `tty.log` can be hundreds of MB or
multi-GB. The dump endpoint did `f.read()` with no bound, so each
click materialized the whole file in memory. Repeated clicks OOM the
server.

Fix: cap to a 32 MiB tail via `seek(size - cap)` + drop the leading
partial line. Companion constant `TTY_DUMP_MAX_BYTES`.

### TTY WebSocket backlog read was unbounded

**File:** `routes/jobs.py:339-379`

Initial connect read from offset 0 to EOF in one `f.read()` and tried
to send it as a single `ws.send_bytes`. For multi-GB logs, the
process held the entire backlog in memory before sending the first
chunk. Also blocked the asyncio event loop during the read.

Fix: read in 1 MiB chunks (`TTY_BACKLOG_CHUNK_BYTES`) and send each
chunk before reading the next. EOF detection moved into the inner
read loop.

### `dataset_ops --features` allowed argparse-injection

**File:** `dataset_ops.py:67-69`

`--features` is `nargs='*'`. Argparse stops consuming when it sees a
flag-shaped token, so an attacker-controlled feature name beginning
with `-` would inject argparse flags into the dataset CLI invocation
(e.g. `["normal_feature", "--tokenizer-path", "/evil"]`).

Fix: validate each feature name; reject empty or `-`-prefixed
strings with a clear `ValueError`.

## Medium severity

### `job_control` did not catch trainer endpoint errors

**File:** `routes/jobs.py:295-299`

When the trainer's HTTP control endpoint was unreachable (network
error, trainer hung), the exception bubbled up as a generic 500
stack trace. Sibling `job_status` already wrapped this as 502 — fix
mirrors that behavior on the control path.

### Schema-load failure silently bypassed required-field enforcement

**File:** `routes/queue.py:161-165`

`config_ops.load_dynamic_args` was wrapped in `except Exception:
schema = []`. An empty schema means the required-field check
trivially passes, so any malformed `project_dir` / `config` enqueued
without validation and only failed at execution.

Fix: surface the schema-load error as HTTP 400 with the diagnostic
message instead of silently swallowing it.

### NaN / Inf could bypass dynamic-arg numeric bounds

**File:** `routes/queue.py:182-201`

`float("NaN") > a.max` is False, `float("NaN") < a.min` is False —
NaN slipped past both checks. Inf is similarly weird (legal `>` /
`<` but rarely a sane value).

Fix: explicitly reject non-finite values via `math.isnan` /
`math.isinf` and surface them as a constraint violation.

### `generation_configs.put_preset` was not crash-atomic

**File:** `routes/generation_configs.py:149-161`

Used `tmp.write_text(...)` + `os.replace(...)` without an `fsync`
between write and replace. Codebase has `_atomic.atomic_write_text`
exactly for this; consistent with the project's stated atomic-write
contract.

Fix: route through `_atomic.atomic_write_text`. Also added a 64 KB
body-size cap so a malformed/abusive PUT can't fill the user's
preset directory, and tightened `_validate_name` to reject `..` even
though traversal is also caught downstream.

### `list_model_evaluations` sort key crashed on mixed None/str timestamps

**File:** `models_catalog.py:414-417`

`(ev.result.timestamp if ev.result else None) or ev.eval_id`
correctly handles `result is None` and falsy `result.timestamp`, but
when *some* entries have `None` from the outer guard and *others*
have a real string, the sorted comparison raises `TypeError`. The
practical hit is "partial run wrote a result without a timestamp".

Fix: unconditionally fall back to `""` then `eval_id` so the key is
always a string.

### Discovery walks descended into `output_models/`

**File:** `discovery.py:154-178`

Both `_iter_workspace_dirs` and `_iter_project_dirs` only pruned
hidden dirs and `forgather_workspace/`. They walked the entire
`output_models/` tree (every run, every checkpoint dir) on every
discovery pass, which is slow and risks picking up a stray
`meta.yaml` under a checkpoint as a real project.

Fix: added `_PRUNED_DIR_NAMES = {"output_models", "node_modules",
"__pycache__", ".git"}` filtered out of `dirnames[:]` in both walks.

### `fs.py` symlink-rejection guards were dead code

**Files:** `routes/fs.py:154-263, 426-469`, `routes/projects.py:147-159`

Every caller built the target via
`Path(os.path.expanduser(req.path)).resolve()` *before* invoking
`_check_path_safe` / `_reject_unsafe`. `Path.resolve()` collapses
symlinks, so the helpers' `target.is_symlink()` test was always
False. The README and helper docstrings claimed "refuses to follow
or delete symlink", but the only actual protection was the depth
floor / denylist.

Fix: added `_reject_symlink_in_chain(raw)` that walks the
*unresolved* path's ancestor chain via `os.path.islink`. Helpers now
take an optional `raw` parameter (the user-supplied string) and run
the symlink check on it before any resolve. Callers updated to pass
`raw=req.path` / `raw=req.src` / `raw=req.dest_dir`. The dead
`is_symlink()` branch in the asset endpoint was removed (the
containment check on resolved paths already enforces "stays inside
project_dir", which is the actually-load-bearing protection).

### `inference_proxy._proxy_streaming_post` leaked AsyncClient on non-RequestError

**File:** `routes/inference_proxy.py:126-141`

`httpx.AsyncClient` was constructed before the try/except, but the
except only caught `httpx.RequestError`. Anything else (bad URL,
runtime errors, unsupported protocol) propagated without `aclose`
on the client — connection pool lingered until GC.

Fix: catch-and-aclose-and-reraise any exception thrown before the
generator takes over. Also wrapped the error-body `aread` in
try/finally so a broken upstream response can't leak the client
either.

### Optimistic-concurrency mtime tolerance was too loose

**File:** `routes/configs.py:403-418`

The 1ms tolerance on `expected_mtime` round-tripping was much wider
than ext4/xfs/btrfs mtime resolution (nanoseconds). An external
write within 1ms of the user's GET would silently pass as "no
concurrent edit" — exactly the case the check was meant to catch.

Fix: tightened to 1µs, plenty of slack for JSON float round-trips
without hiding genuine edits.

## Notes / not fixed

- **CSRF / no-auth posture.** The unrestricted-path PUT/GET for
  template source, the `copy_from` parameter on project creation,
  the inference proxy's any-host SSRF allowance, and the missing
  Origin checks are all in scope of "single-user localhost
  prototype, no auth, do not expose the port". Mitigation requires
  an auth/origin design, not a code edit.
- **Shared file lock for `queue.json` / `job_records.json`.** Today
  only the server process writes these via HTTP; the CLI enqueues
  through HTTP too. If direct file writes ever become a workflow,
  add `fcntl.flock` to the read-modify-write paths.
- **PID-reuse 10s window in `_reattach_or_cleanup_on_startup`.**
  Persisting `psutil.Process.create_time()` at launch time and
  comparing for equality on restart would be tighter than the
  upper-bound slack, but the current behavior is correct under
  normal pid_max settings.
- **Dispatcher loop runs `build_command` (Jinja preprocess)
  synchronously.** Slow projects can stall the asyncio event loop
  for hundreds of ms. Moving the preprocess to `asyncio.to_thread`
  would un-stall it but isn't required for correctness.
