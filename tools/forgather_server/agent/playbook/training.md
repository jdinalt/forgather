# training — run/schedule a training job (single-node, multi-node, DiLoCo)

run_train is the SINGLE composable submit entry point (the equivalent of
`forgather submit`, i.e. `train --schedule`) — it SCHEDULES a background job,
not a foreground run. run_construct is the sibling for "build/inspect a named
target" (defaults target=main, gpus=0); pick run_train only when the user
actually wants to train.

BASIC (the common case) is trivial: project_dir + config_name [+ gpus]. Defaults
are fine — gpus defaults to the config's nproc_per_node (else 1; 0 = CPU
smoke-test), dataset defaults to mode-aware 'auto'. DON'T overthink it.

BEFORE training, check resolve_output_dir. If the output dir already has
runs/checkpoints, do NOT silently train into it — a finished run already at its
max_steps just resumes and exits immediately (no progress, confusing). Tell the
user what's there and offer: resume/extend, evaluate it, generate from it (see
the `inference` playbook), or start fresh (delete_path the output dir).

OVERRIDES: to change config parameters, pass dynamic_args (keyed by dest, from
inspect_config). There are many — if unfamiliar, read
`docs/project-templates/lm-training-projects.md` first.

MULTI-NODE: pass members ('HOST:GPUS[:IFACE]', repeatable), optionally rdzv_host
/ allow_version_mismatch. Read `docs/guides/multi-node-training.md` first.
DiLoCo workers: pass diloco_server (+ backend / diloco_worker_count / ...).
Read `docs/trainers/diloco.md` first. If the user hasn't said how they want a
complex job run, ASK focused questions before composing the call.

WATCH: list_jobs / job_status (live step/loss) / read_job_output (raw TTY), or
block with wait_for_job(queue_id). Blocking is fine for a short/tutorial run (to
show the result); for a long multi-hour run do NOT block — check periodically.
Never report finished until status is terminal (done/failed/aborted).

TIDY UP: once a short-lived job you started is terminal and reported, clean it up
with cleanup_jobs(queue_ids=[...the ids YOU spawned...]). Don't use all_terminal
(clears everyone's jobs) unless asked; never remove a job whose output the user
may still want.
