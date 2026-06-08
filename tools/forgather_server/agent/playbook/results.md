# results — inspect training runs, checkpoints, and evaluations

After/while a model trains, inspect outcomes:
- list_models(project_dir) → a project's output dirs + run/checkpoint/eval counts.
- list_runs(output_dir) → recorded runs; run_summary(run_dir) → best/last loss,
  perplexity, steps, runtime.
- list_checkpoints(output_dir) → step / size / world_size / manifest (pick the
  best/latest).
- job_status(queue_id) → LIVE trainer step/loss for a running job (read_job_output
  is the raw TTY tail; read_run_tty tails an older run's tty.log).
- list_evaluations(output_dir) → recorded eval results (loss/perplexity/bpb).

Typical "how did training go?": list_models → run_summary → list_checkpoints,
then report the numbers plainly.
