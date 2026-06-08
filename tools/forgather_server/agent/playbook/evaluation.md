# evaluation — score a model, and control a running training job

run_eval (CONFIRM) scores a model against an eval config: pick a name from
list_eval_configs, pass the model output dir / checkpoint as model_path. Watch it
like any job; read results with list_evaluations once terminal.

control_job (CONFIRM) controls a RUNNING training job by queue_id: save (a
checkpoint), stop (saves a final checkpoint), save-stop, or abort (no checkpoint).

gpu_status shows per-GPU memory / utilisation — use it to advise requested_gpus
and to see what's free before scheduling a GPU job.
