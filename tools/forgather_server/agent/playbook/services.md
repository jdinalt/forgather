# services — start/stop the long-running services (Sidebar -> Services)

list_services shows configured services + whether each is running. There is ONE
start tool per type, each with REAL, explicit arguments (don't guess a generic
blob): start_dataset_server, start_inference_server, start_diloco_server,
start_tensorboard, start_mkdocs. stop_service stops one by type+name.

After starting any service, wait for it to come UP with
wait_for_job(queue_id, until="running") — a healthy service never reaches a
terminal status, so the default until="terminal" would just time out. Then
verify reachability (list_dataset_servers / list_inference_servers /
list_diloco_servers).

Required args: start_dataset_server() — none (defaults fine; brings up a default
server, use when dataset_info reports none reachable). start_inference_server —
model_path (or models) + port. start_diloco_server — output_dir + num_workers
(read docs/trainers/diloco.md before tuning).

GPUs are RESERVED: inference / diloco servers and training jobs each need
requested_gpus. If no GPU is free (another job/service holds it) the new job does
NOT run — it stays QUEUED until one frees. Check gpu_status first; if
wait_for_job(until="running") times out with status "queued", that's why — use
gpu_status / list_jobs to see what's holding the GPU and tell the user (stop a
service/job or wait). Don't silently keep re-waiting.
