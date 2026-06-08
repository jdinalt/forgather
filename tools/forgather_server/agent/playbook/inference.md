# inference — serve a model and generate from it

To prove a trained model works: start an inference server, then query_model.

1. start_inference_server (CONFIRM). To serve a model you TRAINED (a Forgather
   output dir like output_models/<name>), set from_checkpoint=true — it loads the
   latest native checkpoint. A bare model_path expects an already-HF-format model
   and will FAIL to load a raw output dir. (Equivalent CLI:
   `forgather inf server -m <dir> --from-checkpoint`.) Pass a port; on
   unified-memory hardware (DGX Spark / Grace-Hopper) set keep_on_gpu=true.
2. wait_for_job(queue_id, until="running"). Note: "running" means the process
   spawned — if generation then fails with "no inference server is known", the
   server likely crashed on load (often the from_checkpoint issue above); check
   list_inference_servers / read_job_output.
3. query_model(prompt=... or messages=...) (CONFIRM) → the model's reply. Omit
   server_id to use the first reachable server.

It reserves a GPU; if none is free it QUEUES (see the `services` playbook GPU note).
list_inference_servers shows running model servers; cluster_status reports
node/master/members on a multi-node setup.
