# diloco — monitor and control a DiLoCo parameter server

DiLoCo (distributed low-communication training): a parameter server that workers
sync against. Read `docs/trainers/diloco.md` for concepts before tuning.

- Start/stop the SERVER: start_diloco_server (output_dir + num_workers required;
  common knobs exposed, the rest via `advanced`) / stop_service. After starting,
  wait_for_job(queue_id, until="running").
- list_diloco_servers → known servers + reachability/health.
- diloco_status(server_id) → round/step, synced + known workers.
- diloco_control (CONFIRM) → save_state (checkpoint the server), shutdown, or
  relay a worker command (save_checkpoint | save_and_stop | abort), optionally to
  one worker_id.
- Launch DiLoCo WORKERS via run_train(diloco_server=...) — see the `training`
  playbook.
