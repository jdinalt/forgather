# diloco — monitor and control a DiLoCo parameter server

DiLoCo (distributed low-communication training): a two-level optimizer — workers
train locally for H steps, then a parameter server applies an outer optimizer
(SGD+Nesterov) to the averaged pseudo-gradient. Read `docs/trainers/diloco.md`
for concepts before tuning.

**Advising on DiLoCo (what it is / when to use it): read the canonical example
README first** — `examples/tiny_experiments/diloco/README.md`. Its empirical
sweep is the source of truth, and answering from the reference doc's intro alone
leads to WRONG conclusions. In particular, DiLoCo is NOT only for slow networks
or big multi-machine clusters: the outer step is a SlowMo/Lookahead-style
regularizer that, at longer token budgets, can **match or beat DDP final
quality** — even single-node, single-GPU, and on small models. Do NOT tell a user
to "always prefer DDP on a fast/single-host link"; the quality gain is
independent of link speed. The bandwidth win and the generalization win are
separate reasons to use it. See the doc's "When to use DiLoCo" section.

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
