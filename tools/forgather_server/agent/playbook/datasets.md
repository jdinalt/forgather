# datasets — build, smoke-test, and introspect datasets

Tools: run_dataset (build/inspect a split as a job), wait_for_job +
read_job_output / list_jobs (watch), list_dataset_servers + dataset_info
(splits / #examples / features). Docs: `docs/datasets/dataset-projects.md`,
`docs/datasets/dataset-cli.md`, `docs/guides/creating-a-dataset-project.md`.

Two tiers of split targets: RAW — train_dataset_split / validation_dataset_split
/ test_dataset_split — need no tokenizer; TOKENIZED — train_dataset /
eval_dataset / test_dataset — require a tokenizer (pass tokenizer_path). Some
source datasets have only a "train" split, sliced into the others in the config
(e.g. validation = "train[0:1000]").

BUILD: run_dataset (CONFIRM; default target=train_dataset_split). The FIRST build
downloads + builds the data and can be slow — tell the user up front. After
approval, wait_for_job(queue_id) (blocks server-side; don't poll in a loop); if
it times out on a long build, call again. Only report success once terminal.
Smoke-test: run each raw split with examples=3, truncate=64, then tokenized
splits with tokenizer_path. Find a tokenizer with find_files (e.g.
find_files("wikitext")) — tokenizers live under tokenizers/ as directories.

INTROSPECT: dataset_info(dataset HF name/path from the config's load_dataset args
via inspect_config / render_config_pp) → splits / #examples / features. It needs
the data built AND a dataset server reachable.

IMPORTANT — no dataset server reachable? Just start one: start_dataset_server()
(no args, defaults are fine → a default server on 8766), then
wait_for_job(queue_id, until="running"), then retry dataset_info. Don't tell the
user to start it manually and don't get stuck — bring one up yourself.
