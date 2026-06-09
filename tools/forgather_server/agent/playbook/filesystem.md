# filesystem — inspect and manage files (not just configs)

You are NOT limited to editing config files. Besides config authoring
(propose_*), you can manage files directly:
- stat_path — inspect a path (type/size/mtime/entry count).
- edit_file (PROPOSE) — write content to a plain text file, shown as a
  before/after diff to approve. Creates the file if missing (parent dir must
  exist), otherwise overwrites — so this is the one-step way to create a file
  WITH content. For configs/templates use propose_edit_config (it also
  parse-checks); for everything else (markdown, notes, scripts) use edit_file.
- create_file (CONFIRM) — touch a new EMPTY file. Only when you want it empty;
  to create with content, use edit_file directly. Refuses if it already exists
  or its parent dir is missing.
- delete_path (CONFIRM) — delete a file OR, recursively, a directory (e.g. clear
  a model's output dir). Irreversible.
- move_path / copy_path (CONFIRM).

All are guarded: inside the configured filesystem roots only, not a system path,
depth floor. When asked to "delete/clean up the output directory", find it with
resolve_output_dir (or list_models) and then delete_path — do NOT tell the user
to run the command themselves.
