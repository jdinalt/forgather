# Evaluation workspace

Holds named evaluation configurations discovered by `forgather eval list/show/test`.

Each config inside a project here is tagged `type.evaluation` in its meta block.
The CLI scans this directory (and any additional paths in
`~/.forgather/config.yaml` under `eval.search_paths`) for projects containing
eval configs.
