# External Workspace Configuration

This workspace configures access to the Forgather templatelib from an external project directory.

When a Project is constructed, the enclosing directories are recursively searched for a directory named 'forgather_workspace' and, if found, this directory is implicitly added to the template search path of all enclosed 'meta.yaml' files.

This allows external projects to access Forgather's template library and define common workspace configurations.