# Workspace Configuration

A Forgather workspace defines the common configuraiton for the set of projects within the same directory (or sub-directory) as the workspace.

When a Project is constructed, the enclosing directories are recursively searched for a directory named 'forthater_workspace' and, if found, 
this directory is implicilty added to the template search path of all enclosed 'meta.yaml' files. This allows this directory to contain
meta-templates, which can be included or extended by project meta-files.

A typical use-case is in defining the paths to common directories used by the enclosed projects. Thus, if any of these change, you only need
to updte a single file.
