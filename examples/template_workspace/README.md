# Workpace Template: Copy me as "forgaher_worksapce" to create a new workspace

A Forgather workspace defines the common configuraiton for the set of projects within the same directory (or sub-directory) as the workspace.

When a Project is constructed, the enclosing directories are recursively searched for a directory named 'forthater_workspace' and, if found, 
this directory is implicilty added to the template search path of all enclosed 'meta.yaml' files. This allows this directory to contain
meta-templates, which can be included or extended by project meta-files.

A typical use-case is in defining the paths to common directories used by the enclosed projects. Thus, if any of these change, you only need
to updte a single file.

### base_directories.yaml

- ns.forgather_dir : This defines the location of the "forgather" root directory, which is defined as being relative to the "workspace_root" directory.

#### Predefined Symbolic Directories

- workspace_root : The directory in which the "forgather_workspace" is located.
- user_home_dir() : The present user's home directory. (~ or $HOME on Linux).
- forgather_config_dir() : A user-specific configuration directory. ("~/.config/forgather" on Linux)

### meta_defaults.yaml

All of the templates in the templates library assume the existance of a common Jinja2 namespace, named "ns." We declare the root-namespace within meta_defaults.yaml, which is needed to address a partiularity of Jinja2:

>Please keep in mind that it is not possible to set variables inside a block and have them show up outside of it. This also applies to loops. The only exception to that rule are if statements which do not introduce a scope. -- Jinja2 Docs

Another exception to this is when the variable is enclosed within a namespace, which sidesteps this restriction.
