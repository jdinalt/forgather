from forgather.meta_config import MetaConfig


def ws_cmd(args):
    """Workspace management commands."""
    if hasattr(args, "ws_subcommand"):
        if args.ws_subcommand == "create":
            ws_create_cmd(args)
    else:
        # Default behavior - show workspace directory
        workspace_dir = MetaConfig.find_workspace_dir(args.project_dir)
        print(f"Workspace Directory: {workspace_dir}")


def ws_create_cmd(args):
    """Initialize a new forgather workspace."""
    import os

    from jinja2 import BaseLoader, Environment

    # Determine project directory name
    if args.workspace_dir:
        workspace_dir = args.workspace_dir
    else:
        # Default: replace spaces with underscores
        workspace_dir = args.name.replace(" ", "_").lower()
        workspace_dir = workspace_dir.replace(".", "")
        workspace_dir = os.path.join(args.project_dir, workspace_dir)

    forgather_dir = args.forgather_dir
    if not os.path.exists(forgather_dir):
        print(f"Error: Directory '{forgather_dir}' does not exist")

    # ``init_existing`` (server-side flag, not exposed via CLI args): when
    # true, treat ``workspace_dir`` as an *already-existing* directory the
    # caller wants to populate with workspace metadata, instead of a
    # not-yet-created directory we should mkdir. Used by the Forgather
    # server's "init workspace here" right-click flow on the Files tree —
    # the user has already created the directory and right-clicked it; we
    # just write ``forgather_workspace/`` into it.
    init_existing = getattr(args, "init_existing", False)

    if init_existing:
        if not os.path.isdir(workspace_dir):
            print(f"Error: Directory '{workspace_dir}' does not exist")
            return 1
        if os.path.exists(os.path.join(workspace_dir, "forgather_workspace")):
            print(f"Error: 'forgather_workspace' already exists in '{workspace_dir}'")
            return 1
    else:
        if os.path.exists(workspace_dir):
            print(f"Error: Directory '{workspace_dir}' already exists")
            return 1
        os.makedirs(workspace_dir)

    # Determine target directory - create forgather_workspace subdirectory
    target_dir = os.path.join(workspace_dir, "forgather_workspace")

    # Create the directory
    os.makedirs(target_dir, exist_ok=True)
    print(f"Creating forgather workspace in: {target_dir}")

    fg_relative = False
    if not os.path.isabs(forgather_dir):
        forgather_dir = os.path.normpath(os.path.relpath(forgather_dir, workspace_dir))
        fg_relative = True

    # Template context
    context = {
        "workspace_name": args.name,
        "worspace_description": args.description,  # Note: keeping typo to match template
        "forgather_dir": forgather_dir,
        "fg_relative": fg_relative,
        "search_paths": args.search_path or [],
        "libraries": args.lib or [],
    }

    # Template content inlined from the example files
    readme_template = """# {{ workspace_name }}

{{ worspace_description }}
"""

    # Structured workspace metadata consumed by tooling (e.g. the Forgather
    # server). Plain YAML, so human editors and parsers both read it easily.
    workspace_yaml_template = """name: "{{ workspace_name | replace('"', '\\\\"') }}"
description: "{{ worspace_description | replace('"', '\\\\"') }}"
"""

    base_directories_template = """
{% if fg_relative -%}
-- set ns.forgather_dir = joinpath(workspace_root, "{{ forgather_dir }}")
{% else -%}
-- set ns.forgather_dir = "{{ forgather_dir }}"
{% endif -%}
"""

    meta_defaults_template = """{% raw -%}
-- set ns = namespace()

## Import default paths; common to meta and regular templates.
-- include "base_directories.yaml"
-- set ns.templatelib_dir = joinpath(ns.forgather_dir, "templatelib")

## Search these directories for templates
searchdir:
-- block searchdir_project
    - "{{ joinpath(project_dir, 'templates') }}"
-- endblock searchdir_project

-- block searchdir_common
{% endraw -%}
{% for search_path in search_paths %}
    - "{{ search_path }}"
{% endfor %}
{% raw %}    - "{{ joinpath(workspace_root, 'forgather_workspace') }}"{% endraw +%}
{% for lib in libraries %}
{% raw %}    - "{{ joinpath(ns.templatelib_dir, '{% endraw %}{{ lib }}{% raw %}') }}"{% endraw +%}
{% endfor %}
-- endblock searchdir_common


-- block configs
-- endblock configs

"""

    # Render templates with whitespace control
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

    # Create files
    files_to_create = {
        "README.md": readme_template,
        "base_directories.yaml": base_directories_template,
        "meta_defaults.yaml": meta_defaults_template,
        "workspace.yaml": workspace_yaml_template,
    }

    for filename, template_content in files_to_create.items():
        template = env.from_string(template_content)
        rendered_content = template.render(**context)

        file_path = os.path.join(target_dir, filename)
        with open(file_path, "w") as f:
            f.write(rendered_content)
        print(f"Created: {filename}")

    print(f"\nForgather workspace '{args.name}' initialized successfully!")
    print(f"Workspace directory: {target_dir}")
    return 0
