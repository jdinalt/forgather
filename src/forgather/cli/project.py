import os
import shutil

from forgather.meta_config import MetaConfig

default_config_template = """-- set ns = namespace()
-- include "base_directories.yaml"

meta: &meta_output !dict:@meta
    config_name: "Default"
    config_description: "Default Config"
    config_class: "none"
    project_dir: "{{ project_dir }}"
    workspace_root: "{{ workspace_root }}"
    forgather_dir: "{{ ns.forgather_dir }}"

dynamic_args: []

main: "Main Output"
"""


def project_cmd(args):
    """Workspace management commands."""
    if hasattr(args, "project_subcommand"):
        match args.project_subcommand:
            case "create":
                project_create_cmd(args)
            case "show":
                project_show_cmd(args)
            case "new_config":
                project_new_config_cmd(args)
            case _:
                raise Exception(f"Unrecognized sub-command {args.project_subcommand}")
    else:
        # Default behavior - show workspace directory
        project_dir = MetaConfig.find_project_dir(args.project_dir)
        print(f"Project Directory: {project_dir}")


def project_create_cmd(args):
    """Create a new forgather project in the workspace."""
    import os
    from jinja2 import Environment, BaseLoader

    # Determine project directory name
    if args.project_dir_name:
        project_dir_name = args.project_dir_name
    else:
        # Default: replace spaces with underscores
        project_dir_name = args.name.replace(" ", "_").lower()

    # Create target directory
    target_dir = os.path.join(args.project_dir, project_dir_name)

    if os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' already exists")
        return 1

    # Create directory structure
    os.makedirs(target_dir, exist_ok=True)
    print(f"Creating forgather project in: {target_dir}")

    # Template context
    context = {
        "project_name": args.name,
        "project_description": args.description,
        "config_prefix": args.config_prefix,
        "default_config": args.default_config,
    }

    # Template content inlined from the example files
    readme_template = """# {{ project_name }}

{{ project_description }}
"""

    meta_template = """-- extends "meta_defaults.yaml"

-- block configs
name: "{{ project_name }}"
description: "{{ project_description }}"
config_prefix: "{{ config_prefix }}"
default_config: "{{ default_config }}"
<< endblock configs
"""

    # Render templates with whitespace control
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

    # Create files
    files_to_create = {
        "README.md": readme_template,
        "meta.yaml": meta_template,
    }

    for filepath, template_content in files_to_create.items():
        template = env.from_string(template_content)
        rendered_content = template.render(**context)

        full_path = os.path.join(target_dir, filepath)
        with open(full_path, "w") as f:
            f.write(rendered_content)
        print(f"Created: {filepath}")

    meta = MetaConfig(target_dir)

    config_path = os.path.join(
        meta.searchpath[0], meta.config_prefix, args.default_config
    )

    os.makedirs(os.path.dirname(config_path))
    create_config(config_path, args.copy_from)

    print(f"\nForgather project '{args.name}' created successfully!")
    print(f"Project directory: {target_dir}")
    return 0


def project_show_cmd(args):
    project_dir = MetaConfig.find_project_dir(args.project_dir)
    meta = MetaConfig(project_dir)
    print(meta)


def create_config(config_path, copy_from):
    if copy_from:
        shutil.copyfile(copy_from, config_path)
        print(f"Copied config {copy_from} to {config_path}")
    else:
        with open(config_path, "w") as f:
            f.write(default_config_template)
        print(f"Created new config at {config_path}")


def project_new_config_cmd(args):
    meta = MetaConfig(MetaConfig.find_project_dir(args.project_dir))
    match args.type:
        case "config":
            config_path = os.path.join(
                meta.searchpath[0], meta.config_prefix, args.config_name
            )
        case "project":
            config_path = os.path.join(meta.searchpath[0], args.config_name)
        case "ws":
            config_path = os.path.join(
                meta.workspace_root, "forgather_workspace", args.config_name
            )

    config_dir = os.path.dirname(config_path)
    os.makedirs(config_dir, exist_ok=True)

    if os.path.exists(config_path):
        raise FileExistsError(f"{config_path} already exists")

    create_config(config_path, args.copy_from)
