import os
import subprocess
from pprint import pformat
import traceback

from forgather.project import Project
from forgather.config import fconfig
from forgather.codegen import generate_code
from forgather.yaml_encoder import to_yaml
from forgather.latent import Latent
from forgather.preprocess import debug_pp
from forgather.meta_config import MetaConfig

from .dynamic_args import get_dynamic_args
from .utils import (
    write_output_or_edit,
    should_use_absolute_paths,
    set_default_template,
    get_env,
    get_config,
    BaseCommand,
)

"""Command Implementations"""


def ls_cmd(args):
    """List available configurations."""
    if args.project:
        for project_dir in args.project:
            if args.recursive:
                list_project_recurse(project_dir, args.debug)
            else:
                try:
                    print(f"\nProject Path: {project_dir}\n")
                    list_project(project_dir, args.debug)
                except ValueError as e:
                    print(e)
    elif args.recursive:
        list_project_recurse(args.project_dir, args.debug)
    else:
        try:
            # Search upward for the nearest project directory
            project_dir = MetaConfig.find_project_dir(args.project_dir)
            if os.path.realpath(project_dir) != os.path.realpath(args.project_dir):
                print(f"\nProject Path: {project_dir}\n")
            list_project(project_dir, args.debug)
        except Exception as e:
            print(e)


def list_project_recurse(project_dir, debug):
    for root, dirs, files in os.walk(project_dir):
        for file_name in files:
            if file_name == "meta.yaml":
                print(f"\nProject Path: {root}\n")
                try:
                    list_project(root, debug)
                except Exception as e:
                    print(f"PARSE ERROR: {os.path.join(root, file_name)} '{e}'")


def list_project(project_dir, debug=False):
    """List configurations for a single project."""
    meta = MetaConfig(project_dir)
    meta_config = meta.config_dict
    project_name = meta_config.get("name", "Anonymous")
    project_description = meta_config.get("description", "No Description")
    print(f"{project_name} : {project_description}")
    env = get_env(meta, project_dir)
    config_names = [name for name, _ in meta.find_templates(meta.config_prefix)]
    for config_name in sorted(config_names):
        try:
            config, pp_config = get_config(meta, env, config_name)
            config_meta = Latent.materialize(config.meta)
            config_long_name = config_meta.get("config_name", "Anonymous")
            config_description = config_meta.get("config_description", "No Description")
        except Exception as e:
            config_long_name = "PARSE ERROR"
            config_description = "An error occured while parsing the configuration."
            if debug:
                traceback.print_exc()

        if config_name == meta.default_config():
            config_name = f"[{config_name}]"
        print(f"    {config_name:<30} {config_long_name} : {config_description}")


def targets_cmd(args):
    """List available output targets."""
    cmd = BaseCommand(args)
    config, pp_config = cmd.get_config()
    s = ""
    for target in config.keys():
        s += f"{target}\n"
    write_output_or_edit(args, s, ".txt", "targets")


def pp_cmd(args):
    """Show preprocessed configuration."""
    with debug_pp(args.debug):
        # Get dynamic arguments (configuration-specific args)
        dynamic_args = get_dynamic_args(args)
        if dynamic_args:
            print(f"# Dynamic arguments received: {dynamic_args}")
            print()

        cmd = BaseCommand(args)
        pp_config = cmd.env.preprocess(
            cmd.meta.config_path(args.config_template), **dynamic_args
        )
        write_output_or_edit(args, pp_config, ".yaml", "pp")


def code_cmd(args):
    """Generate Python code from configuration."""
    cmd = BaseCommand(args)
    dynamic_args = get_dynamic_args(args)
    config, pp_config = cmd.get_config(**dynamic_args)
    code = generate_code(config[args.target])
    write_output_or_edit(args, code, ".py", "code")


def construct_cmd(args):
    """Materialize and print a target."""
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    dynamic_args = get_dynamic_args(args)
    proj = Project(args.config_template, args.project_dir, **dynamic_args)
    target = proj(args.target)
    if args.call:
        target = target()
    write_output_or_edit(args, pformat(target), ".txt")


def meta_cmd(args):
    """Show meta configuration."""
    import forgather.nb.notebooks as nb

    cmd = BaseCommand(args)
    md = nb.render_meta(cmd.meta, "# Meta Config\n")
    write_output_or_edit(args, md, ".md")


def graph_cmd(args):
    """Preprocess and parse into node graph."""
    with debug_pp(args.debug):
        cmd = BaseCommand(args)
        dynamic_args = get_dynamic_args(args)
        config, pp_config = cmd.get_config(**dynamic_args)
        match args.format:
            case "none":
                pass
            case "fconfig":
                write_output_or_edit(args, fconfig(config), ".txt")
            case "repr":
                write_output_or_edit(args, repr(config), ".txt")
            case "yaml":
                write_output_or_edit(args, to_yaml(config), ".yaml")
            case "python":
                write_output_or_edit(args, generate_code(config["main"]), ".py")
            case _:
                raise Exception(f"Unrecognized format {args.format}")


def tb_cmd(args):
    """Start Tensorboard for project."""
    cmd = BaseCommand(args)
    dynamic_args = get_dynamic_args(args)
    config, pp_config = cmd.get_config(**dynamic_args)
    config_meta = Latent.materialize(config.meta)

    if args.all:
        output_dir = os.path.abspath(config_meta["models_dir"])
    else:
        output_dir = os.path.abspath(config_meta["output_dir"])

    cmd_args = [
        "tensorboard",
        "--logdir",
        output_dir,
    ]

    if len(args.remainder) > 1 and args.remainder[0] == "--":
        cmd_args.extend(args.remainder[1:])

    cmd_str = ""
    for arg in cmd_args:
        cmd_str += f"{arg} "

    print(f"{cmd_str}")

    # Run the command
    if not args.dry_run:
        subprocess.run(cmd_args)


def template_list(args):
    """List available templates."""
    cmd = BaseCommand(args)
    match args.format:
        case "md":
            import forgather.nb.notebooks as nb

            use_absolute_paths = should_use_absolute_paths(args)
            md_content = nb.render_extends_graph(
                cmd.meta, use_absolute_paths=use_absolute_paths
            )
            write_output_or_edit(args, md_content, ".md", "tlist")
        case "files":
            data = ""
            for template_name, template_path in cmd.meta.find_templates():
                data += template_path + "\n"
            write_output_or_edit(args, data, ".txt")
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def index_cmd(args):
    """Show project index."""
    import forgather.nb.notebooks as nb

    md = nb.render_project_index(args.project_dir)
    write_output_or_edit(args, md, ".md")
