import os
import subprocess
from pprint import pp

from forgather.project import Project
import forgather.nb.notebooks as nb
from forgather.config import fconfig
from forgather.codegen import generate_code
from forgather.yaml_encoder import to_yaml
from forgather.latent import Latent
from forgather.meta_config import preprocessor_globals, MetaConfig
from forgather.config import ConfigEnvironment

"""shared command utils"""


def set_default_template(meta, args):
    """Set default template if not specified in args."""
    if not args.config_template:
        default_config = meta.default_config()
        args.config_template = default_config


def get_env(meta, project_dir):
    """Create new config environment and load configuration."""
    environment = ConfigEnvironment(
        searchpath=meta.searchpath,
        global_vars=preprocessor_globals(project_dir, meta.workspace_root),
    )
    return environment


def get_config(meta, env, config_template):
    """Load and return configuration from template."""
    return env.load(meta.config_path(config_template)).get()


class BaseCommand:
    """Base class for CLI commands with common setup patterns."""

    def __init__(self, args):
        self.args = args
        self.meta = MetaConfig(args.project_dir)
        set_default_template(self.meta, args)
        self.env = get_env(self.meta, args.project_dir)

    def get_config(self):
        """Get configuration for current template."""
        return get_config(self.meta, self.env, self.args.config_template)


"""Command Implementations"""


def ls_cmd(args):
    """List available configurations."""
    if args.recursive:
        for root, dirs, files in os.walk(args.project_dir):
            for file_name in files:
                if file_name == "meta.yaml":
                    print(f"\nProject Path: {root}\n")
                    try:
                        list_project(root)
                    except:
                        print(f"PARSE ERROR: {os.path.join(root, file_name)}")
    else:
        list_project(args.project_dir)


def list_project(project_dir):
    """List configurations for a single project."""
    meta = MetaConfig(project_dir)
    meta_config = meta.config_dict
    project_name = meta_config.get("name", "Anonymous")
    project_description = meta_config.get("description", "No Description")
    print(f"{project_name} : {project_description}")
    env = get_env(meta, project_dir)
    for config_name, path in meta.find_templates(meta.config_prefix):
        try:
            config, pp_config = get_config(meta, env, config_name)
            config_meta = Latent.materialize(config.meta)
            config_long_name = config_meta.get("config_name", "Anonymous")
            config_description = config_meta.get("config_description", "No Description")
        except Exception as e:
            config_long_name = "PARSE ERROR"
            config_description = "An error occured while parsing the configuration."
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
    print(s)


def pp_cmd(args):
    """Show pp_cmded configuration."""
    cmd = BaseCommand(args)
    pp_config = cmd.env.preprocess(cmd.meta.config_path(args.config_template))
    print(pp_config)


def code_cmd(args):
    """Generate Python code from configuration."""
    cmd = BaseCommand(args)
    config, pp_config = cmd.get_config()
    code = generate_code(config[args.target])
    print(code)


def construct_cmd(args):
    """Materialize and print a target."""
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    proj = Project(args.config_template, args.project_dir)
    target = proj(args.target)
    if args.call:
        target = target()
    pp(target)


def meta_cmd(args):
    """Show meta configuration."""
    cmd = BaseCommand(args)
    md = nb.render_meta(cmd.meta, "# Meta Config\n")
    print(md)


def trefs_cmd(args):
    """List templates referenced by configuration."""
    cmd = BaseCommand(args)

    match args.format:
        case "md":
            print(
                nb.render_referenced_templates_tree(
                    cmd.env, cmd.meta.config_path(args.config_template)
                )
            )
        case "files":
            # Yields # tuple(level: int, name: str, path: str)
            for level, name, path in cmd.env.find_referenced_templates(
                cmd.meta.config_path(args.config_template)
            ):
                print(os.path.relpath(path))
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def graph_cmd(args):
    """Preprocess and parse into node graph."""
    cmd = BaseCommand(args)
    config, pp_config = cmd.get_config()
    match args.format:
        case "none":
            pass
        case "fconfig":
            print(fconfig(config))
        case "repr":
            print(repr(config))
        case "yaml":
            print(to_yaml(config))
        case "python":
            print(generate_code(config["main"]))
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def tb_cmd(args):
    """Start Tensorboard for project."""
    cmd = BaseCommand(args)
    config, pp_config = cmd.get_config()
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


def train_cmd(args):
    """Run configuration with train script."""
    cmd = BaseCommand(args)
    config, pp_config = cmd.get_config()
    config_meta = Latent.materialize(config.meta)
    nproc_per_node = config_meta["nproc_per_node"]
    train_script_path = os.path.join(
        config_meta["forgather_dir"], "scripts", "train_script.py"
    )

    cmd_args = ["torchrun"]

    if len(args.remainder) > 1 and args.remainder[0] == "--":
        cmd_args.extend(args.remainder[1:])
    else:
        # Apply defaults, if not specified
        cmd_args.extend(
            [
                "--standalone",
                "--nproc-per-node",
                str(nproc_per_node),
            ]
        )

    # Apply path to script and project directory argument to script.
    cmd_args.extend(
        [
            os.path.normpath(train_script_path),
            "-p",
            os.path.normpath(args.project_dir),
        ]
    )

    # Optionally, apply system search path from meta.
    if cmd.meta.system_path is not None:
        cmd_args.extend(["-s", cmd.meta.system_path])

    # Add the config template name
    cmd_args.append(args.config_template)

    # Generate equivalent command string
    cmd_str = ""

    if args.devices:
        cmd_str += f'CUDA_VISIBLE_DEVICES="{args.devices}" '
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

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
            print(nb.render_extends_graph(cmd.meta))
        case "files":
            for template_name, template_path in cmd.meta.find_templates():
                print(template_path)
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def index_cmd(args):
    """Show project index."""
    md = nb.render_project_index(args.project_dir)
    print(md)
