import os
import subprocess
from pprint import pp, pformat
import traceback

from forgather.project import Project
import forgather.nb.notebooks as nb
from forgather.config import fconfig
from forgather.codegen import generate_code
from forgather.yaml_encoder import to_yaml
from forgather.latent import Latent
from forgather.meta_config import preprocessor_globals, MetaConfig
from forgather.config import ConfigEnvironment
from forgather.preprocess import debug_pp
from forgather.trainer_control import get_default_client, HTTPTrainerControlClient

from .dynamic_args import get_dynamic_args
from .utils import write_output, write_output_or_edit, should_use_absolute_paths

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


def get_config(meta, env, config_template, **kwargs):
    """Load and return configuration from template."""
    return env.load(meta.config_path(config_template), **kwargs).get()


class BaseCommand:
    """Base class for CLI commands with common setup patterns."""

    def __init__(self, args, search_for_project=True):
        if search_for_project:
            args.project_dir = MetaConfig.find_project_dir(args.project_dir)
        self.args = args
        self.meta = MetaConfig(args.project_dir)
        set_default_template(self.meta, args)
        self.env = get_env(self.meta, args.project_dir)

    def get_config(self, **kwargs):
        """Get configuration for current template."""
        return get_config(self.meta, self.env, self.args.config_template, **kwargs)


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
    for config_name, path in meta.find_templates(meta.config_prefix):
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
    cmd = BaseCommand(args)
    md = nb.render_meta(cmd.meta, "# Meta Config\n")
    write_output_or_edit(args, md, ".md")


def trefs_cmd(args):
    """List templates referenced by configuration."""
    cmd = BaseCommand(args)

    match args.format:
        case "md":
            use_absolute_paths = should_use_absolute_paths(args)
            md_content = nb.render_referenced_templates_tree(
                cmd.env,
                cmd.meta.config_path(args.config_template),
                use_absolute_paths=use_absolute_paths,
            )
            write_output_or_edit(args, md_content, ".md", "trefs")
        case "files":
            data = ""
            # Yields # tuple(level: int, name: str, path: str)
            for level, name, path in cmd.env.find_referenced_templates(
                cmd.meta.config_path(args.config_template)
            ):
                data += os.path.relpath(path) + "\n"
            write_output_or_edit(args, data, ".txt")
        case "tree":
            data = render_template_hierarchy_tree(
                cmd.env, cmd.meta.config_path(args.config_template)
            )
            write_output_or_edit(args, data, ".txt")
        case "dot":
            data = render_template_hierarchy_dot(
                cmd.env, cmd.meta.config_path(args.config_template)
            )
            write_output_or_edit(args, data, ".dot")
        case "svg":
            svg_data = render_template_hierarchy_svg(
                cmd.env, cmd.meta.config_path(args.config_template)
            )
            write_output_or_edit(args, svg_data, ".svg")
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def render_template_hierarchy_tree(environment, template_path):
    """Render template hierarchy as a clean tree structure"""

    template_data = list(environment.find_referenced_templates(template_path))

    output = []
    output.append(f"Template Hierarchy for {template_path}")
    output.append("=" * (len(f"Template Hierarchy for {template_path}")))
    output.append("")

    # Group by hierarchies (detect separate root-level trees)
    current_hierarchy = []
    last_level = -1

    for level, name, path in template_data:
        # Detect new hierarchy starting
        if level == 0 and last_level >= 0:
            # End previous hierarchy
            if current_hierarchy:
                output.extend(_render_hierarchy_tree(current_hierarchy))
                output.append("")
            current_hierarchy = []

        current_hierarchy.append((level, name, path))
        last_level = level

    # Render final hierarchy
    if current_hierarchy:
        output.extend(_render_hierarchy_tree(current_hierarchy))

    return "\n".join(output)


def _render_hierarchy_tree(hierarchy_data):
    """Render a single hierarchy as a tree"""
    output = []

    # Use proper tree characters
    for i, (level, name, path) in enumerate(hierarchy_data):
        # Determine tree characters based on position and level
        indent = "    " * level

        if level == 0:
            # Root level
            prefix = "📁 "
        else:
            # Child levels - use tree characters
            is_last_at_level = True
            # Check if this is the last item at this level
            for j in range(i + 1, len(hierarchy_data)):
                next_level, _, _ = hierarchy_data[j]
                if next_level == level:
                    is_last_at_level = False
                    break
                elif next_level < level:
                    break

            # Build prefix with proper tree characters
            prefix = ""
            for l in range(level):
                if l == level - 1:
                    prefix += "└── " if is_last_at_level else "├── "
                else:
                    # Check if we need a vertical line at this level
                    has_sibling_below = False
                    for j in range(i + 1, len(hierarchy_data)):
                        check_level, _, _ = hierarchy_data[j]
                        if check_level == l:
                            has_sibling_below = True
                            break
                        elif check_level < l:
                            break
                    prefix += "│   " if has_sibling_below else "    "

        # Add file type emoji and path info
        if "trainer" in name.lower():
            emoji = "🏃 "
        elif "model" in name.lower():
            emoji = "🧠 "
        elif "dataset" in name.lower():
            emoji = "📊 "
        elif "callback" in name.lower():
            emoji = "🔄 "
        elif name.endswith(".yaml"):
            emoji = "📄 "
        else:
            emoji = "📋 "

        # Show relative path if different from name
        path_info = ""
        if path != name and not path.endswith(name):
            rel_path = os.path.relpath(path)
            if len(rel_path) < 50:  # Only show if not too long
                path_info = f" → {rel_path}"

        output.append(f"{prefix}{emoji}{name}{path_info}")

    return output


def render_template_hierarchy_dot(environment, template_path):
    """Render template hierarchy as Graphviz DOT format"""

    # Get the raw dependency relationships instead of using hierarchy levels
    load_sequence, dependencies = environment.get_template_dependencies(template_path)

    output = []
    output.append("digraph TemplateHierarchy {")
    output.append("    rankdir=TB;")
    output.append("    node [shape=box, style=rounded];")
    output.append("    edge [color=gray60];")
    output.append("")

    # Define node styles for different template types
    styles = {
        "config": 'fillcolor=lightblue, style="rounded,filled"',
        "trainer": 'fillcolor=lightgreen, style="rounded,filled"',
        "model": 'fillcolor=lightyellow, style="rounded,filled"',
        "dataset": 'fillcolor=lightcoral, style="rounded,filled"',
        "callback": 'fillcolor=lightgray, style="rounded,filled"',
        "default": 'fillcolor=white, style="rounded,filled"',
    }

    # Add nodes with styles
    all_templates = set()
    for name, path in load_sequence:
        all_templates.add(name)

    for name in sorted(all_templates):
        # Determine node style based on name
        style = styles["default"]
        if "trainer" in name.lower():
            style = styles["trainer"]
        elif "model" in name.lower():
            style = styles["model"]
        elif "dataset" in name.lower():
            style = styles["dataset"]
        elif "callback" in name.lower():
            style = styles["callback"]
        elif "config" in name.lower() or name.endswith(".yaml"):
            style = styles["config"]

        # Clean node name for DOT format
        clean_name = name.replace(".", "_").replace("/", "_").replace("-", "_")
        display_name = name.replace('"', '\\"')

        output.append(f'    {clean_name} [label="{display_name}", {style}];')

    output.append("")

    # Add edges based on actual dependency relationships
    # The dependencies dict shows: parent -> [children] relationships
    # In DOT format, we want: parent -> child (parent depends on/includes child)
    for parent, children in dependencies.items():
        parent_clean = parent.replace(".", "_").replace("/", "_").replace("-", "_")
        for child in sorted(children):
            child_clean = child.replace(".", "_").replace("/", "_").replace("-", "_")
            output.append(f"    {parent_clean} -> {child_clean};")

    output.append("}")

    return "\n".join(output)


def render_template_hierarchy_svg(environment, template_path):
    """Render template hierarchy as SVG using Graphviz"""
    import subprocess
    import tempfile
    import shutil

    # Check if dot command is available
    if not shutil.which("dot"):
        raise Exception(
            "Graphviz 'dot' command not found. Please install Graphviz to use SVG output."
        )

    dot_content = render_template_hierarchy_dot(environment, template_path)

    try:
        # Use dot command to generate SVG
        result = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot_content,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to generate SVG with Graphviz: {e.stderr}")
    except Exception as e:
        raise Exception(f"Error generating SVG: {e}")


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


def train_cmd(args):
    """Run configuration with train script."""
    import json

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

    # Add dynamic arguments as JSON if any exist
    dynamic_args = get_dynamic_args(args)
    if dynamic_args:
        # Serialize dynamic args to JSON and pass to training script
        dynamic_args_json = json.dumps(dynamic_args)
        cmd_args.extend(["--dynamic-args", dynamic_args_json])

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
    md = nb.render_project_index(args.project_dir)
    write_output_or_edit(args, md, ".md")


def ws_cmd(args):
    """Workspace management commands."""
    if hasattr(args, "ws_subcommand"):
        if args.ws_subcommand == "init":
            ws_init_cmd(args)
        elif args.ws_subcommand == "project":
            ws_project_cmd(args)
    else:
        # Default behavior - show workspace directory
        workspace_dir = MetaConfig.find_workspace_dir(args.project_dir)
        print(f"Workspace Directory: {workspace_dir}")


def ws_init_cmd(args):
    """Initialize a new forgather workspace."""
    import os
    from jinja2 import Environment, BaseLoader

    # Determine target directory - create forgather_workspace subdirectory
    target_dir = os.path.join(os.getcwd(), "forgather_workspace")

    if os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' already exists")
        return 1

    # Create the directory
    os.makedirs(target_dir, exist_ok=True)
    print(f"Creating forgather workspace in: {target_dir}")

    # Template context
    context = {
        "workspace_name": args.name,
        "worspace_description": args.description,  # Note: keeping typo to match template
        "forgather_dir": os.path.abspath(args.forgather_dir),
        "search_paths": args.search_paths or [],
        "no_defaults": args.no_defaults,
    }

    # Template content inlined from the example files
    readme_template = """# {{ workspace_name }}

{{ worspace_description }}
"""

    base_directories_template = """-- set ns.forgather_dir = "{{ forgather_dir }}"
"""

    meta_defaults_template = """{% raw -%}
-- set ns = namespace()

## Import default paths; common to meta and regular templates.
-- include "base_directories.yaml"
-- set ns.forgather_templates_dir = joinpath(ns.forgather_dir, "templatelib")

## Search these directories for templates
## The list is split, which makes it easier to selectively append or prepend.
searchdir:
-- block searchdir_project
    - "{{ joinpath(project_dir, 'templates') }}"
-- endblock searchdir_project

-- block searchdir_common
{% endraw -%}
{% for search_path in search_paths %}
    - "{{ search_path }}"
{%- endfor %}
{% raw %}
    - "{{ joinpath(workspace_root, 'forgather_workspace') }}"
{%- endraw %}
{% if not no_defaults %}
{%- raw %}
    - "{{ joinpath(ns.forgather_templates_dir, 'examples') }}"
    - "{{ joinpath(ns.forgather_templates_dir, 'base') }}"
{%- endraw %}
{% endif %}

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


def ws_project_cmd(args):
    """Create a new forgather project in the workspace."""
    import os
    from jinja2 import Environment, BaseLoader

    # Determine project directory name
    if args.project_dir:
        project_dir_name = args.project_dir
    else:
        # Default: replace spaces with underscores
        project_dir_name = args.name.replace(" ", "_").lower()

    # Create target directory
    target_dir = os.path.join(os.getcwd(), project_dir_name)

    if os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' already exists")
        return 1

    # Create directory structure
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "templates", "configs"), exist_ok=True)
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

    default_config_template = """-- set ns = namespace()
-- include "base_directories.yaml"

meta: &meta_output !dict:@meta
    config_name: "Default"
    config_description: "Default Config"
    config_class: "none"
{%- raw %}
    project_dir: "{{ project_dir }}"
    workspace_root: "{{ workspace_root }}"
    forgather_dir: "{{ ns.forgather_dir }}"
{%- endraw %}


dynamic_args: []

main: "Main Output"
"""

    # Render templates with whitespace control
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

    # Create files
    files_to_create = {
        "README.md": readme_template,
        "meta.yaml": meta_template,
        os.path.join(
            "templates", "configs", f"{args.default_config}"
        ): default_config_template,
    }

    for filepath, template_content in files_to_create.items():
        template = env.from_string(template_content)
        rendered_content = template.render(**context)

        full_path = os.path.join(target_dir, filepath)
        with open(full_path, "w") as f:
            f.write(rendered_content)
        print(f"Created: {filepath}")

    print(f"\nForgather project '{args.name}' created successfully!")
    print(f"Project directory: {target_dir}")
    return 0


def control_cmd(args):
    """Handle trainer control commands."""
    try:
        client = get_default_client()

        if args.control_subcommand == "list":
            jobs = client.list_jobs()

            if hasattr(args, "remote") and args.remote:
                # Parse remote host:port
                try:
                    host, port = args.remote.split(":")
                    port = int(port)
                    if hasattr(client, "list_jobs_remote"):
                        remote_jobs = client.list_jobs_remote(host, port)
                        jobs.extend(remote_jobs)
                    else:
                        print(
                            f"Warning: Remote job listing not supported by current client"
                        )
                except ValueError:
                    print(
                        f"Error: Invalid remote format '{args.remote}'. Use HOST:PORT format."
                    )
                    return 1

            if not jobs:
                print("No discoverable training jobs found.")
                return 0

            # Check which jobs are still alive and mark dead ones
            import psutil

            alive_jobs = []
            dead_jobs = []

            for job in jobs:
                try:
                    # Check if process is still running
                    if psutil.pid_exists(job.pid):
                        proc = psutil.Process(job.pid)
                        if proc.is_running():
                            alive_jobs.append((job, "✓"))
                        else:
                            dead_jobs.append((job, "✗"))
                    else:
                        dead_jobs.append((job, "✗"))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    dead_jobs.append((job, "✗"))

            # Display results
            all_jobs = alive_jobs + dead_jobs

            print("Discoverable training jobs:")
            print(
                f"{'Status':<3} {'Job ID':<30} {'Host':<15} {'Port':<6} {'PID':<8} {'Started'}"
            )
            print("-" * 75)
            for job, status in all_jobs:
                import datetime

                started = datetime.datetime.fromtimestamp(job.started_at).strftime(
                    "%m/%d %H:%M"
                )
                print(
                    f"{status:<3} {job.job_id:<30} {job.host:<15} {job.port:<6} {job.pid:<8} {started}"
                )

            if dead_jobs:
                print(f"\n✗ = Process not running ({len(dead_jobs)} dead job(s) found)")
                print("Tip: Use 'forgather control cleanup' to remove dead job files")

        elif args.control_subcommand == "status":
            status = client.get_status(args.job_id)
            print("Job Status:")
            for key, value in status.items():
                if key == "timestamp":
                    import datetime

                    value = datetime.datetime.fromtimestamp(value).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                print(f"  {key}: {value}")

        elif args.control_subcommand == "stop":
            response = client.graceful_stop(args.job_id)
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return 1

        elif args.control_subcommand == "save":
            response = client.save_checkpoint(args.job_id)
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return 1

        elif args.control_subcommand == "save-stop":
            response = client.save_and_stop(args.job_id)
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return 1

        elif args.control_subcommand == "cleanup":
            jobs = client.list_jobs()
            if not jobs:
                print("No job files found.")
                return 0

            # Find dead jobs
            import psutil
            import shutil
            from pathlib import Path

            dead_jobs = []

            for job in jobs:
                try:
                    if not psutil.pid_exists(job.pid):
                        dead_jobs.append(job)
                    else:
                        proc = psutil.Process(job.pid)
                        if not proc.is_running():
                            dead_jobs.append(job)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    dead_jobs.append(job)

            if not dead_jobs:
                print("No dead job files found.")
                return 0

            print(f"Found {len(dead_jobs)} dead job(s):")
            for job in dead_jobs:
                import datetime

                started = datetime.datetime.fromtimestamp(job.started_at).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"  {job.job_id} (PID {job.pid}, started {started})")

            if not args.force:
                response = input(f"\nRemove {len(dead_jobs)} dead job file(s)? [y/N]: ")
                if response.lower() not in ["y", "yes"]:
                    print("Cleanup cancelled.")
                    return 0

            # Remove dead job directories
            removed_count = 0
            jobs_dir = Path.home() / ".forgather" / "jobs"

            for job in dead_jobs:
                job_dir = jobs_dir / job.job_id
                try:
                    if job_dir.exists():
                        shutil.rmtree(job_dir)
                        removed_count += 1
                        print(f"✓ Removed {job.job_id}")
                except Exception as e:
                    print(f"✗ Failed to remove {job.job_id}: {e}")

            print(f"\nCleanup complete: {removed_count} job file(s) removed.")

        elif args.control_subcommand == "abort":
            # Show warning and ask for confirmation unless --force is used
            if not hasattr(args, "force") or not args.force:
                print(
                    "⚠️  WARNING: Abort will stop training immediately WITHOUT saving!"
                )
                print(
                    "   This action cannot be undone and will lose all unsaved progress."
                )
                response = input(f"\nAbort training job '{args.job_id}'? [y/N]: ")
                if response.lower() not in ["y", "yes"]:
                    print("Abort cancelled.")
                    return 0

            response = client.abort(args.job_id)
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return 1

        else:
            print(f"Unknown control subcommand: {args.control_subcommand}")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0
