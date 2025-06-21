import os
from dataclasses import fields
from typing import Iterator, Tuple, Any
from pprint import pformat

from IPython import display as ds

from forgather.latent import Latent, CallableNode
from forgather.dynamic import (
    parse_module_name_or_path,
    parse_dynamic_import_spec,
    import_dynamic_module,
    walk_package_modules,
)
from forgather.meta_config import preprocessor_globals, MetaConfig
from forgather.config import ConfigEnvironment
from forgather.codegen import generate_code
from forgather.yaml_encoder import to_yaml
from forgather.ml.utils import count_parameters


def display_md(md: str):
    display(ds.Markdown(md))


def find_file_specs(config):
    """
    Generate all referenced file names in config
    """
    spec_set = set()
    for level, node, sub_nodes in Latent.walk(config):
        if not isinstance(node, CallableNode):
            continue

        # Skip built-ins
        if not ":" in node.constructor:
            continue
        try:
            module_name_or_path, symbol_name = parse_dynamic_import_spec(
                node.constructor
            )
            module_name, module_path = parse_module_name_or_path(module_name_or_path)
        except:
            # Skip on parse errors; not all values are import specs
            continue
        # Skip, if not file-spec
        if module_path is None:
            continue
        spec = (module_path, symbol_name)

        # Skip, if already seen
        if spec in spec_set:
            continue
        spec_set.add(spec)
        yield module_path, symbol_name, node.submodule_searchpath


def render_meta(meta, title=""):
    md = f"{title}"
    md += f"Meta Config: [{os.path.abspath(meta.meta_path)}]({os.path.relpath(meta.meta_path)})\n\n"
    for level, name, path in meta.environment.find_referenced_templates(meta.name):
        md += f"{' ' * 4 * level}- [{name}]({os.path.relpath(path)})\n"
    md += "\n"
    md += f"Template Search Paths:\n"
    for path in meta.searchpath:
        md += f"- [{os.path.abspath(path)}]({os.path.relpath(path)})\n"
    md += "\n"
    return md


def display_meta(meta, title=""):
    display(ds.Markdown(render_meta(meta, title)))


def render_template_list(
    templates: Iterator[Tuple[str, str]], title: str = "", with_paths=False
):
    """
    Given a template iterator, display a list of templates

    The iterator is expected to yield (template_name, template_path)
    """
    md = f"{title}"
    for template_name, template_path in templates:
        md += f"- [{template_name}]({os.path.relpath(template_path)})"
        if with_paths:
            md += " " + os.path.abspath(template_path)
        md += "\n"
    md += "\n"
    return md


def list_templates(templates: Iterator[Tuple[str, str]], title: str = ""):
    display(ds.Markdown(render_template_list(templates, title)))


def render_referenced_templates_tree(environment, path, title=""):
    md = f"{title}"
    # Yields # tuple(level: int, name: str, path: str)
    for level, name, path in environment.find_referenced_templates(path):
        md += f"{' ' * 4 * level}- [{name}]({os.path.relpath(path)})\n"
    return md


def display_referenced_templates_tree(environment, path, title=""):
    display(ds.Markdown(render_referenced_templates_tree(environment, path, title)))


# Render code via Markdown render
def display_codeblock(language, source, header=None):
    display(ds.Markdown(render_codeblock(language, source, header)))


# Render code via Markdown render
def display_markdown(text):
    display(ds.Markdown(text))


# An alias for display_codeblock()... until it is fully depricated.
def show_codeblock(*args, **kwargs):
    display_codeblock(*args, **kwargs)


def render_codeblock(language, source, header=None):
    header = header + "\n" if header is not None else ""
    return f"{header}```{language}\n{source}\n\n```\n\n"


def render_referenced_source_list(config, title="", deep=False):
    """
    Setting the 'deep' flag requires actually loading the modules
    """
    visited_modules = set()
    md = f"{title}"
    for file, callable, searchpath in find_file_specs(config):
        md += f"- [{file}]({os.path.relpath(file)}) : {callable}\n"
        if deep:
            mod = import_dynamic_module(file, searchpath=searchpath)
            for level, submod in walk_package_modules(mod):
                module_name = submod.__name__
                origin = submod.__spec__.origin
                hasht = (module_name, origin)
                if hasht in visited_modules:
                    continue
                visited_modules.add(hasht)
                md += f"{' ' * 4 * (level + 1)}- [{origin}]({os.path.relpath(origin)}) : {module_name}\n"
    return md


def render_model(model: "nn.Module") -> str:
    params = count_parameters(model)
    md = render_codeblock("yaml", repr(model))
    md += f"\n- Total Parameters: {params['total']}\n- Trainable Parameters: {params['trainable']}"
    return md


def display_model(model: "nn.Module"):
    display_md(render_model(model))


def render_output_targets(config, title=""):
    md = f"{title}"

    for target in config.keys():
        md += f"- {target}\n"
    md += "\n"
    return md


def display_referenced_source_list(config, title="", deep=False):
    display(ds.Markdown(render_referenced_source_list(config, title, deep)))


def render_filelink(path, title="", name=None):
    if name is None:
        name = path
    md = f"{title}" f"[{name}]({path})\n"
    return md


def display_filelink(path, title="", name=None):
    display(ds.Markdown(render_filelink(path, title, name)))


def get_train_cmdline(train_script_path, meta, nproc="gpu", cuda_devices=None):
    s = (
        f"torchrun --standalone --nproc-per-node '{nproc}' '{train_script_path}'"
        + f" -p '{os.path.abspath(meta.project_dir)}'"
    )
    if meta.system_path is not None:
        s += f" -s '{os.path.abspath(meta.system_path)}'"
    if cuda_devices is not None:
        s = f"CUDA_VISIBLE_DEVICES='{cuda_devices}' " + s
    return s


def make_train_script(
    train_script_path,
    project_directory,
    config_template=None,
    script_name="train.sh",
    nproc="gpu",
    cuda_devices=None,
):
    """
    Generate a bash training script from a project meta-config

    The generated script will be written to 'project_directory' and all paths will be
    relative to this location.

    project_directory: The project directory. Assumes meta-config is 'meta_config.yaml'
    script_name: The name of the output script. If none, the script can be specified on the command-line.
    nproc: Number of processes; 'gpu' is number of available GPUs
    cuda_devices: List of CUDA devices to limit training to.
        i.e. If you wish to only CUDA 0 and 1, then "0,1"
    """
    import stat
    from forgather.meta_config import MetaConfig

    prev_cwd = os.getcwd()

    try:
        os.chdir(project_directory)
        project_directory = "."
        meta = MetaConfig(project_directory)

        if config_template is None:
            config_template = r"${@}"
        cmdline = get_train_cmdline(train_script_path, meta, nproc, cuda_devices)

        with open(script_name, "w") as f:
            f.write("#!/bin/bash\n" + cmdline + f' "{config_template}"\n')
            os.chmod(
                f.fileno(), stat.S_IREAD | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            )
    finally:
        os.chdir(prev_cwd)


def get_tb_command(project, local_host=True):
    cmd = "tensorboard "
    if not local_host:
        cmd += "--bind_all "

    directory = os.path.abspath(project("meta")["output_dir"])
    cmd += f'--logdir "{directory}"'
    return cmd


def display_tb_command(project, local_host=True):
    cmd = get_tb_command(project, local_host)
    md = f"#### Tensorboard Command\n\n" f"```bash\n{cmd}\n```"
    display_md(md)


def generate_trainingscript(project, cuda_devices=None):
    """
    A high-level version of make_train_script(), for use in a notebook.

    Given a project and target cuda devices, generates a trainscript in the project directory,
    named after the project's configuration name.

    Finally, reads back the generated script and renders it as Markdown.
    """
    meta = project("meta")
    train_script_path = os.path.join(
        meta["forgather_dir"], "scripts", "train_script.py"
    )
    script_name = os.path.splitext(os.path.basename(project.config_name))[0] + ".sh"
    make_train_script(
        train_script_path=os.path.abspath(train_script_path),
        project_directory=project.project_dir,
        config_template=project.config_name,
        script_name=script_name,
        cuda_devices=cuda_devices,
    )

    # Read back to verify
    script_path = os.path.join(project.project_dir, script_name)
    with open(script_path, "r") as f:
        md = (
            f"#### Generated Shell Script\n"
            f"[{script_name}]({os.path.relpath(script_path)})\n"
            f"```bash\n{f.read()}\n```"
        )
        display_md(md)


def render_project_readme(project_dir):
    md_path = os.path.join(project_dir, "README.md")
    if os.path.exists(md_path):
        with open(md_path, "r") as f:
            md = f.read() + "\n\n"
        return md
    else:
        return ""


def show_project_readme(project_dir):
    md = render_project_readme(project_dir)
    if len(md):
        display(ds.Markdown(md))


def delete_dir(target, prompt):
    """
    Recursively remove a directory, with interactive confirmation.
    """
    from shutil import rmtree

    print(prompt)
    print(f"This will delete '{target}'")
    response = input("Enter 'YES' to confirm: ")
    if response == "YES":
        rmtree(target, ignore_errors=True)
        print("deleted")
    else:
        print("aborted")


def delete_output_dir(proj):
    meta = proj("meta")
    delete_dir(meta["output_dir"], "Delete the configuration's output directory?")


def render_project_index(
    project_dir: str = ".",
    /,
    config_template: str = "",
    show_available_templates: bool = False,
) -> str:
    """
    Render project information

    project_dir: The location of the project directory
    show_available_templates: List all templates available on the search path.
    """
    try:
        md = ""
        md += render_project_readme(project_dir)
        md += f'#### Project Directory: "{os.path.abspath(project_dir)}"\n\n'

        # Load project meta and get default config
        meta = MetaConfig(project_dir)

        md += render_meta(meta, "## Meta Config\n")
        md += render_template_list(
            meta.find_templates(meta.config_prefix), "## Available Configurations\n"
        )

        config_template_path = meta.config_path(config_template)
        default_config = meta.default_config()
        md += f"Default Configuration: {default_config}\n\n"
        if show_available_templates:
            md += render_template_list(
                sorted(meta.find_templates("")), "## Available Templates\n"
            )
        return md
    except Exception as e:
        md += render_codeblock("", repr(e), "# RAISED EXCEPTION\n\n")
        setattr(e, "markdown", md)
        raise e


def display_project_index(
    project_dir: str = ".",
    /,
    **kwargs,
) -> None:
    try:
        display(ds.Markdown(render_project_index(project_dir, **kwargs)))
    except Exception as e:
        display(ds.Markdown(e.markdown))
        delattr(e, "markdown")
        raise e


def render_config(
    project_dir: str = ".",
    /,
    config_template: str = "",
    show_pp_config: bool = False,
    show_loaded_config: bool = False,
    show_generated_code: bool = False,
    pp_first: bool = False,
    **kwargs,
) -> str:
    """
    Render config information

    project_dir: The location of the project directory
    config_template: The configuration to display. If "", the default is shown.
    show_pp_config: Show the preprocessed configuration.
    show_loaded_config: After loading the preprocessed configuration, render the node-graph as YAML.
    show_generated_code: After loading the preprocessed configuration, render the node-graph as Python code.
    pp_first: Normally a config is preprocessed and loaded in a single step. This breaks the process down
        into seperate steps, which can be useful for debugging YAML errors.
    kwargs: These kwargs are passed in to the Jinja2 environment and are available to templates. Usually not
        required.
    """
    try:
        md = ""
        meta = MetaConfig(project_dir)

        config_template_path = meta.config_path(config_template)
        default_config = meta.default_config()

        active_config = config_template if len(config_template) else default_config

        # Create new config environment and load configuration
        environment = ConfigEnvironment(
            searchpath=meta.searchpath,
            global_vars=preprocessor_globals(project_dir, meta.workspace_root),
        )

        md += render_referenced_templates_tree(
            environment, config_template_path, "## Included Templates\n"
        )

        # Perform discrete pp-step, if set.
        # Useful, should there be a failure in YAML processing.
        if pp_first and show_pp_config:
            pp_config = environment.preprocess(config_template_path, **kwargs)
            md += render_codeblock("yaml", pp_config, "## Preprocessed Config\n")

        config, pp_config = environment.load(config_template_path, **kwargs).get()

        # TODO: Add API to only materialize 'safe' objects and use it here.
        config_meta = Latent.materialize(config.meta)
        md += f"### Config Metadata:\n\n"
        md += render_codeblock("python", pformat(config_meta))

        md += render_referenced_source_list(config, title="## Modules\n", deep=False)
        md += render_output_targets(config, title="## Output Targets\n")

        if not pp_first and show_pp_config:
            md += render_codeblock("yaml", pp_config, "## Preprocessed Config\n")

        if show_loaded_config:
            md += render_codeblock("yaml", to_yaml(config), "## Loaded Configuration\n")
        if show_generated_code:
            md += render_codeblock(
                "python", generate_code(config["main"]), "## Generated Code\n"
            )
        return md
    except Exception as e:
        md += render_codeblock("", repr(e), "# RAISED EXCEPTION\n\n")
        setattr(e, "markdown", md)
        raise e


def display_config(
    project_dir=".",
    /,
    **kwargs,
) -> None:
    try:
        display(ds.Markdown(render_config(project_dir, **kwargs)))
    except Exception as e:
        display(ds.Markdown(e.markdown))
        delattr(e, "markdown")
        raise e


def display_model_project_index(project_dir="."):
    md = ""
    try:
        md += render_project_readme(project_dir)
        md += f'#### Project Directory: "{os.path.abspath(project_dir)}"\n\n'
        meta = MetaConfig(project_dir)
        md += render_meta(meta, "## Meta Config\n")
        template_iter = filter(
            lambda x: not x[0].startswith("abstract/"),
            meta.find_templates(prefix="models"),
        )
        md += render_template_list(
            template_iter, "## Available Models\n", with_paths=True
        )
    finally:
        display_md(md)
