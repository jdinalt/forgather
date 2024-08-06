import os
from dataclasses import fields
from typing import Iterator, Tuple

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


def display_meta(meta, title=""):
    md = f"{title}"
    md += f"Project Directory: {os.path.abspath(meta.project_dir)}\n\n"
    md += f"Meta Config: [{os.path.abspath(meta.meta_path)}]({os.path.relpath(meta.meta_path)})\n\n"
    md += f"Template Search Paths:\n"
    for path in meta.searchpath:
        md += f"- [{os.path.abspath(path)}]({os.path.relpath(path)})\n"
    display(ds.Markdown(md))


def list_templates(templates: Iterator[Tuple[str, str]], title: str = ""):
    """
    Given a template iterator, display a list of templates

    The iterator is expected to yield (template_name, template_path)
    """
    md = f"{title}"
    for template_name, template_path in templates:
        md += f"- [{template_name}]({os.path.relpath(template_path)})\n"
    display(ds.Markdown(md))


def display_referenced_templates_tree(environment, path, title=""):
    s = f"{title}"
    # Yields # tuple(level: int, name: str, path: str)
    for level, name, path in environment.find_referenced_templates(path):
        s += f"{' ' * 4 * level}- [{name}]({os.path.relpath(path)})\n"
    display(ds.Markdown(s))


def display_preprocessed_template(environment, template, title=""):
    md = f"{title}" f"```yaml\n{environment.preprocess(template)}\n```\n"
    display(ds.Markdown(md))


def display_referenced_templates(environment, template, title=""):
    visited_set = set()
    md = f"{title}"

    for _, name, path in environment.find_referenced_templates(template):
        if path in visited_set:
            continue
        visisted_set.add(path)

        with open(path, "r") as f:
            data = f.read()

        md += f"#### [{name}]({os.path.relpath(path)})\n" f"```yaml\n{data}\n```\n---\n"

    display(ds.Markdown(md))


def display_referenced_source_list(config, title="", deep=False):
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

    display(ds.Markdown(md))


def display_filelink(path, title="", name=None):
    if name is None:
        name = path
    md = f"{title}" f"[{name}]({path})\n"
    display(ds.Markdown(md))


def get_train_cmdline(train_script_path, meta, nproc="gpu", cuda_devices=None):
    s = (
        f"torchrun --standalone --nproc-per-node '{nproc}' '{train_script_path}'"
        + f" -p '{os.path.abspath(meta.project_dir)}' -s '{os.path.abspath(meta.system_path)}'"
    )
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


def show_project_readme(project_dir):
    md_path = os.path.join(project_dir, "README.md")
    if os.path.exists(md_path):
        with open(md_path, "r") as f:
            md = f.read()
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


def display_project_index(
    project_dir=".", config_template="", materialize=True, pp_first=False
):
    """
    Display project information

    project_dir: The location of the project directory
    config_template: The configuration to display. If "", the default is shown.
    materialize: Materialize the configuration. Without doing so, dynamic imports may not show up.
    pp_first: Preprocess before loading. This can be useful for debugging, if loading raises and exception.
    """
    show_project_readme(project_dir)

    # Load project meta and get default config
    meta = MetaConfig(project_dir)
    config_template_path = meta.config_path(config_template)
    default_config = meta.default_config()

    display_meta(meta, "### Meta Config\n")
    list_templates(
        meta.find_templates(meta.config_prefix), "### Available Configurations\n"
    )
    display(ds.Markdown(f"Default Configuration: {default_config}\n\n"))
    list_templates(meta.find_templates(""), "### Available Templates\n")

    # Create new config environment and load configuration
    environment = ConfigEnvironment(
        searchpath=meta.searchpath,
        global_vars=preprocessor_globals(project_dir),
    )
    display_referenced_templates_tree(
        environment, config_template_path, "### Included Templates\n"
    )

    if pp_first:
        pp_config = environment.preprocess(config_template_path)
        display(
            ds.Markdown(f"#### Preprocessed Config\n" f"```yaml\n{pp_config}\n```\n")
        )

    config, pp_config = environment.load(config_template_path).get()

    # Materialize the configuration
    if materialize:
        main_output = Latent.materialize(config, pp_config=pp_config)

    display_referenced_source_list(config, title="### Modules\n", deep=True)

    if not pp_first:
        display(
            ds.Markdown(f"#### Preprocessed Config\n" f"```yaml\n{pp_config}\n```\n")
        )
    display(
        ds.Markdown(
            f"### Loaded Configuration to YAML\n```yaml\n{to_yaml(config)}\n```"
        )
    )
    display(
        ds.Markdown(
            f"### Generated Source Code\n```python\n{generate_code(config)}\n```"
        )
    )
