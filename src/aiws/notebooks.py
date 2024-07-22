import os
from dataclasses import fields
from typing import Iterator, Tuple

from forgather.dynamic import parse_module_name_or_path, parse_dynamic_import_spec
from forgather import Latent
from IPython import display


def find_file_specs(config):
    """
    Generate all referenced file names in config
    """
    spec_set = set()
    for latent in Latent.all_latents(config):
        # Skip, if not string
        if not isinstance(latent.constructor, str):
            continue
        try:
            module_name_or_path, symbol_name = parse_dynamic_import_spec(
                latent.constructor
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
        yield spec


def display_meta(meta, title=""):
    md = f"{title}"
    relpath = os.path.relpath(meta.project_dir)
    md += f"Project Directory: {relpath}\n\n"
    relpath = os.path.relpath(meta.meta_path)
    md += f"Meta Config: [{relpath}]({relpath})\n\n"
    md += f"Template Search Paths:\n"
    for path in meta.searchpath:
        relpath = os.path.relpath(path)
        md += f"- [{relpath}]({relpath})\n"
    display.display(display.Markdown(md))


def list_templates(templates: Iterator[Tuple[str, str]], title: str = ""):
    """
    Given a template iterator, display a list of templates

    The iterator is expected to yield (template_name, template_path)
    """
    md = f"{title}"
    for template_name, template_path in templates:
        md += f"- [{template_name}]({os.path.relpath(template_path)})\n"
    display.display(display.Markdown(md))


def display_referenced_templates_tree(environment, path, title=""):
    s = f"{title}"
    # Yields # tuple(level: int, name: str, path: str)
    for level, name, path in environment.find_referenced_templates(path):
        s += f"{' ' * 4 * level}- [{name}]({os.path.relpath(path)})\n"
    display.display(display.Markdown(s))


def display_preprocessed_template(environment, template, title=""):
    md = f"{title}" f"```yaml\n{environment.preprocess(template)}\n```\n"
    display.display(display.Markdown(md))


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

    display.display(display.Markdown(md))


def display_referenced_source_list(config, title=""):
    md = f"{title}"
    for file, callable in sorted(find_file_specs(config)):
        md += f"- [{file}]({file}) : {callable}\n"
    display.display(display.Markdown(md))


def display_filelink(path, title="", name=None):
    if name is None:
        name = path
    md = f"{title}" f"[{name}]({path})\n"
    display.display(display.Markdown(md))


def get_train_cmdline(train_script_path, meta, nproc="gpu", cuda_devices=None):
    s = (
        f"torchrun --standalone --nproc-per-node '{nproc}' '{train_script_path}'"
        + f" -p '{meta.project_dir}' -s '{meta.system_path}'"
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
    from aiws.config import MetaConfig

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
        display.display(display.Markdown(md))


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
