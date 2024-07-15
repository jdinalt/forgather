import os

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


def display_meta_config(path, meta_config, title=""):
    md = f"{title}"
    md += f"[{path}]({path})\n"
    for level, key, value in Latent.all_items(meta_config):
        if level == 0:
            continue
        level -= 1
        if not isinstance(value, str):
            value = str(type(value))
        else:
            value = f"'{value}'"

        if isinstance(key, str):
            md += f"{' ' * 4 * level}- {key}: {value}\n"
        elif isinstance(key, int):
            md += f"{' ' * 4 * level}- {value}\n"
    display.display(display.Markdown(md))


def display_referenced_templates_tree(environment, path, title=""):
    s = f"{title}"
    # Yields # tuple(level: int, name: str, path: str)
    for level, name, path in environment.find_referenced_templates(path):
        s += f"{' ' * 4 * level}- [{name}]({path})\n"
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

        md += f"#### [{name}]({path})\n" f"```yaml\n{data}\n```\n---\n"

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


def training_loop(project_directory, config_template_path):
    """
    A mini-training-loop for use with accelerate.notebook_launcher

    project_directory: The location of the project directory, relative to CWD
    config_template_path: Path to a configuration template in the project
    ```
    from accelerate import notebook_launcher
    from aiws.notebooks import training_loop

    notebook_launcher(
        training_loop,
        args=(project_directory, config_template_path,),
        num_processes=2
    )
    ```
    """
    import os
    from forgather.config import load_config, ConfigEnvironment, fconfig, pconfig
    from aiws.config import base_preprocessor_globals
    from transformers import set_seed

    # Ensure that initialization is deterministic.
    set_seed(42)

    # Get Torch Distributed parameters from environ.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Load meta-config
    meta_config_path = os.path.join(project_directory, "meta_config.yaml")
    metacfg = load_config(meta_config_path, project_directory=project_directory)

    # Initialize the pre-processor globals
    pp_globals = base_preprocessor_globals() | dict(
        project_directory=project_directory,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )

    # Create configuration envrionment
    cfg_environment = ConfigEnvironment(
        searchpath=metacfg.search_paths, globals=pp_globals
    )

    # Load the target configuration
    loaded_config = cfg_environment.load(config_template_path)

    # Materialize the configuration
    config = loaded_config.materialize()

    # In a distriubted environment, we only want one process to print messages
    is_main_process = local_rank == 0

    if is_main_process:
        print("**** Training Started *****")
        print(f"experiment_name: {config.experiment_name}")
        print(f"experiment_description: {config.experiment_description}")
        print(f"output_dir: {config.output_dir}")
        print(f"logging_dir: {config.logging_dir}")

    # This is where the actual 'loop' is.
    metrics = config.trainer.train().metrics

    if is_main_process:
        print("**** Training Completed *****")
        print(metrics)

    metrics = config.trainer.evaluate()

    if is_main_process:
        print("**** Evaluation Completed *****")
        print(metrics)

    if config.do_save:
        config.trainer.save_model()
        if is_main_process:
            print(f"Model saved to: {config.trainer.args.output_dir}")


def get_train_cmdline(meta_config, nproc="gpu", cuda_devices=None):
    includes = "".join(f"-I '{inc}' " for inc in meta_config.search_paths)
    s = (
        f"torchrun --standalone --nproc-per-node '{nproc}' '{meta_config.train_script_path}'"
        + f" {includes} -p '{meta_config.project_dir}' -s '{meta_config.src_dir}'"
    )
    if cuda_devices is not None:
        s = f"CUDA_VISIBLE_DEVICES='{cuda_devices}' " + s
    return s


def make_train_script(
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
    from forgather.config import load_config

    prev_cwd = os.getcwd()
    try:
        os.chdir(project_directory)
        project_directory = "."
        meta_config_path = "meta_config.yaml"
        meta_config = load_config(meta_config_path, project_directory=project_directory)

        if config_template is None:
            config_template_path = r"${@}"
        else:
            config_template_path = os.path.join(
                meta_config.project_templates, config_template
            )
        cmdline = get_train_cmdline(meta_config, nproc, cuda_devices)

        with open(script_name, "w") as f:
            f.write("#!/bin/bash\n" + cmdline + f' "{config_template_path}"\n')
            os.chmod(
                f.fileno(), stat.S_IREAD | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            )
    finally:
        os.chdir(prev_cwd)


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
