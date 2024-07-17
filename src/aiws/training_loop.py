from typing import Any
from dataclasses import dataclass, field
import os


@dataclass(kw_only=True)
class TrainingScriptConfig:
    output_dir: os.PathLike | str
    logging_dir: os.PathLike | str
    experiment_name: str = "Undefined"
    experiment_description: str = "Undefined"
    trainer: Any
    do_save: bool = False
    do_train: bool = True
    do_eval: bool = False

    def __post_init__(self):
        if int(os.environ.get("RANK", 0)) == 0:
            self.validate_dirs()

    def validate_dirs(self):
        """
        Ensure that output directory exists and make it more difficult to accidentally
        overwrite a model directory.
        """
        for dir in (self.output_dir, self.logging_dir):
            if dir is not None and not os.path.isdir(dir):
                print(f"Creating directory: {dir}")
                os.makedirs(dir, exist_ok=True)


def training_loop(project_directory, config_template_path, backend=None):
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

    project_directory: The relative path to the project directory (with meta_config.yaml)
    config_template_path: the relative path (to project_directory) to a
        configuration template.
    backend: The torch backend to use.
        https://pytorch.org/docs/stable/distributed.html#backends
    ```
    """
    import os
    from forgather.config import load_config, ConfigEnvironment, fconfig, pconfig
    from aiws.config import base_preprocessor_globals
    from torch.distributed import init_process_group

    # Get Torch Distributed parameters from environ.
    # Provide single-process defautls, if variables are not set.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Only init torch-distributed, if more than one process is used.
    if world_size > 1:
        init_process_group(backend=backend)

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
    config = TrainingScriptConfig(**loaded_config.config)

    # In a distriubted environment, we only want one process to print messages
    is_main_process = local_rank == 0

    if is_main_process:
        print("**** Training Started *****")
        print(f"experiment_name: {config.experiment_name}")
        print(f"experiment_description: {config.experiment_description}")
        print(f"output_dir: {config.output_dir}")
        print(f"logging_dir: {config.logging_dir}")

    # Materialize the trainer
    config.trainer = config.trainer(pp_config=loaded_config.pp_config)

    if config.do_train:
        # This is where the actual 'loop' is.
        metrics = config.trainer.train().metrics

        if is_main_process:
            print("**** Training Completed *****")
            print(metrics)

    if config.do_eval:
        metrics = config.trainer.evaluate()

        if is_main_process:
            print("**** Evaluation Completed *****")
            print(metrics)

    if config.do_save:
        config.trainer.save_model()
        if is_main_process:
            print(f"Model saved to: {config.trainer.args.output_dir}")
