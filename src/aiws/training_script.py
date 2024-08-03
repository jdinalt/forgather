from typing import Any, Dict
from dataclasses import dataclass, field
import os
import sys
from pprint import pformat
from random import seed as py_seed

from torch import manual_seed as torch_seed
from torch import distributed
from numpy.random import seed as np_seed
from torch.distributed.elastic.multiprocessing.errors import record

from forgather.config import ConfigEnvironment, fconfig, pconfig
from aiws.config import MetaConfig, preprocessor_globals
from aiws.distributed import DistributedEnvironment
from aiws.dotdict import DotDict
from forgather import Latent


def set_seed(seed: int):
    torch_seed(seed)
    np_seed(seed)
    py_seed(seed)


@dataclass(kw_only=True)
class TrainingScript:
    meta: dict
    do_save: bool = True
    do_train: bool = True
    do_eval: bool = False
    distributed_env: DistributedEnvironment
    trainer: Any

    def __post_init__(self):
        self.meta = DotDict(self.meta)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            self.validate_dirs()

    def validate_dirs(self):
        """
        Ensure that output directory exists and make it more difficult to accidentally
        overwrite a model directory.
        """
        for dir in (self.meta.output_dir, self.meta.logging_dir):
            if dir is not None and not os.path.isdir(dir):
                print(f"Creating directory: {dir}")
                os.makedirs(dir, exist_ok=True)

    @record
    def run(self, pp_config: str = None):
        # In a distriubted environment, we only want one process to print messages
        is_main_process = self.distributed_env.local_rank == 0

        if is_main_process:
            print("**** Training Script Started *****")
            print(f"config_name: {self.meta.config_name}")
            print(f"config_description: {self.meta.config_description}")
            print(f"output_dir: {self.meta.output_dir}")
            print(f"logging_dir: {self.meta.logging_dir}")

        if pp_config is not None:
            # Store a copy of the pre-processed configuration in the logging directory.
            os.makedirs(self.meta.logging_dir, exist_ok=True)
            with open(os.path.join(self.meta.logging_dir, "config.yaml"), "w") as f:
                f.write(pp_config)

        if self.do_train:
            # This is where the actual 'loop' is.
            metrics = self.trainer.train().metrics
            if self.distributed_env.world_size > 1:
                distributed.barrier()

            if is_main_process:
                print("**** Training Completed *****")
                print(metrics)

        if self.do_eval:
            metrics = self.trainer.evaluate()

            if is_main_process:
                print("**** Evaluation Completed *****")
                print(metrics)

        if self.do_save:
            self.trainer.save_model(self.meta.output_dir)
            if is_main_process:
                print(f"Model saved to: {self.meta.output_dir}")


def training_loop(project_directory, config_template=None):
    """
    A mini-training-loop for use with accelerate.notebook_launcher

    project_directory: The location of the project directory, relative to CWD
    config_template_path: Path to a configuration template in the project
    ```
    from accelerate import notebook_launcher
    from aiws.training_script import training_loop

    notebook_launcher(
        training_loop,
        args=(project_directory, config_template_path,),
        num_processes=2
    )

    project_directory: The relative path to the project directory (with meta_config.yaml)
    config_template_path: the relative path (to project_directory) to a
        configuration template.
    log_level: The log-level to use.
    ```
    """

    set_seed(42)

    # Load meta-config
    meta = MetaConfig(project_directory)

    # Create configuration envrionment
    environment = ConfigEnvironment(
        searchpath=meta.searchpath,
        global_vars=preprocessor_globals(project_directory),
    )

    # Load the target configuration
    config, pp_config = environment.load(meta.config_path(config_template)).get()

    # Materialize the configuration
    main = config.main(pp_config=pp_config)

    # Run it!
    main.run(pp_config=pp_config)
