import logging
from dataclasses import dataclass, field
from typing import Any

from torch.distributed.elastic.multiprocessing.errors import record

from forgather.dotdict import DotDict
from forgather.ml.distributed import (
    DistributedEnvInterface,
    from_env,
    prefix_logger_rank,
)
from forgather.project import Project

logger = logging.getLogger(__name__)
prefix_logger_rank(logger)


@dataclass(kw_only=True)
class TrainingScript:
    """
    This is just a wrapper around a trainer class which calls the requested
    trainer methods.
    """

    meta: dict
    do_save: bool = False
    do_train: bool = True
    do_eval: bool = False
    distributed_env: DistributedEnvInterface = field(default_factory=from_env)
    trainer: Any

    def __post_init__(self):
        self.meta = DotDict(self.meta)

    @record
    def run(self):
        # In a distriubted environment, we only want one process to print messages

        logger.info("**** Training Script Started *****")
        logger.info(f"config_name: {self.meta.config_name}")
        logger.info(f"config_description: {self.meta.config_description}")
        logger.info(f"output_dir: {self.meta.output_dir}")
        if "logging_dir" in self.meta:
            logger.info(f"logging_dir: {self.meta.logging_dir}")

        if self.do_train:
            # This is where the actual 'loop' is.
            output = self.trainer.train()

            logger.info("**** Training Completed *****")
            logger.info(output)

        if self.do_eval:
            output = self.trainer.evaluate()

            logger.info("**** Evaluation Completed *****")
            logger.info(output)

        if self.do_save:
            self.trainer.save_model(self.meta.output_dir)
            logger.info(f"Model saved to: {self.meta.output_dir}")


def training_loop(project_directory, config_template=""):
    """
    A mini-training-loop for use with accelerate.notebook_launcher

    project_directory: The location of the project directory, relative to CWD
    config_template_path: Path to a configuration template in the project
    ```
    from accelerate import notebook_launcher
    from forgather.ml.training_script import training_loop

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

    # Load the project
    proj = Project(config_template, project_directory)

    # Materialize the config
    training_script = proj()

    # Run it!
    training_script.run()
