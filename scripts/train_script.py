import os
import argparse
from argparse import RawTextHelpFormatter
import random
import sys
from typing import Any, Callable, Iterable, Optional, Union, List, Type
import types
import importlib
import time
import datetime
import platform
import yaml
import gc

from pprint import pformat
import numpy as np
from loguru import logger
import torch
import torch.distributed as distributed
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed import get_rank, get_world_size
import transformers
import datasets

# Default environment, when not using torchrun or accelerate to start the process.
DEFAULT_ENVIRON = (
    ("LOCAL_RANK", "0"),
    ("RANK", "0"),
    ("WORLD_SIZE", "1"),
    ("LOCAL_WORLD_SIZE", "1"),
    ("MASTER_ADDR", "localhost"),
    ("MASTER_PORT", "29501"),
)

def init_process(args):
    """
    Check requirements, show process specific info, and set process specific variables

    This is called before torch-distributed and logging have been initialized.
    """
    # https://pytorch.org/docs/stable/distributed.html
    assert distributed.is_available(), "This script requires Torch Distributed"

    # See: https://pytorch.org/docs/stable/elastic/run.html
    if "LOCAL_RANK" not in os.environ:
        logger.warning("LOCAL_RANK is undefined. Script was not launched with torchrun or accelerate. "
            "Defining default environment variables for single process execution...")
        for key, value in DEFAULT_ENVIRON:
            os.environ[key] = value
    
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = local_rank
    torch.cuda.set_device(device)

    logger.info(
        f"Distributed Process: world_size={world_size}, "
        f"global rank={global_rank}, local rank={local_rank}, "
        f"default_device={torch.cuda.current_device()}"
    )
    # Instantiating 'transformers.TrainingArguments' initializes torch-distributed.
    # Skip, if it has already performed the initialization.
    if not distributed.is_initialized():
        logger.info("Initializing torch.distributed")
        distributed.init_process_group(backend=args.torch_backend)
    else:
        logger.info("Torch distributed has already been initialized.")

def init_logging(args):
    if int(os.environ['RANK']) == 0:
        log_level = args.log_level
        # TODO: Is there a version which takes a string?
        transformers.utils.logging.set_verbosity_info()
        if args.debug_ctor:
            logger.enable("aiws.latent")
    else:
        log_level = args.secondary_log_level

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Extensible model training script.",
        epilog=(
            "This script should be run with torchrun or accelerate...\n"
            "    torchrun --nproc-per-node 1 --standalone train_script.py -I ./config my_config.yaml\n"
            "    accelerate launch train_script.py my_config.yaml"
        )
    )
    parser.add_argument(
        'config',
        type=str,
        metavar="config-file",
        help="Path to yaml configuration file"
    )
    parser.add_argument(
        '-I', '--include',
        required=False,
        action='append',
        metavar="include-path",
        #default=[os.getcwd()],
        help="One or more search paths for Jinja2 include files used in configuration file."
    )
    parser.add_argument(
        '--debug-ctor',
        action='store_true',
        help="Enable dynamic constructor debugging."
    )
    parser.add_argument(
        '-l', '--log-level',
        default="INFO",
        help="Set the log level for the main process: INFO, WARNING, DEBUG, ...; default=INFO"
    )
    parser.add_argument(
        '--secondary-log-level',
        default="WARNING",
        help="Set the log level for the secondary processes, if any: INFO, WARNING, DEBUG, ...; default=WARNING"
    )
    parser.add_argument(
        '-s',
        '--syspath',
        type=str,
        default=None,
        help="Add sys.path for relative imports"
    )
    parser.add_argument(
        '-p',
        '--project-dir',
        type=str,
        default='.',
        help="The relative path to the project directory."
    )
    parser.add_argument(
        '--torch-backend',
        type=str,
        default=None,
        help="The torch backend to use."
    )
    parser.add_argument(
        '-d',
        '--dryrun',
        action='store_true',
        help="Proccess configuration and exit without train, eval, save, etc."
    )

    args = parser.parse_args(args)
    logger.info(f"args: {args}")

    # We only resolve these modules after we know where to look for them.pconfig
    if args.syspath is not None:
        sys.path.insert(0, args.syspath)
    # We only resolve these modules after we know where to look for them.
    global ConfigEnvironment, base_preprocessor_globals, format_line_numbers
    global TrainingScriptConfig
    
    from forgather.utils import format_line_numbers
    from forgather.config import ConfigEnvironment
    from aiws.config import base_preprocessor_globals
    from aiws.training_loop import TrainingScriptConfig
    return args

class TrainingScriptConfigError(Exception):
    pass


def load_config(args):
    config_path = args.config
    search_path = args.include
    project_directory = args.project_dir
    env_globals = base_preprocessor_globals() | dict(
        script_args=pformat(args),
        project_directory=project_directory,
        world_size=os.environ['WORLD_SIZE'],
        rank=int(os.environ['RANK']),
        local_rank=int(os.environ['LOCAL_RANK']),
    )

    cfg_environment = ConfigEnvironment(
        searchpath=search_path,
        globals=env_globals,
    )

    loaded_config = cfg_environment.load(config_path)
    return loaded_config


def write_configuration_log(args, config, preprocessed_config):
    if config.logging_dir is None:
        return
    log_name = config.experiment_name + '_' + str(time.time_ns()) + ".yaml"
    config_log = os.path.join(config.logging_dir, log_name)
    
    with open(config_log, 'w') as file:
        file.write(preprocessed_config)

def set_seed(seed):
    """
    Set random seeds on all libraries we may potentially use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(config):
    trainer = config.trainer
    train_output = trainer.train()
    distributed.barrier()
    if hasattr(trainer, "log_metrics"):
        metrics = train_output.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    if hasattr(trainer, "save_state"):
        trainer.save_state()
    
def eval(config):
    trainer = config.trainer
    gc.collect()
    torch.cuda.empty_cache()
    metrics = trainer.evaluate()
    if hasattr(trainer, "log_metrics"):
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

def save(config):
    config.trainer.save_model(config.output_dir)
    
@record # Improves diagnostics for distributed training
def main():
    args = parse_args()
    init_process(args)
    init_logging(args)
    loaded_config = load_config(args)
    config = TrainingScriptConfig(**loaded_config.config)

    logger.info("*" * 40)
    logger.info(f"Training started with world-size of {get_world_size()}")
    logger.debug(f"preprocessed-config:\n{format_line_numbers(loaded_config.pp_config)}")
    logger.debug(f"config:\n{pformat(config)}")
    logger.info("*" * 40)

    # Materialize the trainer
    config.trainer = config.trainer(pp_config=loaded_config.pp_config)
    
    if args.dryrun:
        sys.exit(0)

    logger.debug(config)
    write_configuration_log(args, config, loaded_config.pp_config)

    if config.do_train:
        train(config)
    if config.do_eval:
        eval(config)
    if config.do_train and config.do_save:
        save(config)

if __name__ == "__main__":
    main()