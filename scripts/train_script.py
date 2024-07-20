import os
import argparse
from argparse import RawTextHelpFormatter
import sys
from typing import Any, Callable, Iterable, Optional, Union, List, Type
import types

from loguru import logger
import torch
import torch.distributed as distributed
from torch.distributed.elastic.multiprocessing.errors import record
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

def init_logging(args):
    if int(os.environ['RANK']) == 0:
        log_level = args.log_level
        # TODO: Is there a version which takes a string?
        transformers.utils.logging.set_verbosity_info()
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
        'config_template',
        type=str,
        metavar="config-template",
        help="Configuration Template Name"
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

    args = parser.parse_args(args)
    logger.info(f"args: {args}")

    # We only resolve these modules after we know where to look for them.pconfig
    if args.syspath is not None:
        sys.path.insert(0, args.syspath)
    # We only resolve these modules after we know where to look for them.
    global training_loop
    from aiws.training_loop import training_loop
    return args

def save(config):
    config.trainer.save_model(config.output_dir)
    
@record # Improves diagnostics for distributed training
def main():
    args = parse_args()
    init_process(args)
    init_logging(args)
    training_loop(args.project_dir, args.config_template)

if __name__ == "__main__":
    main()