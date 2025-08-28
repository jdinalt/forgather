import os
import argparse
from argparse import RawTextHelpFormatter
import sys

from torch.distributed.elastic.multiprocessing.errors import record
import logging
import transformers
import datasets

import torch

from forgather.ml.training_script import training_loop

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_logging(args):
    # Default to zero, if not set.
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        log_level = args.log_level
        # TODO: Is there a version which takes a string?
        transformers.utils.logging.set_verbosity_info()
    else:
        log_level = args.secondary_log_level

    logger.setLevel(log_level)
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
        ),
    )
    parser.add_argument(
        "config_template",
        type=str,
        metavar="config-template",
        help="Configuration Template Name",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        help="Set the log level for the main process: INFO, WARNING, DEBUG, ...; default=INFO",
    )
    parser.add_argument(
        "--secondary-log-level",
        default="WARNING",
        help="Set the log level for the secondary processes, if any: INFO, WARNING, DEBUG, ...; default=WARNING",
    )
    parser.add_argument(
        "-s",
        "--syspath",
        type=str,
        default=None,
        help="Add sys.path for relative imports",
    )
    parser.add_argument(
        "-p",
        "--project-dir",
        type=str,
        default=".",
        help="The relative path to the project directory.",
    )
    parser.add_argument(
        "--dynamic-args",
        type=str,
        default=None,
        help="JSON-encoded dynamic arguments to pass to the Project constructor",
    )

    args = parser.parse_args(args)
    logger.info(f"args: {args}")

    if args.syspath is not None:
        sys.path.insert(0, args.syspath)

    return args


@record
def main():
    import json
    from forgather.project import Project
    
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    init_logging(args)
    
    # Parse dynamic args from JSON if provided
    dynamic_args = {}
    if args.dynamic_args:
        try:
            dynamic_args = json.loads(args.dynamic_args)
            logger.info(f"Parsed dynamic args: {dynamic_args}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dynamic args JSON: {e}")
            sys.exit(1)
    
    # Create Project with dynamic args and run training
    proj = Project(args.config_template, args.project_dir, **dynamic_args)
    training_script = proj()
    training_script.run()
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
