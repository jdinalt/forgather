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

    args = parser.parse_args(args)
    logger.info(f"args: {args}")

    if args.syspath is not None:
        sys.path.insert(0, args.syspath)

    return args


@record
def main():
    args = parse_args()
    init_logging(args)
    training_loop(args.project_dir, args.config_template)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        # This can trigger a seg-fault on process exit. I get a warning without it and a crash with it... I'll take
        # the warning, unitl I can figure out what the issue is.
        # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
