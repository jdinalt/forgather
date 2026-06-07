import argparse
import faulthandler
import logging
import os
import signal
import sys
from argparse import RawTextHelpFormatter

import torch
import transformers
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import datasets

logger = logging.getLogger(__name__)


def _enable_faulthandler() -> None:
    """Wire Python's faulthandler so a hung or crashed worker leaves a trace.

    Two complementary behaviours, both essential for diagnosing
    multi-node hangs that produce no Python traceback today:

    * ``faulthandler.enable()`` — install C-level signal handlers
      (SIGSEGV, SIGFPE, SIGABRT, SIGBUS, SIGILL) that dump the Python
      stack of every thread to stderr before the process dies. Without
      this, a worker that hits a CUDA driver assertion or segfaults in
      a background thread exits with just an exit code and no clue
      where it died — exactly the silent-rank-death pattern we hit on
      the 7B PP run.

    * ``faulthandler.register(SIGUSR1)`` — on SIGUSR1, dump every
      thread's Python stack to stderr but *don't* kill the process.
      Lets us ``kill -USR1 <pid>`` against a hung rank to see where
      it's stuck (most often blocked in a torch.distributed collective)
      without disturbing it. Same idiom as ``py-spy dump``, but works
      from inside a container that doesn't allow ptrace.

    Idempotent — safe to call multiple times. The dump destination is
    stderr, which torchrun routes to the per-rank TTY log, so the
    output lands in the same place the operator is already looking.
    """
    faulthandler.enable()
    # SIGUSR1 is reserved for application use on Linux and isn't
    # claimed by torch / NCCL / huggingface internals at the time of
    # writing, so it's safe to repurpose for our dump-on-demand. Keep
    # ``chain=False`` so we don't fall through to the default action
    # (which would terminate the process).
    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, chain=False)
        except (ValueError, OSError):
            # Some environments (Windows, restricted containers) don't
            # let us register; not fatal.
            pass


def init_logging(args):
    # Default to zero, if not set.
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        log_level = args.log_level
    else:
        log_level = args.secondary_log_level

    # Convert log-level string to numeric value
    log_level = log_level.upper()
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        logger.warning(f"Invalid log level: {log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO

    # Change local local level
    logger.setLevel(log_level)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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

    # Enable before anything heavy so a crash during model construction,
    # NCCL init, or the first pipeline collective still produces a
    # traceback. Goes ahead of init_logging so even errors during arg
    # parsing benefit.
    _enable_faulthandler()

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    init_logging(args)

    # Collective DiLoCo (issue #154): the N replicas share this torchrun world's
    # env, so make DILOCO_WORKER_ID per-replica-distinct BEFORE the Project below
    # preprocesses the config (the output dir is derived from it). No-op unless
    # DILOCO_REPLICATE > 1.
    from forgather.ml.diloco import diloco_apply_collective_worker_id

    diloco_apply_collective_worker_id()

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

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
