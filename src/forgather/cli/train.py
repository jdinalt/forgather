import json
import os
import subprocess

from forgather.latent import Latent

from .dynamic_args import get_dynamic_args
from .utils import BaseCommand, assert_project_class


def train_cmd(args):
    """Run configuration with train script."""
    assert_project_class(args, "type.training_script")

    cmd = BaseCommand(args)
    config, pp_config = cmd.get_config()
    config_meta = Latent.materialize(config.meta)
    nproc_per_node = config_meta["nproc_per_node"]
    train_script_path = os.path.join(
        config_meta["forgather_dir"], "scripts", "train_script.py"
    )

    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    if "env" in config:
        config_env = Latent.materialize(config.env)
        print(f"Config Environment: {config_env}")
        env |= config_env

    cmd_args = ["torchrun"]

    if len(args.remainder) > 1 and args.remainder[0] == "--":
        cmd_args.extend(args.remainder[1:])
    else:
        # Apply defaults, if not specified
        cmd_args.extend(
            [
                "--standalone",
                "--nproc-per-node",
                str(nproc_per_node),
            ]
        )

    # Apply path to script and project directory argument to script.
    cmd_args.extend(
        [
            os.path.normpath(train_script_path),
            "-p",
            os.path.normpath(args.project_dir),
        ]
    )

    # Optionally, apply system search path from meta.
    if cmd.meta.system_path is not None:
        cmd_args.extend(["-s", cmd.meta.system_path])

    # Add dynamic arguments as JSON if any exist
    dynamic_args = get_dynamic_args(args)
    if dynamic_args:
        # Serialize dynamic args to JSON and pass to training script
        dynamic_args_json = json.dumps(dynamic_args)
        cmd_args.extend(["--dynamic-args", dynamic_args_json])

    # Add the config template name
    cmd_args.append(args.config_template)

    # Generate equivalent command string
    cmd_str = ""

    for arg in cmd_args:
        cmd_str += f"{arg} "

    print(f"{cmd_str}")

    # Run the command
    if not args.dry_run:
        subprocess.run(cmd_args, env=env)
