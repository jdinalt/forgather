import json
import os
import signal
import subprocess
import sys

from forgather.latent import Latent

from .utils import BaseCommand, assert_project_class


def train_cmd(args):
    """Run configuration with train script.

    Default: run locally in the foreground via torchrun. With --schedule
    (or the deprecated --enqueue alias) the job is submitted to the
    forgather-server scheduler instead — in the background unless
    --foreground attaches to stream its output. --local-only / a
    --local-fallback against an unreachable server drop back to the
    foreground torchrun path.
    """
    from . import submit_orch

    # --enqueue is the legacy spelling of --schedule.
    if args.enqueue and not args.schedule:
        print(
            "note: --enqueue is deprecated; use --schedule.",
            file=sys.stderr,
        )
    schedule = args.schedule or args.enqueue
    local_only = getattr(args, "local_only", False)

    if schedule and args.devices and not local_only:
        print(
            "error: --schedule and --devices are mutually exclusive "
            "(the scheduler picks GPUs)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    assert_project_class(args, "type.training_script")

    cmd = BaseCommand(args)
    # Dynamic args may override values consumed by the meta block (e.g.
    # nproc_per_node), so they must be supplied to the preprocessor before
    # we materialize meta — otherwise we read template defaults and pass
    # the wrong --nproc-per-node to torchrun. collect_dynamic_args also
    # enforces required/bounds (the same checks train has always done).
    dynamic_args = submit_orch.collect_dynamic_args(args)
    config, _ = cmd.get_config(**dynamic_args)
    config_meta = Latent.materialize(config.meta)
    nproc_per_node = config_meta["nproc_per_node"]

    # --nproc wins over the config value. Honor the override even for the
    # scheduled path so users can submit jobs at a specific GPU count
    # without having to teach the config about it.
    if args.nproc is not None:
        nproc_per_node = args.nproc
    elif nproc_per_node == "gpu":
        # torchrun's "gpu" sentinel asks the launcher to count visible
        # CUDA devices. On a CPU-only host (no driver, --gpus none under
        # docker, or torch.cuda unavailable) torchrun aborts immediately
        # with "invalid literal for int() with base 10: 'gpu'". Detect
        # that here and fall back to a single rank so the CPU path keeps
        # working — useful for debugging training on a laptop. Operators
        # who want a specific count should pass --nproc N explicitly.
        try:
            import torch

            cuda_visible = torch.cuda.is_available() and torch.cuda.device_count() > 0
        except (ImportError, RuntimeError):
            cuda_visible = False
        if not cuda_visible:
            print(
                "warning: config requests --nproc-per-node 'gpu' but no CUDA"
                " device is visible; falling back to --nproc-per-node 1."
                " Pass --nproc N to override.",
                file=sys.stderr,
            )
            nproc_per_node = 1

    if schedule:
        # use_orchestrator returns a client to enqueue through, or None to
        # act locally (--local-only, or --local-fallback when the server is
        # down). None falls through to the foreground torchrun path below.
        from .server_client import ServerUnreachable

        try:
            client = submit_orch.use_orchestrator(args)
        except ServerUnreachable as e:
            print(str(e), file=sys.stderr)
            raise SystemExit(1)
        if client is not None:
            if args.requested_gpus is not None:
                requested_gpus = args.requested_gpus
            else:
                try:
                    requested_gpus = int(nproc_per_node)
                except (TypeError, ValueError):
                    print(
                        "error: couldn't infer the GPU count from config "
                        f"nproc_per_node={nproc_per_node!r}; pass --requested-gpus N.",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
            dataset_source = submit_orch.resolve_dataset_source(client, args)
            try:
                item = submit_orch.submit_single(
                    client,
                    project_dir=os.path.abspath(args.project_dir),
                    config=args.config_template,
                    dynamic_args=dynamic_args,
                    priority=args.priority,
                    requested_gpus=requested_gpus,
                    dataset_source=dataset_source,
                )
            except (ServerUnreachable, RuntimeError) as e:
                print(str(e), file=sys.stderr)
                raise SystemExit(1)
            queue_id = item["queue_id"]
            if args.foreground:
                submit_orch.attach_submitted(client, queue_id)
            else:
                print(
                    f"queued: {queue_id} (priority={item['priority']}, "
                    f"gpus={item['requested_gpus']})"
                )
            return
        # client is None → fall through to the foreground torchrun path.

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
        proc = subprocess.Popen(cmd_args, env=env, preexec_fn=os.setsid)

        def _sigint_handler(sig, frame):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        signal.signal(signal.SIGINT, _sigint_handler)
        proc.wait()
