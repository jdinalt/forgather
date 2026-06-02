"""`forgather submit` — submit the current project's config to the scheduler.

Single-node by default (delegates to the same path as `train --schedule`);
`--global` fans out across the cluster via submit_orch.submit_global.
"""

import os
import sys


def submit_cmd(args):
    from . import submit_orch

    # Resolve the config: explicit -t, else the project's default_config.
    config = getattr(
        args, "config_template", None
    ) or submit_orch.resolve_default_config(args)
    if not config:
        print(
            "error: no config to submit. Pass -t <config> (a global forgather "
            "flag, before 'submit') or set the project's default_config.",
            file=sys.stderr,
        )
        return 1
    args.config_template = config

    if getattr(args, "run_global", False):
        return _submit_global(args, submit_orch, config)
    return _submit_single(args, submit_orch, config)


def _submit_single(args, submit_orch, config):
    """Single-node submit: identical to `forgather train --schedule`."""
    from .train import train_cmd

    # Drive train's scheduled path. submit has no foreground torchrun fallback
    # of its own — --local-only / --local-fallback resolve through train_cmd.
    args.schedule = True
    args.enqueue = False
    args.devices = None
    args.nproc = None
    args.dry_run = False
    train_cmd(args)
    return 0


def _submit_global(args, submit_orch, config):
    """Multi-node fan-out across the cluster."""
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient(getattr(args, "via_server", None) or None)
    try:
        dataset_source = submit_orch.resolve_dataset_source(client, args)
        dynamic_args = submit_orch.collect_dynamic_args(args)
        # Multi-node needs an absolute project path (every peer resolves it);
        # submit_global validates and reports.
        return submit_orch.submit_global(
            client,
            args,
            project_dir=getattr(args, "project_dir", None),
            config=config,
            dynamic_args=dynamic_args,
            dataset_source=dataset_source,
        )
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
