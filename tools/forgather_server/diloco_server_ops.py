"""Server-side wrappers for launching ``forgather diloco server``.

The DiLoCo server is a long-lived parameter server: it holds global model
parameters in memory, accepts pseudo-gradient submissions over HTTP, and
applies the outer optimizer. It uses **no GPU** (everything runs on CPU)
so the scheduler reserves zero GPUs for these spawns. Its lifecycle
mirrors mkdocs / tensorboard / dataset_server: starts when enqueued,
runs until killed.

Mirrors the argv from :mod:`forgather.cli.diloco_args` so the same code
path runs as a manual ``forgather diloco server …`` invocation. Invokes
the CLI as ``python -m forgather.cli`` (same pattern as model_ops /
inference_ops) to inherit the exact same module-resolution behavior.

Auth / TLS is intentionally out of scope for the initial cut. When TLS
lands we'll add ``--cert-file`` / ``--auth-token-file`` style args here
without rearranging the existing surface.
"""

from __future__ import annotations

import sys
from typing import List, Optional


def build_diloco_server_command(
    *,
    output_dir: str,
    num_workers: int,
    port: int = 8512,
    host: str = "127.0.0.1",
    async_mode: bool = False,
    dn_buffer_size: int = 0,
    dylu: bool = False,
    dylu_base_sync_every: int = 500,
    from_checkpoint: Optional[str] = None,
    save_every: int = 10,
    save_total_limit: int = 3,
    outer_lr: Optional[float] = None,
    outer_momentum: Optional[float] = None,
    no_nesterov: bool = False,
    heartbeat_timeout: Optional[float] = None,
    min_workers: Optional[int] = None,
    auth_token_file: Optional[str] = None,
    no_auth: bool = False,
    bulk_port: Optional[int] = None,
    bulk_tls: Optional[bool] = None,
    bulk_auth: Optional[bool] = None,
) -> List[str]:
    """Build argv for ``forgather diloco server``.

    Every flag the diloco server CLI exposes is surfaced here so the
    webui's service modal can drive any deployment. Optionals that map
    to CLI defaults are passed only when explicitly set, keeping the
    spawned argv readable in ``ps`` and the TTY log.
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "forgather.cli",
        "diloco",
        "server",
        "-o",
        output_dir,
        "-n",
        str(int(num_workers)),
        "--port",
        str(int(port)),
        "--host",
        host,
    ]
    if async_mode:
        cmd.append("--async")
    if dn_buffer_size and int(dn_buffer_size) > 0:
        cmd.extend(["--dn-buffer-size", str(int(dn_buffer_size))])
    if dylu:
        cmd.append("--dylu")
        # base sync is only meaningful in dylu mode; the server's default
        # (500) is fine, but surface the explicit value so the spawn
        # argv reflects the operator's intent.
        cmd.extend(["--dylu-base-sync-every", str(int(dylu_base_sync_every))])
    if from_checkpoint:
        cmd.extend(["--from-checkpoint", from_checkpoint])
    # save_every: 0 disables periodic save — the CLI accepts 0, so pass
    # explicitly when the operator set anything other than the CLI default.
    cmd.extend(["--save-every", str(int(save_every))])
    if save_total_limit is not None:
        cmd.extend(["--save-total-limit", str(int(save_total_limit))])
    if outer_lr is not None:
        cmd.extend(["--outer-lr", str(float(outer_lr))])
    if outer_momentum is not None:
        cmd.extend(["--outer-momentum", str(float(outer_momentum))])
    if no_nesterov:
        cmd.append("--no-nesterov")
    if heartbeat_timeout is not None:
        cmd.extend(["--heartbeat-timeout", str(float(heartbeat_timeout))])
    if min_workers is not None:
        cmd.extend(["--min-workers", str(int(min_workers))])
    # Security (issue #90). Mirrors the dataset_server spawn pattern:
    # token persisted to per-port file, passed by *file path* via
    # --auth-token-file so it never appears in argv. --no-auth opts
    # out for the legacy single-LAN-trust case.
    if no_auth:
        cmd.append("--no-auth")
    elif auth_token_file:
        cmd.extend(["--auth-token-file", str(auth_token_file)])
    # Two-port bulk plane. Only emit flags when the operator opted in.
    if bulk_port is not None:
        cmd.extend(["--bulk-port", str(int(bulk_port))])
        if bulk_tls is True:
            cmd.append("--bulk-tls")
        elif bulk_tls is False:
            cmd.append("--no-bulk-tls")
        if bulk_auth is True:
            cmd.append("--bulk-auth")
        elif bulk_auth is False:
            cmd.append("--no-bulk-auth")
    return cmd
