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
    grace_period: float = 0.0,
    token_budget: int = 0,
    verbose_sync: bool = False,
    dylu: bool = False,
    dylu_base_sync_every: int = 500,
    sync_every: int = 500,
    bf16_comm: Optional[bool] = None,
    upload_dtype: Optional[str] = None,
    upload_sr: bool = False,
    download_dtype: str = "fp32",
    download_sr: bool = False,
    wire_format: str = "pickle",
    grpc_enabled: bool = False,
    backend: str = "http",
    num_fragments: int = 1,
    fragment_assignment: str = "strided",
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
    quiet_tokens: bool = False,
    bulk_cleartext: bool = False,
    run_name: Optional[str] = None,
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
        # CRITICAL: run the parameter server in-process. Without this the
        # spawned `forgather diloco server` would hit its own orchestrator
        # auto-detect, see the forgather server is up, and enqueue ANOTHER
        # diloco_server job — a self-enqueue loop of dead jobs. --local-only
        # makes it run foreground (which is exactly what the scheduler wants
        # from the process it spawned).
        "--local-only",
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
    if grace_period and float(grace_period) > 0:
        cmd.extend(["--grace-period", str(float(grace_period))])
    if token_budget and int(token_budget) > 0:
        cmd.extend(["--token-budget", str(int(token_budget))])
    if verbose_sync:
        cmd.append("--verbose-sync")
    if dylu:
        cmd.append("--dylu")
        # base sync is only meaningful in dylu mode; the server's default
        # (500) is fine, but surface the explicit value so the spawn
        # argv reflects the operator's intent.
        cmd.extend(["--dylu-base-sync-every", str(int(dylu_base_sync_every))])
    # Group-wide worker settings the server is authoritative for (issue
    # #53 follow-up + #130 precision refactor). sync_every is always
    # meaningful (the non-DyLU cadence); precision/fragments knobs only
    # when they diverge from the CLI default, keeping argv readable.
    cmd.extend(["--sync-every", str(int(sync_every))])
    if num_fragments and int(num_fragments) > 1:
        cmd.extend(["--num-fragments", str(int(num_fragments))])
        # Only emit a non-default assignment (server defaults to strided), and
        # only when streaming is on (it's meaningless without fragments).
        if fragment_assignment and fragment_assignment != "strided":
            cmd.extend(["--fragment-assignment", str(fragment_assignment)])
    # Wire precision (issue #130). Prefer the four explicit knobs;
    # fall back to the legacy ``--no-bf16`` shortcut when only the
    # deprecated ``bf16_comm`` is set (pre-#130 callers). Surface
    # ``--upload-dtype`` only when it differs from the CLI default so
    # argv stays readable on the common path.
    if upload_dtype is not None and upload_dtype != "bf16":
        cmd.extend(["--upload-dtype", upload_dtype])
    if upload_sr:
        cmd.append("--upload-sr")
    if download_dtype and download_dtype != "fp32":
        cmd.extend(["--download-dtype", download_dtype])
    if download_sr:
        cmd.append("--download-sr")
    if upload_dtype is None and bf16_comm is False:
        # Pre-#130 caller path: surface the deprecated alias.
        cmd.append("--no-bf16")
    # Bulk transport (issue #154). The wire codec only when it diverges from
    # the CLI default (pickle); --grpc when the gRPC bulk listener is requested
    # (it supersedes --bulk-cleartext server-side).
    if wire_format and wire_format != "pickle":
        cmd.extend(["--wire-format", wire_format])
    if grpc_enabled:
        cmd.append("--grpc")
    # Declared group backend (issue #154); only when it diverges from the
    # default, keeping the argv readable.
    if backend and backend != "http":
        cmd.extend(["--backend", backend])
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
    if quiet_tokens and not no_auth:
        # Belt + suspenders: the demo-mode webui sets this so the
        # spawned server's TTY log doesn't echo the token. The token
        # is still discoverable via the per-port file for legitimate
        # peers — only the launch banner is suppressed.
        cmd.append("--quiet-tokens")
    # Cleartext bulk plane. Single opt-in flag; the server picks the
    # ephemeral port and advertises it to workers over the control plane.
    if bulk_cleartext:
        cmd.append("--bulk-cleartext")
    # Run label for the per-run stats log dir (runs/<ts>_<run_name>). The
    # server sanitizes it to a safe path component; honored only on a fresh
    # start (a resume continues the checkpoint's recorded run dir).
    if run_name:
        cmd.extend(["--run-name", str(run_name)])
    return cmd
