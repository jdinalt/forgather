"""Server-side wrappers for launching a TensorBoard instance.

TensorBoard is a pure viewer over trainer_logs / TF event files — it needs
no GPUs, no forgather imports, and no special launch harness. The scheduler
treats it as a job type so it queues alongside training / eval / inference
and shows up in the unified Jobs view with TTY + kill controls; the only
actual difference is ``requested_gpus`` is typically zero.

The server shells out to the stock ``tensorboard`` CLI (same binary the
``forgather tb`` CLI wraps) rather than re-implementing anything from it.
"""

from __future__ import annotations

from typing import List, Optional


def build_tensorboard_command(
    *,
    logdir: str,
    port: int,
    host: Optional[str] = None,
    bind_all: bool = False,
    window_title: Optional[str] = None,
    reload_interval: Optional[int] = None,
    reload_multifile: bool = False,
    samples_per_plugin: Optional[str] = None,
    path_prefix: Optional[str] = None,
) -> List[str]:
    """Build the argv for ``tensorboard``.

    Only ``logdir`` and ``port`` are required. ``bind_all`` and ``host``
    are mutually exclusive on the CLI — if both are passed, ``--bind_all``
    wins (matching the CLI's own precedence).

    Default bind: ``127.0.0.1`` when neither ``bind_all`` nor ``host`` is
    set. Loopback ports on multi-user hosts are reachable by every other
    local user — TB job dirs may contain training metadata they shouldn't
    see. The forgather-server's auth-gated ``/api/tb/{job_id}/`` reverse
    proxy is the supported path for browser access. If you want
    LAN-accessible TB, pass ``bind_all=True`` (or a non-loopback ``host``)
    explicitly via the queue submit modal.

    ``path_prefix`` maps to TB's ``--path_prefix``: when the proxy mounts
    TB at ``/api/tb/<queue_id>``, TB has to generate matching internal
    links or browser-side asset / data URLs 404. Pass the same prefix
    here that the proxy strips on inbound requests.

    The "advanced" knobs (``reload_interval``, ``reload_multifile``,
    ``samples_per_plugin``) surface the options a Forgather user
    actually tends to reach for during long-running experiments; the rest
    of TensorBoard's sprawling flag surface is left to anyone who needs
    it to run their own ``tensorboard`` directly.
    """
    cmd: List[str] = [
        "tensorboard",
        "--logdir",
        logdir,
        "--port",
        str(port),
    ]
    if bind_all:
        cmd.append("--bind_all")
    elif host:
        cmd.extend(["--host", host])
    else:
        # Default to loopback; see docstring.
        cmd.extend(["--host", "127.0.0.1"])
    if window_title:
        cmd.extend(["--window_title", window_title])
    if reload_interval is not None:
        cmd.extend(["--reload_interval", str(reload_interval)])
    if reload_multifile:
        cmd.extend(["--reload_multifile", "true"])
    if samples_per_plugin:
        cmd.extend(["--samples_per_plugin", samples_per_plugin])
    if path_prefix:
        cmd.extend(["--path_prefix", path_prefix])
    return cmd
