"""Training-subprocess launcher.

Mirrors the spawn logic in ``src/forgather/cli/train.py`` but routes
stdout/stderr to a TTY log file and keeps the ``Popen`` handle so the
scheduler can poll exit status and kill the whole process group on abort.

Staying outside the CLI lets us own the ``torchrun`` process group
directly — the CLI's ``train_cmd`` wraps torchrun in *its own* setsid group,
which makes cleanup trickier than owning torchrun ourselves.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from forgather.cli.utils import get_env
from forgather.latent import Latent
from forgather.meta_config import MetaConfig

from . import (
    convert_ops,
    eval_ops,
    finalize_ops,
    inference_ops,
    mkdocs_ops,
    tensorboard_ops,
)

log = logging.getLogger("forgather_server.launcher")


@dataclass
class LaunchResult:
    proc: subprocess.Popen
    pid: int
    pgid: int
    cmd: List[str]
    tty_log_path: Path


def build_command(
    project_dir: str,
    config_name: str,
    dynamic_args: Dict[str, Any],
) -> List[str]:
    """Preprocess the config enough to build the ``torchrun`` command.

    Reads ``nproc_per_node`` and ``forgather_dir`` from the config's
    materialized meta block. Dynamic args are passed into the preprocess so
    they can influence those values (e.g. an experiment that overrides
    nproc_per_node via a dynamic arg).

    ``nproc_per_node`` is passed to torchrun verbatim. It may be an int, or
    one of the strings ``"gpu"`` / ``"cpu"`` / ``"auto"`` — the latter tell
    torchrun to auto-detect the worker count from CUDA_VISIBLE_DEVICES (or
    CPU count). Mirrors ``src/forgather/cli/train.py``'s behavior.
    """
    meta = MetaConfig(project_dir)
    env = get_env(meta, project_dir)
    config_path = meta.config_path(config_name)
    loaded = env.load(config_path, **dynamic_args)
    config_meta = Latent.materialize(loaded.config.meta)

    nproc_per_node = config_meta["nproc_per_node"]
    forgather_dir = config_meta["forgather_dir"]
    train_script_path = os.path.join(forgather_dir, "scripts", "train_script.py")

    cmd: List[str] = [
        "torchrun",
        "--standalone",
        "--nproc-per-node",
        str(nproc_per_node),
        os.path.normpath(train_script_path),
        "-p",
        os.path.normpath(project_dir),
    ]
    if meta.system_path is not None:
        cmd.extend(["-s", meta.system_path])
    if dynamic_args:
        cmd.extend(["--dynamic-args", json.dumps(dynamic_args)])
    cmd.append(config_name)
    return cmd


def _spawn_subprocess(
    cmd: List[str],
    gpu_indices: List[int],
    tty_log_path: Path,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Shared spawn plumbing: GPU pinning, TTY capture, own session.

    Used by both training and eval spawn paths. The caller owns the
    returned ``Popen`` and is responsible for reaping it. Anything
    generic to "subprocess-style Forgather job" lives here; anything
    type-specific (argv construction) lives in its callers.
    """
    proc_env = os.environ.copy()
    if gpu_indices:
        proc_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
    if extra_env:
        proc_env.update(extra_env)

    tty_log_path.parent.mkdir(parents=True, exist_ok=True)
    tty_file = open(tty_log_path, "wb", buffering=0)
    log.info("spawning: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        env=proc_env,
        stdout=tty_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    return LaunchResult(
        proc=proc,
        pid=proc.pid,
        pgid=os.getpgid(proc.pid),
        cmd=cmd,
        tty_log_path=tty_log_path,
    )


def spawn_training_process(
    project_dir: str,
    config_name: str,
    dynamic_args: Dict[str, Any],
    gpu_indices: List[int],
    tty_log_path: Path,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a training run."""
    cmd = build_command(project_dir, config_name, dynamic_args)
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_eval_process(
    *,
    eval_project: str,
    eval_template: str,
    model_path: str,
    gpu_indices: List[int],
    tty_log_path: Path,
    checkpoint_path: Optional[str] = None,
    no_checkpoint: bool = False,
    trainer: str = "ddp",
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    max_steps: int = -1,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    compile: bool = False,
    output_dir: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn an evaluation run via ``scripts/eval_script.py``.

    Mirrors the argv that ``forgather eval test`` would build so the
    behavior is identical to the CLI — just with GPUs restricted via
    ``CUDA_VISIBLE_DEVICES`` and stdout/stderr captured to a TTY log
    that the web UI can stream. Unlike training, eval processes do not
    register with ``TrainerControlClient``; the scheduler correlates
    lifecycle by PID only.
    """
    cmd = eval_ops.build_eval_command(
        eval_project=eval_project,
        eval_template=eval_template,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        no_checkpoint=no_checkpoint,
        trainer=trainer,
        batch_size=batch_size,
        max_length=max_length,
        max_steps=max_steps,
        dtype=dtype,
        attn_implementation=attn_implementation,
        compile=compile,
        output_dir=output_dir,
    )
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_inference_process(
    *,
    model_path: str,
    port: int,
    gpu_indices: List[int],
    tty_log_path: Path,
    host: str = "127.0.0.1",
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    from_checkpoint: bool = False,
    compile: bool = False,
    disable_kv_cache: bool = False,
    ignore_eos: bool = False,
    chat_template: Optional[str] = None,
    cache_implementation: Optional[str] = None,
    compile_args: Optional[str] = None,
    log_level: str = "INFO",
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn an OpenAI-compatible inference server.

    Long-lived — runs until killed. No control protocol; the only
    supported lifecycle actions are kill / force-kill, like eval.
    """
    cmd = inference_ops.build_inference_command(
        model_path=model_path,
        port=port,
        host=host,
        dtype=dtype,
        attn_implementation=attn_implementation,
        checkpoint_path=checkpoint_path,
        from_checkpoint=from_checkpoint,
        compile=compile,
        disable_kv_cache=disable_kv_cache,
        ignore_eos=ignore_eos,
        chat_template=chat_template,
        cache_implementation=cache_implementation,
        compile_args=compile_args,
        log_level=log_level,
    )
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_tensorboard_process(
    *,
    logdir: str,
    port: int,
    tty_log_path: Path,
    host: Optional[str] = None,
    bind_all: bool = False,
    window_title: Optional[str] = None,
    reload_interval: Optional[int] = None,
    reload_multifile: bool = False,
    samples_per_plugin: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a TensorBoard instance.

    CPU-only viewer; the caller allocates zero GPUs, and we don't touch
    ``CUDA_VISIBLE_DEVICES`` so the process inherits whatever the server
    started with — TensorBoard doesn't initialize CUDA regardless.
    Long-lived; terminates only on kill.
    """
    cmd = tensorboard_ops.build_tensorboard_command(
        logdir=logdir,
        port=port,
        host=host,
        bind_all=bind_all,
        window_title=window_title,
        reload_interval=reload_interval,
        reload_multifile=reload_multifile,
        samples_per_plugin=samples_per_plugin,
    )
    return _spawn_subprocess(cmd, [], tty_log_path, extra_env)


def spawn_mkdocs_process(
    *,
    config_file: str,
    port: int,
    tty_log_path: Path,
    host: str = "127.0.0.1",
    strict: bool = False,
    livereload: bool = True,
    dirty: bool = False,
    watch: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn an ``mkdocs serve`` instance.

    CPU-only viewer; the caller allocates zero GPUs. Long-lived;
    terminates only on kill, same lifecycle as TensorBoard.
    """
    cmd = mkdocs_ops.build_mkdocs_command(
        config_file=config_file,
        host=host,
        port=port,
        strict=strict,
        livereload=livereload,
        dirty=dirty,
        watch=watch,
    )
    return _spawn_subprocess(cmd, [], tty_log_path, extra_env)


def spawn_convert_process(
    *,
    src_model_path: str,
    dst_model_path: str,
    gpu_indices: List[int],
    tty_log_path: Path,
    reverse: bool = False,
    model_type: Optional[str] = None,
    dtype: Optional[str] = None,
    max_length: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    generation_test: bool = False,
    dry_run: bool = False,
    prompt: Optional[str] = None,
    compare_text_file: Optional[str] = None,
    debug_params: bool = False,
    chat_template_path: Optional[str] = None,
    add_tokens: Optional[str] = None,
    skip_default_tokens: bool = False,
    converter_paths: Optional[List[str]] = None,
    log_level: str = "INFO",
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a ``forgather convert`` run.

    Fire-and-forget like eval; no trainer-control protocol. May or may
    not need a GPU depending on ``device`` / ``generation_test`` —
    callers that don't want a GPU pass ``gpu_indices=[]``.
    """
    cmd = convert_ops.build_convert_command(
        src_model_path=src_model_path,
        dst_model_path=dst_model_path,
        reverse=reverse,
        model_type=model_type,
        dtype=dtype,
        max_length=max_length,
        checkpoint_path=checkpoint_path,
        device=device,
        generation_test=generation_test,
        dry_run=dry_run,
        prompt=prompt,
        compare_text_file=compare_text_file,
        debug_params=debug_params,
        chat_template_path=chat_template_path,
        add_tokens=add_tokens,
        skip_default_tokens=skip_default_tokens,
        converter_paths=converter_paths,
        log_level=log_level,
    )
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_finalize_process(
    *,
    source: str,
    dest: str,
    gpu_indices: List[int],
    tty_log_path: Path,
    checkpoint: Optional[str] = None,
    add_tokens: Optional[str] = None,
    skip_default_tokens: bool = False,
    chat_template_path: Optional[str] = None,
    no_auto_stop_tokens: bool = False,
    stop_tokens: Optional[str] = None,
    generation_config: Optional[str] = None,
    keep_optimizer: bool = False,
    root_copy: bool = False,
    safetensors: bool = False,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a ``forgather finalize`` run.

    Fire-and-forget like eval / convert. Defaults to CPU
    (``--device cpu`` is the script's own default); pass ``gpu_indices``
    if the user picked one.
    """
    cmd = finalize_ops.build_finalize_command(
        source=source,
        dest=dest,
        checkpoint=checkpoint,
        add_tokens=add_tokens,
        skip_default_tokens=skip_default_tokens,
        chat_template_path=chat_template_path,
        no_auto_stop_tokens=no_auto_stop_tokens,
        stop_tokens=stop_tokens,
        generation_config=generation_config,
        keep_optimizer=keep_optimizer,
        root_copy=root_copy,
        safetensors=safetensors,
        dtype=dtype,
        device=device,
        dry_run=dry_run,
        log_level=log_level,
    )
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def kill_process_group(pid: int, sig: int = signal.SIGTERM) -> bool:
    """Kill the whole process group led by ``pid``.

    Returns ``False`` if the group is already gone (nothing to kill).
    """
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
