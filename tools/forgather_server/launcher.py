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
    construct_ops,
    convert_ops,
    dataset_ops,
    dataset_server_ops,
    diloco_server_ops,
    eval_ops,
    finalize_ops,
    inference_ops,
    mkdocs_ops,
    model_ops,
    tensorboard_ops,
    update_ops,
)

log = logging.getLogger("forgather_server.launcher")


def _isolate_cuda_extra_env(
    gpu_indices: List[int], extra_env: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Merge ``CUDA_VISIBLE_DEVICES=""`` into extra_env when gpu_indices is empty.

    Used by spawn paths whose subprocess should never see CUDA when no
    GPU is reserved (eval / training / inference for CPU debugging) --
    without this, the subprocess inherits the parent's (typically
    unset) CUDA_VISIBLE_DEVICES and torchrun's "gpu" sentinel grabs
    every GPU on the box, defeating the point of reserving 0. Other
    zero-GPU job types (convert / finalize / model with --device
    cuda:0) deliberately opt INTO host CUDA without a reservation;
    those callers don't invoke this helper.

    Caller-supplied ``CUDA_VISIBLE_DEVICES`` wins over our empty-string
    override -- explicit operator intent shouldn't be silently
    rewritten.
    """
    if gpu_indices:
        return extra_env
    merged = dict(extra_env) if extra_env else {}
    merged.setdefault("CUDA_VISIBLE_DEVICES", "")
    return merged


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
    rdzv_args: Optional[Dict[str, Any]] = None,
    gpu_indices: Optional[List[int]] = None,
    nproc_override: Optional[str] = None,
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

    Multi-node mode: when ``rdzv_args`` is provided (cluster-coordinator
    submit, see Phase 3), the ``--standalone`` flag is replaced by an
    explicit rendezvous block — ``--nnodes``, ``--node-rank``,
    ``--rdzv-backend``, ``--rdzv-endpoint``, ``--rdzv-id`` — and
    ``nproc_per_node`` is overridden by the cluster-supplied value
    (different peers may have different GPU counts). Single-node mode
    (``rdzv_args=None``) retains the existing ``--standalone`` form
    so non-cluster training is unaffected.

    The c10d backend autodetects "am I the rendezvous host?" by
    resolving ``socket.gethostname()`` and comparing the result to
    ``rdzv_endpoint``. On Debian/Ubuntu the system hostname resolves
    to ``127.0.1.1`` via ``/etc/hosts``, so the comparison silently
    fails and *no* node binds the store — every peer sits as a client
    and the rendezvous times out. To work around this we accept an
    optional ``is_host`` boolean in ``rdzv_args``: when set, we emit
    ``--rdzv-conf is_host=true|false`` so torch skips the broken
    autodetection entirely.

    Similarly, rank 0's elastic agent publishes ``MASTER_ADDR`` to
    every peer via the c10d store; without ``--local-addr`` it falls
    back to ``socket.getfqdn()`` (torch elastic
    ``RendezvousStoreInfo.build``). On LANs without DNS that yields a
    bare hostname like ``hal9000`` which other ranks then fail to
    resolve (``gai error: -3``). ``rdzv_args["local_addr"]`` lets the
    cluster master ship each peer's own routable address; we emit it
    as ``--local-addr <addr>`` so MASTER_ADDR is an IP every peer can
    dial.
    """
    meta = MetaConfig(project_dir)
    env = get_env(meta, project_dir)
    config_path = meta.config_path(config_name)
    loaded = env.load(config_path, **dynamic_args)
    config_meta = Latent.materialize(loaded.config.meta)

    nproc_per_node = config_meta["nproc_per_node"]
    forgather_dir = config_meta["forgather_dir"]
    train_script_path = os.path.join(forgather_dir, "scripts", "train_script.py")

    # Caller-supplied override (e.g. from the SubmitModal's nproc
    # field, or a `forgather train --nproc N` style call) wins over
    # everything else. Single-node only -- cluster dispatches use
    # rdzv_args's cluster_nproc.
    if rdzv_args is None and nproc_override is not None:
        nproc_per_node = nproc_override
    # CPU dispatch fallback: a zero-GPU reservation paired with
    # nproc_per_node="gpu" would crash torchrun with "invalid literal
    # for int() with base 10: 'gpu'" (the "gpu" sentinel resolves to
    # 0 visible devices because _spawn_subprocess sets
    # CUDA_VISIBLE_DEVICES="" for empty gpu_indices). Drop to 1 worker
    # so CPU debugging of training works. Mirrors the train CLI's
    # fallback in src/forgather/cli/train.py. Only applies in
    # single-node (standalone) mode; cluster dispatches use
    # cluster_nproc from rdzv_args instead.
    elif (
        rdzv_args is None
        and gpu_indices is not None
        and not gpu_indices
        and nproc_per_node == "gpu"
    ):
        nproc_per_node = 1

    cmd: List[str] = ["torchrun"]
    if rdzv_args:
        # Cluster-coordinated rendezvous. The cluster-supplied
        # nproc_per_node wins over the config's because each peer
        # likely has a different GPU count and the master computed
        # an explicit per-peer value at submit time.
        cluster_nproc = rdzv_args.get("nproc_per_node", nproc_per_node)
        cmd.extend(
            [
                "--nnodes",
                str(rdzv_args["nnodes"]),
                "--node-rank",
                str(rdzv_args["node_rank"]),
                "--rdzv-backend",
                str(rdzv_args.get("rdzv_backend", "c10d")),
                "--rdzv-endpoint",
                str(rdzv_args["rdzv_endpoint"]),
                "--rdzv-id",
                str(rdzv_args["rdzv_id"]),
                "--nproc-per-node",
                str(cluster_nproc),
            ]
        )
        is_host = rdzv_args.get("is_host")
        if is_host is not None:
            cmd.extend(
                [
                    "--rdzv-conf",
                    f"is_host={'true' if is_host else 'false'}",
                ]
            )
        local_addr = rdzv_args.get("local_addr")
        if local_addr:
            cmd.extend(["--local-addr", str(local_addr)])
    else:
        cmd.extend(
            [
                "--standalone",
                "--nproc-per-node",
                str(nproc_per_node),
            ]
        )
    cmd.extend(
        [
            os.path.normpath(train_script_path),
            "-p",
            os.path.normpath(project_dir),
        ]
    )
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
    # NOTE: we deliberately do NOT set CUDA_VISIBLE_DEVICES="" on the
    # empty-gpu_indices path here. Some zero-GPU job types
    # (convert / finalize / model with ``--device cuda:0``) opt into
    # using a host GPU without a scheduler reservation -- hiding CUDA
    # blanket-wise would silently break that combo. Spawn paths that
    # need the subprocess truly isolated from CUDA (eval / training /
    # inference for CPU debugging) call ``_isolate_cuda_in_env`` on
    # their proc_env before invoking _spawn_subprocess, or pass
    # ``extra_env={"CUDA_VISIBLE_DEVICES": ""}`` themselves.
    if extra_env:
        proc_env.update(extra_env)

    tty_log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("spawning: %s", " ".join(cmd))
    # Use a `with` block so the parent's fd is closed once Popen has
    # dup'd it into the child. Without close-after-spawn, every launched
    # job leaks one fd in the server process.
    with open(tty_log_path, "wb", buffering=0) as tty_file:
        # TTY logs may capture checkpoint paths, dataset locations, and
        # config details. Other local users on the host shouldn't read
        # them. Chmod before spawn so the child never inherits a window
        # at looser perms.
        try:
            os.chmod(tty_log_path, 0o600)
        except OSError as e:
            log.warning("could not chmod %s to 0600: %s", tty_log_path, e)
        proc = subprocess.Popen(
            cmd,
            env=proc_env,
            stdout=tty_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # The child is in its own session (start_new_session=True) so pgid==pid.
    # Reading os.getpgid(pid) can race with an immediate-exit child; fall back
    # to pid (== pgid by construction) if the leader is already gone.
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid

    return LaunchResult(
        proc=proc,
        pid=proc.pid,
        pgid=pgid,
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
    rdzv_args: Optional[Dict[str, Any]] = None,
    nproc_override: Optional[str] = None,
) -> LaunchResult:
    """Spawn a training run.

    ``rdzv_args`` enables multi-node mode — see ``build_command``. When
    a cluster job is fanned out, the master sets ``rdzv_args`` and
    typically also passes ``NCCL_SOCKET_IFNAME`` through ``extra_env``
    so the NCCL backend picks the right interface. Both default to
    None so single-node submits are unchanged.
    """
    cmd = build_command(
        project_dir,
        config_name,
        dynamic_args,
        rdzv_args,
        gpu_indices=gpu_indices,
        nproc_override=nproc_override,
    )
    extra_env = _isolate_cuda_extra_env(gpu_indices, extra_env)
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_eval_process(
    *,
    eval_project: str,
    eval_template: str,
    model_path: str,
    gpu_indices: List[int],
    tty_log_path: Path,
    extra_env: Optional[Dict[str, str]] = None,
    nproc_override: Optional[str] = None,
    **passthrough,
) -> LaunchResult:
    """Spawn an evaluation run via ``scripts/eval_script.py``.

    Mirrors the argv that ``forgather eval test`` would build so the
    behavior is identical to the CLI — just with GPUs restricted via
    ``CUDA_VISIBLE_DEVICES`` and stdout/stderr captured to a TTY log
    that the web UI can stream. Unlike training, eval processes do not
    register with ``TrainerControlClient``; the scheduler correlates
    lifecycle by PID only.

    All passthrough flags (``--trainer``, ``--checkpoint``, ``--fused-loss``,
    etc.) are forwarded to :func:`eval_ops.build_eval_command`, which is
    driven by the shared spec in
    ``forgather.cli.eval_args._EVAL_SCRIPT_ARGS``.
    """
    cmd = eval_ops.build_eval_command(
        eval_project=eval_project,
        eval_template=eval_template,
        model_path=model_path,
        gpu_indices=gpu_indices,
        nproc_override=nproc_override,
        **passthrough,
    )
    extra_env = _isolate_cuda_extra_env(gpu_indices, extra_env)
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_inference_process(
    *,
    port: int,
    gpu_indices: List[int],
    tty_log_path: Path,
    model_path: Optional[str] = None,
    models: Optional[List[Dict[str, Any]]] = None,
    host: str = "127.0.0.1",
    # Explicit device override. None -> derive from gpu_indices ("cpu"
    # when no GPU is reserved, else let the server's _default_device()
    # pick within CUDA_VISIBLE_DEVICES). Pass "auto" for HF's
    # device_map='auto' (multi-GPU sharding, HF loader only); pass an
    # explicit "cpu" / "cuda:N" / "xpu:N" for direct pinning.
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    from_checkpoint: bool = False,
    compile: bool = False,
    disable_kv_cache: bool = False,
    ignore_eos: bool = False,
    keep_on_gpu: bool = False,
    chat_template: Optional[str] = None,
    cache_implementation: Optional[str] = None,
    compile_args: Optional[str] = None,
    log_level: str = "INFO",
    auth_token_file: Optional[str] = None,
    no_auth: bool = False,
    quiet_tokens: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn an OpenAI-compatible inference server.

    Long-lived — runs until killed. No control protocol; the only
    supported lifecycle actions are kill / force-kill, like eval.

    Pass either ``model_path`` (single-model) or ``models`` (multi-model,
    a list of ``{"name", "path"}`` dicts). Exactly one is required.
    """
    # Pin the spawned server to a real device. Caller's explicit value
    # (e.g. "auto" for HF sharding) wins; otherwise derive:
    #   - zero-GPU dispatch  -> "-d cpu"  (unambiguous in argv/TTY log)
    #   - GPU dispatch       -> omit "-d" and let the server's own
    #     _default_device() resolve to ``<accelerator>:0`` against the
    #     CUDA_VISIBLE_DEVICES set by _spawn_subprocess (subprocess
    #     sees only the reserved GPU(s), so index-0 IS the reserved
    #     device).
    if device is None:
        device = "cpu" if not gpu_indices else None
    # No CUDA isolation here: the inference server is single-process
    # (no torchrun "gpu" sentinel to silently expand to all visible
    # GPUs) and its ``-d`` flag pins the placement directly. An
    # operator explicitly picking device="cuda:0" with
    # requested_gpus=0 (the no-reservation host-CUDA escape hatch,
    # same pattern as convert/finalize/model) should still see host
    # CUDA.
    cmd = inference_ops.build_inference_command(
        model_path=model_path,
        models=models,
        port=port,
        host=host,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        checkpoint_path=checkpoint_path,
        from_checkpoint=from_checkpoint,
        compile=compile,
        disable_kv_cache=disable_kv_cache,
        ignore_eos=ignore_eos,
        keep_on_gpu=keep_on_gpu,
        chat_template=chat_template,
        cache_implementation=cache_implementation,
        compile_args=compile_args,
        log_level=log_level,
        auth_token_file=auth_token_file,
        no_auth=no_auth,
        quiet_tokens=quiet_tokens,
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
    path_prefix: Optional[str] = None,
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
        path_prefix=path_prefix,
    )
    return _spawn_subprocess(cmd, [], tty_log_path, extra_env)


def spawn_diloco_server_process(
    *,
    output_dir: str,
    num_workers: int,
    tty_log_path: Path,
    port: int = 8512,
    host: str = "127.0.0.1",
    async_mode: bool = False,
    dn_buffer_size: int = 0,
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
    num_fragments: int = 1,
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
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a DiLoCo parameter server.

    CPU-only (no GPUs reserved); long-lived like dataset_server /
    mkdocs / tensorboard. The DiLoCo server holds global model
    parameters in memory and applies the outer optimizer when workers
    submit pseudo-gradients over HTTP.

    Auth (issue #90): the scheduler resolves a per-port bearer token
    and writes it to the per-port file before invoking us; we forward
    that path as ``--auth-token-file`` so the token never lands in
    ``argv``. ``no_auth=True`` opts the spawn out of bearer enforcement.
    """
    cmd = diloco_server_ops.build_diloco_server_command(
        output_dir=output_dir,
        num_workers=num_workers,
        port=port,
        host=host,
        async_mode=async_mode,
        dn_buffer_size=dn_buffer_size,
        dylu=dylu,
        dylu_base_sync_every=dylu_base_sync_every,
        sync_every=sync_every,
        bf16_comm=bf16_comm,
        upload_dtype=upload_dtype,
        upload_sr=upload_sr,
        download_dtype=download_dtype,
        download_sr=download_sr,
        wire_format=wire_format,
        grpc_enabled=grpc_enabled,
        num_fragments=num_fragments,
        from_checkpoint=from_checkpoint,
        save_every=save_every,
        save_total_limit=save_total_limit,
        outer_lr=outer_lr,
        outer_momentum=outer_momentum,
        no_nesterov=no_nesterov,
        heartbeat_timeout=heartbeat_timeout,
        min_workers=min_workers,
        auth_token_file=auth_token_file,
        no_auth=no_auth,
        quiet_tokens=quiet_tokens,
        bulk_cleartext=bulk_cleartext,
        run_name=run_name,
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
    quantize: Optional[str] = None,
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
        quantize=quantize,
    )
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_update_process(
    *,
    src_model_path: str,
    dst_model_path: str,
    gpu_indices: List[int],
    tty_log_path: Path,
    arch: Optional[str] = None,
    from_version: Optional[str] = None,
    to_version: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    no_strict: bool = False,
    safetensors: bool = False,
    converter_paths: Optional[List[str]] = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a ``forgather update`` run.

    Fire-and-forget like convert / finalize. Defaults to CPU
    (``--device cpu`` is the script's own default); pass ``gpu_indices``
    if the user picked one.
    """
    cmd = update_ops.build_update_command(
        src_model_path=src_model_path,
        dst_model_path=dst_model_path,
        arch=arch,
        from_version=from_version,
        to_version=to_version,
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        no_strict=no_strict,
        safetensors=safetensors,
        converter_paths=converter_paths,
        dry_run=dry_run,
        log_level=log_level,
    )
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_model_process(
    *,
    project_dir: str,
    config_name: str,
    subcommand: str,
    dynamic_args: Dict[str, Any],
    gpu_indices: List[int],
    tty_log_path: Path,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    no_init_weights: bool = False,
    load_from_checkpoint: Optional[str] = None,
    gradient_checkpointing: bool = False,
    fuse_optim_with_backward: bool = False,
    refresh_model: bool = False,
    save_checkpoint: bool = False,
    safetensors: bool = False,
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    steps: Optional[int] = None,
    lr: Optional[float] = None,
    dataset_project: Optional[str] = None,
    dataset_config: Optional[str] = None,
    packed: bool = False,
    compile: bool = False,
    compile_backend: Optional[str] = None,
    compile_mode: Optional[str] = None,
    compile_dynamic: Optional[bool] = None,
    compile_fullgraph: bool = False,
    amp: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a ``forgather model construct/test`` run.

    Fire-and-forget like eval / convert; no trainer-control protocol.
    May or may not need a GPU depending on ``device`` — callers that
    don't want a GPU pass ``gpu_indices=[]``.
    """
    cmd = model_ops.build_model_command(
        project_dir=project_dir,
        config_name=config_name,
        subcommand=subcommand,
        dynamic_args=dynamic_args,
        device=device,
        dtype=dtype,
        no_init_weights=no_init_weights,
        load_from_checkpoint=load_from_checkpoint,
        gradient_checkpointing=gradient_checkpointing,
        fuse_optim_with_backward=fuse_optim_with_backward,
        refresh_model=refresh_model,
        save_checkpoint=save_checkpoint,
        safetensors=safetensors,
        batch_size=batch_size,
        sequence_length=sequence_length,
        steps=steps,
        lr=lr,
        dataset_project=dataset_project,
        dataset_config=dataset_config,
        packed=packed,
        compile=compile,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        compile_fullgraph=compile_fullgraph,
        amp=amp,
    )
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_construct_process(
    *,
    project_dir: str,
    config_name: str,
    dynamic_args: Dict[str, Any],
    gpu_indices: List[int],
    tty_log_path: Path,
    target: str = "main",
    call: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a ``forgather construct`` run.

    Fire-and-forget like model / eval / convert. ``gpu_indices`` may be
    empty — most targets don't need a GPU, but the modal lets the user
    reserve one for targets that allocate real tensors (e.g. a tokenizer
    trainer that runs on CUDA).
    """
    cmd = construct_ops.build_construct_command(
        project_dir=project_dir,
        config_name=config_name,
        target=target,
        dynamic_args=dynamic_args,
        call=call,
    )
    return _spawn_subprocess(cmd, gpu_indices, tty_log_path, extra_env)


def spawn_dataset_server_process(
    *,
    host: str,
    port: int,
    tty_log_path: Path,
    log_level: str = "INFO",
    no_hf: bool = False,
    allow_paths: bool = False,
    allow_downloads: bool = False,
    locals_: Optional[List[tuple]] = None,
    config_file: Optional[str] = None,
    auth_token_file: Optional[str] = None,
    no_auth: bool = False,
    quiet_tokens: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a Forgather dataset server.

    CPU-only (no GPUs reserved); long-lived like inference / tensorboard.
    Token, when present, is passed via 0600 file rather than argv so it
    isn't visible in ``ps``.
    """
    cmd = dataset_server_ops.build_dataset_server_command(
        host=host,
        port=port,
        log_level=log_level,
        no_hf=no_hf,
        allow_paths=allow_paths,
        allow_downloads=allow_downloads,
        locals_=locals_,
        config_file=config_file,
        auth_token_file=auth_token_file,
        no_auth=no_auth,
        quiet_tokens=quiet_tokens,
    )
    return _spawn_subprocess(cmd, [], tty_log_path, extra_env)


def spawn_dataset_process(
    *,
    project_dir: str,
    config_name: str,
    dynamic_args: Dict[str, Any],
    tty_log_path: Path,
    tokenizer_path: Optional[str] = None,
    pp: bool = False,
    histogram: bool = False,
    target: Optional[str] = None,
    histogram_samples: Optional[int] = None,
    examples: Optional[int] = None,
    features: Optional[List[str]] = None,
    tokenized: bool = False,
    num_shards: Optional[int] = None,
    shard_index: Optional[int] = None,
    select_range: Optional[str] = None,
    seed: Optional[int] = None,
    example_stride: Optional[int] = None,
    truncate: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> LaunchResult:
    """Spawn a ``forgather dataset`` run.

    CPU-only; the caller allocates zero GPUs. Fire-and-forget.
    """
    cmd = dataset_ops.build_dataset_command(
        project_dir=project_dir,
        config_name=config_name,
        dynamic_args=dynamic_args,
        tokenizer_path=tokenizer_path,
        pp=pp,
        histogram=histogram,
        target=target,
        histogram_samples=histogram_samples,
        examples=examples,
        features=features,
        tokenized=tokenized,
        num_shards=num_shards,
        shard_index=shard_index,
        select_range=select_range,
        seed=seed,
        example_stride=example_stride,
        truncate=truncate,
    )
    return _spawn_subprocess(cmd, [], tty_log_path, extra_env)


def kill_process_group(pid: int, sig: int = signal.SIGTERM) -> bool:
    """Kill the whole process group led by ``pid``.

    Returns ``False`` only when no signal could be delivered. If the
    leader pid is gone, we still try ``killpg(pid, sig)`` since
    ``start_new_session=True`` guarantees ``pgid == pid`` for our
    spawns — worker children may still be alive in the same group
    after the leader exits (e.g. torchrun ranks outliving torchrun).
    """
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        # Leader exited, but pgid == pid by construction — try anyway.
        pgid = pid
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
