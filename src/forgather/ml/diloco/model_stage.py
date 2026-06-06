"""Stage a DiLoCo server's model-definition bundle into a worker-local
directory, so the worker can build the model with no shared filesystem and
no operator-supplied model path (issue #53).

The staging is *lazy and output-dir-scoped*: a training config wires
``stage_model_def`` as a cached ``!singleton`` closing over the worker's
computed ``output_dir`` and the server address. Nothing happens until the
model assets materialize; the first reference (tokenizer or model config)
triggers a single fetch into ``<output_dir>/diloco_model_def/``, and the
result is cached for every later reference in the same process. Because the
target lives under the worker's own output dir, an operator debugging a run
finds the exact code/config that worker built from next to its logs, and
there is no global cache to invalidate across servers.

Reuse and concurrency:

* A ``.forgather_model_hash`` stamp records the bundle identity. On a later
  run (or a restart resuming the same server) a matching stamp short-circuits
  the network fetch; a mismatch — the server was restarted on a *different*
  model — forces a clean re-fetch (no stale reuse, per the no-silent-fallback
  rule).
* ``file_lock_build`` serializes concurrent workers / DDP ranks sharing one
  host so exactly one fetches while the rest wait, then reuse.

Fail-loud: any fetch / hash-mismatch / extraction error propagates. Under
DiLoCo there is no offline fallback — a worker that cannot obtain the
definition from the server must not silently build something else.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

from forgather.ml.construct import file_lock_build

logger = logging.getLogger(__name__)

#: Subdirectory under the worker's output dir where the bundle is staged.
STAGE_SUBDIR = "diloco_model_def"

#: Stamp file (inside the staged dir) recording the bundle's model hash.
#: Excluded from the bundle itself by ``model_def`` policy, so a fetch never
#: overwrites or ships it.
STAMP_NAME = ".forgather_model_hash"


def _read_stamp(stamp_path: str) -> str:
    try:
        with open(stamp_path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def stage_model_def(
    server_addr: str,
    output_dir: str,
    *,
    token: str | None = None,
    verify_tls: bool = True,
    timeout: float = 120.0,
    lock_timeout: float = 300.0,
) -> str:
    """Fetch the server's model-definition bundle into ``output_dir`` and
    return the staged directory path.

    Args:
        server_addr: DiLoCo server URL or ``host:port``. Token and TLS
            verification are auto-discovered by ``DiLoCoClient`` (per-port
            loopback file / ``FORGATHER_DILOCO_SERVER_TOKEN``) when not
            passed explicitly — matching how the worker authenticates.
        output_dir: The worker's computed output directory. The bundle is
            staged into ``<output_dir>/diloco_model_def/``.

    Returns:
        Absolute path to the staged directory, suitable as a model id-or-path
        for ``AutoConfig`` / ``AutoTokenizer`` / ``AutoModel`` with
        ``trust_remote_code=True``.
    """
    # Imported lazily: client.py pulls in torch, and this module is imported
    # by template materialization where keeping the import local avoids a
    # torch import on paths that never stage.
    from forgather.ml.diloco.client import DiLoCoClient
    from forgather.ml.diloco.coordinator import CoordinatorClient

    output_dir = os.path.abspath(output_dir)
    local_dir = os.path.join(output_dir, STAGE_SUBDIR)
    stamp_path = os.path.join(local_dir, STAMP_NAME)

    # Model staging (get_info + fetch_model_def) is coordination, not bulk
    # transport — go through the coordinator surface (#154).
    client = CoordinatorClient(
        DiLoCoClient(server_addr, timeout=timeout, token=token, verify_tls=verify_tls)
    )
    # The authoritative bundle identity for this server right now.
    want_hash = client.get_info().get("model_hash") or ""

    # Fast path: an existing staging whose stamp matches the live server is
    # reused without taking the lock or hitting the network for the tar.
    if want_hash and _read_stamp(stamp_path) == want_hash:
        logger.info(
            "DiLoCo: reusing staged model definition at %s (hash %s)",
            local_dir,
            want_hash[:12],
        )
        return local_dir

    os.makedirs(output_dir, exist_ok=True)
    # force_lock so we serialize even when local_dir already exists (its
    # stamp may be stale); re-check the stamp once we hold the lock.
    with file_lock_build(local_dir, timeout=lock_timeout, force_lock=True):
        if want_hash and _read_stamp(stamp_path) == want_hash:
            logger.info(
                "DiLoCo: staged model definition appeared while waiting "
                "for the lock; reusing %s",
                local_dir,
            )
            return local_dir

        logger.info(
            "DiLoCo: staging model definition from %s into %s",
            server_addr,
            local_dir,
        )
        # Fetch into a sibling temp dir, then atomically swap it into place,
        # writing the stamp last so an interrupted fetch leaves no
        # match-looking directory behind.
        tmp_dir = tempfile.mkdtemp(prefix=".diloco_model_def.", dir=output_dir)
        try:
            fetched_hash = client.fetch_model_def(tmp_dir)
            # Fail loud on a definition-less bundle rather than stamping an
            # empty dir as valid (which poisons the cache and resurfaces as a
            # cryptic "Unrecognized model … no model_type key" at config
            # load). A server restarted off a weights-only checkpoint used to
            # serve an empty bundle; the server now refuses that, but validate
            # here too so no future empty fetch is ever cached.
            if not os.path.exists(os.path.join(tmp_dir, "config.json")):
                raise RuntimeError(
                    f"DiLoCo: model-definition bundle from {server_addr} has "
                    "no config.json — the server has no model definition to "
                    "serve (it may have been started from a weights-only "
                    "checkpoint). Refusing to stage an empty definition."
                )
            with open(os.path.join(tmp_dir, STAMP_NAME), "w", encoding="utf-8") as fh:
                fh.write(fetched_hash or want_hash)
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            os.replace(tmp_dir, local_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    return local_dir
