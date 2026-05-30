"""DiLoCo model-definition bundle: the non-weight files a worker needs to
*construct* the model the server holds, without ever shipping weights.

A DiLoCo worker no longer points at a local checkpoint. Instead the server
serves the model **definition** — ``config.json``, the custom modeling /
configuration ``.py`` files (HF ``trust_remote_code`` sources), and the
tokenizer — from the same self-contained checkpoint directory it was
started from. The worker stages that bundle locally, builds an empty model
on the meta device, and fills it from the server's parameter sync. Weights
(``*.safetensors`` / ``*.bin`` / ``*.pt``), shard indices, server state,
and the audit log are excluded: they're large, they're authority the
server owns, and the worker never persists them.

Custom HF models commonly split the configuration and modeling classes
across two files (``configuration_x.py`` + ``modeling_x.py``), referenced
from ``config.json``'s ``auto_map``. We make no single-file assumption:
the whole directory tree is walked and *every* ``.py`` is included
(relative paths preserved), so both halves of a split definition ride
along and ``trust_remote_code`` resolves each.

This module is the single source of truth for the include/exclude policy,
the deterministic bundle hash, and traversal-safe packing/unpacking, shared
by the server (``_handle_model_def``) and the client
(``DiLoCoClient.fetch_model_def``).
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from typing import List, Tuple

#: Header carrying the bundle's content hash on the ``/model_def`` response.
#: The client validates it against the server's ``/info`` ``model_hash`` so a
#: bundle is never silently paired with a mismatched parameter set.
MODEL_HASH_HEADER = "X-Forgather-Model-Hash"

#: Files included verbatim when present at any level of the checkpoint dir.
#: ``.py`` is handled separately (any module file is included).
_INCLUDE_NAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "added_tokens.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    }
)

#: Weight / state artifacts always excluded — the worker pulls weights from
#: the server's parameter sync, never from the bundle.
_EXCLUDE_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".index.json",
)

#: Exact filenames excluded regardless of suffix matching above.
_EXCLUDE_NAMES = frozenset(
    {
        "server_state.pt",
        "diloco_audit.log",
        ".package_files_copied",
        ".forgather_model_hash",
    }
)


def _is_included(rel_path: str) -> bool:
    """Decide whether a checkpoint-relative file belongs in the bundle.

    Include any ``.py`` (custom-code closure, possibly split across files)
    and the known config/tokenizer names; exclude weights, shard indices,
    server state, and bookkeeping files.
    """
    name = os.path.basename(rel_path)
    if name in _EXCLUDE_NAMES:
        return False
    if any(name.endswith(suffix) for suffix in _EXCLUDE_SUFFIXES):
        return False
    if name.endswith(".py"):
        return True
    return name in _INCLUDE_NAMES


def enumerate_model_def_files(checkpoint_dir: str) -> List[Tuple[str, str]]:
    """Return ``(abs_path, arcname)`` pairs for the bundle, sorted by arcname.

    Walks the tree so a custom-code package in a subdirectory is captured,
    but skips nested ``checkpoint-*`` rollout dirs (a server checkpoint dir
    can contain prior-round checkpoints) and refuses symlinks / anything
    whose real path escapes ``checkpoint_dir`` (defense against a tampered
    or hand-assembled directory). Deterministic order makes the bundle hash
    stable across calls and machines.
    """
    root = os.path.realpath(checkpoint_dir)
    out: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune nested checkpoint rollouts and symlinked subdirs in place.
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith("checkpoint-")
            and not os.path.islink(os.path.join(dirpath, d))
        ]
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            if os.path.islink(abs_path):
                continue
            real = os.path.realpath(abs_path)
            if real != abs_path or os.path.commonpath([root, real]) != root:
                # Symlink target or path escapes the checkpoint dir.
                continue
            rel = os.path.relpath(abs_path, root)
            if _is_included(rel):
                out.append((abs_path, rel))
    out.sort(key=lambda pair: pair[1])
    return out


def compute_bundle_hash(checkpoint_dir: str) -> str:
    """Deterministic SHA-256 over the bundle's contents.

    Hashes each member's arcname and bytes in sorted order, so it changes
    when any included file's name or content changes (a new tokenizer, an
    edited modeling ``.py``, a config tweak) but is stable across restarts
    and hosts for the same definition. Folded into the server's advertised
    ``model_hash`` so the worker's cache stamp invalidates on any
    definition change, not just a parameter-shape change.
    """
    h = hashlib.sha256()
    for abs_path, arcname in enumerate_model_def_files(checkpoint_dir):
        h.update(arcname.encode("utf-8"))
        h.update(b"\0")
        with open(abs_path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


def pack_model_def(checkpoint_dir: str) -> bytes:
    """Build an uncompressed tar of the bundle with deterministic member
    order and normalized metadata (no mtime/uid/gid/mode variation), so the
    same definition packs to the same bytes regardless of on-disk
    timestamps or ownership.
    """
    buf = io.BytesIO()
    # Fixed mtime keeps the archive byte-stable; the bundle hash (content,
    # not the tar wrapper) is the authority clients validate against.
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for abs_path, arcname in enumerate_model_def_files(checkpoint_dir):
            info = tar.gettarinfo(abs_path, arcname=arcname)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with open(abs_path, "rb") as fh:
                tar.addfile(info, fh)
    return buf.getvalue()


def _is_within(base: str, target: str) -> bool:
    base = os.path.realpath(base)
    target = os.path.realpath(target)
    return target == base or target.startswith(base + os.sep)


def extract_model_def(data: bytes, dest_dir: str) -> List[str]:
    """Traversal-safe extraction of a bundle tar into ``dest_dir``.

    Rejects absolute members, ``..`` traversal, symlinks, hardlinks, and
    any non-regular entry — a malicious server must not be able to write
    outside the staging dir or plant a link. Returns the list of extracted
    relative paths.
    """
    os.makedirs(dest_dir, exist_ok=True)
    extracted: List[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        for member in tar.getmembers():
            name = member.name
            if not member.isfile():
                raise ValueError(
                    f"refusing non-regular bundle member: {name!r} "
                    f"(type {member.type!r})"
                )
            if os.path.isabs(name) or name.startswith(("/", "\\")):
                raise ValueError(f"refusing absolute bundle member: {name!r}")
            target = os.path.join(dest_dir, name)
            if not _is_within(dest_dir, target):
                raise ValueError(f"refusing out-of-tree bundle member: {name!r}")
            os.makedirs(os.path.dirname(target) or dest_dir, exist_ok=True)
            with tar.extractfile(member) as src:
                with open(target, "wb") as dst:
                    dst.write(src.read() if src is not None else b"")
            extracted.append(name)
    return extracted
