"""Per-project model output catalog.

Enumerates the *models* a project produces by materializing each config's
``output_dir`` — not by globbing ``<project>/output_models/``. Finetune
projects commonly redirect ``output_dir`` to an external path, so deriving
from configs is the only way to find them.

Backing APIs used (native, not CLI subprocess):

- ``MetaConfig.find_templates`` — enumerate configs.
- ``config_ops.load_output_dir_info`` — resolves the absolute ``output_dir``
  for a single config (shares the overrides-aware materialization the
  Clean-Output dialog already uses).
- ``TrainingLog.from_file`` + ``compute_summary_statistics`` — run summary.

All functions are pure reads; no mutation of on-disk state.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from forgather.eval_config import EvalResult
from forgather.meta_config import MetaConfig
from forgather.ml.analysis import TrainingLog, compute_summary_statistics

from . import config_ops

log = logging.getLogger("forgather_server.models_catalog")

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
# Run dirs are written as ``<time.time_ns()>_<platform.node()>`` — the
# base_trainer keeps both parts. We try to parse the ns prefix for a
# precise ``started_at`` and fall back to stat mtime if the pattern fails
# (legacy run-dir layouts, user renames, etc.).
_RUN_PREFIX_RE = re.compile(r"^(\d+)_(.+)$")


@dataclass
class ModelEntry:
    """One row in the Models tree for a project.

    A "model" here = a distinct absolute ``output_dir`` produced by one or
    more configs in the project. Multiple configs can share an output dir
    (e.g., continue-training configs extending a base) — the ``configs``
    list captures that many-to-one mapping so the UI can show them all.
    """

    output_dir: str
    model_name: str
    configs: List[str] = field(default_factory=list)
    exists: bool = False
    run_count: int = 0
    checkpoint_count: int = 0
    eval_count: int = 0
    total_size_bytes: int = 0
    # Populated only when a config's meta fails to materialize — we still
    # surface the config name under a synthetic model entry so the user
    # doesn't silently lose it.
    parse_errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunEntry:
    run_dir: str
    run_id: str
    started_at: float
    has_logs: bool
    hostname: Optional[str] = None
    # Absolute path to ``<run_dir>/tty.log`` when present. Server-launched
    # jobs symlink their captured stdout/stderr here so historical output
    # is inspectable after the job finishes. ``None`` for runs started
    # before TTY capture existed or for runs from other entry points.
    tty_log_path: Optional[str] = None


@dataclass
class EvalEntry:
    """Thin wrapper around :class:`forgather.eval_config.EvalResult`.

    The canonical schema for ``results.json`` contents lives in the
    library as ``EvalResult``. This wrapper adds filesystem metadata
    (``eval_dir``, ``eval_id``) that only makes sense server-side, plus
    a ``parse_error`` for partial / aborted eval runs so they still show
    up in the list rather than silently disappearing.
    """

    eval_dir: str
    eval_id: str  # basename of the eval dir
    result: Optional[EvalResult] = None
    parse_error: Optional[str] = None


@dataclass
class CheckpointEntry:
    checkpoint_dir: str
    step: int
    size_bytes: int = 0
    world_size: Optional[int] = None
    timestamp: Optional[str] = None
    manifest_present: bool = False


def _dir_size(path: str) -> int:
    """Best-effort recursive size. Silent on per-entry stat failures."""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.stat(
                    os.path.join(root, name), follow_symlinks=False
                ).st_size
            except OSError:
                continue
    return total


def get_project_models(project_dir: str) -> List[ModelEntry]:
    """Group every config in ``project_dir`` by its resolved ``output_dir``.

    Each config's meta is materialized via ``load_output_dir_info`` (same
    path the Clean-Output dialog uses, so overrides are honored). Configs
    whose meta fails to materialize are bucketed under a synthetic entry
    keyed by the error so they're visible in the UI rather than dropped.
    """
    project_dir = os.path.abspath(project_dir)
    try:
        meta = MetaConfig(project_dir)
    except Exception as e:
        log.warning("MetaConfig failed for %s: %s", project_dir, e)
        return []

    try:
        config_names = [name for name, _ in meta.find_templates(meta.config_prefix)]
    except Exception as e:
        log.warning("find_templates failed for %s: %s", project_dir, e)
        return []

    by_output: Dict[str, ModelEntry] = {}
    errors: Dict[str, str] = {}
    for name in sorted(config_names):
        try:
            info = config_ops.load_output_dir_info(project_dir, name)
        except Exception as e:
            errors[name] = str(e)
            continue
        key = info.output_dir
        entry = by_output.get(key)
        if entry is None:
            entry = ModelEntry(
                output_dir=key,
                model_name=os.path.basename(os.path.normpath(key)) or key,
                exists=info.output_dir_exists,
            )
            by_output[key] = entry
        entry.configs.append(name)

    # Populate per-entry run/checkpoint counts and disk usage once per
    # distinct output_dir. We use a single os.walk via ``_dir_size`` plus
    # a couple of scandirs rather than recomputing info.output_dir_size_*
    # (which load_output_dir_info already does but costs one walk per
    # config — duplicating work when configs share output).
    for entry in by_output.values():
        if not entry.exists:
            continue
        try:
            entry.total_size_bytes = _dir_size(entry.output_dir)
        except OSError:
            pass
        entry.run_count = _count_runs(entry.output_dir)
        entry.checkpoint_count = _count_checkpoints(entry.output_dir)
        entry.eval_count = _count_evals(entry.output_dir)

    out = sorted(by_output.values(), key=lambda m: m.model_name.lower())
    if errors:
        # Surface parse errors as a single synthetic entry at the end so
        # they don't mix with healthy models or get silently dropped.
        out.append(
            ModelEntry(
                output_dir="",
                model_name="(configs with parse errors)",
                configs=sorted(errors.keys()),
                exists=False,
                parse_errors=errors,
            )
        )
    return out


def _count_runs(output_dir: str) -> int:
    runs_dir = os.path.join(output_dir, "runs")
    if not os.path.isdir(runs_dir):
        return 0
    try:
        return sum(1 for e in os.scandir(runs_dir) if e.is_dir())
    except OSError:
        return 0


def _checkpoint_dirs(output_dir: str) -> List[str]:
    """Return every ``checkpoint-N`` dir under ``output_dir``.

    Two conventions exist in Forgather:
        * Built-in trainer writes ``<output_dir>/checkpoint-N/`` directly.
        * ``sharded_checkpoint`` writes ``<output_dir>/checkpoints/checkpoint-N/``.
    This helper looks in both locations and returns a flat list.
    """
    found: List[str] = []
    for base in (output_dir, os.path.join(output_dir, "checkpoints")):
        if not os.path.isdir(base):
            continue
        try:
            for e in os.scandir(base):
                if e.is_dir() and _CHECKPOINT_RE.match(e.name):
                    found.append(e.path)
        except OSError:
            continue
    return found


def _count_checkpoints(output_dir: str) -> int:
    return len(_checkpoint_dirs(output_dir))


def _count_evals(output_dir: str) -> int:
    evals_root = os.path.join(output_dir, "evals")
    if not os.path.isdir(evals_root):
        return 0
    try:
        return sum(1 for e in os.scandir(evals_root) if e.is_dir())
    except OSError:
        return 0


def get_model_runs(output_dir: str) -> List[RunEntry]:
    """List every run under ``<output_dir>/runs/``.

    Sorted newest-first. Run IDs follow base_trainer.py's format
    ``<time.time_ns()>_<hostname>``; we parse the ns prefix for
    ``started_at`` (falls back to mtime).
    """
    output_dir = os.path.abspath(output_dir)
    runs_dir = os.path.join(output_dir, "runs")
    if not os.path.isdir(runs_dir):
        return []

    out: List[RunEntry] = []
    try:
        entries = list(os.scandir(runs_dir))
    except OSError:
        return []

    for e in entries:
        if not e.is_dir():
            continue
        run_id = e.name
        started_at = 0.0
        hostname: Optional[str] = None
        m = _RUN_PREFIX_RE.match(run_id)
        if m:
            try:
                # The trainer writes time.time_ns(); convert to seconds so
                # the frontend can feed it straight into Date.
                started_at = int(m.group(1)) / 1e9
            except ValueError:
                started_at = 0.0
            hostname = m.group(2)
        if started_at == 0.0:
            try:
                started_at = e.stat(follow_symlinks=False).st_mtime
            except OSError:
                pass
        logs_path = os.path.join(e.path, "trainer_logs.json")
        tty_path = os.path.join(e.path, "tty.log")
        # ``tty.log`` is typically a symlink into ~/.forgather/server/jobs/;
        # accept either a regular file or a symlink that resolves to one.
        has_tty = os.path.isfile(tty_path) or (
            os.path.islink(tty_path) and os.path.isfile(os.path.realpath(tty_path))
        )
        out.append(
            RunEntry(
                run_dir=os.path.abspath(e.path),
                run_id=run_id,
                started_at=started_at,
                has_logs=os.path.isfile(logs_path),
                hostname=hostname,
                tty_log_path=os.path.abspath(tty_path) if has_tty else None,
            )
        )

    out.sort(key=lambda r: r.started_at, reverse=True)
    return out


def get_run_summary(run_dir: str) -> Dict[str, Any]:
    """Return the run's summary + paths to its artifacts.

    Shape: ``{summary, log_path, config_path, pp_path}``. Each path is
    ``None`` when the file doesn't exist so the UI can disable the link.
    """
    run_dir = os.path.abspath(run_dir)
    log_path = os.path.join(run_dir, "trainer_logs.json")
    config_path = os.path.join(run_dir, "config.yaml")
    pp_path = os.path.join(run_dir, "preprocessed_config.yaml")

    summary: Dict[str, Any] = {}
    if os.path.isfile(log_path):
        try:
            tlog = TrainingLog.from_file(log_path)
            summary = compute_summary_statistics(tlog)
        except Exception as e:
            summary = {"error": str(e)}

    return {
        "summary": summary,
        "log_path": log_path if os.path.isfile(log_path) else None,
        "config_path": config_path if os.path.isfile(config_path) else None,
        "pp_path": pp_path if os.path.isfile(pp_path) else None,
    }


def read_run_tty(run_dir: str, max_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the contents of ``<run_dir>/tty.log`` (or its symlink target).

    Capped at ``max_bytes`` from the *end* of the file — training TTY
    captures can be gigabytes for long runs, and the terminal portion is
    almost always what you want. Raises ``FileNotFoundError`` if the file
    is absent.
    """
    run_dir = os.path.abspath(run_dir)
    tty_path = os.path.join(run_dir, "tty.log")
    real = os.path.realpath(tty_path)
    if not os.path.isfile(real):
        raise FileNotFoundError(f"no tty log at {tty_path}")

    size = os.path.getsize(real)
    offset = max(0, size - max_bytes)
    truncated = offset > 0
    with open(real, "rb") as f:
        if offset:
            f.seek(offset)
            # Drop a potential partial first line so the output is clean.
            _ = f.readline()
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text = f"[…truncated {offset} bytes from start…]\n{text}"
    return text


def list_model_evaluations(output_dir: str) -> List[EvalEntry]:
    """Enumerate ``<output_dir>/evals/*/results.json``, newest-first.

    Each entry is a parsed ``results.json`` plus pulled-out primary
    metrics. Directories without a readable ``results.json`` are still
    returned with ``parse_error`` populated so partial / aborted eval runs
    show up rather than disappearing.
    """
    output_dir = os.path.abspath(output_dir)
    evals_root = os.path.join(output_dir, "evals")
    if not os.path.isdir(evals_root):
        return []

    out: List[EvalEntry] = []
    try:
        entries = list(os.scandir(evals_root))
    except OSError:
        return []

    # Cache the authoritative field set once per call so we don't scan
    # the dataclass on every parse.
    known_fields = {f.name for f in fields(EvalResult)}

    for e in entries:
        if not e.is_dir():
            continue
        eval_dir = os.path.abspath(e.path)
        entry = EvalEntry(eval_dir=eval_dir, eval_id=e.name)
        results_path = os.path.join(eval_dir, "results.json")
        if not os.path.isfile(results_path):
            entry.parse_error = "missing results.json"
            out.append(entry)
            continue
        try:
            data = json.loads(Path(results_path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            entry.parse_error = f"failed to parse results.json: {exc}"
            out.append(entry)
            continue
        if not isinstance(data, dict):
            entry.parse_error = "results.json is not a JSON object"
            out.append(entry)
            continue
        # Accept any subset of EvalResult's required fields — a partial
        # run may have written outcome fields as None but still carries
        # identity fields. Unknown keys (from schema evolution) are
        # ignored here and surfaced to the UI as "extra" fields.
        filtered = {k: v for k, v in data.items() if k in known_fields}
        try:
            entry.result = EvalResult(**filtered)
        except TypeError as exc:
            entry.parse_error = f"schema mismatch: {exc}"
        out.append(entry)

    # Sort newest-first. ``timestamp`` is ISO-8601 when present; the
    # eval_id basename embeds yyyymmddThhmmss so lexicographic ordering
    # lines up with chronological as a fallback. Using `or ""` (rather
    # than `or None`) keeps the key always-string so a result with a
    # missing-but-not-None timestamp doesn't leak `None` into the sort
    # and produce a TypeError when compared with a string-keyed entry.
    def _sort_key(ev: EvalEntry) -> str:
        ts = ev.result.timestamp if ev.result else None
        return ts or ev.eval_id

    out.sort(key=_sort_key, reverse=True)
    return out


def list_run_checkpoints(output_dir: str) -> List[CheckpointEntry]:
    """List ``<output_dir>/checkpoint-N/`` dirs, newest-step-first.

    Parses ``checkpoint_manifest.json`` when present for world_size and
    timestamp — purely informational; the trainer records it automatically.
    """
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        return []

    out: List[CheckpointEntry] = []
    for ckpt_path in _checkpoint_dirs(output_dir):
        m = _CHECKPOINT_RE.match(os.path.basename(ckpt_path))
        if not m:
            continue
        step = int(m.group(1))
        size = 0
        try:
            size = _dir_size(ckpt_path)
        except OSError:
            pass
        ck = CheckpointEntry(
            checkpoint_dir=os.path.abspath(ckpt_path),
            step=step,
            size_bytes=size,
        )
        manifest_path = os.path.join(ckpt_path, "checkpoint_manifest.json")
        if os.path.isfile(manifest_path):
            ck.manifest_present = True
            try:
                data = json.loads(Path(manifest_path).read_text())
                if isinstance(data, dict):
                    ws = data.get("world_size")
                    if isinstance(ws, int):
                        ck.world_size = ws
                    ts = data.get("timestamp")
                    if isinstance(ts, str):
                        ck.timestamp = ts
            except (OSError, json.JSONDecodeError):
                pass
        out.append(ck)

    out.sort(key=lambda c: c.step, reverse=True)
    return out
