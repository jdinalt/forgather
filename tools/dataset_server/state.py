"""
In-process state for the dataset server.

Owns the loaded-backend cache, the `local/<name>` -> filesystem-path
mapping, and the policy flags (HF cache enabled, path loading
allowed, downloads allowed). The FastAPI route handlers query and
mutate this object; the server entry point constructs it from
parsed CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from forgather.ml.datasets.iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


# Fields the /v1/load endpoint accepts and forwards to the local loader.
# Anything else in the request body is silently ignored.
LOAD_FIELDS = (
    "path",
    "name",
    "split",
    "data_files",
    "revision",
    "force_reindex",
    "num_proc",
)


def canonical_handle(load_args: Dict[str, Any]) -> str:
    """Stable short hash of a normalized load_args dict."""
    normalized = {k: v for k, v in load_args.items() if v is not None}
    canonical = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class PolicyError(Exception):
    """Raised when a load request is denied by policy.

    Carries an HTTP status code so the route handler can translate
    directly without re-classifying the failure.
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class HandleEntry:
    backend: IterableDatasetBackend
    load_args: Dict[str, Any]
    source: str  # "local" | "path" | "hf"


@dataclass
class ServerState:
    """All mutable server state lives here.

    Attributes
    ----------
    hf_cache_enabled : bool
        If False, ``/v1/load`` rejects HF dataset ids outright.
    allow_paths : bool
        If False, ``/v1/load`` rejects requests whose ``path`` is an
        existing filesystem path. Local-mapping requests are unaffected.
    allow_downloads : bool
        If False, the server runs HF loads with ``HF_DATASETS_OFFLINE=1``,
        so a cache miss surfaces as 404 instead of triggering a download.
    local_datasets : dict[str, str]
        Mapping of ``local/<name>`` -> resolved filesystem path. Populated
        from ``--local`` flags. Path existence is verified on register.
    auth_required : bool
        Reflected via ``/v1/auth/status`` so clients can probe.
    """

    hf_cache_enabled: bool = True
    allow_paths: bool = False
    allow_downloads: bool = False
    local_datasets: Dict[str, str] = field(default_factory=dict)
    auth_required: bool = True

    # Loaded backends, keyed by canonical_handle(load_args).
    _handles: Dict[str, HandleEntry] = field(default_factory=dict)
    # Single short-critical-section lock guarding the handle registry
    # and the inflight-load table. We never hold this across an actual
    # loader call — see ``load_on_demand`` for the per-handle dedup
    # pattern that keeps loads of distinct handles parallel.
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # ``handle -> Event`` for loads currently in flight. Deduplicates
    # concurrent ``/v1/load`` requests for the same canonical args.
    _inflight: Dict[str, threading.Event] = field(default_factory=dict)

    # Hook for tests: override the loader call so we don't need real
    # Arrow files. Production leaves this as the default
    # _local_load_iterable_dataset import.
    loader: Optional[Callable[..., Any]] = None

    # ----- handle registry -----

    def list_handles(self) -> List[str]:
        return sorted(self._handles.keys())

    def get(self, handle: str) -> Optional[IterableDatasetBackend]:
        entry = self._handles.get(handle)
        return entry.backend if entry is not None else None

    def get_entry(self, handle: str) -> Optional[HandleEntry]:
        return self._handles.get(handle)

    def register(
        self,
        handle: str,
        backend: IterableDatasetBackend,
        load_args: Optional[Dict[str, Any]] = None,
        source: str = "registered",
    ) -> None:
        """Register a backend under a chosen handle. Used by tests and
        by load_on_demand."""
        if not handle or "/" in handle:
            raise ValueError(f"Invalid handle: {handle!r}")
        with self._lock:
            self._handles[handle] = HandleEntry(
                backend=backend,
                load_args=dict(load_args or {}),
                source=source,
            )

    def unregister(self, handle: str) -> None:
        with self._lock:
            self._handles.pop(handle, None)

    # ----- local mappings -----

    def add_local(self, name: str, path: str) -> None:
        if "/" in name:
            raise ValueError(f"Local dataset name {name!r} must not contain '/'")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local dataset path does not exist: {path}")
        self.local_datasets[name] = os.path.abspath(path)

    def list_locals(self) -> Dict[str, str]:
        return dict(self.local_datasets)

    # ----- the policy gate -----

    def load_on_demand(self, load_args: Dict[str, Any]) -> str:
        """Resolve ``load_args`` against policy, lazy-load if necessary,
        return the handle. Raises ``PolicyError`` if the request is
        rejected (the route handler maps that to an HTTP status).

        Per-handle deduplication: concurrent requests for the SAME
        canonical args wait for one shared load. Concurrent requests
        for DIFFERENT handles run independently — the loader is
        called outside ``self._lock``, so a long HF download does
        not block /v1/load for any other dataset.
        """
        path = load_args.get("path")
        if not path:
            raise PolicyError(400, "Missing required field 'path'")

        # Resolve the request into (resolved_args, source) per the
        # documented policy. May raise PolicyError synchronously.
        resolved_args, source = self._resolve_request(load_args)

        # Canonical handle is computed from RESOLVED args so e.g.
        # `local/stories` and `/data/tinystories` (same target after
        # `--allow-paths`) hash to the same key — no duplicate loads.
        handle = canonical_handle(resolved_args)

        # Inflight coordination: short critical section to either
        # find a cached entry, find a load already in progress and
        # wait on its Event, or register ourselves as the loader.
        with self._lock:
            if handle in self._handles:
                return handle
            existing = self._inflight.get(handle)
            if existing is not None:
                event = existing
                we_load = False
            else:
                event = threading.Event()
                self._inflight[handle] = event
                we_load = True

        if not we_load:
            # Another thread is loading this exact handle — wait for
            # it to finish, then re-check the registry.
            event.wait()
            with self._lock:
                if handle in self._handles:
                    return handle
            raise PolicyError(
                500,
                f"Concurrent load of {path!r} failed; see server logs",
            )

        # We are the loader. Run OUTSIDE the lock so distinct-handle
        # loads stay parallel.
        try:
            backend = self._do_load(path, source, resolved_args)
        except BaseException:
            # On any failure, signal waiters and remove the inflight
            # marker so a future request can retry.
            with self._lock:
                self._inflight.pop(handle, None)
            event.set()
            raise

        with self._lock:
            self._handles[handle] = HandleEntry(
                backend=backend,
                # Store the RESOLVED args so introspection
                # (/v1/datasets/{handle}) tells the operator what
                # was actually loaded, not what the client typed.
                load_args=dict(resolved_args),
                source=source,
            )
            self._inflight.pop(handle, None)
        event.set()
        return handle

    def _do_load(
        self, path: str, source: str, resolved_args: Dict[str, Any]
    ) -> IterableDatasetBackend:
        """Run the actual loader. Called WITHOUT ``self._lock`` held."""
        loader = self.loader or _default_loader
        env_overrides: Dict[str, str] = {}
        if source == "hf" and not self.allow_downloads:
            env_overrides["HF_DATASETS_OFFLINE"] = "1"

        logger.info(
            "loading dataset on demand: source=%s args=%s",
            source,
            resolved_args,
        )
        with _temp_env(env_overrides):
            try:
                ds = loader(**resolved_args)
            except (FileNotFoundError, ConnectionError) as exc:
                if source == "hf":
                    raise PolicyError(
                        404,
                        f"HF dataset {path!r} not in cache "
                        f"(downloads disabled): {exc}",
                    ) from exc
                raise PolicyError(404, str(exc)) from exc
            except Exception as exc:
                # OfflineModeIsEnabled and friends — match by name to
                # avoid hard import dependency on huggingface_hub.
                name = type(exc).__name__
                if source == "hf" and name in (
                    "OfflineModeIsEnabled",
                    "LocalEntryNotFoundError",
                    "RepositoryNotFoundError",
                ):
                    raise PolicyError(
                        404,
                        f"HF dataset {path!r} not available: {exc}",
                    ) from exc
                raise
        return getattr(ds, "backend", ds)

    def _resolve_request(self, load_args: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Apply the loading policy. Returns (effective_load_args, source).

        Raises ``PolicyError`` for denied requests.
        """
        path = load_args["path"]

        # 1. local/<name> mapping
        if isinstance(path, str) and path.startswith("local/"):
            name = path[len("local/") :]
            if not name:
                raise PolicyError(400, "Local dataset name must follow 'local/'")
            if name not in self.local_datasets:
                raise PolicyError(404, f"Unknown local dataset: local/{name}")
            resolved = dict(load_args)
            resolved["path"] = self.local_datasets[name]
            return resolved, "local"

        # 2. existing filesystem path
        if isinstance(path, str) and os.path.exists(path):
            if not self.allow_paths:
                raise PolicyError(
                    403,
                    "Loading by filesystem path is disabled "
                    "(start the server with --allow-paths)",
                )
            return dict(load_args), "path"

        # 3. otherwise treat as an HF dataset id
        if not self.hf_cache_enabled:
            raise PolicyError(
                403,
                "HF cache loading is disabled on this server " "(--no-hf was passed)",
            )
        return dict(load_args), "hf"


def _default_loader(**kwargs):
    """Lazy import to avoid pulling the loader at module load time."""
    from forgather.ml.datasets.fast_hf_loader import _local_load_iterable_dataset

    return _local_load_iterable_dataset(**kwargs)


class _temp_env:
    """Context manager that sets/restores environment variables."""

    def __init__(self, overrides: Dict[str, str]):
        self.overrides = overrides
        self._saved: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self._saved[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, prev in self._saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
