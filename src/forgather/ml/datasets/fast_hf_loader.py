"""
Fast HuggingFace Dataset Loader.

Builds a compact JSON index over the Arrow cache files behind a
HuggingFace dataset on first call, then reads that index in
milliseconds on every subsequent call — bypassing the 10-30 minute
startup that ``datasets.load_dataset(...).to_iterable_dataset()``
imposes on large datasets like C4.

The loader returns a `ComposableIterableDataset` wrapping an
`ArrowBackend` over the indexed files. All higher-level operations
(`map`, `filter`, `select`, `slice`, `shard`, `shuffle`,
`set_epoch`, `state_dict`/`load_state_dict`, multi-worker support,
length estimation) live on the wrapper; the backend is purely the
file-walking storage layer.
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, load_dataset

from .arrow_backend import ArrowBackend
from .composable_iterable_dataset import ComposableIterableDataset

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

# Metadata version - increment when index format changes
METADATA_VERSION = 2  # v2: Added per-file example counts and version check


def _parse_split_notation(split: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse HuggingFace split notation into base split and slice bounds.

    Examples:
        "train" → ("train", None, None)
        "train[100:]" → ("train", 100, None)
        "train[:1000]" → ("train", None, 1000)
        "train[100:1000]" → ("train", 100, 1000)
        "train[10%:20%]" → ("train", "10%", "20%")
    """
    if not split:
        return split, None, None

    match = re.match(r"^([^[]+)(?:\[([^:]*):([^\]]*)\])?$", split)
    if not match:
        return split, None, None

    base_split = match.group(1)
    start_str = match.group(2)
    end_str = match.group(3)

    if start_str:
        start_idx = start_str if "%" in start_str else int(start_str)
    else:
        start_idx = None

    if end_str:
        end_idx = end_str if "%" in end_str else int(end_str)
    else:
        end_idx = None

    return base_split, start_idx, end_idx


def _make_dataset(
    arrow_files: List[str],
    file_lengths: Optional[List[int]],
    *,
    length_estimate: str,
    reset_length_on_iter: bool,
    slice_start=None,
    slice_end=None,
) -> ComposableIterableDataset:
    """
    Build the public-facing wrapper around an `ArrowBackend`. Applies
    the optional split-notation slice if present.
    """
    backend = ArrowBackend(arrow_files, file_lengths)
    ds = ComposableIterableDataset(
        backend,
        length_estimate=length_estimate,
        reset_length_on_iter=reset_length_on_iter,
    )
    if slice_start is not None or slice_end is not None:
        ds = ds.slice(slice_start, slice_end)
    return ds


class FastDatasetLoaderSimple:
    """
    Fast HuggingFace dataset loader backed by an Arrow file index.

    On the first call for a given dataset/split combination the loader
    downloads (or locates) the dataset via the HuggingFace ``datasets``
    library, records the paths and per-file example counts of the
    underlying Arrow cache files in a compact JSON index, and returns
    a `ComposableIterableDataset` wrapping an `ArrowBackend`. All
    subsequent calls for the same configuration load in milliseconds
    by reading the index directly.

    Both HuggingFace Hub datasets and locally saved datasets (produced
    by ``Dataset.save_to_disk()``) are supported.

    Parameters
    ----------
    index_dir : str, optional
        Directory in which the JSON index files are stored. Defaults to
        ``~/.cache/fast_hf_indexes_simple``.

    Examples
    --------
    >>> loader = FastDatasetLoaderSimple()
    >>> ds = loader.load_iterable("allenai/c4", name="en", split="train")
    >>> ds = ds.shuffle(seed=42).shard(num_shards=4, index=0)
    >>> for example in ds:
    ...     pass
    """

    def __init__(self, index_dir: Optional[str] = None):
        if index_dir is None:
            index_dir = os.path.expanduser("~/.cache/fast_hf_indexes_simple")

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _get_config_hash(
        self,
        path: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        data_files: Optional[Union[str, list]] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> str:
        config = {
            "path": path,
            "name": name,
            "split": split,
            "data_files": data_files,
            "revision": revision,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_index_file(self, config_hash: str) -> Path:
        return self.index_dir / f"{config_hash}.json"

    def _get_arrow_files(self, dataset_obj: Dataset) -> Optional[list]:
        """Get Arrow file paths from dataset."""
        if hasattr(dataset_obj, "cache_files") and dataset_obj.cache_files:
            return [cf["filename"] for cf in dataset_obj.cache_files]
        if hasattr(dataset_obj, "_data_files") and dataset_obj._data_files:
            return [df["filename"] for df in dataset_obj._data_files]
        return None

    def _get_file_lengths_from_metadata(
        self, arrow_files: List[str], split: str
    ) -> Optional[List[int]]:
        """
        Try to extract file lengths from HuggingFace's ``dataset_info.json``.

        Avoids opening each Arrow file individually to read metadata. For
        datasets with thousands of files this is significantly faster.
        Returns ``None`` on any mismatch / missing data — caller falls
        back to opening files individually.
        """
        if not arrow_files:
            return None

        try:
            cache_dir = Path(arrow_files[0]).parent
            dataset_info_path = cache_dir / "dataset_info.json"
            if not dataset_info_path.exists():
                return None

            with open(dataset_info_path, "r") as f:
                dataset_info = json.load(f)

            splits = dataset_info.get("splits", {})
            if split not in splits:
                return None

            shard_lengths = splits[split].get("shard_lengths", [])

            if len(shard_lengths) != len(arrow_files):
                logger.warning(
                    f"Shard count mismatch: dataset_info.json has "
                    f"{len(shard_lengths)} shards, but found "
                    f"{len(arrow_files)} Arrow files. Falling back to "
                    f"file-by-file indexing."
                )
                return None

            if not shard_lengths:
                return None

            logger.info(
                f"Loaded file lengths from dataset_info.json: "
                f"{len(shard_lengths)} files, "
                f"{sum(shard_lengths):,} total examples"
            )
            return shard_lengths

        except Exception as e:
            logger.debug(
                f"Could not load file lengths from dataset_info.json: {e}. "
                f"Falling back to file-by-file indexing."
            )
            return None

    def _is_saved_dataset_path(self, path: str) -> bool:
        """
        Check if path is a local directory containing a saved dataset.

        Multi-split form has ``dataset_dict.json`` at the root and
        per-split subdirectories. Single-split form has ``state.json``
        at the root.
        """
        if not path:
            return False

        dataset_path = Path(path)
        if not dataset_path.is_dir():
            return False

        if (dataset_path / "dataset_dict.json").exists():
            return True
        if (dataset_path / "state.json").exists():
            return True

        return False

    def _load_saved_dataset(
        self,
        path: str,
        split: str,
        force_reindex: bool = False,
        length_estimate: str = "dynamic",
        reset_length_on_iter: bool = False,
    ) -> Optional[ComposableIterableDataset]:
        """Load a saved dataset directly from disk."""
        dataset_path = Path(path)

        # Determine split directory.
        dataset_dict_path = dataset_path / "dataset_dict.json"
        if dataset_dict_path.exists():
            with open(dataset_dict_path, "r") as f:
                dataset_dict = json.load(f)
            available_splits = dataset_dict.get("splits", [])
            if split not in available_splits:
                logger.warning(
                    f"Split '{split}' not found in saved dataset. "
                    f"Available splits: {available_splits}"
                )
                return None
            split_dir = dataset_path / split
        else:
            split_dir = dataset_path

        # Read state.json to get data files.
        state_path = split_dir / "state.json"
        if not state_path.exists():
            logger.warning(f"state.json not found in {split_dir}")
            return None

        with open(state_path, "r") as f:
            state = json.load(f)

        data_files = state.get("_data_files", [])
        if not data_files:
            logger.warning(f"No data files listed in {state_path}")
            return None

        # Build full paths to Arrow files.
        arrow_files = [
            str(split_dir / df["filename"])
            for df in data_files
            if df.get("filename", "").endswith(".arrow")
        ]
        if not arrow_files:
            logger.warning(f"No Arrow files found in {split_dir}")
            return None

        missing = [f for f in arrow_files if not Path(f).exists()]
        if missing:
            logger.warning(f"Missing Arrow files: {missing[:5]}...")
            return None

        num_files = len(arrow_files)
        logger.info(f"Found saved dataset with {num_files} Arrow file(s)")

        # Check for cached index.
        config_hash = self._get_config_hash(path, split=split)
        if not force_reindex:
            index_data = self._load_index(config_hash)
            if index_data is not None:
                cached_files = index_data.get("arrow_files", [])
                if cached_files == arrow_files:
                    logger.info("Loading from cached index")
                    file_lengths = index_data.get("file_lengths")
                    return _make_dataset(
                        arrow_files,
                        file_lengths,
                        length_estimate=length_estimate,
                        reset_length_on_iter=reset_length_on_iter,
                    )

        # Get file lengths — try dataset_info.json first.
        file_lengths = self._get_file_lengths_from_metadata(arrow_files, split)

        if file_lengths is None:
            logger.info("Computing per-file example counts...")
            file_lengths = []

            use_progress = HAS_TQDM and sys.stderr.isatty()
            iterator = (
                tqdm(arrow_files, desc="Indexing files", unit="file")
                if use_progress
                else arrow_files
            )

            for arrow_file in iterator:
                ds_file = Dataset.from_file(arrow_file)
                file_lengths.append(len(ds_file))

        total_examples = sum(file_lengths)
        logger.info(f"Total examples: {total_examples:,}")

        metadata = {
            "dataset_path": path,
            "split": split,
            "source": "saved_dataset",
            "num_arrow_files": num_files,
            "total_examples": total_examples,
        }
        self._save_index(config_hash, arrow_files, file_lengths, metadata)

        return _make_dataset(
            arrow_files,
            file_lengths,
            length_estimate=length_estimate,
            reset_length_on_iter=reset_length_on_iter,
        )

    def _save_index(
        self,
        config_hash: str,
        arrow_files: list,
        file_lengths: list,
        metadata: Dict[str, Any],
    ):
        index_data = {
            "version": METADATA_VERSION,
            "arrow_files": arrow_files,
            "file_lengths": file_lengths,
            "metadata": metadata,
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        index_file = self._get_index_file(config_hash)
        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=2)

    def _load_index(self, config_hash: str) -> Optional[Dict[str, Any]]:
        index_file = self._get_index_file(config_hash)
        if not index_file.exists():
            return None

        with open(index_file, "r") as f:
            index_data = json.load(f)

        # Force reindex on version mismatch.
        stored_version = index_data.get("version", 1)
        if stored_version != METADATA_VERSION:
            logger.info(
                f"Index version mismatch (stored: v{stored_version}, "
                f"current: v{METADATA_VERSION}). Forcing reindex..."
            )
            return None

        return index_data

    def load_iterable(
        self,
        path: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        data_files: Optional[Union[str, list]] = None,
        revision: Optional[str] = None,
        force_reindex: bool = False,
        num_proc: Optional[int] = None,
        length_estimate: str = "dynamic",
        reset_length_on_iter: bool = False,
        **load_dataset_kwargs,
    ) -> ComposableIterableDataset:
        """
        Load a dataset as a `ComposableIterableDataset` over an
        `ArrowBackend`.

        Parameters
        ----------
        path : str
            HuggingFace Hub identifier or a local saved-dataset path.
        name : str, optional
            Dataset configuration name.
        split : str, optional
            Split, with optional slice notation (e.g. ``"train[10000:]"``).
        data_files, revision, num_proc : optional
            Forwarded to ``datasets.load_dataset`` on the slow path.
        force_reindex : bool, optional
            Rebuild the Arrow file index even when a valid cached index
            already exists.
        length_estimate : {"dynamic", "static", "exact"}, optional
            Length-estimation mode for the wrapper. Default ``"dynamic"``.
        reset_length_on_iter : bool, optional
            Reset wrapper length-estimation counters at the start of each
            new iteration. Default ``False``.
        **load_dataset_kwargs
            Forwarded to ``datasets.load_dataset`` on the slow path.

        Returns
        -------
        ComposableIterableDataset
            Wrapper around an `ArrowBackend` ready for shuffling,
            sharding, mapping, and checkpointing.
        """
        # Saved-dataset path?
        if self._is_saved_dataset_path(path):
            logger.info(f"Detected saved dataset at: {path}")
            base_split, slice_start, slice_end = (
                _parse_split_notation(split) if split else (split, None, None)
            )
            effective_split = base_split or "train"

            result = self._load_saved_dataset(
                path=path,
                split=effective_split,
                force_reindex=force_reindex,
                length_estimate=length_estimate,
                reset_length_on_iter=reset_length_on_iter,
            )
            if result is not None:
                if slice_start is not None or slice_end is not None:
                    result = result.slice(slice_start, slice_end)
                return result
            else:
                logger.warning(
                    "Failed to load saved dataset, falling back to load_from_disk"
                )

        # Hub-style path with optional slice notation.
        base_split, slice_start, slice_end = (
            _parse_split_notation(split) if split else (split, None, None)
        )

        config_hash = self._get_config_hash(
            path, name, base_split, data_files, revision
        )
        index_data = self._load_index(config_hash) if not force_reindex else None

        if index_data is not None:
            arrow_files = index_data["arrow_files"]
            file_lengths = index_data.get("file_lengths")

            if all(Path(f).exists() for f in arrow_files):
                start_time = time.time()
                logger.debug(f"Dataset: {path}" + (f"/{name}" if name else ""))
                if split:
                    logger.debug(f"Split: {split}")

                ds = _make_dataset(
                    arrow_files,
                    file_lengths,
                    length_estimate=length_estimate,
                    reset_length_on_iter=reset_length_on_iter,
                    slice_start=slice_start,
                    slice_end=slice_end,
                )

                elapsed = time.time() - start_time
                logger.debug(
                    f"Loaded as IterableDataset in {elapsed:.3f}s "
                    f"Arrow files: {len(arrow_files)} (natural shards)"
                )
                return ds

            else:
                logger.warning("Arrow files missing. Re-indexing...")

        # Slow path: initial load.
        logger.info(
            f"{'Re-indexing' if index_data else 'First-time indexing'} dataset..."
        )
        logger.info(f"Dataset: {path}" + (f"/{name}" if name else ""))
        logger.info("This will be slow, but only happens once...")

        start_time = time.time()
        ds = load_dataset(
            path,
            name=name,
            split=base_split,
            data_files=data_files,
            revision=revision,
            num_proc=num_proc,
            **load_dataset_kwargs,
        )
        load_time = time.time() - start_time
        logger.info(f"Dataset loaded in {load_time:.1f}s")

        arrow_files = self._get_arrow_files(ds)

        if arrow_files:
            num_files = len(arrow_files)
            logger.info(f"Found {num_files} Arrow file(s) in HF cache")

            file_lengths = self._get_file_lengths_from_metadata(arrow_files, base_split)

            if file_lengths is None:
                logger.info("Computing per-file example counts...")
                file_lengths = []

                use_progress = HAS_TQDM and sys.stderr.isatty()
                iterator = (
                    tqdm(arrow_files, desc="Indexing files", unit="file")
                    if use_progress
                    else arrow_files
                )

                for arrow_file in iterator:
                    ds_file = Dataset.from_file(arrow_file)
                    file_lengths.append(len(ds_file))

            total_examples = sum(file_lengths)
            logger.info(f"Total examples: {total_examples:,}")

            metadata = {
                "dataset_path": path,
                "dataset_name": name,
                "split": base_split,
                "load_time": load_time,
                "num_arrow_files": num_files,
                "total_examples": total_examples,
            }
            self._save_index(config_hash, arrow_files, file_lengths, metadata)

            total_size = sum(Path(f).stat().st_size for f in arrow_files)
            size_gb = total_size / (1024**3)
            logger.info(
                f"Index saved: {num_files} Arrow files = {num_files} "
                f"natural shards, Data size: {size_gb:.2f} GB"
            )

            return _make_dataset(
                arrow_files,
                file_lengths,
                length_estimate=length_estimate,
                reset_length_on_iter=reset_length_on_iter,
                slice_start=slice_start,
                slice_end=slice_end,
            )

        else:
            logger.warning("Could not find Arrow files")
            # Fallback: use regular to_iterable_dataset.
            result_ds = ds.to_iterable_dataset(num_shards=1)
            # Note: split-notation slice is not applied to this fallback.
            return result_ds


# Global default instance.
_default_loader = None


def get_default_loader() -> FastDatasetLoaderSimple:
    global _default_loader
    if _default_loader is None:
        _default_loader = FastDatasetLoaderSimple()
    return _default_loader


#: Environment variable name. When set to a server URL
#: (e.g. ``http://host:8765``) `fast_load_iterable_dataset` routes
#: through that server instead of loading data locally. The server
#: must be running with ``allow_load=True`` (CLI: ``--allow-load``).
DATASET_SERVER_ENV_VAR = "FORGATHER_DATASET_SERVER"


def _local_load_iterable_dataset(
    path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    data_files: Optional[Union[str, list]] = None,
    revision: Optional[str] = None,
    force_reindex: bool = False,
    num_proc: Optional[int] = None,
    index_dir: Optional[str] = None,
    length_estimate: str = "dynamic",
    reset_length_on_iter: bool = False,
    **load_dataset_kwargs,
) -> ComposableIterableDataset:
    """
    Always load locally — bypasses the FORGATHER_DATASET_SERVER
    environment variable check. Used by the dataset server itself
    (`tools.dataset_server.DatasetServer.load_on_demand`) so the
    server process can have the variable set in its own environment
    without recursively dialing back into itself.

    Public callers should use `fast_load_iterable_dataset` instead.
    """
    if index_dir is not None:
        loader = FastDatasetLoaderSimple(index_dir=index_dir)
    else:
        loader = get_default_loader()
    return loader.load_iterable(
        path=path,
        name=name,
        split=split,
        data_files=data_files,
        revision=revision,
        force_reindex=force_reindex,
        num_proc=num_proc,
        length_estimate=length_estimate,
        reset_length_on_iter=reset_length_on_iter,
        **load_dataset_kwargs,
    )


def _remote_load_iterable_dataset(
    server_url: str,
    *,
    path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    data_files: Optional[Union[str, list]] = None,
    revision: Optional[str] = None,
    length_estimate: str = "dynamic",
    reset_length_on_iter: bool = False,
) -> ComposableIterableDataset:
    """
    Load via a remote `tools.dataset_server`. Sends a POST /v1/load
    request with the load args, receives a handle, then constructs a
    `RemoteBackend(handle)` wrapped in `ComposableIterableDataset`.

    The split-notation slice (e.g. ``"train[10000:]"``) is parsed
    client-side and applied as a wrapper-level `slice` after
    construction — the server only loads the base split, exactly
    matching the local-loader behavior.
    """
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    from .remote_backend import RemoteBackend, _make_ssl_context, resolve_auth_token

    base_split, slice_start, slice_end = (
        _parse_split_notation(split) if split else (split, None, None)
    )

    load_args = {
        "path": path,
        "name": name,
        "split": base_split,
        "data_files": data_files,
        "revision": revision,
    }
    payload = json.dumps({k: v for k, v in load_args.items() if v is not None}).encode(
        "utf-8"
    )
    url = server_url.rstrip("/") + "/v1/load"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = resolve_auth_token(server_url, explicit=None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, data=payload, method="POST", headers=headers)
    # SSLContext is built from the shared CA bundle for https:// URLs;
    # None for http://. Without this, urlopen falls back to the system
    # trust store and rejects our self-signed certs.
    ssl_context = _make_ssl_context(server_url)
    try:
        with urlopen(req, timeout=300.0, context=ssl_context) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        # Surface the server's error message for debuggability.
        try:
            err_body = exc.read().decode("utf-8")
        except Exception:
            err_body = ""
        raise RuntimeError(
            f"Dataset server load failed ({exc.code}): {err_body or exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach dataset server at {server_url}: {exc.reason}"
        ) from exc

    handle = body["handle"]
    backend = RemoteBackend(
        server_url,
        handle,
        token=token,
        column_names=body.get("column_names"),
    )
    ds = ComposableIterableDataset(
        backend,
        length_estimate=length_estimate,
        reset_length_on_iter=reset_length_on_iter,
    )
    if slice_start is not None or slice_end is not None:
        ds = ds.slice(slice_start, slice_end)
    return ds


def fast_load_iterable_dataset(
    path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    data_files: Optional[Union[str, list]] = None,
    revision: Optional[str] = None,
    force_reindex: bool = False,
    num_proc: Optional[int] = None,
    index_dir: Optional[str] = None,
    length_estimate: str = "dynamic",
    reset_length_on_iter: bool = False,
    **load_dataset_kwargs,
) -> ComposableIterableDataset:
    """
    Load a HuggingFace dataset as a fast iterable with sharding and
    checkpoint support.

    Routing
    -------
    - If the ``FORGATHER_DATASET_SERVER`` environment variable is set
      to a URL (e.g. ``http://host:8765``), the load is routed
      transparently through the dataset server and a
      `RemoteBackend`-wrapped dataset is returned. The server must
      have been started with ``--allow-load``. Server-only options
      (``force_reindex``, ``num_proc``, ``index_dir``,
      ``**load_dataset_kwargs``) are not forwarded over the wire and
      take effect only on the local path.
    - Otherwise, loads locally via `FastDatasetLoaderSimple`. The
      first call for a given dataset is slow (it builds an Arrow
      file index); all subsequent calls are instant.

    Parameters
    ----------
    path : str
        HuggingFace Hub identifier (e.g. ``"allenai/c4"``) **or** a local
        path to a dataset saved with ``Dataset.save_to_disk()``.
    name : str, optional
        Dataset configuration name (e.g. ``"en"`` for C4 English).
    split : str, optional
        Split to load. Supports HuggingFace slice notation such as
        ``"train[10000:]"`` or ``"validation[:500]"``.
    data_files : str or list of str, optional
        Specific data files to load (forwarded to ``load_dataset``).
    revision : str, optional
        Dataset revision or commit hash (forwarded to ``load_dataset``).
    force_reindex : bool, optional
        Rebuild the Arrow file index from scratch (local path only).
    num_proc : int, optional
        Number of processes for the initial dataset download/indexing
        step (local path only).
    index_dir : str, optional
        Directory where JSON index files are stored (local path only).
    length_estimate : {"dynamic", "static", "exact"}, optional
        Length-estimation mode for the wrapper.
    reset_length_on_iter : bool, optional
        Whether to reset length-estimation counters at the start of each
        new iteration pass.
    **load_dataset_kwargs
        Extra keyword arguments forwarded to ``datasets.load_dataset``
        on the initial (slow-path) local load. Not forwarded to the
        remote server.

    Returns
    -------
    ComposableIterableDataset
        Iterable dataset (wrapper over `ArrowBackend` locally or
        `RemoteBackend` when routed through the server) supporting:

        - `.shuffle(seed)` for backend-level + buffer-level shuffling
        - `.shard(num_shards, index)` for DDP data partitioning
        - `.map(fn)` for lazy transformations
        - `.slice()` / `.select()` for virtual splits
        - `state_dict` / `load_state_dict` for stateful checkpointing

    Examples
    --------
    >>> ds = fast_load_iterable_dataset("allenai/c4", name="en", split="train")
    >>> ds = ds.shuffle(seed=42)
    >>> ds = ds.shard(num_shards=world_size, index=rank)
    >>> ds = ds.map(tokenize)
    >>> for example in ds:
    ...     pass
    """
    server_url = os.environ.get(DATASET_SERVER_ENV_VAR)
    if server_url:
        if load_dataset_kwargs:
            logger.warning(
                "Ignoring load_dataset_kwargs %s when routing through "
                "%s — server-only on the local path.",
                list(load_dataset_kwargs.keys()),
                DATASET_SERVER_ENV_VAR,
            )
        return _remote_load_iterable_dataset(
            server_url,
            path=path,
            name=name,
            split=split,
            data_files=data_files,
            revision=revision,
            length_estimate=length_estimate,
            reset_length_on_iter=reset_length_on_iter,
        )
    return _local_load_iterable_dataset(
        path=path,
        name=name,
        split=split,
        data_files=data_files,
        revision=revision,
        force_reindex=force_reindex,
        num_proc=num_proc,
        index_dir=index_dir,
        length_estimate=length_estimate,
        reset_length_on_iter=reset_length_on_iter,
        **load_dataset_kwargs,
    )
