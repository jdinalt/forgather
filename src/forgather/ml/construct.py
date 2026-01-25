import fcntl
import filecmp
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from types import NoneType
from typing import Any, Callable, List, Optional

from forgather.dynamic import walk_package_modules
from forgather.latent import Undefined
from forgather.ml.distributed import (
    get_barrier_fn,
    get_global_process_group,
    get_local_process_group,
    get_rank,
    get_world_size,
)

# Import torch.distributed lazily to avoid issues when torch isn't installed
_dist = None


def _get_dist():
    """Lazy import of torch.distributed."""
    global _dist
    if _dist is None:
        from torch import distributed as _dist
    return _dist


from forgather.project import Project

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


@contextmanager
def file_lock_build(
    target: str | os.PathLike, timeout: float = 300.0, force_lock: bool = False
):
    """
    Context manager for file-based synchronization during object construction.

    Uses file locking to ensure only one process per node constructs the target,
    avoiding torch.distributed initialization requirements and handling non-shared
    filesystems across nodes.

    Args:
        target: Path to the target file/directory being built
        timeout: Maximum time to wait for lock acquisition (seconds)
        force_lock: If True, acquire lock even if target exists (for dependency checking)

    The lock file is created alongside the target with .lock suffix.
    If the target already exists when entering the context and force_lock is False,
    construction is skipped.

    IMPORTANT: Secondary processes must wait for the lock to be released before
    proceeding, even if the target appears to exist. The lock release is the signal
    that the build is complete, not the existence of the target (which may be a
    directory that exists but contains incomplete files).
    """
    target_path = os.path.abspath(target)
    lock_path = f"{target_path}.lock"

    # If target already exists and we're not forcing lock, no need to acquire lock
    if os.path.exists(target_path) and not force_lock:
        yield False  # False indicates no construction needed
        return

    # Ensure lock directory exists
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    # Try to acquire exclusive lock
    lock_fd = None
    start_time = time.time()

    try:
        while time.time() - start_time < timeout:
            try:
                # Open lock file for writing (create if not exists)
                lock_fd = os.open(
                    lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644
                )

                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Write process info to lock file for debugging
                os.write(
                    lock_fd,
                    f"pid:{os.getpid()}\nrank:{os.environ.get('LOCAL_RANK', 'unknown')}\n".encode(),
                )
                os.fsync(lock_fd)

                # Check again if target exists (another process might have created it)
                # But if force_lock is True, we always proceed with construction
                if force_lock or not os.path.exists(target_path):
                    yield True  # Proceed with construction
                else:
                    yield False  # Construction not needed

                break

            except (OSError, IOError) as e:
                # Lock acquisition failed, another process has the lock
                if lock_fd is not None:
                    try:
                        os.close(lock_fd)
                    except:
                        pass
                    lock_fd = None

                # DO NOT exit early just because the target exists!
                # The target might be a directory that exists but contains incomplete files.
                # We must wait for the lock to be released, which signals the build is complete.
                # Only then can we safely check if construction is needed.

                # Wait a bit before retrying
                time.sleep(0.1)
        else:
            # Timeout reached
            raise TimeoutError(
                f"Failed to acquire lock for {target_path} within {timeout} seconds"
            )

    finally:
        # Release lock and cleanup
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except:
                pass

        # Clean up lock file (best effort)
        try:
            if os.path.exists(lock_path):
                os.unlink(lock_path)
        except:
            pass


@contextmanager
def build_sync(target: str | os.PathLike, local: bool = False, timeout: float = 300.0):
    """
    Unified synchronization context manager for build operations.

    This context manager ensures only one process performs a build operation,
    using the most appropriate synchronization mechanism available:

    1. When torch.distributed is initialized: Uses barrier-based synchronization.
       Rank 0 of the process group builds while others wait at a barrier.
       After rank 0 completes, all processes proceed together.

    2. When torch.distributed is not initialized: Falls back to file-based locking.
       The first process to acquire the lock builds, others wait for the lock
       to be released before proceeding.

    Args:
        target: Path to the target being built (used for file locking fallback)
        local: If False (default), synchronize globally across all ranks.
               If True, synchronize only within the local node.
        timeout: Maximum time to wait for synchronization (seconds), used for
                 file locking fallback.

    Yields:
        True if this process should perform the build, False otherwise.

    Example:
        with build_sync("output_models/my_model") as should_build:
            if should_build:
                # Only one process executes this
                generate_model_code()
        # All processes continue here after build is complete
        model = load_model()
    """
    dist = _get_dist()

    # Use distributed barriers when available
    if dist.is_available() and dist.is_initialized() and get_world_size() > 1:
        logger.debug(
            f"[Rank {get_rank()}] build_sync: distributed available and initialized, "
            f"world_size={get_world_size()}, local={local}"
        )
        group = get_local_process_group() if local else get_global_process_group()

        if group is not None:
            logger.debug(
                f"[Rank {get_rank()}] build_sync: using distributed barriers "
                f"(group={'local' if local else 'global'})"
            )
            group_rank = dist.get_group_rank(group, get_rank())
            barrier = get_barrier_fn(group)

            if group_rank == 0:
                # Main process: build first, then signal completion
                logger.debug(
                    f"[Rank {get_rank()}] build_sync: builder (rank 0 in group)"
                )
                yield True
                barrier()
            else:
                # Other processes: wait for main process to complete
                logger.debug(
                    f"[Rank {get_rank()}] build_sync: waiter (rank {group_rank} in group)"
                )
                barrier()
                yield False
            # Final barrier ensures all processes are synchronized before continuing
            barrier()
            logger.debug(f"[Rank {get_rank()}] build_sync: exiting after barriers")
            return
        else:
            logger.warning(
                f"[Rank {get_rank()}] build_sync: distributed initialized but "
                f"{'local' if local else 'global'} process group is None, "
                f"falling back to file locking"
            )

    # Fallback: file locking (distributed not available, not initialized, or single process)
    #
    # Strategy: The first process to acquire the lock becomes the builder (yields True).
    # Other processes wait for the lock. When they acquire it, they check if the target
    # was modified while they were waiting. If so, someone else built it (yield False).
    # If not, they become the builder (yield True).
    #
    # This works for both new builds (target doesn't exist) and rebuilds (target exists
    # but needs updating). The target's mtime after building will be newer than when
    # waiting processes started.
    logger.debug(
        f"[Rank {get_rank()}] build_sync: falling back to file locking "
        f"(dist.is_available={dist.is_available()}, "
        f"dist.is_initialized={dist.is_initialized()}, "
        f"world_size={get_world_size()})"
    )
    target_path = os.path.abspath(target)
    lock_path = f"{target_path}.build_lock"

    # Ensure directory exists
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    # Record target state before trying to acquire lock
    pre_lock_exists = os.path.exists(target_path)
    pre_lock_mtime = os.path.getmtime(target_path) if pre_lock_exists else None

    lock_fd = None
    start_time = time.time()

    try:
        # Acquire lock (blocking with timeout)
        while time.time() - start_time < timeout:
            try:
                lock_fd = os.open(
                    lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644
                )
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, IOError):
                if lock_fd is not None:
                    try:
                        os.close(lock_fd)
                    except OSError:
                        pass
                    lock_fd = None
                time.sleep(0.05)
        else:
            raise TimeoutError(f"Failed to acquire build lock for {target_path}")

        # Lock acquired. Check if target was modified while we were waiting.
        post_lock_exists = os.path.exists(target_path)
        post_lock_mtime = os.path.getmtime(target_path) if post_lock_exists else None

        logger.debug(
            f"[Rank {get_rank()}] build_sync: acquired lock, "
            f"pre_exists={pre_lock_exists}, post_exists={post_lock_exists}, "
            f"pre_mtime={pre_lock_mtime}, post_mtime={post_lock_mtime}"
        )

        # Determine if another process built the target while we waited:
        # - Target was created (didn't exist before, exists now)
        # - Target was modified (mtime changed)
        target_was_built = False
        if not pre_lock_exists and post_lock_exists:
            target_was_built = True
        elif pre_lock_exists and post_lock_exists and post_lock_mtime != pre_lock_mtime:
            target_was_built = True

        if target_was_built:
            # Another process built the target while we waited
            logger.debug(
                f"[Rank {get_rank()}] build_sync: target was built by another process"
            )
            yield False
        else:
            # We're the builder
            logger.debug(f"[Rank {get_rank()}] build_sync: we are the builder")
            yield True

    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except OSError:
                pass

            # Clean up lock file (best effort)
            try:
                os.unlink(lock_path)
            except OSError:
                pass


def register_for_auto_class(object, /, *args, **kwargs):
    """
    Register an object as a HF AutoClass

    PretrainedModel and PretrainedConfig both support this method. When applied,
    the source code for the respective objects will be automatically saved
    with the model weights.

    This is very useful for custom models, as it simplifies things if the code is
    stored with the model.

    The following macro demonstrates how it can be used in a configuration script.
    ```
    ## Custom model constructor
    ## Defines a constructor for a custom model and registers both the
    ## configuration and model class for AutoConfig/AutoModel construction.
    -- macro custom_model(model_path, model_cls, config_cls, model_config)
    !object:forgather.ml.construct:register_for_auto_class
        - !object:{{model_path}}:{{model_cls}}
            - !object:forgather.ml.construct:register_for_auto_class
                - !object:{{model_path}}:{{config_cls}}
                    kwargs: *{{model_config}}
    -- endmacro
    ```
    """
    object.register_for_auto_class(*args, **kwargs)
    return object


def add_special_tokens(tokenizer, token_map):
    """
    Add additional special tokens to a tokenizer

    Useful when a predefined tokenizer is missing a required token.
    """
    tokenizer.add_special_tokens(token_map)
    return tokenizer


def _check_needs_build(
    target: str | os.PathLike, prerequisites: List[str | os.PathLike]
) -> bool:
    """Check if target needs to be built based on existence and prerequisite modification times."""
    if not os.path.exists(target):
        return True
    target_mtime = os.path.getmtime(target)
    for dependency in prerequisites:
        if os.path.exists(dependency):
            if target_mtime < os.path.getmtime(dependency):
                return True
    return False


def build_rule(
    target: str | os.PathLike,
    recipe: List[Callable] | Callable,
    loader: Callable,
    prerequisites: List[str | os.PathLike] = [],
) -> Any:
    """
    Build a target with automatic multiprocess synchronization.

    This function acts like a lightweight makefile rule: it checks if the target
    needs to be built (based on existence and prerequisite modification times),
    and if so, ensures only one process performs the build while others wait.

    Synchronization is handled by `build_sync`, which uses torch.distributed
    barriers when available, or falls back to file locking when distributed
    isn't initialized (e.g., with third-party training frameworks like Torch Titan).

    Args:
        target: Path to the target file/directory to build
        recipe: A callable or list of callables to execute to construct the target
        loader: Callable that loads and returns the constructed target
        prerequisites: List of dependency files; if any are newer than target,
                       the target will be rebuilt

    Returns:
        The result of calling loader()

    Example:
        model = build_rule(
            target="output_models/my_model",
            recipe=lambda: generate_model_code(),
            loader=lambda: load_model("output_models/my_model"),
            prerequisites=["templates/model.yaml"],
        )
    """
    assert isinstance(recipe, Callable | list)
    assert isinstance(loader, Callable)

    # Quick check: if target is up-to-date, skip synchronization entirely.
    # IMPORTANT: Only safe when distributed is NOT initialized, because with
    # distributed barriers, ALL ranks must enter build_sync to participate in
    # the barriers. Otherwise, we get a TOCTOU race where rank 0 creates the
    # target, then other ranks see it exists and skip build_sync, causing rank 0
    # to hang at the barrier waiting for them.
    dist = _get_dist()
    distributed_active = (
        dist.is_available() and dist.is_initialized() and get_world_size() > 1
    )

    if not distributed_active and not _check_needs_build(target, prerequisites):
        logger.debug(f"[Rank {get_rank()}] build_rule: target is up to date: {target}")
        return loader()

    logger.debug(f"[Rank {get_rank()}] build_rule: target needs building: {target}")
    # Target may need building; use build_sync to coordinate across processes
    with build_sync(target) as should_build:
        if should_build:
            # Double-check after acquiring synchronization (another process or
            # concurrent run may have built the target while we were waiting)
            if _check_needs_build(target, prerequisites):
                logger.debug(f"Building target: {target}")
                if isinstance(recipe, Callable):
                    recipe()
                else:
                    for fn in recipe:
                        assert isinstance(
                            fn, Callable
                        ), f"Item in recipe list is not callable {type(fn)=}"
                        fn()
            else:
                logger.debug(f"Target became up-to-date while waiting: {target}")
        else:
            logger.debug(f"Target was built by another process: {target}")

    return loader()


torch_dtype_map = None


def torch_dtype(type: str):
    global torch_dtype_map
    if torch_dtype_map is None:
        import torch

        torch_dtype_map = {
            "float32": torch.float32,
            "float": torch.float,
            "float64": torch.float64,
            "double": torch.double,
            "float16": torch.float16,
            "half": torch.half,
            "bfloat16": torch.bfloat16,
            "complex32": torch.complex32,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
            "uint8": torch.uint8,
            "uint16": torch.uint16,
            "uint32": torch.uint32,
            "uint64": torch.uint64,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
            "quint8": torch.quint8,
            "qint8": torch.qint8,
            "qint32": torch.qint32,
            "quint4x2": torch.quint4x2,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
        }
    return torch_dtype_map[type]


def module_to_dtype(module_ctor, dtype: str, **kwargs):
    logger.info(f"Constructing module and converting to dytpe={dtype}")
    m = module_ctor(**kwargs)
    return m.to(dtype=torch_dtype(dtype))


# Depricated: use forgather.from_project()
def load_from_config(
    project_dir: str,
    config_template: str | NoneType = None,
    targets: str | List[str] = "",
    **config_kwargs,
):
    """
    Construct an object from a project configuration

    project_directory: Path to project.
    config_template: Config template name; if None, use default config
    """

    proj = Project(config_template, project_dir)
    return proj(targets, **config_kwargs)


def _should_write_file(file_path: str, exists: str) -> bool:
    """
    Process file overwriting policy

    exists: One of ['ok', 'warn', 'skip', 'raise']
        ok: Quietly overwrite file
        warn: Warn if files are not the same, but allow overwrite.
        skip: Warn if files are not the same and skip overwrite.
        raise: Raise exception if files are not the same.

    """
    if os.path.isfile(file_path):
        match exists:
            case "warn":
                logger.warning(
                    f"The source data for '{file_path}' has changed; the file will be overwritten."
                )
                return True
            case "skip":
                logger.warning(
                    f"The source data for '{file_path}' has changed; the file will NOT be overwritten. "
                    "This could lead to unexpected results!"
                )
                return False
            case "raise":
                raise RuntimeError(
                    f"The source data for '{file_path}' has changed; overwrite is prohibited. "
                    "Delete the destination file or change the overwrite policy."
                )
            case "ok":
                return True
            case _:
                raise ValueError(
                    f"File overwrite policy must be one of: ['ok', 'warn', 'skip', 'raise']; found {exists}"
                )
    else:
        return True


def copy_package_files(
    dest_dir: str | os.PathLike, obj: Any, exists: Optional[str] = "raise"
) -> Any:
    """
    Given an object, copy the source files for those objects,
        and all referenced source files within the same package, to the
        desitnation directory.

    returns the input object, unaltered

    ```
    # Copy the source code for a custom model instance to the model output
    # directory.

    custom_model = copy_package_files('output_models/my_model', custom_model)
    ```

    The underlying implementation only copies imported files from the same
    module as the object (recursively). Duplicates are eliminated before
    the copy.

    While not perfect, it's less broken than the attempt at something similar
    within the Transformers library. Included modules can be in sub-directories,
    which makes it easy to symlink a 'model-bits' directory and have this only
    copy the referenced bits.
    """

    # Use file locking to ensure only one process per node copies files
    dest_dir = os.path.abspath(dest_dir)
    lock_marker = os.path.join(dest_dir, ".package_files_copied")

    with file_lock_build(lock_marker) as should_build:
        if should_build:
            # Get module for object
            pkg = sys.modules[obj.__module__]
            for level, value in walk_package_modules(pkg):
                # Ignore namespaces
                if value.__spec__.origin is None:
                    continue
                origin = value.__spec__.origin
                package_name = value.__package__

                file_name = os.path.basename(origin)
                module_prefix = package_name.split(".")[1:]
                module_dir = os.path.join(dest_dir, *module_prefix)
                dest_path = os.path.join(module_dir, file_name)

                if os.path.exists(dest_path) and filecmp.cmp(origin, dest_path):
                    continue

                if _should_write_file(dest_path, exists):
                    os.makedirs(module_dir, exist_ok=True)
                    shutil.copy2(origin, module_dir, follow_symlinks=True)

            # Create marker file to indicate completion
            os.makedirs(os.path.dirname(lock_marker), exist_ok=True)
            with open(lock_marker, "w") as f:
                f.write(f"Package files copied at {time.time()}\n")

    return obj


def dependency_list(*args):
    """
    A passthrough-node, which resolves "phantom" dependencies.

    This returns the first element in the list, untouched.

    The primary use-case is chaining additional dependencies, like file generation, which
    don't pass directly through the graph.

    ```yaml
    !singleton:forgather.ml.construct.dependency_list
        - *pass_through_node
        - copy_package_files
            ...
    ```
    """
    return args[0]


def _compare_file_to_str(file_path: str, string: str):
    """
    Compare the contents of a file and a string for equality
    """
    if not os.path.isfile(file_path):
        return False
    with open(file_path, "r") as f:
        return f.read() == string


def write_file(
    data,
    output_file: Optional[str | os.PathLike] = None,
    return_value: Optional[Any] = Undefined,
    exists: Optional[str] = "raise",
):
    """
    Write unicode data to a file using file-locking synchronization.

    Uses file locking to ensure only one process per node writes the file,
    supporting multi-node setups with non-shared filesystems while avoiding
    torch.distributed initialization requirements.

    data: The data to write
    output_file: If specified, write the generated code to the specified file path.
        Missing directories will automatically be created.
        If running in a multiprocess environment, only one process per node will write the file,
        while the other processes will wait for the file to be written.
    exists: one of [ "ok", "warn", "skip", "raise" ]; see _should_write_file()
    return_value: Override passthrough of the data by returning this value instead.
    """
    if isinstance(data, Callable):
        data = data()

    if output_file is not None:
        with file_lock_build(output_file) as should_build:
            if should_build:
                if not _compare_file_to_str(output_file, data) and _should_write_file(
                    output_file, exists
                ):
                    module_dir = os.path.dirname(output_file)
                    if len(module_dir):
                        os.makedirs(module_dir, exist_ok=True)
                    with open(output_file, "w") as f:
                        f.write(data)

    if return_value is not Undefined:
        return return_value
    else:
        return data
