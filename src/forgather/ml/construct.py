from typing import Callable, List, Any, Optional
from types import NoneType
import os
import shutil
import sys
import filecmp
import logging

from .distributed import main_process_first
from forgather.meta_config import MetaConfig
from forgather.project import Project
from forgather.config import ConfigEnvironment
from forgather.dynamic import walk_package_modules
from forgather.latent import Undefined

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


@main_process_first()
def build_rule(
    target: str | os.PathLike,
    recipe: Callable,
    loader: Callable,
    prerequisites: List[str | os.PathLike] = [],
) -> Any:
    assert isinstance(recipe, Callable)
    assert isinstance(loader, Callable)

    if os.path.exists(target):
        build_target = False
        target_mtime = os.path.getmtime(target)
        for dependency in prerequisites:
            dep_mtime = os.path.getmtime(dependency)
            if target_mtime < dep_mtime:
                build_target = True
                break
    else:
        build_target = True

    if build_target:
        output = recipe()
        if output is not None:
            return output
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


def load_from_config(project_dir: str, config_template: str | NoneType = None):
    """
    Construct an object from a project configuration

    project_directory: Path to project.
    config_template: Config template name; if None, use default config
    """

    proj = Project(config_template, project_dir)
    return proj()


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


@main_process_first()
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

    # Only do this on the main process
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return obj

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


@main_process_first()
def write_file(
    data,
    output_file: Optional[str | os.PathLike] = None,
    return_value: Optional[Any] = Undefined,
    exists: Optional[str] = "raise",
):
    """
    Write unicode data to a file, with the main-process first and only with the main process

    data: The data to write
    output_file: If specified, write the generated code to the specified file path.
        Missing directories will automatically be created.
        If running in a multiprocess environment, only the main local process will write the file,
        while the other processes will wait for the file to be written.
    exits: one of [ "ok", "warn", "skip", "raise" ]; see _should_write_file()
    return_value: Override passthrough of the data by returning this value instead.
    """
    if isinstance(data, Callable):
        data = data()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
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
