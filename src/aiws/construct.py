from typing import Callable, List, Any, Tuple
from types import NoneType, ModuleType
import os
import shutil
import sys

from .distributed import main_process_first

from aiws.config import MetaConfig
from forgather.config import ConfigEnvironment
from forgather.dynamic import walk_package_modules


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
    !object:aiws.construct:register_for_auto_class
        - !object:{{model_path}}:{{model_cls}}
            - !object:aiws.construct:register_for_auto_class
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


def load_from_config(project_dir: str, config_template: str | NoneType = None):
    """
    Construct an object from a project configuration

    project_directory: Path to project.
    config_template: Config template name; if None, use default config.

    TODO: Add ability to pass args to pre-processor and constructor
    """
    meta = MetaConfig(project_dir)
    # Get default
    if config_template is None:
        config_template = meta.default_config()
    environment = ConfigEnvironment(
        searchpath=meta.searchpath,
        globals={"project_dir": project_dir},
    )
    config = environment.load(meta.config_path(config_template)).config
    return config.main()


@main_process_first()
def copy_package_files(dest_dir: str | os.PathLike, obj: Any) -> Any:
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
    module as the object -- recursively. Duplicates are eliminated before
    the copy.

    While not perfect, it's less broken than the attempt at something similar
    within the Transformers library. Includes modules can be in sub-directories,
    which makes it easy to symlink a 'model-bits' directory and have this only
    copy the referenced bits.

    Why do this?

    I have been burned multiple times by using imported models, only
    to 'fix' the library, resulting in incompatible model weights. This being
    followed by the hilarity of trying to piece back together the original
    source code. Fun times!

    This allows one to keep a snap-shot of working code with the model weights.
    Even if you don't save the weights, it makes the experiment reproducible.
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

        # Skip is the source and destination are the same file
        # This can happen when dynamically generating source code.
        if os.path.exists(dest_path) and os.path.samefile(origin, dest_path):
            continue

        os.makedirs(module_dir, exist_ok=True)
        shutil.copy2(origin, module_dir, follow_symlinks=True)
    return obj


@main_process_first()
def generate_factory_file(
    passthrough: Any,
    code: Any,
    factory_name,
    output_file: str | os.PathLike,
) -> Any:
    # Only do this on the main process
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return passthrough

    source_code = format_latent_to_py(code, factory_name)
    module_dir = os.path.dirname(output_file)

    os.makedirs(module_dir, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(source_code)
    return passthrough


DYNAMIC_IMPORT_DEF = """
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys

def dynimport(module, name, searchpath):
    module_path = module
    module_name = os.path.basename(module).split(".")[0]
    module_spec = spec_from_file_location(
        module_name,
        module_path,
        submodule_search_locations=searchpath,
    )
    mod = module_from_spec(module_spec)
    sys.modules[module_name] = mod
    module_spec.loader.exec_module(mod)
    for symbol in name.split("."):
        mod = getattr(mod, symbol)
    return mod

""".strip()


def format_imports(imports):
    s = ""
    dynamic_modules = []

    for args in imports:
        module, name, searchpath = args

        if module.endswith(".py"):
            dynamic_modules.append(args)
        else:
            s += f"from {module} import {name.split('.')[-1]}\n"
    if len(dynamic_modules):
        s += DYNAMIC_IMPORT_DEF + "\n\n"

        for args in dynamic_modules:
            attr_str = args[1].split(".")[-1]
            args_str = ", ".join([repr(x) for x in args])
            s += f"{attr_str} = dynimport({args_str})\n"

    return s


def indent_block(block, level):
    indent = " " * level * 4
    s = "".join(map(lambda s: indent + s + "\n", block.split("\n")))
    return s[:-1]


def format_factory(name, variables, main_body):
    var_block = ""
    for var in variables:
        var_block += var + ",\n"

    # Allow additional args
    var_block += "**kwargs"
    s = f"def {name}(\n"
    s += indent_block(var_block, 1).rstrip() + "\n"
    s += "):\n"
    s += indent_block("return " + main_body, 1)
    return s


def format_latent_to_py(code, factory_name="construct"):
    s = ""
    s += format_imports(code["imports"])
    s += "\n\n"
    s += "# Code automatically generated from config by Forgather\n"
    s += format_factory(factory_name, code["variables"], code["main_body"])
    return s

def dependency(passthrough: Any, /, *args):
    return passthrough