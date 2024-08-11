from typing import Any, Callable, Iterable, Optional, Union, List, Type, Tuple
from collections.abc import Iterable
from types import ModuleType
import importlib
import sys
import os


def walk_package_modules(mod: ModuleType, level=0) -> Iterable[Tuple[int, ModuleType]]:
    """
    Given a package, walk all referenced sub-modules within the same package

    yields Tuple[recursion_depth: int, module: ModuleType]
    """
    yield level, mod
    # Is it a package; only packages have a '__path__' attr
    if not hasattr(mod, "__path__"):
        return
    for value in mod.__dict__.values():
        # Ignore items which are not modules or which don't have a __package__ attr
        # Ignore things which don't start with the package prefix
        if (
            type(value) != ModuleType
            or not hasattr(value, "__package__")
            or not value.__package__.startswith(mod.__package__)
        ):
            continue
        yield from walk_package_modules(value, level + 1)


def parse_module_name_or_path(module_name_or_path: Union[os.PathLike, str]):
    """
    module_name_or_path -> tuple(module_name, module_path)

    module-name-or-path ::= <module-path> | <module-name>
    module-path ::= [ <os-path> ] <module-name> '.py'
    """
    if module_name_or_path.endswith(".py"):
        module_path = module_name_or_path
        module_name = os.path.basename(module_name_or_path).split(".")[0]
    else:
        module_path = None
        module_name = module_name_or_path
    return module_name, module_path


def parse_dynamic_import_spec(import_spec: str):
    """
    Parse import-spec and return tuple(module_path, module_name, symbol_name)

    import-spec ::=  <module-name-or-path> : <symbol-name>
    symbol-name ::= <symbol> [ '.' <symbol-name> ]
    """
    if not isinstance(import_spec, str):
        raise TypeError(
            f"import_spec must be of type 'str'; found type {type(import_spec)}"
        )

    split_spec = import_spec.split(":")
    if len(split_spec) != 2:
        raise DynamicImportParseError(
            f"Expected import-spec ::=  <module-name-or-path> : <symbol-name>; found {import_spec}"
        )
    module_name_or_path, symbol_name = split_spec
    return module_name_or_path, symbol_name


def encode_import_spec(module_name_or_path: str, symbol_name: str):
    """
    Given a name-or-path and a symbol-name, return the equivalent import spec.
    """
    return module_name_or_path + ":" + symbol_name


def normalize_import_spec(import_spec: str):
    """
    If import spec is a path, normalize the path

    This ensures that equivalent paths are resolved to the same name.

    ```
    assert(
        normalize_import_spec("foobar/../foobar/file.py:Foo") ==
        normalize_import_spec("foobar/file.py:Foo")
    )
    ```
    """
    module_name_or_path, symbol_name = parse_dynamic_import_spec(import_spec)
    module_name, module_path = parse_module_name_or_path(module_name_or_path)
    if module_path is not None:
        return encode_import_spec(os.path.abspath(module_path), symbol_name)
    else:
        return import_spec


class DynamicImportParseError(Exception):
    pass


def import_dynamic_module(
    module_name_or_path: os.PathLike | str,
    *,
    searchpath: List[os.PathLike | str] = [],
) -> ModuleType:
    """
    Given a module-name-or-path return the associated module

    ```
    nn = import_dynamic_module("torch.nn")
    ```
    """
    module_name, module_path = parse_module_name_or_path(module_name_or_path)

    importlib.invalidate_caches()
    mod = sys.modules.get(module_name, None)
    if mod is None:
        if module_path is None:
            mod = importlib.import_module(module_name)
        else:
            module_spec = importlib.util.spec_from_file_location(
                module_name,
                module_path,
                submodule_search_locations=searchpath,
            )
            mod = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = mod
            module_spec.loader.exec_module(mod)
    return mod


def from_dynamic_module_import(
    module_name_or_path: Union[os.PathLike, str],
    symbol_name: str,
    *,
    searchpath: List[os.PathLike | str] = [],
) -> Any:
    """
    Given a module-name-or-path and a symbol name in that module,
        return the associated object.

    ```
    tensor_cls = from_dynamic_module_import("torch", "tensor")
    tensor = tensor_cls([1, 2, 3])
    ```
    """
    mod = import_dynamic_module(module_name_or_path, searchpath=searchpath)
    for symbol in symbol_name.split("."):
        mod = getattr(mod, symbol)
    return mod


def get_builtin(name):
    mod = sys.modules["builtins"]
    for s in name.split("."):
        mod = getattr(mod, s, None)
        if mod is None:
            return None
    return mod


def dynamic_import(
    import_spec: str,
    *,
    searchpath: List[os.PathLike | str] = [],
):
    """
    Given an import spec, return the associated object

    ```
    torch_module_cls = dynamic_import("torch.nn:Module")
    ```
    """
    return from_dynamic_module_import(
        *parse_dynamic_import_spec(import_spec), searchpath=searchpath
    )


def test_parse():
    """
    A simple uinit test for the parser

    TODO: Move the a real unit-test location.
    """
    test_cases = (
        ("my_module:MyClass", (None, "my_module", "MyClass")),
        ("my_package.my_module:MyClass", (None, "my_package.my_module", "MyClass")),
        (
            "my_package.my_module:MyClass.my_method",
            (None, "my_package.my_module", "MyClass.my_method"),
        ),
        (
            "./stuff/my_module.py:MyClass",
            ("./stuff/my_module.py", "my_module", "MyClass"),
        ),
    )
    for test_case in test_cases:
        import_spec, (expect_module_path, expect_module_name, expect_symbol) = test_case
        print(
            f"{import_spec} -> ({expect_module_path}, {expect_module_name}, {expect_symbol})"
        )

        module_name_or_path, symbol_name = parse_dynamic_import_spec(import_spec)
        module_name, module_path = parse_module_name_or_path(module_name_or_path)

        assert module_path == expect_module_path
        assert module_name == expect_module_name
        assert symbol_name == expect_symbol

    try:
        parse_dynamic_import_spec("The quick brown fox...")
    except DynamicImportParseError as e:
        print(f"Caught expected exception '{e}')")
    else:
        raise Exception("Exception not raised for invalid syntax!")
