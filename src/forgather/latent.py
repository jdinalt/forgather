from typing import Any, Callable, Union, Container, Tuple, Optional, Hashable, List
from collections.abc import MutableSequence, MutableMapping
from types import NoneType
from itertools import chain
import sys
import os

from pprint import pformat

from forgather.dynamic import (
    dynamic_import,
    normalize_import_spec,
)


class LatentException(Exception):
    pass


class Latent:
    """
    A Latent [object] abstracts what to create from when to create it
    """

    def __init__(
        self,
        constructor: str,
        /,
        *args,
        as_lambda: Optional[bool] = False,
        identity: Hashable = None,
        submodule_searchpath: Optional[List[str | os.PathLike]] = None,
        **kwargs,
    ):
        assert isinstance(constructor, str)
        assert isinstance(as_lambda, bool)
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs

        if as_lambda:
            self.as_lambda = as_lambda

        if submodule_searchpath is not None:
            self.submodule_searchpath = submodule_searchpath

        if as_lambda or (isinstance(identity, int) and identity == 0):
            self.identity = None
        elif identity is None:
            self.identity = id(self)
        else:
            self.identity = identity

    def is_anonymous(self):
        return self.identity is None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({repr(self.constructor)}, *{repr(self.args)}, "
            f"**{repr(self.kwargs)}, identity={repr(self.identity)}, "
            f"as_lambda={getattr(self, 'as_lambda', False)})"
        )

    def __call__(self, **mapping):
        """
        Alias for calling materialize() on self
        """
        return Latent.materialize(self, **mapping)

    @staticmethod
    def materialize(obj: Any, /, **kwargs):
        """
        Traverse the graph of objects, replacing all Latent objects with concrete instances
        """
        if len(kwargs):
            Latent._resolve_standins(obj, **kwargs)
        return Latent._materialize(obj, dict(), 0)

    @staticmethod
    def to_serailizable(obj: Any, **kwargs):
        """
        Convert Latent graph to serializable format

        ```
        # Convert map of latents to JSON
        serialized_latents = json.dumps(Latent.to_serailizable(map_with_latents))
        ```
        Object with the same identity will be encoded as such, allowing them
        to be reconstructed as intended.

        Note: This only support 'string' callables at present.

        There is a reference 'decoder' for materializing the output format at the bottom of this file.
        """
        if len(kwargs):
            Latent._resolve_standins(obj, **kwargs)
        encoding = Latent._to_serailizable(obj, set(), 0)

        # Add versioning header
        return {"!forgather_version": "1.0", "encoding": encoding}

    @staticmethod
    def contains_only_pods(obj):
        """
        Return True, if contains only Plain Old Datatypes (PODs). else False
        """
        for level, key, value in Latent.all_items(value):
            if (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
            ):
                continue
            return False
        return True

    @staticmethod
    def _to_serailizable(obj, idmap, level):
        if isinstance(obj, list):
            return [Latent._to_serailizable(value, idmap, level + 1) for value in obj]
        elif isinstance(obj, dict):
            return {
                key: Latent._to_serailizable(value, idmap, level + 1)
                for key, value in obj.items()
            }
        elif isinstance(obj, tuple):
            # Convert tuples to list with !tuple tag, as this type may not be natively supported.
            # And... some arguments require this type.
            return {
                "!tuple": list(
                    Latent._to_serailizable(value, idmap, level + 1) for value in obj
                )
            }
        elif isinstance(obj, Latent):
            # If in identity map (we have seen it before) and not anonymous, return id tag.
            if not obj.is_anonymous() and obj.identity in idmap:
                return {"!id": obj.identity}

            # If value has been resolved contains only PODs, return literal value
            if hasattr(obj, "value"):
                assert contains_only_pods(
                    obj.value
                ), "Encounterd resolved symbol which is not a POD {obj}"
                return obj.value

            # If stand-in, encode as key
            if ":" not in obj.constructor:
                return {"!key": obj.constructor}

            # Encode callable and arguments
            encoding = {
                "!callable": obj.constructor,
            }

            # If object is not anonymous, add identity to id-map and add identity
            if not obj.is_anonymous():
                encoding["id"] = obj.identity
                idmap.add(obj.identity)

            if hasattr(obj, "as_lambda"):
                encoding["as_lambda"] = True
            if len(obj.args):
                encoding["args"] = Latent._to_serailizable(obj.args, idmap, level + 1)
            if len(obj.kwargs):
                encoding["kwargs"] = Latent._to_serailizable(
                    obj.kwargs, idmap, level + 1
                )
            return encoding
        else:
            return obj

    @staticmethod
    def _resolve_callable(latent):
        # The object has not yet been constructed. Verify that the
        # constructor has been resolved.
        callable = getattr(latent, "callable", None)
        if callable is not None:
            return callable

        searchpath = getattr(latent, "submodule_searchpath", [])
        callable = dynamic_import(latent.constructor, searchpath=searchpath)
        if not isinstance(callable, Callable):
            raise LatentException(
                f"Imported constructor is not Callable: [{type(callable)}] {latent.constructor}"
            )
        # Cache the resolved symbol
        setattr(latent, "callable", callable)
        return callable

    @staticmethod
    def _materialize(obj, idmap, level):
        if isinstance(obj, list):
            return [Latent._materialize(value, idmap, level + 1) for value in obj]
        elif isinstance(obj, dict):
            return {
                key: Latent._materialize(value, idmap, level + 1)
                for key, value in obj.items()
            }
        elif isinstance(obj, tuple):
            return tuple(Latent._materialize(value, idmap, level + 1) for value in obj)
        elif isinstance(obj, Latent):
            # If flagged 'as_lambda,' and not the root node, skip materialization.
            if hasattr(obj, "as_lambda") and level > 0:
                return obj
            # Have we already constructed this object?
            value = idmap.get(obj.identity, None)
            if value is not None:
                return value

            # If injected 'stand-in' value, return it
            value = getattr(obj, "value", None)
            if value is not None:
                return value

            # Materialize the arguments
            args = Latent._materialize(obj.args, idmap, level + 1)
            kwargs = Latent._materialize(obj.kwargs, idmap, level + 1)

            # Materialize the object
            value = Latent._resolve_callable(obj)(*args, **kwargs)

            # Cache materialized object
            if not obj.is_anonymous():
                idmap[obj.identity] = value

            return value
        # We return the original reference for everything else, which means that
        # there will only be a single instance of all other object types.
        else:
            return obj

    @staticmethod
    def all_items(value, key=None, level=0) -> Tuple[int, int | str | NoneType, Any]:
        """
        Iterate over all objects in graph

        yields tuple(level: int, key: int|str|NoneType, value: Any)
        """
        yield (level, key, value)

        # Supported sequence types
        if isinstance(value, list) or isinstance(value, tuple):
            generator = enumerate(value)
        # Supported mapping types
        elif isinstance(value, dict):
            generator = value.items()
        elif isinstance(value, Latent):
            generator = chain(enumerate(value.args), value.kwargs.items())
        else:
            return
        for k, v in generator:
            yield from Latent.all_items(v, k, level + 1)

    @staticmethod
    def all_latents(value: Any) -> "_Latent":
        for _, _, value in filter(
            lambda x: isinstance(x[2], Latent), Latent.all_items(value)
        ):
            yield value

    @staticmethod
    def _resolve_standins(obj: Any, **mapping):
        """
        Replace all stand-ins with the corresonding Callables from the mapping.
        """
        for latent in Latent.all_latents(obj):
            if ":" in latent.constructor:
                continue
            if latent.constructor in mapping:
                value = mapping[latent.constructor]
                setattr(latent, "value", value)


"""
This section contains a reference implementation for materializing the output
from Latent.to_serializable().

This code is intended to be embedded in other projects, where this library
may not be available.
"""
import importlib
import sys


def dynamic_import_module(name):
    module_name, symbol_name = name.split(":")
    package = sys.modules[__name__].__package__
    mod = importlib.import_module(module_name, package=package)
    for symbol in symbol_name.split("."):
        mod = getattr(mod, symbol)
    return mod


def materialize_config(obj, idmap: dict[int, Any] = None):
    if idmap is None:
        idmap = {}
    if isinstance(obj, list):
        return [materialize_config(value, idmap) for value in obj]
    elif isinstance(obj, tuple):
        return tuple(materialize_config(value, idmap) for value in obj)
    elif isinstance(obj, dict):
        if "!id" in obj:
            return idmap[obj["!id"]]
        elif "!callable" in obj:
            as_lambda = obj.get("as_lambda", False)
            args = obj.get("args", tuple())
            kwargs = obj.get("kwargs", {})
            obj_id = obj.get("id", None)
            if len(args):
                args = tuple(materialize_config(obj["args"], idmap))
            if len(kwargs):
                kwargs = materialize_config(obj["kwargs"], idmap)
            callable = dynamic_import_module(obj["!callable"])
            if as_lambda:
                return lambda: callable(*args, **kwargs)
            else:
                value = callable(*args, **kwargs)
            if obj_id is not None:
                idmap[obj_id] = value
            return value
        else:
            return {key: materialize_config(value, idmap) for key, value in obj.items()}
    else:
        return obj


"""
This section contains a reference implementation for materializing the output
from Latent.to_serializable().

This code is intended to be embedded in other projects, where this library
may not be available.
"""
import importlib
import sys


def dynamic_import_module(name):
    module_name, symbol_name = name.split(":")
    package = sys.modules[__name__].__package__
    mod = importlib.import_module(module_name, package=package)
    for symbol in symbol_name.split("."):
        mod = getattr(mod, symbol)
    return mod


def materialize_config(config: dict, **kwargs):
    MAJOR_VERSION = 1
    assert isinstance(config, dict)
    version_string = config.get("!forgather_version", None)
    if version_string is None:
        raise KeyError("This does not appear to be a Forgather encoded object.")
    major, minor = version_string.split(".")
    if int(major) > MAJOR_VERSION:
        raise RuntimeError(
            "The encoded data was encoded by a newer version"
            f"{int(major)} > {MAJOR_VERSION}"
        )
    return _materialize_config(config["encoding"], kwargs, {}, 0)


def _materialize_config(obj, mapping, idmap, level):
    if isinstance(obj, list):
        return [_materialize_config(value, mapping, idmap, level + 1) for value in obj]
    # A dictionary /may/ contain a type tag. If so, construct the type.
    elif isinstance(obj, dict):
        # Convert tag '!tuple' back to tuple
        if "!tuple" in obj:
            return tuple(
                _materialize_config(obj["!tuple"], mapping, idmap, level=level + 1)
            )
        # This indicates that we /should/ have seen the definition for 'id' already
        # Find the value cached in the idmap and return value
        elif "!id" in obj:
            key = obj["!id"]
            return idmap[key]

        # Is it a kwargs substition
        elif "!key" in obj:
            key = obj["!key"]
            return mapping[key]

        # Is it a callable definition?
        elif "!callable" in obj:
            as_lambda = obj.get("as_lambda", False)

            # If this is not the root and it is a lambda, stop traversal and
            # return a lambda for deferred construction.
            # If level is zero and it's a lambda, then
            # it is being called as a lambda; construct it!
            if as_lambda and level > 0:
                return lambda: _materialize_config(obj, mapping, {}, 0)

            # Get the arguments
            obj_id = obj.get("id", None)
            args = obj.get("args", tuple())
            kwargs = obj.get("kwargs", {})
            if len(args):
                args = tuple(
                    _materialize_config(obj["args"], mapping, idmap, level=level + 1)
                )
            if len(kwargs):
                kwargs = _materialize_config(
                    obj["kwargs"], mapping, idmap, level=level + 1
                )

            # Resolve the callable name
            fn = dynamic_import_module(obj["!callable"])

            # Call it with the args to get the value
            value = fn(*args, **kwargs)

            # If it is not an anonymous object, cache the result
            if obj_id is not None:
                idmap[obj_id] = value

            # Return the constructed object.
            return value
        else:  # It's an ordinary mapping
            return {
                key: _materialize_config(value, mapping, idmap, level=level + 1)
                for key, value in obj.items()
            }
    else:  # It is a basic type (i.e. int, float, etc.)
        return obj
