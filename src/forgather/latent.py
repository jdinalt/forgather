from typing import Any, Callable, Union, Container, Tuple
from collections.abc import MutableSequence, MutableMapping
from types import NoneType
from itertools import chain

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

    __slots__ = (
        "constructor",
        "args",
        "kwargs",
        "as_callable",
        "is_singleton",
        "singleton",
    )

    def __init__(
        self,
        constructor: Callable | str,
        /,
        *args,
        as_callable=False,
        is_singleton=False,
        **kwargs,
    ):
        assert isinstance(constructor, str) or isinstance(constructor, Callable)
        assert isinstance(as_callable, bool)
        assert isinstance(is_singleton, bool)
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs

        # When this attribute is True, the object will not be materialized, unless it is the
        # root of the graph. This can be useful for passing a Latent object to another
        # Latent object, as-is -- for example, as a factory object.
        self.as_callable = as_callable

        # A 'singleton' latent object always returns the same instance of the object,
        # otherwise, a new instance is created on each materialization.
        self.is_singleton = is_singleton

        # The object's singleton instance of the materialized object.
        self.singleton = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({repr(self.constructor)}, *{repr(self.args)}, "
            f"**{repr(self.kwargs)}, as_callable={self.as_callable}, is_singleton={self.is_singleton})"
        )

    def __call__(self, **mapping):
        """
        Alias for calling materialize() on self
        """
        return Latent.materialize(self, **mapping)

    """
    Traverse the graph of objects, replacing all Latent objects with concrete instances

    whitelist: A Container object which is used to verify if the constructor is allowed.
    **mapping: A dict[str, Any], which will substitue any stand-in constructors with
        the corresponding objects from the map.
    """

    @staticmethod
    def materialize(obj: Any, **mapping):
        Latent._resolve_standins(obj, **mapping)
        Latent._resolve_dynamic_imports(obj)
        return Latent._materialize(obj, dict())

    @staticmethod
    def _materialize(obj: Any, idmap: dict[int, Any], level: int = 0):
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
            # If flagged 'as_callable,' and not the root node, skip materialization.
            if obj.as_callable and level > 0:
                return obj
            # Have we already constructed this object?
            value = idmap.get(id(obj), None)
            if value is not None:
                return value

            # If the object has already been constructed
            if obj.singleton is not None:
                idmap[id(obj)] = obj.singleton
                return obj.singleton

            # The object has not yet been constructed. Verify that the
            # constructor has been resolved.
            if not isinstance(obj.constructor, Callable):
                if isinstance(obj.constructor, str) and ":" not in obj.constructor:
                    raise LatentException(
                        f"Found unresolved symbol '{obj.constructor}' in Latent: {obj}"
                        + " This is likely either a missing stand-in or a syntax error."
                    )
                else:
                    raise LatentException(
                        f"Constructor must be Callable, but found [{type(obj.constructor)}] {obj.constructor}"
                        + "; see _resolve_dynamic_imports() and _resolve_standins()"
                    )

            # Materialize the arguments
            args = Latent._materialize(obj.args, idmap, level + 1)
            kwargs = Latent._materialize(obj.kwargs, idmap, level + 1)

            # Materialize the object
            value = obj.constructor(*args, **kwargs)

            # Cache materialized object
            idmap[id(obj)] = value

            # If the object is a singleton, add the instance to the node.
            if obj.is_singleton:
                obj.singleton = value
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
    def _resolve_dynamic_imports(obj: Any):
        """
        Try to resolve all dynamic imports, replacing the constructor string with the
        corresponding Callable.
        """
        for latent in Latent.all_latents(obj):
            # If not plausibly an import spec, skip it
            if not isinstance(latent.constructor, str) or not ":" in latent.constructor:
                continue
            constructor = dynamic_import(latent.constructor)
            if not isinstance(constructor, Callable):
                raise LatentException(
                    f"Imported constructor is not Callable: [{type(self.constructor)}] {self.constructor}"
                )
            latent.constructor = constructor

    @staticmethod
    def _resolve_standins(obj: Any, **mapping):
        """
        Replace all stand-ins with the corresonding Callables from the mapping.
        """
        for latent in Latent.all_latents(obj):
            if not isinstance(latent.constructor, str) or ":" in latent.constructor:
                continue
            value = mapping.get(latent.constructor, None)
            # If we found the key in the map...
            # and either the value has not been set OR the node is not a singleton, then set the value
            if value is not None and (
                latent.singleton is None or not latent.is_singleton
            ):
                latent.singleton = value
