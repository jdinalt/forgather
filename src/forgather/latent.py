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

    Example:
    ```
    from forgather import Latent

    latent_tensor = Latent("torch:Tensor", [1 ,2, 3])
    print(latent_tensor)
    > Latent('torch:Tensor', *([1, 2, 3],), **{})

    # ... some time later
    tensor = latent_tensor()
    print(tensor)
    > tensor([1., 2., 3.])
    ```

    This can also extend to graphs of objects:
    ```
    data = dict(
        total = Latent("torch:sum", Latent("torch:Tensor", [1 ,2, 3]))
    )
    print(data)
    > {'total': Latent('torch:sum', *(Latent('torch:Tensor', *([1, 2, 3],), **{}),), **{})}

    Latent.materialize(data)
    print(data)
    > {'total': tensor(6.)}
    ```

    In addion to modules within the current system path, you can specify the target
    symbol via a path to a Python file:
    ```
    latent = Latent("../../aiws/utils.py:format_mapping", dict(foo="bar", baz=2.0))
    print(latent())
    > foo: bar
    > baz: 2.0
    ```

    You can restrict which types of objects can be materialized:
    ```
    whitelist = set((
        "torch:Tensor",
    ))
    print(latent_tensor(whitelist=whitelist))
    # Success!
    > tensor([1., 2., 3.])

    latent = Latent("../../aiws/utils.py:format_mapping", dict(foo="bar", baz=2.0))
    print(latent(whitelist=whitelist))
    # Failure!
    > LatentException: The following dynamic imports were not found in the whitelist: {'/home/dinalt/ai_assets/aiworkshop/aiws/utils.py:format_mapping'}

    # Alternatively...
    invalid_set = validate_whitelist(data, whitelist)
    if len(invalid_set):
        # Show all disallowed types in the graph
        print(f"Disallowed: {invalid_set}")
    > Disallowed: {'torch:sum'}
    ```

    After an object has been materialized, subsequent 'materializations' return the
    same object instance:
    ```
    latent_tensor = Latent("torch:Tensor", [1 ,2, 3])
    tensor = latent_tensor()
    assert(id(tensor) == id(latent_tensor()))
    ```

    Finally, you can inject arguments at the point of materialization:
    ```
    import torch
    deferred_sum = Latent("torch:sum", Latent("sum_input"))
    deferred_sum(sum_input=lambda: torch.tensor([1 ,2, 3]))
    > tensor(6)
    ```

    How is this useful?

    A fair question. The primary intended use-case is for safely constructing objects from a
    configuration file. Consider the case where a configuration file may define objects which
    can take a considerable amount of time to construct (i.e. processing a dataset).

    In this case, its useful to allow the complete file to be parsed before attempting a
    lengthy task, as there may still be errors present which will cause the operation to
    abort. It's much better to first fully parse the file, validate the safety of
    the all the types, and only then then, materialize the definiton. This is far less
    painful than having to fix a single error, wait for the long operation to complete (again)
    and then hit another error. Fun times...

    Allowing deferal can also avoid materializing expensive objects which are not needed, as per
    runtime logic. For example, a definition may define several datasets, where-as only a single
    one is actually selected, contingent upon 'whatever.'

    If an object is never materialized, this also avoids loading the associated modules.

    Finally, this allows one two lazilly construct objects in whatever order makes sense.

    Background:

    This project started as a 'Lazy' object implementation, where the Lazy objects were
    specified as they are now, but would self-materialize on their first non-trivial access.

    For example, getting an attribute or using the subscript operator.

    When this occurred, the Lazy object would "transmute" into the real one by replacing the
    object __dict__ and __class__ with those of the newly instantiated object.

    Superficially, this worked. The problem is that there are many corner cases where it does not.
    An example is when a Lazy object was passed into something which checks its type with
    "isinstance()," before rejecting it.

    This led to a deep-dive down the rabbit hole of the Python Data Model:
    https://docs.python.org/3/reference/datamodel.html

    For lack of clarity, I even ended up diving into the 'C' code, which constitues
    the reference version of Python.

    There are two main issues:
    1. Not all Python objects have the same representation internally, thus making it impossible
    to just replace the __dict__ and __class__ values. This does not work for many internal types
    or for types which make uses of the "slots" feature.

    2. Many Python operations bypass attribute lookup, so it's not possible to intercept everything
    from one central place. For example, when implementing the __getattribute__ method, Python
    bypasses calling it for performance reasons. These can only be intercepted by defining every
    corresponding dunder method at the class level.

    The first problem can be solved by 'wrapping' the object, should transmutation fail. This is
    not ideal, as it adds an additional layers of indirection to most accesses and consequent to
    issue 2, the only way to intercept every possible access is to override every conveivable
    dunder method at the class level.

    While in theory this is possible, the list of all possible dunder methods (which can bypass
    ___getattribute__) it long, poorly documented, and a moving target. You can see where this
    has led with the 'wrapd' Python package.

    All of this ammounts to a big mantainance nightmare using a fragile implementation which
    depends upon poorly documented internal implementation details to function properly.

    So now the objects are Latent, rather than Lazy. Rather than becomming the materialized object,
    they return it. The code-base is far smaller and easier to understand and maintain.

    And speaking of __slots__ causing incompatible internal object layout issues, now that I am
    aware of this esoteric feature...
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

    def __call__(self, *, whitelist: Container = None, **mapping):
        """
        Alias for calling materialize() on self
        """
        return Latent.materialize(self, whitelist=whitelist, **mapping)

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
