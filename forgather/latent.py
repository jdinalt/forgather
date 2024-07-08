from typing import Any, Callable, Union, Container, Set
from collections.abc import MutableSequence, MutableMapping

from pprint import pformat

from forgather.dynamic import (
    dynamic_import,
    normalize_import_spec,
)

class LatentException(Exception):
    pass

class Latent:
    class MappingNode:
        """
        An indirection descriptor object which provides an abstraction
        for updating the reference held by the parent ojbect (a Sequence or Mapping).
        """
        class IndirectDescriptor:
            def __get__(self, obj, type=None):
                if obj is None:
                    return None
                mapping = getattr(obj, "mapping")
                key = getattr(obj, "key")
                return mapping.__getitem__(key)
                
            def __set__(self, obj, value):
                mapping = getattr(obj, "mapping")
                key = getattr(obj, "key")
                mapping.__setitem__(key, value)
                
            def __delete__(self, obj):
                mapping = getattr(obj, "mapping")
                key = getattr(obj, "key")
                mapping.__delitem__(key)
        
        child = IndirectDescriptor()
    
        def __init__(self, mapping, key):
            self.mapping = mapping
            self.key = key
    
        def __str__(self):
            return f"{type(self.mapping).__name__}[{key}] -> {self.mapping[self.key]}"
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
    __slots__ = ("constructor", "args", "kwargs", "obj")

    def __init__(self, constructor: Callable | str, /, *args, **kwargs):
        assert isinstance(constructor, str) or isinstance(constructor, Callable)
        self.constructor = constructor
        # Args are initially a tuple. We need it to be mutable.
        self.args = list(args)
        self.kwargs = kwargs
        self.obj = None

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.constructor)}, *{repr(self.args)}, **{repr(self.kwargs)})"

    def __iter__(self) -> MappingNode:
        """
        Iteration over latent yields all nodes but self in a depth first traversal.

        See Latent.generate(), as this has limited utility on its own.
        """
        for node in Latent.generate(self.args):
            yield node
        for node in Latent.generate(self.kwargs):
            yield node

    def __call__(self, *, whitelist: Container=None, **mapping):
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
    def materialize(obj: Any, *, whitelist: Container=None, **mapping):
        if whitelist is not None:
            invalid_set = Latent.validate_whitelist(obj, whitelist)
            if len(invalid_set):
                raise LatentException(
                    f"The following dynamic imports were not found in the whitelist: {pformat(invalid_set)}")
        Latent._resolve_standins(obj, **mapping)
        Latent._resolve_dynamic_imports(obj)
        return Latent._construct_all(obj)
        
    """
    Walk the graph, checking each constructor against the whitelist, and return the set of disallowed objects.
    """
    @staticmethod
    def validate_whitelist(obj: Any, whitelist: Container) -> Set[str]:
        invalid_set = set()
        for node in Latent.generate([obj]):
            latent = node.child
            # If not plausibly an import spec, skip it
            if not isinstance(latent.constructor, str) or not ':' in latent.constructor:
                continue
            import_spec = normalize_import_spec(latent.constructor)
            if import_spec not in whitelist:
                invalid_set.add(import_spec)
        return invalid_set

    """
    Return a generator for walking the graph of Latent objects.

    The yielded type, Latent.MappingNode, provides the indirection required modify the 
    references of the parent node.

    Example: Replace all notes with their repr() string...
    ```
    container = [ 1, Latent("a"), { "foo": Latent("b"), "bar": Latent("c", Latent("d")) } ]
    print(container)
    > [1, Latent('a', *[], **{}), {'foo': Latent('b', *[], **{}), 'bar': Latent('c', *[Latent('d', *[], **{})], **{})}]
    for node in Latent.generate(container):
        node.child = repr(node.child)
    print(container)
    > [1, "Latent('a', *[], **{})", {'foo': "Latent('b', *[], **{})", 'bar': 'Latent(\'c\', *["Latent(\'d\', *[], **{})"], **{})'}]
    ```
    """
    @staticmethod
    def generate(obj) -> MappingNode:
        """
        Construct a generator for walking a graph of Latent objects

        This performs a depth first traversal of the graph.
        """
        if isinstance(obj, MutableSequence):
            generator = enumerate(obj)
        elif isinstance(obj, MutableMapping):
            generator = obj.items()
        else:
            return
        for key, value in generator:
            if isinstance(value, Latent):
                for node in value:
                    yield node
                yield Latent.MappingNode(obj, key)
            else:
                for node in Latent.generate(value):
                    yield node

    def _construct(self):
        """
        If the object has not been constructed, construct it

        Note: This is not recursive and it will not work if all of the constructors have not been resolved first.
        """
        if self.obj is not None:
            return self.obj
        if not isinstance(self.constructor, Callable):
            raise LatentException(
                f"Constructor must be Callable, but found [{type(self.constructor)}] {self.constructor}"
                + "; see resolve_dynamic_imports() and resolve_standins()")
        
        self.obj = self.constructor(*self.args, **self.kwargs)
        return self.obj 

    @staticmethod
    def _resolve_dynamic_imports(obj: Any):
        """
        Try to resolve all dynamic imports, replacing the constructor string with the 
        corresponding Callable.
        """
        for node in Latent.generate([obj]):
            latent = node.child
            # If not plausibly an import spec, skip it
            if not isinstance(latent.constructor, str) or not ':' in latent.constructor:
                continue
            constructor = dynamic_import(latent.constructor)
            if not isinstance(constructor, Callable):
                raise LatentException(
                    f"Imported constructor is not Callable: [{type(self.constructor)}] {self.constructor}")
            latent.constructor = constructor

    @staticmethod
    def _resolve_standins(obj: Any, **mapping):
        """
        Replace all stand-ins with the corresonding Callables from the mapping.
        """
        for node in Latent.generate([obj]):
            latent = node.child
            if not isinstance(latent.constructor, str) or ':' in latent.constructor:
                continue
            try:
                value = mapping[latent.constructor]
            except KeyError:
                raise LatentException(f"Key '{latent.constructor}' was not found in mapping.")
            if not isinstance(value, Callable):
                raise LatentException(
                    f"Mapping for key '{latent.constructor}' is not a Callable: [{type(value)}] {value}")
            latent.constructor = value

    @staticmethod
    def _construct_all(obj: Any):
        """
        Recursively construct all Latent objects in the graph

        This assumes that all constructors have been fully resolved.
        """
        indirect_obj = [obj]
        for node in Latent.generate(indirect_obj):
            node.child = node.child._construct()
        return indirect_obj[0]
