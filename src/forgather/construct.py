from typing import Sequence, Mapping, Hashable, Sized, Any


# A simple constructor which simply gets the key from the object.
def get_item(obj: Mapping, key: Hashable):
    return obj[key]


def get_attr(obj, attribute):
    return getattr(obj, attribute)


def flatten(*args):
    return [x for xx in args for x in xx]


def values(obj: Mapping):
    return list(obj.values())


def keys(obj: Mapping):
    return list(obj.keys())


def length(obj: Sized):
    return len(obj)


def items(obj: Mapping):
    return list(obj.items())


def method_call(obj: Any, method_name: str, *args, **kwargs):
    method = getattr(obj, method_name)
    return method(*args, **kwargs)
