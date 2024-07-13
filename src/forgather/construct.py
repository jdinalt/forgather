from typing import Sequence, Mapping, Hashable


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


def items(obj: Mapping):
    return list(obj.items())
