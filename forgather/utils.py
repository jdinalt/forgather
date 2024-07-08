from enum import Enum

class ConversionDescriptor:
    def __init__(self, cls, *, default):
        self._cls = cls
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        setattr(obj, self._name, self._cls(value))

class DiagnosticEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}; choose one of {cls._value2member_map_.keys()}"
        )