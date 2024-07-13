class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, name):
        return self[name]

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value
