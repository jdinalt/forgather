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
