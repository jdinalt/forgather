def parse_dynamic_args(parser, global_args):
    if global_args.no_dyn:
        return
    from forgather import Project, MetaConfig
    import traceback
    import os

    def _convert_type_string(type_str):
        """Convert type string to appropriate callable for argparse.

        Args:
            type_str: String representation of the type (e.g., 'int', 'str', 'float', 'path')

        Returns:
            Callable that can be used as argparse type parameter
        """
        if type_str == "int":
            return int
        elif type_str == "str":
            return str
        elif type_str == "float":
            return float
        elif type_str == "bool":
            # For bool, we'll use a custom converter that handles common boolean strings
            def bool_converter(value):
                if isinstance(value, bool):
                    return value
                if value.lower() in ("true", "1", "yes", "on"):
                    return True
                elif value.lower() in ("false", "0", "no", "off"):
                    return False
                else:
                    raise ValueError(f"Invalid boolean value: {value}")

            return bool_converter
        elif type_str == "path":
            # For path type, use os.path.expanduser to handle ~ expansion
            return os.path.expanduser
        else:
            # Unknown type string - return as-is and let argparse handle it
            return type_str

    dynamic_arg_names = []
    try:
        global_args.project_dir = MetaConfig.find_project_dir(global_args.project_dir)
        proj = Project(
            config_name=global_args.config_template,
            project_dir=global_args.project_dir,
        )
        dynamic_args = []

        if "dynamic_args" in proj.config:
            dynamic_args = proj("dynamic_args")
        if dynamic_args:
            for dynamic_arg in dynamic_args:
                # The names in add_args() are positional only
                # To simplify the interface, we just support a single name
                names = dynamic_arg.pop("names")
                if isinstance(names, str):
                    names = [names]
                else:
                    assert isinstance(
                        names, list
                    ), "names must be either str or list[str]"

                # Remove 'default' from kwargs to let template defaults take precedence
                # This prevents argparse from setting unspecified args to a default value
                dynamic_arg.pop("default", None)

                # Handle type conversion for string-based types
                if "type" in dynamic_arg and isinstance(dynamic_arg["type"], str):
                    dynamic_arg["type"] = _convert_type_string(dynamic_arg["type"])

                parser.add_argument(
                    *names,
                    **dynamic_arg,
                )
                # Track the dynamic argument name (use the long form if available)
                for name in names:
                    if name.startswith("--"):
                        # Convert --max-steps to max_steps (argparse destination format)
                        dynamic_arg_names.append(name[2:].replace("-", "_"))
                        break
                else:
                    # No long form, use short form
                    if arg_names and arg_names[0].startswith("-"):
                        dynamic_arg_names.append(arg_names[0][1:])
    except:
        print("Loading dynamic args failed!")
        traceback.print_exc()

    # Attach dynamic arg names to the parser for later use
    parser._dynamic_arg_names = dynamic_arg_names


def partition_args(args_namespace, dynamic_arg_names):
    """Partition parsed arguments into built-in and dynamic arguments.

    Args:
        args_namespace: The parsed arguments namespace
        dynamic_arg_names: List of dynamic argument names (in argparse format, e.g., ['max_steps'])

    Returns:
        tuple: (built_in_args_dict, dynamic_args_dict)
    """
    built_in_args = {}
    dynamic_args = {}

    for key, value in vars(args_namespace).items():
        if key in dynamic_arg_names:
            dynamic_args[key] = value
        else:
            built_in_args[key] = value

    return built_in_args, dynamic_args


def get_dynamic_args(args, filter_none=True):
    """Extract dynamic arguments from the parsed args namespace.

    This function handles the common pattern where CLI arguments should override
    template defaults, but unspecified CLI arguments should fall back to template
    defaults rather than argparse defaults.

    Args:
        args: The parsed arguments namespace (from parse_args)
        filter_none: If True, remove arguments with None values (default: True).
                    This allows template defaults (e.g., {{ max_steps | default(100) }})
                    to take precedence when arguments are not specified.

    Returns:
        dict: Dictionary of dynamic arguments and their values

    Example:
        # In your command implementation:
        dynamic_args = get_dynamic_args(args)
        # Pass to template: max_steps={{ max_steps | default(-1) }}
        # - If --max-steps 500 provided: max_steps gets 500
        # - If --max-steps not provided: max_steps gets -1 (template default)
    """
    dynamic_args = getattr(args, "_dynamic_args", {})

    if filter_none:
        # Filter out None values to let template defaults take precedence
        return {k: v for k, v in dynamic_args.items() if v is not None}

    return dynamic_args
