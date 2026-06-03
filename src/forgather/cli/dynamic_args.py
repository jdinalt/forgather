def parse_dynamic_args(parser, global_args):
    if global_args.no_dyn:
        return
    import argparse
    import os
    import sys
    import traceback

    from forgather import MetaConfig, Project

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
            # Render config-derived (dynamic) args under their own argparse
            # group(s), created here (after the command's native args) so they
            # appear AFTER the native flags in --help and clearly mark where the
            # config args begin. Sub-group by each arg's optional ``group``
            # metadata (the same hierarchical label the webui uses, e.g.
            # ``Trainer:Schedule``) so long arg lists are easier to scan.
            config_label = global_args.config_template or "config"
            _arg_groups = {}

            def _arg_target(group):
                key = group or ""
                if key not in _arg_groups:
                    title = (
                        f"Config arguments — {group}"
                        if group
                        else f"Config arguments (from {config_label})"
                    )
                    _arg_groups[key] = parser.add_argument_group(title)
                return _arg_groups[key]

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

                # Webui-only metadata: argparse rejects unknown kwargs. ``required``
                # is intentionally NOT forwarded — argparse-required would break
                # ``pp`` and other read-only actions where the user hasn't filled
                # in the value yet. Required is enforced at action time instead
                # (see require_dynamic_args). ``min`` / ``max`` likewise are
                # webui constraints; the action-time check (validate_dynamic_arg_bounds)
                # is the canonical CLI enforcement point.
                group = dynamic_arg.pop("group", None)
                dynamic_arg.pop("required", None)
                dynamic_arg.pop("min", None)
                dynamic_arg.pop("max", None)

                # Handle type conversion for string-based types
                if "type" in dynamic_arg and isinstance(dynamic_arg["type"], str):
                    dynamic_arg["type"] = _convert_type_string(dynamic_arg["type"])

                try:
                    _arg_target(group).add_argument(
                        *names,
                        **dynamic_arg,
                    )
                except argparse.ArgumentError as exc:
                    # A config dynamic arg whose option string collides with a
                    # built-in flag of this subcommand (e.g. `forgather diloco
                    # worker` defines --resume-workers / --count / --server,
                    # which a config's dynamic_args might also use), or a
                    # genuinely malformed option string. Skip just that arg —
                    # the built-in takes precedence — instead of aborting the
                    # whole dynamic-arg load. Don't add it to dynamic_arg_names
                    # either (it's not a dynamic dest here). Note it so a
                    # malformed config isn't dropped completely silently.
                    print(
                        f"Note: skipping dynamic arg {names}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                # Track the dynamic argument name (use the long form if available)
                for name in names:
                    if name.startswith("--"):
                        # Convert --max-steps to max_steps (argparse destination format)
                        dynamic_arg_names.append(name[2:].replace("-", "_"))
                        break
                else:
                    # No long form, use short form
                    if names and names[0].startswith("-"):
                        dynamic_arg_names.append(names[0][1:])
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


def validate_dynamic_arg_bounds(project_dir, config_template, dynamic_args):
    """Return a list of human-readable bound-violation messages.

    Only checks args the user actually supplied (``dynamic_args`` is the
    post-filter dict from ``get_dynamic_args``). Empty list = clean.
    Loads the schema the same way ``required_dynamic_arg_dests`` does and
    returns silently on any schema-load failure so the action's own error
    path surfaces the underlying problem.
    """
    from forgather import MetaConfig, Project

    try:
        project_dir = MetaConfig.find_project_dir(project_dir)
        proj = Project(config_name=config_template, project_dir=project_dir)
        if "dynamic_args" not in proj.config:
            return []
        schema = proj("dynamic_args")
    except Exception:
        return []
    out = []
    for entry in schema or []:
        if not isinstance(entry, dict):
            continue
        type_str = entry.get("type")
        if type_str not in ("int", "float"):
            continue
        lo = entry.get("min")
        hi = entry.get("max")
        if lo is None and hi is None:
            continue
        names = entry.get("names")
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not names:
            continue
        long_form = next((n for n in names if n.startswith("--")), names[0])
        dest = long_form.lstrip("-").replace("-", "_")
        if dest not in dynamic_args:
            continue
        v = dynamic_args[dest]
        if isinstance(v, bool) or v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if isinstance(lo, (int, float)) and fv < lo:
            out.append(f"{long_form} >= {lo}")
        if isinstance(hi, (int, float)) and fv > hi:
            out.append(f"{long_form} <= {hi}")
    return out


def required_dynamic_arg_dests(project_dir, config_template):
    """Return the dests of dynamic args marked ``required: true``.

    Resolves the schema the same way ``parse_dynamic_args`` does, but skips
    argparse construction. Used by action commands (e.g. ``forgather train``)
    to refuse to launch when a required arg has no user-supplied value.
    Returns an empty list on any failure so the action's own error path
    surfaces the underlying problem instead of a confusing required-arg
    error.
    """
    from forgather import MetaConfig, Project

    try:
        project_dir = MetaConfig.find_project_dir(project_dir)
        proj = Project(config_name=config_template, project_dir=project_dir)
        if "dynamic_args" not in proj.config:
            return []
        schema = proj("dynamic_args")
    except Exception:
        return []
    out = []
    for entry in schema or []:
        if not isinstance(entry, dict):
            continue
        if not entry.get("required"):
            continue
        names = entry.get("names")
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not names:
            continue
        long_form = next((n for n in names if n.startswith("--")), names[0])
        dest = long_form.lstrip("-").replace("-", "_")
        out.append(dest)
    return out


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
